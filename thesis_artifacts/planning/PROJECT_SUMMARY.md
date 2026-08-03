# Project Summary — View-Invariant 3D Human Pose Estimation

## Project Context

Final-year CS undergraduate thesis: **View-Invariant 3D Human Pose Estimation from RGB Images**.

- **Model:** MotionAGFormer-XS (WACV 2024), 2.2M params, 27-frame window
- **Dataset:** Human3.6M (MPJPE 45.15 mm, P-MPJPE 36.89 mm)
- **Environment:** Python 3.11, PyTorch 2.13, CPU-only
- **Repository:** github.com/MHKhanCou/ViewInvariant-3D-Pose

## Current Thesis Direction (Examiner-Approved)

> A lightweight, training-free, reliability-aware geometric canonicalization framework for cross-view comparability of frozen monocular 3D pose predictions.

**Not competing with MotionAGFormer.** Solving a different problem:
- Input: frozen 3D pose prediction from any monocular estimator
- Output: reliability-scored canonical representation
- Evaluation: cross-view consistency on synchronized multi-camera data

## What Exists Today

### Engineering (Strong)
- Reproduced MotionAGFormer-XS baseline with <0.1mm match
- Complete RGB → YOLOv8 → H36M → MotionAGFormer → 3D pose pipeline
- 50-image benchmark with full stage-by-stage diagnostics
- Interactive Plotly 3D viewer with coordinate space switching
- Canonical body-frame module (36 tests passing)
- Cross-view evaluation infrastructure on MPI-INF-3DHP

### Scientific (Weak — needs strengthening)
- Geometric canonical body-frame: Gram-Schmidt orthonormalization of torso+hip axes
- Cross-view evaluation on 1 subject, 2 cameras, 50 frames (too thin)

### Fixed Bugs (as of latest session)
1. **Flip augmentation bug (FIXED)**: Code was flipping OUTPUT after prediction instead of INPUT before prediction. Codex fixed: flip input → model → un-flip output → average. Matches official eval protocol.
2. **Double camera_to_world (FIXED)**: Commit 24be19b.
3. **Double normalization in canonical mode (FIXED)**: `lift_sequence()` already returns [0,1] normalized poses; canonical mode was applying extra normalization.
4. **Make_frame height mismatch (FIXED)**: Explicit resize before hstack.
5. **Rotation keypoint remapping (FIXED)**: `detect_with_rotation()` remaps keypoints for 90/180/270 rotations.

### Remaining Known Issues
1. **No `evaluation/` directory** — evaluation code is scattered
2. **No reliability score exists** — the `valid` flag is binary, not continuous
3. **MPI data is limited** — only S1/Seq1 has RGB frames from 2 cameras (50 frames each). But `annot.mat` has GT 3D for all 6416 frames × 14 cameras.
4. **Canonicalizer state leak** — `mpi_cross_view.py` calls canonicalizer on pred_a then pred_b without `reset()`.
5. **Base MPI model swap not done** — needs proper 81-frame pipeline, not ad-hoc swap
6. **MPI cross-view claims INVALID** — "67% reduction" and "40% reduction" are exploratory diagnostics, not calibrated experiments

### Codex Changes (Uncommitted)
- `demo_live/lifter.py`: `apply_display_postprocess` flag + `build_base_model()`
- `backend/inference.py`: Fixed flip augmentation + canonical path uses apply_display_postprocess=False
- These changes are correct and should be committed (with user approval per D13)

## Reference Systems Compared

| Aspect | Ours | MotionAGFormer Official | KelvinHong | MocapNET |
|--------|------|------------------------|------------|----------|
| 2D Detector | YOLOv8-nano | YOLOv3+HRNet | AlphaPose | MobileNet/OpenPose |
| Lifter | XS (2.2M, 27f) | Base (9.8M, 243f) | MotionBert-lite | SNN ensemble |
| Post-processing | None (frozen) | None | Blender rigging | IK refinement |
| Output | 3D joints (17,3) | 3D joints (17,3) | BVH rotations | BVH rotations |
| Training-free | Yes | Yes | No | No |

## Key Files

| File | Purpose |
|------|---------|
| `canonical/body_frame.py` | Core canonicalization (Gram-Schmidt) |
| `canonical/canonicalizer.py` | Stateful wrapper with temporal consistency |
| `canonical/metrics.py` | Cross-view consistency, bone-length stability |
| `canonical/mpi_eval.py` | MPI-INF-3DHP data loader |
| `demo_live/lifter.py` | MotionAGFormer inference + post-processing (has apply_display_postprocess) |
| `demo_live/pose_detector.py` | YOLOv8 detection + rotation robustness |
| `backend/inference.py` | App inference (estimate_poses, predict_video) — flip fix applied |
| `backend/model_loader.py` | Singleton model loading (XS model) |
| `demo_live/plotly_renderer.py` | Interactive 3D viewer |
| `app.py` | Gradio web application |
| `scripts/mpi_cross_view.py` | Cross-view evaluation script (has state leak bug) |
| `scripts/image_pipeline_diagnostic.py` | Benchmark diagnostic pipeline |

## Session History

- **Baseline reproduction:** MPJPE 45.15mm confirmed
- **Canonical module:** 36 tests, synthetic rotation validation (max error 1.19e-07)
- **MPI-INF-3DHP evaluation:** Cross-view consistency measured (40% improvement — exploratory, not thesis evidence)
- **Engineering fixes:** make_frame height, rotation remapping, flip augmentation, double-normalization
- **Milestone 1:** 50-image benchmark, all stages pass
- **Milestone 2:** Interactive Plotly 3D viewer, coordinate space selector
- **Examiner review:** Thesis scientifically weak, needs reliability-aware canonicalization + proper evaluation
- **Codex session:** Fixed flip augmentation, added apply_display_postprocess flag, attempted Base MPI swap (reverted)
