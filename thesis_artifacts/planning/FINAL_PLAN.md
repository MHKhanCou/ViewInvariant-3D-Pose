# Revised Plan — Reliability-Aware Geometric Canonicalization

## Thesis Title

> A Lightweight, Training-Free, Reliability-Aware Geometric Canonicalization Framework for Cross-View Comparability of Frozen Monocular 3D Pose Predictions.

---

## Current Status: Phases 1-3 Complete, Phase 4 Pending

### Completed

| Phase | What | Status | Evidence |
|-------|------|--------|----------|
| **1. Evaluation Foundation** | Raw output extraction, flip augmentation fix, canonical routing | **DONE** | 61/61 tests pass |
| **2. Reliability Score** | 6 components, hard geometric gates, SO(3) temporal stability | **DONE** | 13/13 reliability tests pass |
| **3. Cross-View Evaluation** | MPI-INF-3DHP S1/Seq1, 50 frame pairs, raw + canonical comparison | **DONE** | 28.4% improvement, 0 hard failures |

### Not Yet Done

| Phase | What | Priority |
|-------|------|----------|
| **4. Coverage-Error Curves** | Coverage vs error at different reliability thresholds | P0 |
| **5. Expanded Evaluation** | More diverse images, hard failure cases | P0 |
| **6. Thesis Writing** | Figures, tables, limitations, defense preparation | P1 |
| **7. Presentation** | Interactive viewer, defense demo | P1 |

---

## Phase 1: Evaluation Foundation (COMPLETE)

### What Was Built

1. **`apply_display_postprocess` flag** in `lift_sequence()` — when False, returns raw root-relative pose without camera_to_world, z-shift, or global normalization.

2. **Flip augmentation fix** in `lift_sequence()` — was flipping the OUTPUT (wrong). Now correctly feeds flipped INPUT to the model and un-flips the OUTPUT before averaging.

3. **`raw_root_relative` key** in `estimate_poses()` — returns the true model output before any display transforms.

4. **Canonical routing fixed** — both image and video canonical paths now pass `apply_display_postprocess=False`, receiving clean raw root-relative output.

5. **`motionagformer_display_pose` key** renamed from misleading `camera_pose`. This is a display convention (root-zero + camera_to_world quaternion + normalization), not a physical camera-coordinate pose.

### Files Changed
- `demo_live/lifter.py` — flip augmentation fix, apply_display_postprocess flag, build_base_model (81 frames)
- `backend/inference.py` — raw_root_relative key, display pose rename, canonical video routing fix
- `backend/model_loader.py` — unchanged (loads XS)
- `scripts/mpi_cross_view.py` — uses use_flip=True + apply_display_postprocess=False

---

## Phase 2: Reliability Score (COMPLETE)

### Components

| # | Component | What It Measures | Normalization |
|---|-----------|-----------------|---------------|
| 1 | Axis conditioning | Torso + hip axis lengths relative to median bone length | Too short (<0.1x) or long (>3x) penalized |
| 2 | Torso-hip angle | sin(angle between torso and hip axes) | 0 = collinear, 1 = orthogonal |
| 3 | Bilateral symmetry | Left/right bone-length ratio (6 pairs) | Mean × (0.5 + 0.5 × min) |
| 4 | Abnormal bone ratio | Fraction of bones within 0.3x-2.5x median | 1 - outlier_count/16 |
| 5 | Detector confidence | Minimum YOLO confidence on 7 key joints (COCO-17) | Key joints: nose, shoulders, hips, knees |
| 6 | Temporal stability | Geodesic angle on SO(3) between consecutive canonical rotations | Uses canonicalize_single() for proper Gram-Schmidt |

**Final score**: Geometric mean of all 6 components. Higher = more reliable.

### Hard Geometric Gates

Before checking the continuous score, `has_hard_geometric_failure()` rejects frames with:
- Torso axis < 5% of median bone length
- Hip axis < 5% of median bone length
- Axes nearly collinear (cos > 0.95)
- Majority of bones degenerate

### Validation Results

| Test Case | Expected | Score | Hard Gate | Pass |
|-----------|----------|-------|-----------|------|
| Good standing pose | Accept | 0.96 | No | Yes |
| Collapsed body (all zeros) | Reject | 0.00 | torso_axis_too_short | Yes |
| Collinear axes | Reject | — | axes_nearly_collinear | Yes |
| Near-zero hip width | Reject | 0.47 | hip_axis_too_short | Yes |
| Near-zero torso | Reject | 0.41 | torso_axis_too_short | Yes |
| Bone outlier (5x) | Reduce score | 0.83 | No | Yes |
| Asymmetry (5x) | Reduce score | 0.85 | No | Yes |
| Low confidence | Reduce score | lower | No | Yes |
| Score range [0,1] | Always valid | — | — | Yes |

### Test Coverage
- 61 total tests (18 canonical + 18 evaluator + 12 inference + 13 reliability validation)
- All passing

---

## Phase 3: Cross-View Evaluation (COMPLETE)

### Protocol

- **Dataset**: MPI-INF-3DHP S1/Seq1
- **Cameras**: cam0 (camera 0) and cam1 (camera 1)
- **Frame pairs**: 50 synchronized frame pairs
- **Inference**: YOLOv8 + MotionAGFormer-XS, flip augmentation, raw root-relative output
- **Canonicalization**: Gram-Schmidt body-frame via canonicalize_single()
- **Metrics**: cross_view_joint_distance (properly named, NOT MPJPE)

### Results

| Metric | Value |
|--------|-------|
| **Raw cross-view distance** | 0.1172 |
| **Canonical cross-view distance** | **0.0839** |
| **Improvement** | **28.4%** |
| Bone-length deviation | 0.0349 |
| Joint-angle difference | 48.1 degrees |
| Reliability (mean) | 0.868 |
| Reliability (min) | 0.853 |
| Hard failures | 0/100 (0%) |
| Coverage at all thresholds | 100% |

### Key Finding
Canonicalization reduces cross-view inconsistency by 28.4% on synchronized MPI-INF-3DHP data. No hard failures in this clean lab dataset.

### Limitations to Report
- Only 1 subject, 1 sequence, 2 cameras (50 frames)
- MPI-INF-3DHP is a clean lab dataset — hard gates never trigger
- Single-frame repetition (not true temporal windows)
- Need expanded evaluation for generalizability

---

## Phase 4: Coverage-Error Curves (NOT YET DONE)

### What's Needed
- Run evaluation at different reliability thresholds
- For each threshold, report: coverage (% frames retained) vs error (cross-view distance)
- Generate the coverage-error curve figure
- Report rejection rate and failure reasons

### Why It Matters
Shows that higher reliability threshold = lower cross-view error (but fewer frames). This is the core evidence for "reliability-aware abstention identifies structurally unsafe cases."

---

## Phase 5: Expanded Evaluation (NOT YET DONE)

### What's Needed
- Test on more diverse images (examples folder has 22 images)
- Generate hard failure cases that trigger the hard gates
- Document failure modes (occlusion, extreme poses, bad detection)
- Add at least one held-out sequence or camera pair

### Why It Matters
The current 0% hard failure rate is because MPI-INF-3DHP is clean. Need real-world images with degenerate cases to validate the hard gates actually work.

---

## Phase 6: Thesis Writing (NOT YET DONE)

### Required Figures
1. Architecture diagram (EXISTS at thesis_artifacts/architecture.md)
2. Cross-view improvement table (raw vs canonical)
3. Coverage-error curve
4. Qualitative examples (success + failure cases)
5. Side-by-side comparison

### Required Sections
1. Algorithm description
2. Reliability metric definition + mathematical formulation
3. Dataset audit (subject, sequence, camera, frame table)
4. Experimental protocol
5. Cross-view evaluation results
6. Coverage-error analysis
7. Failure analysis + limitations
8. Comparison with recent literature (MoViD, 3DPCNet, PoseIRM, etc.)

---

## What This Plan Does NOT Change

- MotionAGFormer model (frozen)
- H36M benchmark evaluation (frozen)
- Canonical body-frame algorithm (unchanged)
- Interactive 3D viewer (keep as-is)

---

## Rules

1. Never push/commit without asking user first
2. Never claim "improved" until experiments prove it (28.4% is measured, not claimed)
3. Never claim "model-agnostic" — only tested on MotionAGFormer-XS
4. Never use camera_to_world, z-shift, or display scaling in quantitative evaluation
5. Always call prediction-vs-prediction metric "cross-view joint distance"
6. Always report coverage alongside error when using abstention
7. Hard gates catch structural failures; continuous score handles subtle degradation — keep them separate
8. Scores_2d in reliability.py is COCO-17 order only

---

## What This Plan Does NOT Assume

- The 28.4% improvement is sufficient for the thesis (supervisor decides)
- The 0.5 threshold is optimal (needs calibration on harder data)
- The evaluation protocol is complete (only 1 subject, 1 sequence, 2 cameras)
- Canonicalization works for all poses (known limitation for extreme poses)
