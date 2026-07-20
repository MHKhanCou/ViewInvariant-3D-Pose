# Final Report: Canonical Pose Contribution

## Date: 2026-07-20

---

## 1. Files Changed/Created

### Created (canonical module)
| File | Purpose |
|------|---------|
| `canonical/__init__.py` | Package exports |
| `canonical/body_frame.py` | Core canonicalization (Gram-Schmidt, temporal consistency) |
| `canonical/canonicalizer.py` | Stateful wrapper with reset |
| `canonical/metrics.py` | canonical_mpjpe, cross_view_consistency_error, bone_length_stability |
| `canonical/visualization.py` | Equal-axis 3D rendering |
| `canonical/test_canonical.py` | 18 geometry unit tests |
| `canonical/test_evaluator.py` | 18 capability-detection tests |
| `canonical/metadata_capability.py` | Cross-view metadata inspection |
| `canonical/README.md` | Documentation |

### Created (thesis artifacts)
| File | Purpose |
|------|---------|
| `thesis_artifacts/baseline_results.json` | Official and reproduced baseline metrics |
| `thesis_artifacts/canonical_validation.json` | Synthetic test results |
| `thesis_artifacts/canonical_validation_report.md` | Validation report |
| `thesis_artifacts/baseline_evaluation.log` | Baseline evaluation output |
| `thesis_artifacts/figures/*.png` | Qualitative visualization panels |
| `thesis_artifacts/figures/README.md` | Figure documentation |

### Modified (web demo)
| File | Change |
|------|--------|
| `backend/inference.py` | Added canonical mode |
| `app.py` | Added visualization mode selector |

### NOT modified (baseline preserved)
- `model/` — untouched
- `configs/` — untouched
- `checkpoint/` — untouched
- `train.py` — only compatibility fix (pkg_resources)
- `train_3dhp.py` — untouched

---

## 2. Test Results

### Canonical module tests (18 tests)
```
Ran 18 tests in 0.186s — OK
```

All tests pass:
- Shape: single frame and batch
- Root: canonical root is zero
- Orthonormality: R^T R = I
- Rotation invariance: 100 random rotations with normalized relative error
- Degenerate: zero pose returns zero, identity R, valid=False
- **Collinear rejection**: torso/hip axes parallel → rejected with valid=False
- Temporal: repeated frames don't flip
- Metrics: canonical_mpjpe, cross_view_consistency, bone_length_stability
- Bone lengths: 16 bones, all positive

### Evaluator tests (18 tests)
```
Ran 18 tests in 0.981s — OK
```

All tests pass:
- **Fixture: H36M two-camera same-timestamp** — exact H36M format, different cameras, same timestamp → can_pair=True
- **Fixture: valid** — synchronized cameras with timestamps allow pairing
- **Fixture: valid GT** — explicit provenance + plausible depth enables floor
- **Fixture: valid GT without provenance** — floor NOT enabled
- **Fixture: missing timestamp** — detected correctly
- **Fixture: invalid GT** — image-plane detected
- **Fixture: single camera** — no pairing possible
- **Fixture: H36M format** — parser handles all formats correctly
- **Real pkl: prediction** — timestamps missing, can't pair
- **Real pkl: GT** — depth plausible but no provenance, floor NOT enabled
- **Report structure** — all required keys present

---

## 3. Baseline Confirmation

| Metric | Official | Reproduced | Difference |
|--------|----------|------------|------------|
| MPJPE | 45.1 mm | 45.149 mm | 0.049 mm |
| P-MPJPE | 36.9 mm | 36.892 mm | 0.008 mm |

Baseline is frozen and untouched.

---

## 4. Artifacts Created

| Artifact | Description |
|----------|-------------|
| `baseline_results.json` | Model, dataset, checkpoint, metrics, command |
| `canonical_validation.json` | 100 rotations, max error 1.19e-07, orthonormality 0 |
| `canonical_validation_report.md` | Thesis-ready validation report |
| `baseline_evaluation.log` | Evaluation output with verified results |
| `figures/` | 4 images × 2 views (camera-relative + canonical) |

---

## 5. Thesis Wording

### Method Section

> We propose a lightweight body-frame view normalization as a deterministic geometric post-processing step. Given root-relative 3D pose predictions from MotionAGFormer-XS, we construct a body-fixed coordinate system aligned with the torso vertical and hip horizontal axes, then express all joints in this canonical frame.
>
> The canonical frame is defined by:
> - Vertical axis: upper torso minus root joint
> - Horizontal axis: left hip minus right hip
> - Forward axis: cross product of horizontal and vertical
>
> A Gram-Schmidt orthonormalization produces a rotation matrix R, and the canonical pose is computed as P_canonical = P_rel @ R. This normalizes camera-orientation variation when body axes are reliably estimated, while preserving intra-body structure.

### Results Section

> The canonical module is validated through synthetic rigid-rotation invariance tests. Given a synthetic human pose, we apply 100 random 3D rotations and canonicalize both the original and rotated poses. The maximum raw synthetic-coordinate error is 1.19e-07, and the maximum relative error (error divided by pose spatial extent) is 5.96e-08, both within floating-point precision, confirming mathematical correctness. The rotation matrix satisfies R^T R = I exactly. Collinear torso/hip configurations are correctly rejected. Degenerate inputs (zero pose) produce zero output with identity rotation matrix.
>
> Exact cross-view correspondence and metric 3D ground-truth semantics could not be verified from the available preprocessed pkl fields. The evaluator correctly identifies that source strings lack temporal timestamps, that depth plausibility alone cannot prove metric 3D units, and that explicit coordinate provenance is required before enabling a GT canonical consistency floor. Human3.6M in its raw form may support cross-view evaluation with different preprocessing; the limitation identified here is specific to this preprocessed pkl's metadata fields.
>
> The canonical mode is available in the web demo for qualitative visualization.

### Limitations Section

> The canonical representation normalizes camera-orientation variation under the assumption that body axes are reliably estimated. It cannot correct detector errors, fix self-occlusion ambiguity, resolve depth ambiguity from monocular input, or recover absolute world-space trajectory. The demo uses YOLOv8-pose on in-the-wild images, which introduces domain shift from the Stacked-Hourglass detections used during training. Exact cross-view quantitative evaluation was not possible with the available preprocessed pkl metadata, though Human3.6M in its raw form may support such evaluation with different preprocessing.

---

## 6. Explicit Confirmation

- **No unsupported cross-view metric is included.** Cross-view prediction metrics are explicitly marked as unavailable.
- **No fabricated results.** All numbers come from verified tests or the official baseline.
- **Baseline is untouched.** The canonical module, web demo, and thesis artifacts do not modify the training code, model architecture, checkpoint, or official evaluation.
- **Blender/avatar rendering is NOT implemented.** It remains qualitative only if added later.
