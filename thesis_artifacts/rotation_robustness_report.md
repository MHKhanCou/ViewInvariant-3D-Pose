# Rotation Robustness Investigation

## Test Setup

Input: Sample frame from `person.mp4` (basketball player)
Rotations tested: 0°, 90°, 180°, 270°

Pipeline stages inspected:
1. Original RGB image
2. YOLOv8 2D detection
3. Normalized 2D keypoints
4. H36M joint conversion
5. MotionAGFormer input tensor
6. MotionAGFormer output (raw 3D joints)
7. Canonical body-frame output
8. Final visualization

---

## Results

| Rotation | Detection | Canonical | Max Error | Raw Extent |
|----------|-----------|-----------|-----------|------------|
| 0° | Success | Valid | 1.04 | 1.30 |
| 90° | **FAILED** | N/A | N/A | N/A |
| 180° | Success | Valid | 2.00 | 1.25 |
| 270° | Success | Valid | 1.46 | 1.44 |

---

## Analysis

### Where the failure occurs:

**90° rotation**: Fails at **YOLOv8 detection** (Stage 2).

YOLOv8-pose is trained on upright human images. When the image is rotated 90°, the person appears sideways, and YOLOv8 fails to detect any person. The pipeline produces zero-filled keypoints, which propagate through to zero 3D output.

### Why this happens:

1. **YOLOv8 detection**: Trained on upright images. Rotated inputs may not be detected.
2. **Keypoint normalization**: Depends on detection succeeding. If detection fails, normalization produces zeros.
3. **MotionAGFormer**: Trained on upright human poses. Even if detection succeeds, rotated inputs produce unreliable 3D predictions.
4. **Canonicalization**: Works correctly when given valid input, but cannot fix garbage input from failed detection.

### Is this expected?

**Yes.** This is a limitation of the underlying model, not an implementation bug:

- MotionAGFormer was trained on **upright Human3.6M poses**
- The training data does not include rotated images
- YOLOv8-pose is trained on **upright person detection**
- Neither model was designed to handle arbitrary image rotations

### Would supporting arbitrary rotation require changes?

**Yes**, to one or more of:
- **Preprocessing**: Auto-detect and correct image orientation before processing
- **Data augmentation**: Train on rotated images (requires retraining, not possible with frozen baseline)
- **Architecture changes**: Add rotation-invariant features (not possible with frozen baseline)

### Recommendation:

For the thesis, document this as a **known limitation** of the current pipeline. The demo assumes upright input images, which is standard for most practical applications (photos, videos are typically upright).

---

## Intermediate Outputs

All intermediate outputs saved to `thesis_artifacts/rotation_test/`:
- `rot_000_01_original.png` through `rot_000_06_final.png`
- `rot_090_01_original.png` through `rot_090_06_final.png`
- `rot_180_01_original.png` through `rot_180_06_final.png`
- `rot_270_01_original.png` through `rot_270_06_final.png`
- `rotation_summary.txt`
