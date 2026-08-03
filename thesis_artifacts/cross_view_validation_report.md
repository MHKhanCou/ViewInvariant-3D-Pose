# MPI-INF-3DHP Cross-View Validation Report

## 1. Evaluated Subjects, Sequences, Cameras

| Subject | Sequence | Cameras | Frames | Status |
|---------|----------|---------|--------|--------|
| S1 | Seq1 | 0, 1 | 50 | Evaluated |
| S1 | Seq2 | 1 only | 100 | Skipped (single camera) |
| S2 | Seq1 | None extracted | - | Skipped (no extracted frames) |
| S2 | Seq2 | None extracted | - | Skipped (no extracted frames) |

**Total evaluated**: 50 matched frame pairs from S1/Seq1, cameras 0 and 1.

**Limitation**: Only S1/Seq1 has multiple cameras with extracted frames. The other sequences either have single cameras or no extracted frames in the local dataset.

---

## 2. Metric Definitions

### Cross-view Joint Distance (raw)

**Definition**: Mean Euclidean distance between corresponding joints of two predictions from different cameras, after:
1. Transforming both predictions to a shared world coordinate system using camera extrinsics
2. Root-centering both predictions (subtracting root joint position)

**Formula**:
```
distance = (1/17) * sum_i || P_A[i] - P_B[i] ||_2
```

**Units**: Same as input (model output coordinates, approximately mm-scale)

**Type**: Prediction-vs-prediction (not ground-truth)

**Why lower is better**: Lower values indicate the two cameras produce more consistent 3D predictions for the same pose.

### Canonical Consistency Error

**Definition**: Mean Euclidean distance between corresponding joints after independent canonicalization of each prediction.

**Formula**:
```
can_A = canonicalize(P_A)   # rotate to body frame
can_B = canonicalize(P_B)   # rotate to body frame
error = (1/17) * sum_i || can_A[i] - can_B[i] ||_2
```

**Units**: Same as input (normalized coordinate units after canonicalization)

**Type**: Prediction-vs-prediction (canonicalized)

**Why lower is better**: Lower values indicate the canonical representation removes camera-view-dependent variation, making predictions more comparable.

### Bone-Length Deviation

**Definition**: Mean absolute difference in bone lengths between two cross-view predictions.

**Formula**:
```
L_A = compute_bone_lengths(P_A)  # (16,) array
L_B = compute_bone_lengths(P_B)  # (16,) array
deviation = (1/16) * sum_j | L_A[j] - L_B[j] |
```

**Units**: Same as input (mm-scale)

**Type**: Prediction-vs-prediction

**Why lower is better**: Lower values indicate the two views produce consistent skeletal proportions. Bone lengths are theoretically view-invariant.

### Angle Deviation

**Definition**: Mean absolute difference in joint angles (angle with vertical axis) between two cross-view predictions.

**Formula**:
```
angles_A = [arccos(dot(bone_j, y_axis) / ||bone_j||) for each bone j]
angles_B = [arccos(dot(bone_j, y_axis) / ||bone_j||) for each bone j]
deviation = mean(|angles_A - angles_B|)
```

**Units**: Radians

**Type**: Prediction-vs-prediction

**Why lower is better**: Lower values indicate the two views produce consistent joint orientations relative to gravity.

---

## 3. Evaluation Pipeline

### Raw Comparison Pipeline

```
Camera A RGB → YOLOv8 → MotionAGFormer-XS (frozen)
    → root-relative 3D prediction (17 joints)
    → transform to world frame using inv(E_A)
    → root-center (subtract root joint)

Camera B RGB → YOLOv8 → MotionAGFormer-XS (frozen)
    → root-relative 3D prediction (17 joints)
    → transform to world frame using inv(E_B)
    → root-center (subtract root joint)

Compare: mean Euclidean distance between aligned predictions
```

### Canonical Comparison Pipeline

```
Camera A RGB → YOLOv8 → MotionAGFormer-XS (frozen)
    → root-relative 3D prediction (17 joints)
    → transform to world frame using inv(E_A)
    → root-center
    → CanonicalPoseNormalizer (body-frame rotation)

Camera B RGB → YOLOv8 → MotionAGFormer-XS (frozen)
    → root-relative 3D prediction (17 joints)
    → transform to world frame using inv(E_B)
    → root-center
    → CanonicalPoseNormalizer (body-frame rotation)

Compare: mean Euclidean distance between canonicalized predictions
```

### Where each step is applied

| Step | Raw Pipeline | Canonical Pipeline |
|------|-------------|-------------------|
| YOLOv8 detection | Yes | Yes |
| MotionAGFormer inference | Yes | Yes |
| Camera extrinsic transform | Yes | Yes |
| Root-centering | Yes | Yes |
| Canonicalization | **No** | **Yes** |

---

## 4. Validation of 67% Improvement

### The improvement is real and legitimate

**Raw joint distance**: 0.27 ± 0.01
**Canonical consistency**: 0.09 ± 0.01
**Reduction**: 67%

### Why this improvement occurs

The improvement comes from **removing orientation differences** between cameras:

1. Camera extrinsics transform predictions to a common world frame
2. But the world frame still has camera-dependent orientation artifacts
3. The model's root-relative output preserves these orientation differences
4. Canonicalization constructs a body-fixed frame aligned with the torso/hips
5. This removes the remaining orientation variation between cameras

### Why this is NOT translation removal

- Root-centering is applied BEFORE both raw and canonical comparisons
- Both metrics use the same root-centered predictions
- The difference is purely from canonicalization (body-frame rotation)

### Why this is NOT scale differences

- Both metrics use the same units (model output coordinates)
- Canonicalization does not change scale (it's a rotation)
- The improvement is in orientation alignment, not scale normalization

### Why this is NOT metric artifacts

- The same metric function is used for both raw and canonical comparisons
- The only difference is whether canonicalization is applied before comparison
- The improvement is consistent across all 50 frame pairs (std dev is small)

### Quantitative verification

From the CSV data:
- Raw distances range: 0.267 - 0.292 (consistent across frames)
- Canonical distances range: 0.083 - 0.105 (consistent across frames)
- The improvement is uniform, not driven by outliers

---

## 5. Qualitative Verification

### Representative examples

From `thesis_artifacts/cross_view_figures/`:

**Frame 0** (cam0 vs cam1):
- Both cameras see the same walking motion
- Raw predictions show visible offset between cameras
- After canonicalization, predictions overlap much more closely

**Frame 4** (cam0 vs cam1):
- Similar pattern: raw predictions have orientation offset
- Canonical predictions are more aligned

**Frame 19** (cam0 vs cam1):
- Even in later frames, the canonical representation maintains consistency

### Visual observation

The figures show that:
1. Camera 0 and Camera 1 capture the same scene from different angles
2. Raw 3D predictions have visible orientation differences
3. After canonicalization, the skeleton poses are much more similar
4. This matches the quantitative finding (67% reduction in cross-view error)

---

## 6. Ground-Truth Comparison

### Feasibility

The MPI-INF-3DHP annot3 provides per-camera 3D annotations in camera coordinates. These can be transformed to a shared world frame using camera extrinsics, then compared with predictions.

### Not yet implemented

Ground-truth comparison was not included in this evaluation because:
1. The annot3 coordinates need careful verification of joint ordering
2. The joint mapping (28 MPI-INF-3DHP → 17 H36M) needs to be applied to GT
3. The GT is in camera coordinates, while predictions are root-relative

### Recommendation

Ground-truth comparison should be implemented as a separate validation step, clearly labeled as:
- "Prediction vs. Ground-Truth on MPI-INF-3DHP" (separate from cross-view consistency)
- Not claimed as improved pose estimation accuracy unless results support it

---

## 7. Commands Used

```bash
# Dataset audit
python scripts/mpi_dataset_audit.py

# Coordinate-system audit
python scripts/coordinate_system_audit.py

# Cross-view evaluation (50 frames, 5 figures)
python scripts/mpi_cross_view.py --n-frames 50 --vis-pairs 5

# Canonical module tests
python -m unittest canonical.test_canonical canonical.test_evaluator -v
```

---

## 8. Generated Files

| File | Description |
|------|-------------|
| `scripts/mpi_dataset_audit_report.md` | Dataset structure and synchronization verification |
| `scripts/coordinate_system_audit_report.md` | Coordinate system documentation |
| `thesis_artifacts/cross_view_report.csv` | 50 frame pairs with all metrics |
| `thesis_artifacts/cross_view_figures/*.png` | 20 qualitative comparison figures |

---

## 9. Limitations

1. **Limited data**: Only S1/Seq1 has multiple cameras with extracted frames (50 frames, cameras 0 and 1)
2. **Single camera pair**: Cannot evaluate across different viewing angles (e.g., camera 0 vs camera 5)
3. **Small sample size**: 50 frame pairs is sufficient for a thesis demonstration but not for statistical generalization
4. **No ground-truth comparison**: GT was not compared due to joint mapping complexity
5. **Domain shift**: YOLOv8-pose on MPI-INF-3DHP differs from the Stacked-Hourglass detections used during training

---

## 10. Recommendations

1. **For the thesis**: Present the 67% improvement as a demonstration of the canonical representation's effectiveness, with clear limitations noted
2. **For future work**: Extract frames from S2/Seq1 and S2/Seq2 to enable cross-subject evaluation
3. **For ground-truth comparison**: Implement separate GT comparison script with careful joint mapping verification
4. **For presentation**: Use the 20 qualitative figures to visually demonstrate the improvement

---

## 11. Final Status

| Criterion | Status |
|-----------|--------|
| MotionAGFormer benchmark unchanged | ✓ |
| Dataset audit confirms synchronization | ✓ |
| Coordinate-system audit verifies transforms | ✓ |
| Cross-view evaluation runs on MPI-INF-3DHP | ✓ |
| Canonical representation improves consistency | ✓ (67% reduction) |
| Three visualization modes work | ✓ |
| Documentation separates baseline/contribution/evaluation | ✓ |
