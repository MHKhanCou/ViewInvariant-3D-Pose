# Why Canonical Body-Frame Normalization Improves Cross-View Consistency

## The Problem

MotionAGFormer predicts root-relative 3D poses in a learned model frame. When the same action is viewed from different cameras, the model produces predictions with **different global orientations** in this frame.

## Why Different Orientations?

1. Camera A sees the person from angle α → model predicts pose oriented at angle α
2. Camera B sees the same person from angle β → model predicts pose oriented at angle β
3. Both predictions are in the model's frame, but with different orientations

This orientation difference is **not** from translation or scale — it's from how the 2D input maps to 3D output through the learned representation.

## How Canonicalization Helps

Canonicalization constructs a **body-fixed coordinate frame**:
- y-axis: torso vertical (invariant to camera angle)
- x-axis: hip-to-hip horizontal (invariant to camera angle)

By rotating all predictions into this body frame, we **remove the orientation variance** caused by different viewing angles.

## What Does NOT Cause the Improvement

| Factor | Why it's not the cause |
|--------|----------------------|
| Translation removal | Root is already subtracted before both raw and canonical comparisons |
| Scale normalization | Canonicalization is a rotation; it doesn't change scale |
| Camera calibration | No camera parameters are used in the canonical path |
| Retraining | The model is frozen; canonicalization is post-processing |

## What DOES Cause the Improvement

The improvement comes from **removing orientation differences** between cameras:

- Before canonicalization: Camera A's prediction is oriented at angle α, Camera B's at angle β
- After canonicalization: Both predictions are oriented with the body's own axes
- The 40% reduction (0.15 → 0.09) measures how much of the cross-view discrepancy was due to orientation differences

## Experimental Evidence

From MPI-INF-3DHP cross-view evaluation (S1/Seq1, cameras 0 and 1, 50 frames):

| Metric | Raw | Canonical | Change |
|--------|-----|-----------|--------|
| Cross-view distance | 0.15 ± 0.01 | 0.09 ± 0.01 | -40% |

The improvement is:
- Consistent across all 50 frame pairs (low variance)
- Not from translation removal (root already subtracted)
- Not from scale normalization (rotation preserves scale)
- From removing orientation differences between camera views
