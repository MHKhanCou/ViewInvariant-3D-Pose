# Canonical Pose Representation

A lightweight, deterministic body-frame view normalization for 3D human pose estimation.

## What This Module Does

Given root-relative 3D pose predictions, this module constructs a **body-fixed orthonormal coordinate system** and re-expresses all joints in that canonical frame. This normalizes camera-orientation variation when body axes are reliably estimated, while preserving intra-body structure.

## Coordinate Systems

### Root-relative (input)
The model outputs 3D joint positions relative to the pelvis (joint 0 at origin). Global translation is removed, but **camera-view-dependent orientation remains** — the same action looks different from different cameras.

### Camera-relative (visualization)
The fixed quaternion `camera_to_world()` rotates the pose for display. This is a visualization convenience, not a calibrated world recovery.

### Canonical body-frame (this module)
Constructs axes aligned with the body itself:
- **y-axis**: body vertical (upper torso - root)
- **x-axis**: hip-to-hip horizontal (left hip - right hip)
- **z-axis**: forward (cross product of x and y)

All joints are projected into this frame, producing a representation that normalizes camera-orientation variation when body axes are reliably estimated.

## Mathematical Convention

```
P_rel = P - P[0]              # root subtraction (defensive)
y_body = normalize(P[8] - P[0])  # vertical axis
x_raw = P[1] - P[4]           # hip horizontal
z_body = normalize(x_raw × y_body)  # forward
x_body = normalize(y_body × z_body)  # right (re-orthogonalized)
R = [x_body | y_body | z_body]       # rotation matrix (columns)
P_canonical = P_rel @ R              # row-vector convention
```

## Validation

### Mathematical validation (synthetic tests)
The canonical module is validated through synthetic rigid-rotation invariance tests:
- Create a valid synthetic human pose
- Rotate it by multiple random 3D rotations
- Canonicalize both original and rotated poses
- Confirm they are approximately equal
- Report both raw error and relative error (error / pose spatial extent)
- Verify orthonormality: R^T R = I
- Verify degenerate handling: zero pose returns zero output, no NaN

These tests are in `canonical/test_canonical.py` (17 tests, all passing).

### Cross-view evaluation status
Exact cross-view correspondence and metric 3D ground-truth semantics could not be verified from the available preprocessed pkl fields. Specifically:

- Source strings lack temporal timestamps suitable for frame-level pairing
- Source strings do not uniquely identify cameras in this preprocessed pkl
- The GT field `joints_2.5d_image` has coordinate semantics that could not be verified as metric 3D

This is detected by `canonical/metadata_capability.py` and verified by `canonical/test_evaluator.py` (tests, all passing).

Note: Human3.6M in its raw form may support cross-view evaluation with different preprocessing. The limitation identified here is specific to this preprocessed pkl's metadata fields.

## What Canonicalization Can and Cannot Solve

**Can:**
- Normalize camera-orientation variation when body axes are reliably estimated
- Provide a consistent coordinate system for visualization
- Reduce viewpoint-dependent variation in predicted poses

**Cannot:**
- Correct detector errors (bad 2D keypoints → bad 3D predictions)
- Fix self-occlusion ambiguity
- Resolve depth ambiguity from monocular input
- Recover absolute world-space trajectory
- Replace learned view-disentangled representations (e.g., MoViD)

## Metrics

### Canonical MPJPE
Mean joint position error after independent canonicalization of prediction and ground truth. **Not directly comparable** to official H36M P1 MPJPE because canonicalization applies independent rotations.

### Cross-View Consistency Error
Canonicalizes predictions from two cameras independently, then computes mean joint distance. Lower values indicate more consistent predictions across viewpoints. **Not computable** with current pkl metadata (see validation section).

### Bone-Length Stability
Compares predicted bone lengths across cross-view pairs. Bone lengths are invariant to rotation, so this measures whether the two views produce consistent skeletal proportions.

## Running Tests

### Canonical module tests (17 tests, should all PASS)
```bash
python -m unittest canonical.test_canonical -v
```

### Evaluator capability-detection tests (should all PASS)
```bash
python -m unittest canonical.test_evaluator -v
```

## Limitations

- **Detector errors**: The demo uses YOLOv8-pose on in-the-wild images, which differs from the Stacked-Hourglass detections used during training.
- **Self-occlusion**: Occluded joints produce unreliable 3D predictions that canonicalization cannot fix.
- **Depth ambiguity**: Monocular depth estimation is inherently ambiguous; canonicalization normalizes orientation but not scale.
- **Degenerate poses**: When the body is nearly horizontal or the torso is collapsed, the canonical frame may be poorly defined (handled gracefully with zero output).
- **Cross-view pairing**: Could not be verified from the available preprocessed pkl fields (see validation section).
- **Blender/avatar rendering**: Qualitative only, not a metric improvement.
