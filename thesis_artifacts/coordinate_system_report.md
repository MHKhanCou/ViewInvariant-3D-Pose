# Coordinate System Analysis

## 1. What Coordinate System Does MotionAGFormer Predict?

MotionAGFormer is trained to predict **root-relative 3D joint positions** in a model-internal coordinate system.

### Training setup (from `configs/h36m/MotionAGFormer-xsmall.yaml`):
```yaml
root_rel: True  # Normalizing joints relative to the root joint
```

### What this means:
- Input: 2D keypoints in normalized screen coordinates `[-1, 1]`
- Output: 3D joint positions where **joint 0 (pelvis/root) is at the origin**
- The model learns to map 2D screen-space positions to 3D root-relative positions

### The coordinate system is:
- **Root-relative**: joint 0 is always at `[0, 0, 0]`
- **Model-internal**: the frame is defined by the training data (H36M camera space with root subtracted)
- **NOT camera coordinates**: the model output is not in any specific camera's coordinate system
- **NOT world coordinates**: no global translation is recovered
- **NOT dataset-normalized**: the output is in the model's learned representation space

### Evidence from code:

**Training** (`train.py` line 112-113):
```python
if args.root_rel:
    predicted_3d_pos[:, :, 0, :] = 0  # Zero root joint
```

**Preprocessing** (`data/reader/h36m.py` lines 60-74):
- 3D ground truth is normalized by camera resolution: `xy / res_w * 2 - [1, res_h/res_w]`
- This maps 3D positions to approximately `[-1, 1]` range
- The model learns this normalized representation

**Model output** (`model/MotionAGFormer.py` lines 266-284):
- Input: `[B, T, 17, 3]` (normalized 2D + confidence)
- Output: `[B, T, 17, 3]` (3D joint positions in model frame)

---

## 2. Why Are Predictions From Different Cameras Directly Comparable?

### The key insight:

MotionAGFormer takes **2D keypoints as input** and outputs **3D positions in a fixed model frame**. The model does NOT know which camera produced the 2D input. It maps 2D screen-space positions to 3D root-relative positions in its learned representation.

### Therefore:
- Camera A's 2D input → model output in model frame
- Camera B's 2D input → model output in model frame
- Both outputs are in the **same coordinate system** (the model's frame)
- No camera extrinsic transform is needed for comparison

### Why camera extrinsics are wrong here:

The camera extrinsics transform between **world coordinates** and **camera coordinates**. But the model output is NOT in camera coordinates — it's in the model's internal frame. Applying different extrinsics to different predictions rotates them by different amounts, which is incorrect.

---

## 3. Relationship to H36M Preprocessing and root_rel=True

### H36M preprocessing:
- 2D keypoints are normalized to `[-1, 1]` by camera resolution
- 3D ground truth is normalized similarly
- Camera names identify which camera produced each frame

### root_rel=True:
- During training, the root joint position is subtracted from all joints
- The model learns to predict root-relative 3D positions
- At evaluation, `predicted_3d_pos[:, :, 0, :] = 0` zeros the root

### Result:
The model's output space is:
1. Root-relative (root at origin)
2. In the model's learned representation (not in any camera's frame)
3. Similar in scale to the normalized training data

---

## 4. Why camera_to_world() Is Used in Demo but Not Evaluation

### In the demo (`demo/vis.py` lines 262-267):
```python
rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
post_out = camera_to_world(post_out, R=rot, t=0)
post_out[:, 2] -= np.min(post_out[:, 2])
post_out /= max_value
```

### Purpose: **Visualization only**
- Applies a **fixed quaternion rotation** (same for all frames)
- Transforms from model output coordinates to a **display-friendly orientation**
- The quaternion is hardcoded, not calibrated to any specific camera
- No translation is applied (t=0)
- The result is then normalized to `[0, 1]` for matplotlib rendering

### Why not in evaluation:
- The official benchmark (`train.py`) compares predictions directly in the model frame
- MPJPE is computed on denormalized predictions without any rotation
- camera_to_world() would add an arbitrary rotation that doesn't correspond to any real camera

---

## 5. Why CanonicalPoseNormalizer Is Still Meaningful

### The problem:
Even though predictions are in the model's frame, the **same pose viewed from different cameras produces different 2D inputs**, which lead to **different 3D predictions** with different orientations.

### Why:
- Camera A sees the person from angle α
- Camera B sees the person from angle β
- The model maps these different 2D views to 3D poses with different orientations
- Both are in the model's frame, but oriented differently

### What canonicalization does:
1. Constructs a body-fixed frame from torso and hip axes
2. Rotates the pose into this canonical orientation
3. This **removes the orientation difference** caused by different viewing angles

### Why it works:
- The body axes (torso vertical, hip horizontal) are **invariant to camera viewpoint**
- A person walking looks the same from any camera in terms of body axis orientation
- Canonicalization aligns these axes, making predictions more comparable

### Mathematical definition:
```
P_canonical = P_rel @ R
```
where R is the rotation matrix from Gram-Schmidt orthogonalization of body axes.

This is a **pure rotation** — it does not change translation (root is already at origin) or scale.

---

## 6. Diagrams

### Model coordinate system:
```
Input: 2D keypoints (screen space)
    ↓
MotionAGFormer (trained with root_rel=True)
    ↓
Output: 3D joints (model frame, root at origin)
```

### Cross-view comparison:
```
Camera A → YOLOv8 → MotionAGFormer → P_A (model frame)
Camera B → YOLOv8 → MotionAGFormer → P_B (model frame)

Both in SAME frame → compare directly (no extrinsic needed)
```

### Canonical comparison:
```
Camera A → ... → P_A → canonicalize(P_A) → P_A_canonical
Camera B → ... → P_B → canonicalize(P_B) → P_B_canonical

Compare P_A_canonical vs P_B_canonical
```

---

## 7. Code References

| Concept | File | Line(s) |
|---------|------|---------|
| root_rel training | `configs/h36m/MotionAGFormer-xsmall.yaml` | 40 |
| Root zeroing in eval | `train.py` | 112-113 |
| camera_to_world in demo | `demo/vis.py` | 262-267 |
| Normalization by camera res | `data/reader/h36m.py` | 29-44 |
| Denormalization | `data/reader/h36m.py` | 133-147 |
| Model forward pass | `model/MotionAGFormer.py` | 266-284 |
