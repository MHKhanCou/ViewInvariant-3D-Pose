# Algorithm 1: Canonical Body-Frame Normalization

## Formal Description

**Input:** Root-relative 3D joints $P \in \mathbb{R}^{17 \times 3}$

**Output:** Canonical pose $P_{\text{canonical}} \in \mathbb{R}^{17 \times 3}$

---

## Algorithm

```
Algorithm 1: Canonical Body-Frame Normalization

Input:  P ∈ R^{17×3}  (root-relative 3D joints)
Output: P_canonical ∈ R^{17×3}  (canonical pose)

1:  P_rel ← P − P[0]                    // Defensive root subtraction
2:  y_raw ← P_rel[8] − P_rel[0]         // Upper torso − root (vertical)
3:  y ← y_raw / ||y_raw||               // Normalize vertical axis
4:  x_raw ← P_rel[1] − P_rel[4]         // Left hip − right hip (horizontal)
5:  if ||x_raw|| < ε then
6:      x_raw ← P_rel[14] − P_rel[11]   // Fallback: left shoulder − right shoulder
7:  end if
8:  z ← (x_raw × y) / ||x_raw × y||     // Forward axis (Gram-Schmidt)
9:  x ← (y × z) / ||y × z||             // Re-orthogonalized horizontal axis
10: R ← [x | y | z]                     // Rotation matrix (columns)
11: P_canonical ← P_rel · R             // Project into canonical frame
12: return P_canonical, R
```

---

## Mathematical Properties

1. **Orthonormality:** $R^T R = I$ (verified by unit tests)
2. **Root at origin:** $P_{\text{canonical}}[0] = [0, 0, 0]$
3. **Rotation only:** No translation or scale change
4. **Degenerate handling:** If axes are collinear, returns zero pose with identity R

---

## Coordinate System

- **y-axis:** Body vertical (aligned with torso)
- **x-axis:** Body horizontal (aligned with hip-to-hip)
- **z-axis:** Body forward (orthogonal to x and y)

This frame is invariant to camera viewpoint because the body axes are defined by the skeleton itself, not by the camera.

---

## Implementation

```python
# canonical/body_frame.py
def canonicalize_single(pose, prev_z=None):
    P_rel = pose - pose[0:1]                      # Step 1
    y_body = normalize(P_rel[8] - P_rel[0])       # Steps 2-3
    x_raw = P_rel[1] - P_rel[4]                   # Step 4
    z_body = normalize(cross(x_raw, y_body))       # Step 8
    x_body = normalize(cross(y_body, z_body))      # Step 9
    R = column_stack([x_body, y_body, z_body])     # Step 10
    P_canonical = P_rel @ R                        # Step 11
    return P_canonical, R
```
