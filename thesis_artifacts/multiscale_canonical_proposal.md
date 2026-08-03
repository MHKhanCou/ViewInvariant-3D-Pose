# Technical Proposal: Multi-Scale Canonical Body-Frame Representation

## 1. Limitation of Current Algorithm

The current canonical body-frame uses a **single global coordinate frame** constructed from:
- y-axis: torso vertical (upper torso − root)
- x-axis: hip horizontal (left hip − right hip)

This captures the **dominant body orientation** but misses **local limb orientations** that differ from the global frame.

### Concrete Example

Consider a person walking with right arm extended sideways:
- Global canonical: aligns torso vertical and hip horizontal
- Problem: the right arm is oriented differently from the torso
- Result: arm orientation varies across cameras even after global canonicalization

### Evidence from Current Results

The residual cross-view distance after canonicalization is **0.09** (down from 0.15). This residual likely comes from:
1. Limb orientation variations not captured by global frame
2. Model prediction noise
3. Detection differences between cameras

### The 40% Improvement Ceiling

The current global canonicalization removes orientation variance that is **aligned with the torso**. Limb orientation variance that **differs from the torso** remains. Multi-scale canonicalization targets this remaining variance.

---

## 2. Supporting Literature

### 3DPCNet (Ekanayake et al., 2025)
- Uses GCN + Transformer to predict 6D rotation for pose canonicalization
- Demonstrates that **per-joint** and **per-limb** features improve canonicalization
- Reduces rotation error from 20° to 3.4° on MM-Fi dataset
- Key insight: hybrid encoder fuses local skeletal features with global context

**Our approach differs:** We use geometric (no training) vs. their learned approach.

### MoViD (Liu et al., 2026)
- Disentangles motion and view features
- View estimator models key joint relationships
- Demonstrates that **different body parts have different view sensitivities**

**Our approach differs:** We use post-processing canonicalization vs. their end-to-end disentanglement.

### V-VIPE (Levy & Shrivastava, 2024)
- VAE-based embedding in canonical coordinate space
- Shows that **multi-scale representations** improve view-invariant retrieval

**Relevance:** Supports the hypothesis that multi-scale canonicalization captures more view-invariant information.

### Key Takeaway from Literature

All successful view-invariant methods use some form of **hierarchical or multi-scale representation**:
- 3DPCNet: GCN (local) + Transformer (global) fusion
- MoViD: per-joint view estimation
- V-VIPE: multi-scale embedding

Our geometric approach can incorporate multi-scale canonicalization without training.

---

## 3. Mathematical Formulation

### Current (Single-Scale)

Given root-relative pose P ∈ R^{17×3}:

```
y_global = normalize(P[8] - P[0])     # torso vertical
x_global = normalize(P[1] - P[4])     # hip horizontal
z_global = cross(x_global, y_global)  # forward
R_global = [x_global | y_global | z_global]
P_canonical = P_rel @ R_global
```

### Proposed (Multi-Scale)

Given root-relative pose P ∈ R^{17×3}, construct K canonical representations:

**Level 0: Global body frame** (existing)
```
y_0 = normalize(P[8] - P[0])    # torso vertical
x_0 = normalize(P[1] - P[4])    # hip horizontal
R_0 = gram_schmidt(x_0, y_0)
P_canonical_0 = P_rel @ R_0
```

**Level 1: Torso frame**
```
P_torso = P[7:11]  # center_torso, upper_torso, neck, head
y_1 = normalize(P[10] - P[7])   # head - center_torso
x_1 = normalize(P[14] - P[11])  # left_shoulder - right_shoulder
R_1 = gram_schmidt(x_1, y_1)
P_torso_canonical = (P - P[7:1]) @ R_1  # relative to center_torso
```

**Level 2: Left arm frame**
```
P_arm = P[14:17]  # left_shoulder, left_elbow, left_hand
y_2 = normalize(P[14] - P[8])   # shoulder - upper_torso (arm direction)
x_2 = normalize(P[15] - P[14])  # elbow - shoulder (arm orientation)
R_2 = gram_schmidt(x_2, y_2)
P_arm_canonical = (P[14:17] - P[14:1]) @ R_2
```

**Level 3: Right arm frame** (symmetric to left)

**Level 4: Left leg frame**
```
P_leg = P[1:4]  # left_hip, left_knee, left_foot
y_3 = normalize(P[1] - P[0])    # hip - root (leg direction)
x_3 = normalize(P[2] - P[1])    # knee - hip (leg orientation)
R_3 = gram_schmidt(x_3, y_3)
P_leg_canonical = (P[1:4] - P[1:1]) @ R_3
```

**Level 5: Right leg frame** (symmetric to left)

### Combined Representation

The multi-scale canonical pose is a concatenation:

```
P_multiscale = [P_canonical_0; P_torso_canonical; P_arm_canonical; P_leg_canonical]
```

Or, for cross-view comparison, use the **residual** between global and local canonicalizations:

```
ΔP = P_canonical_0 - P_local_canonical
```

This captures how much each limb deviates from the global body orientation.

---

## 4. Algorithm and Pseudocode

```
Algorithm 2: Multi-Scale Canonical Body-Frame Normalization

Input:  P ∈ R^{17×3}  (root-relative 3D joints)
Output: Multi-scale canonical representation

1:  // Level 0: Global body frame (existing)
2:  P_rel ← P − P[0]
3:  R_0 ← canonicalize_frame(P_rel, joints=[0,1,4,8])
4:  P_0 ← P_rel @ R_0

5:  // Level 1: Torso frame
6:  P_torso_rel ← P[7:11] − P[7:8]
7:  R_1 ← canonicalize_frame(P_torso_rel, joints=[7,10,11,14])
8:  P_1 ← P_torso_rel @ R_1

9:  // Level 2: Left arm frame
10: P_arm_rel ← P[14:17] − P[14:15]
11: R_2 ← canonicalize_frame(P_arm_rel, joints=[14,15])
12: P_2 ← P_arm_rel @ R_2

13: // Level 3: Right arm frame
14: P_arm_rel ← P[11:14] − P[11:12]
15: R_3 ← canonicalize_frame(P_arm_rel, joints=[11,12])
16: P_3 ← P_arm_rel @ R_3

17: // Level 4: Left leg frame
18: P_leg_rel ← P[1:4] − P[1:2]
19: R_4 ← canonicalize_frame(P_leg_rel, joints=[1,2])
20: P_4 ← P_leg_rel @ R_4

21: // Level 5: Right leg frame
22: P_leg_rel ← P[4:7] − P[4:5]
23: R_5 ← canonicalize_frame(P_leg_rel, joints=[4,5])
24: P_5 ← P_leg_rel @ R_5

25: // Combine
26: P_multiscale ← [P_0; P_1; P_2; P_3; P_4; P_5]

27: return P_multiscale, {R_0, R_1, R_2, R_3, R_4, R_5}


Function canonicalize_frame(P, joints):
    y ← normalize(P[joints[1]] − P[joints[0]])
    x_raw ← P[joints[2]] − P[joints[3]]  (or fallback)
    z ← normalize(x_raw × y)
    x ← normalize(y × z)
    R ← [x | y | z]
    return R
```

---

## 5. Integration into Existing Pipeline

### File Structure
```
canonical/
├── body_frame.py          # Existing (unchanged)
├── multiscale.py          # NEW: Multi-scale canonicalization
├── canonicalizer.py       # MODIFY: Add multiscale option
├── metrics.py             # MODIFY: Add multiscale metrics
├── visualization.py      # MODIFY: Add multiscale visualization
├── test_canonical.py      # MODIFY: Add multiscale tests
```

### Integration Points

1. **`canonical/multiscale.py`**: New file implementing Algorithm 2
2. **`canonical/canonicalizer.py`**: Add `mode="multiscale"` option
3. **`canonical/metrics.py`**: Add `multiscale_cross_view_consistency_error()`
4. **`canonical/visualization.py`**: Add `render_multiscale_canonical_3d()`
5. **`backend/inference.py`**: Add `"multiscale"` mode to web app
6. **`app.py`**: Add multiscale option to Radio selector

### Backward Compatibility
- Existing `"canonical"` mode remains unchanged
- New `"multiscale"` mode is additive
- All existing tests continue to pass

---

## 6. Computational Complexity

### Current (Single-Scale)
- **Time:** O(17) per frame (17 joints, fixed operations)
- **Space:** O(17) for canonical pose, O(9) for rotation matrix

### Multi-Scale
- **Time:** O(17 × K) per frame where K = number of scales (K=6)
- **Space:** O(17 × K) for multi-scale representation, O(9 × K) for rotation matrices

**K = 6** (global + torso + 2 arms + 2 legs)

**Complexity increase:** ~6× per frame

**Practical impact:**
- Current: ~0.1s per frame on CPU
- Multi-scale: ~0.6s per frame on CPU
- For 50 frames: 5s → 30s (acceptable for thesis demo)
- For video: Still viable with frame subsampling

---

## 7. Evaluation Protocol

### Same as current evaluation
- MPI-INF-3DHP, S1/Seq1, cameras 0 and 1
- 50 matched frame pairs
- Metrics: cross-view joint distance, canonical consistency, bone-length deviation

### Additional metrics for multi-scale
- **Per-level consistency**: Cross-view error at each scale level
- **Scale contribution**: How much each level reduces error
- **Residual analysis**: What remains after multi-scale canonicalization

### Comparison
- Raw MotionAGFormer: 0.15
- Single-scale canonical: 0.09
- Multi-scale canonical: target < 0.09

---

## 8. Risks and Failure Cases

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Limb frames too short for stable axis | Medium | Medium | Fallback to parent frame |
| Computational cost too high | Low | Medium | Frame subsampling for video |
| No improvement over single-scale | Medium | High | Document as negative result |
| Degenerate limb configurations | Medium | Low | Graceful fallback (existing pattern) |

### Expected Improvement

Based on literature:
- 3DPCNet achieved 3.4° rotation error (vs. 20° baseline)
- Our single-scale already achieves 40% improvement
- Multi-scale targets the **residual** that single-scale misses

**Conservative estimate:** 5-10% additional improvement (0.09 → 0.08-0.085)
**Optimistic estimate:** 15-20% additional improvement (0.09 → 0.07-0.075)

### Why This Should Improve Cross-View Consistency

The key insight is that **different body parts have different orientations relative to the camera**:
- Torso orientation is captured by global canonicalization
- Limb orientations (arms, legs) may differ from torso
- When Camera A sees an arm extended sideways and Camera B sees it extended forward, global canonicalization doesn't capture this difference
- Multi-scale canonicalization at the limb level captures these local orientation differences

This is supported by:
- 3DPCNet's demonstration that per-joint features improve canonicalization
- MoViD's finding that different body parts have different view sensitivities
- V-VIPE's multi-scale embedding improvements

---

## 9. Success Criteria

### Must achieve
- Cross-view distance < 0.09 (improvement over single-scale)
- All existing tests pass
- No degradation in single-scale performance
- Runtime < 1s per frame on CPU

### Nice to have
- Cross-view distance < 0.08 (significant additional improvement)
- Per-level analysis showing which scales contribute most
- Visualization showing limb-level orientation differences

### Will NOT pursue if
- No measurable improvement over single-scale
- Runtime > 2s per frame (too slow for demo)
- Adds significant complexity without clear benefit

---

## 10. Implementation Timeline

| Task | Time | Dependencies |
|------|------|-------------|
| Implement `multiscale.py` | 2 hours | None |
| Update `canonicalizer.py` | 30 min | multiscale.py |
| Update `metrics.py` | 30 min | multiscale.py |
| Update `visualization.py` | 1 hour | multiscale.py |
| Update `backend/inference.py` | 30 min | multiscale.py |
| Update `app.py` | 30 min | None |
| Add tests | 1 hour | multiscale.py |
| Run evaluation | 30 min | All above |
| **Total** | **~6 hours** | |

---

## 11. Decision

**Recommendation: Proceed with implementation.**

Rationale:
1. Addresses a real limitation of current approach
2. Supported by recent literature (3DPCNet, MoViD, V-VIPE)
3. Training-free (geometric only)
4. Moderate effort (~6 hours)
5. Expected to improve cross-view consistency
6. Fails gracefully (existing mode unchanged if multiscale doesn't help)

**Decision required from user before implementation begins.**
