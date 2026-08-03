# Literature Review & Proposed Extensions

## 1. Relevant Literature

### MoViD (arXiv:2604.03299, 2026)
**View-Invariant 3D Human Pose Estimation via Motion-View Disentanglement**

- Disentangles viewpoint information from motion features
- View estimator models key joint relationships to predict viewpoint
- Orthogonal projection module separates motion and view features
- Physics-grounded contrastive alignment across views
- Frame-by-frame inference with view-aware flip refinement
- **24.2% error reduction** over SOTA on 9 datasets
- Real-time at 15 FPS on edge devices

**Relevance:** Direct comparison target. MoViD learns view-invariant features; we propose geometric canonicalization. MoViD requires training; we don't.

### 3DPCNet (arXiv:2509.23455, 2025)
**Pose Canonicalization for Robust Viewpoint-Invariant 3D Kinematic Analysis**

- GCN + Transformer hybrid encoder for pose canonicalization
- Predicts continuous 6D rotation mapped to SO(3)
- Self-supervised training on MM-Fi dataset
- Reduces rotation error from 20° to 3.4°
- MPJPE from ~64mm to 47mm
- Estimator-agnostic module

**Relevance:** Most directly comparable work. Uses learned canonicalization vs. our geometric approach. Their method requires training; ours doesn't.

### V-VIPE (arXiv:2407.07092, CVPR 2024 Workshop)
**Variational View Invariant Pose Embedding**

- VAE-based embedding in canonical coordinate space
- Encodes 2D and 3D poses for downstream tasks
- Generalizes to unseen camera views
- Uses variational approach for diversity

**Relevance:** Learned canonical representation vs. our geometric approach.

### POEM (arXiv:2010.13321, IJCV 2021)
**View-Invariant, Occlusion-Robust Probabilistic Embedding**

- Probabilistic embedding from 2D keypoints (no 3D prediction)
- View-invariant retrieval across camera views
- Occlusion augmentation strategies

**Relevance:** Different paradigm (embedding vs. pose), but addresses same problem.

### DECA (arXiv:2108.08557, ICCV 2021)
**Viewpoint-Equivariant Human Pose using Capsule Autoencoders**

- Capsule networks for viewpoint equivariance
- Preserves joint hierarchy in feature space
- Reduces data dependency for viewpoint generalization

**Relevance:** Architectural approach vs. our post-processing approach.

### MCSS (arXiv:1908.05293, 2019)
**Multiview-Consistent Semi-Supervised Learning**

- Uses multi-view consistency as weak supervision
- Hard-negative mining based on temporal relations
- View-invariant pose retrieval benchmarks on H36M and MPI-INF-3DHP

**Relevance:** Directly related to cross-view evaluation methodology.

---

## 2. Top 3 Feasible Extensions

### Extension 1: Bone-Length Consistency Regularization

**Idea:** Use bone-length invariance as an additional cross-view constraint.

**How:** After canonicalization, compare bone lengths across frames. If bone lengths vary significantly, apply soft constraints to enforce consistency.

**Scientific value:** Medium — bone lengths are theoretically view-invariant, but this is already partially captured by canonicalization.

**Implementation effort:** Low — post-processing only, ~50 lines of code.

**Compatibility:** Perfect — operates on top of existing canonical output.

**Expected improvement:** Small (5-10%) — bone lengths are already consistent after canonicalization.

**Risk:** Low — no model changes, purely geometric.

### Extension 2: Temporal Canonicalization Smoothing

**Idea:** Apply temporal smoothing to the canonical rotation matrix across video frames to reduce jitter.

**How:** Use exponential moving average or Kalman filter on the rotation matrix R across consecutive frames.

**Scientific value:** Medium — improves temporal consistency but doesn't address the core view-invariance problem.

**Implementation effort:** Low — ~30 lines of code.

**Compatibility:** Perfect — operates on canonical output.

**Expected improvement:** Small — reduces jitter but doesn't improve cross-view consistency.

**Risk:** Low — no model changes.

### Extension 3: Multi-Scale Canonical Representation

**Idea:** Construct canonical representations at multiple body scales (full body, torso, limbs) and combine them.

**How:** Apply canonicalization independently to:
- Full body (current approach)
- Torso only (spine + shoulders)
- Limbs (arms, legs separately)

Then combine the multi-scale representations.

**Scientific value:** High — captures both global orientation and local limb orientations, which may differ from the global body orientation.

**Implementation effort:** Medium — ~100 lines of code, need to define limb groupings.

**Compatibility:** Good — operates on top of existing canonical output, but requires careful design.

**Expected improvement:** Medium (10-20%) — captures orientation variations that global canonicalization misses.

**Risk:** Medium — more complex, needs careful evaluation.

---

## 3. Recommendation

**Rank by impact vs. implementation cost:**

| Rank | Extension | Impact | Cost | Risk |
|------|-----------|--------|------|------|
| 1 | Multi-Scale Canonical | High | Medium | Medium |
| 2 | Bone-Length Regularization | Low | Low | Low |
| 3 | Temporal Smoothing | Low | Low | Low |

**Recommended: Extension 3 (Multi-Scale Canonical)**

Reasons:
- Highest potential impact on cross-view consistency
- Captures orientation variations that global canonicalization misses
- Still a geometric post-processing method (no training required)
- Moderate implementation effort
- Novel contribution not directly addressed by 3DPCNet or MoViD

**Alternative: Skip all extensions**

The current canonical body-frame representation is already a solid contribution. The 40% improvement is meaningful. Adding more complexity risks destabilizing a clean project.

---

## 4. Decision Framework

Before implementing any extension, ask:

1. Does this strengthen the "view-invariant" story? → Yes for Extension 3
2. Can it be implemented without retraining? → Yes for all 3
3. Does it risk destabilizing the baseline? → Low risk for all 3
4. Is the expected improvement significant? → Medium for Extension 3
5. Can it be evaluated properly? → Yes, using existing MPI-INF-3DHP setup
