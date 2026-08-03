# Examiner Feedback Summary

## Verdict

The project is **engineering-strong but scientifically weak** for a thesis titled *View-Invariant 3D Human Pose Estimation from RGB Images*.

## Classification

| Category | Assessment |
|----------|-----------|
| Engineering | Very strong: exact baseline reproduction, stable pipeline, diagnostics, interactive viewer |
| Scientific | Lightweight deterministic body-frame canonical representation |
| Unsupported claims | Any claim of improved accuracy; "world trajectory"; 67% MPI improvement; avatar scientific value |
| Weakest aspect | Novelty and real-data evaluation of canonical representation |

## Key Corrections

### 1. Remove "MoViD-inspired" claim
The canonicalizer is Gram-Schmidt rotation, not MoViD's motion-view disentanglement via orthogonal feature projection. These are fundamentally different.

### 2. Invalidate MPI "67% reduction"
`lift_sequence()` applies demo-only rotation, z-shift, and normalization before comparison. Also repeats one image 27 times. This is exploratory, not calibrated.

### 3. Cannot claim "model-agnostic"
Only tested on MotionAGFormer-XS.

### 4. Cannot pre-claim improvement
Must measure first, then report.

### 5. Cross-view evaluation too thin
1 subject, 2 cameras, 1 sequence, 50 frames cannot support a thesis claim.

### 6. Do not use "canonical" in the UI
Use "View-Invariant Coordinate System" instead. The examiner understands view invariance, not canonical.

## Approved Direction

> A training-free, reliability-aware geometric post-processing method for making frozen monocular 3D-pose predictions more cross-view comparable.

### What the thesis IS
- Post-processing on frozen model output
- Deterministic, no learning
- Reliability-aware: detects when canonicalization fails
- Evaluated on synchronized multi-camera data

### What the thesis IS NOT
- Improved pose estimation
- Learned view disentanglement
- MoViD reproduction
- State-of-the-art performance claim

## Title (Examiner-Approved)

> A Lightweight, Training-Free, Reliability-Aware Geometric Canonicalization Framework for Cross-View Comparability of Frozen Monocular 3D Pose Predictions.

Use "for" (not "improving") until experiments prove improvement.

## Required Evidence Package

1. Audited MPI-INF-3DHP synchronized pairs (subject, sequence, camera, frame table)
2. Raw prediction extraction before camera_to_world, z-shift, or display scaling
3. True 27-frame windows (not repeated single frames)
4. Ablation: raw / canonical / reliability-aware canonical
5. Coverage vs error curves
6. Frame-conditioning and failure analysis
7. At least one held-out sequence or camera pair
8. Limitations paragraph: "training-free post-processing, not learned view disentanglement"

## Reliability Score Components

- Normalized torso/hip-axis conditioning
- Sine of torso-hip angle
- Left/right bone-length asymmetry ratio
- Normalized abnormal bone-length ratio
- Detector confidence
- Temporal frame-rotation change (video only)

Do NOT tune weights on camera pairs used for final reporting.

## Evaluation Rules

1. Before comparing raw cross-view poses, verify raw MotionAGFormer outputs are compatible with MPI camera coordinate convention
2. Do NOT use camera_to_world(), z-shifting, or per-pose display scaling in evaluation
3. Define one fixed body-scale normalization for cross-view comparison
4. Prefer abstention as first low-reliability policy

## Core Claim (Modest)

> "Reliability-aware geometric canonicalization can make a frozen monocular model's predicted poses more comparable across synchronized viewpoints, while explicitly identifying frames where canonicalization is unreliable."

## Reference Papers

| Paper | Year | Approach | vs Ours |
|-------|------|----------|---------|
| MoViD | 2026 | Motion-view disentanglement via orthogonal projection | Retrains full pipeline |
| 3DPCNet | 2025 | Learned SO(3) rotation prediction | Post-process, but learned |
| V-VIPE | 2024 | VAE canonical embedding | Retrains encoder/decoder |
| BLAPose | 2024 | Learned bone-length adjustment | Requires training |
| PoseIRM | 2024 | Camera-setting-invariant training | Requires training |
| CMANet | 2024 | Self-supervised canonical parameter space | Multi-view, SMPL-based |
| Pose Grammar | 2018 | Cross-view evaluation protocol | Foundational protocol |

## Codex Corrections (Applied)

1. Flip augmentation fixed (flips input, not output)
2. `apply_display_postprocess` flag added to lift_sequence()
3. Canonical path now receives clean raw root-relative output
4. Default view behavior unchanged (baseline preserved)
