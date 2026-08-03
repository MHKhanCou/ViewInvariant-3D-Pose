# Reference System Comparison (Updated July 2026)

## Pipeline Comparison Table

| Aspect | Our Pipeline | MotionAGFormer (Official) | KelvinHong | MocapNET |
|--------|-------------|--------------------------|------------|----------|
| **2D Detector** | YOLOv8-nano, conf=0.4 | YOLOv3-SPP + HRNet | YOLOv3-SPP + AlphaPose (ResNet-50) | MobileNet/OpenPose/MediaPipe |
| **Keypoint Format** | COCO-17 | COCO-17 | Halpe-26 (26 joints) | Custom (NSRM encoding) |
| **Skeleton** | H36M-17 | H36M-17 | H36M-17 (via Halpe→H36M) | BVH (Euler rotations) |
| **Lifter** | MotionAGFormer-XS (2.2M, 27f) | MotionAGFormer-Base (9.8M, 243f) | MotionBert-lite (dim=256, 243f) | Ensemble of SNN encoders |
| **Training data** | H36M | H36M | H36M-SH | CMU MoCap (BVH) |
| **Post-processing (inference)** | None | None | None | Hierarchical IK |
| **Post-processing (viz)** | Plotly WebGL | matplotlib 3D | Blender bone rigging | OpenCV/OpenGL/Blender |
| **Confidence threshold** | 0.4 | Not documented | 0.05 (AlphaPose) | Not configurable |
| **Flip augmentation** | Yes (averaged) | Yes (averaged) | Yes (averaged) | No |
| **Output format** | 3D joints (17,3) | 3D joints (17,3) | BVH rotations + armature | BVH rotations |
| **Interactive 3D** | Yes (Plotly WebGL) | No (static matplotlib) | Yes (Blender) | Yes (OpenGL) |
| **Multi-person** | No | No | No | No |
| **Training-free** | Yes | Yes | No | No |

## Recent View-Invariant Pose Papers (2024-2026)

| Paper | Year | Approach | Training? | vs Ours |
|-------|------|----------|-----------|---------|
| **3DPCNet** | 2025 | Learned SO(3) rotation via GCN+Transformer | Yes (self-supervised) | We are training-free; they require training |
| **MoViD** | 2026 | Motion-view disentanglement via orthogonal projection | Yes | We post-process; they modify features |
| **V-VIPE** | 2024 | VAE canonical embedding | Yes | We are deterministic; they are probabilistic |
| **BLAPose** | 2024 | RNN bone-length prediction + post-hoc adjustment | Yes | We are analytical; they need training |
| **DDHPose** | 2024 | Bone length/direction disentanglement in diffusion | Yes | We are training-free; they train |
| **FastDDHPose** | 2026 | Same as DDHPose, faster | Yes | Validated bone-length decomposition as competitive |
| **PriorFormer** | 2025 | Segment lengths as transformer priors | Yes | We use lengths as reliability metric |
| **CMANet** | 2024 | Self-supervised SMPL canonical space | Yes | SMPL-dependent; we work on raw skeletons |
| **COMPOSE** | 2026 | Multi-view hypergraph optimization | **No** | Training-free like us, but multi-view only |
| **RePos** | 2026 | Factorizes relative pose from root | Yes | Similar goal, learned vs. analytical |

## Our Unique Position

| Property | Learned Methods | Our Approach |
|----------|----------------|--------------|
| Training required | Yes (days-weeks) | No |
| Training data needed | Yes (MM-Fi, H36M, etc.) | No |
| Inference time | 10-100ms | <1ms |
| Interpretability | Black box | Full transparency |
| Reliability estimation | Rarely provided | Built-in reliability score + hard gates |
| Deployment cost | GPU + model weights | CPU only |
| Bone-length handling | As training loss (BLAPose, DDHPose) | As reliability metric (analytical) |
| Canonicalization method | Learned rotation (3DPCNet) or feature disentanglement (MoViD) | Gram-Schmidt analytical rotation |
