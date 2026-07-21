# System Architecture

```
RGB Image / Video
        │
        ▼
YOLOv8 Pose Detector
(COCO-17 keypoints)
        │
        ▼
COCO → H36M Conversion
(17 joints, 16 bones)
        │
        ▼
Normalize Screen Coordinates
(x/y normalized to [-1, 1])
        │
        ▼
MotionAGFormer-XS
(2.2M params, 27-frame window)
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
Camera Coordinate      View-Invariant
     System            Coordinate System
        │                       │
        │              MoViD-inspired
        │              Body-Frame
        │              Transform
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
          Interactive 3D Viewer
         (Plotly WebGL, orbit/pan/zoom)
                    │
                    ▼
            ┌───────┴────────┐
            │                │
       Orbit/Pan/Zoom   Coordinate Space
       Camera Controls      Switch
```

## Pipeline Description

### 1. Input
RGB image or video frame.

### 2. 2D Pose Detection (YOLOv8)
Detects 17 COCO-format keypoints per person.
Rotation-robust: tries 0°, 90°, 180°, 270° and selects best detection.
Confidence threshold: 0.4.

### 3. COCO → H36M Conversion
Converts COCO-17 keypoints to Human3.6M-17 joint format.
Creates 4 virtual joints: pelvis (0), spine (7), thorax (8), head top (10).

### 4. Normalization
Linear scaling: x → x/w × 2 − 1, y → y/w × 2 − h/w.
Centers keypoints and preserves aspect ratio.
Confidence scores passed through unchanged.

### 5. 3D Pose Estimation (MotionAGFormer-XS)
Temporal model: 27-frame sliding window (replicate-padded).
Spatial-temporal attention with adaptive fusion.
Output: (17, 3) root-relative 3D joints.
Flip augmentation: original + horizontally flipped, averaged.

### 6a. Camera Coordinate System
Standard MotionAGFormer output.
Root-zeroed, quaternion rotation applied, normalized to [0, 1].
Pose is in the camera's reference frame.

### 6b. View-Invariant Coordinate System (Thesis Contribution)
Root-relative pose transformed into canonical body frame:
- Torso direction → canonical forward
- Hip axis → canonical horizontal
- Gram-Schmidt orthonormalization
Body orientation normalized, viewpoint removed, motion preserved.

### 7. Interactive 3D Viewer
Single Plotly WebGL renderer.
Orbit, pan, zoom, reset — all built-in.
Coordinate space switches between 6a and 6b without re-running inference.
Same renderer, same skeleton, same camera controls — only pose coordinates change.
