# MotionAGFormer-XS Live Demo

End-to-end monocular 3D human pose estimation for a **thesis supervisor demo**:

```
RGB image / video
   -> 2D keypoint detector (RTMPose / YOLOv8-pose, COCO-17)
   -> MotionAGFormer-XS lifting (27 input frames, H36M-17)
   -> 2D skeleton overlay + 3D skeleton render
   -> saved .png / .mp4
```

This demo reuses the **verified official baseline** (MotionAGFormer-XS,
Human3.6M, MPJPE 45.149 mm) unchanged. It only adds a *separate* inference
path that runs the same model on detector keypoints instead of the
Human3.6M ground-truth 2D detections. The model architecture, checkpoint, and
evaluation pipeline are **not** modified.

## Why a separate folder?

The repo's `demo/vis.py` is hardcoded to the **Base** model (243 frames) and
requires YOLOv3 + HRNet checkpoints. This `demo_live/` folder instead:

- uses the **XSmall** model (27 frames) — the exact setup you reproduced,
- runs on **CPU** (the demo machine has no CUDA),
- uses a single lightweight detector (`ultralytics` pose model),
- keeps all demo code away from the training/eval code.

## Setup

The detector dependency is already installed in the project venv:

```bash
# from the MotionAGFormer/ directory, using the project venv
.\venv\Scripts\python.exe -m pip install ultralytics
```

Ultralytics auto-downloads the pose weights (`yolov8n-pose.pt`) on first run.
For higher accuracy you can pass `--det-weights yolov8m-pose.pt` (or an RTMPose
model); all of them output COCO-17, which is what this pipeline expects.

## Usage

```bash
# from the MotionAGFormer/ directory
.\venv\Scripts\python.exe demo_live/infer_image.py --input path/to/person.jpg
.\venv\Scripts\python.exe demo_live/infer_video.py --input path/to/person.mp4
```

Outputs land in `demo_live/output/`:

- image: `demo_live/output/<name>_demo.png` (2D overlay | 3D pose)
- video: `demo_live/output/<name>_demo.mp4` (2D overlay | 3D pose, side by side)

### Options

| flag            | default            | meaning                                  |
|-----------------|--------------------|------------------------------------------|
| `--input`       | required           | input image or video                     |
| `--output`      | auto in `output/`  | output file path                         |
| `--det-weights` | `yolov8n-pose.pt`  | Ultralytics COCO-17 pose model           |
| `--conf`        | `0.4`              | detection confidence threshold           |
| `--det-width`   | `640`              | downscale frames to this width for detection (4K is slow); `0` disables |
| `--out-height`  | `480`              | display height of combined output (keeps memory low on 4K video) |
| `--device`      | `cpu`              | `cpu` or a CUDA device id                |

## Pipeline details (for the thesis write-up)

1. **2D detection** (`pose_detector.py`): `PoseDetector` runs an Ultralytics
   pose model and returns COCO-17 keypoints `(x, y)` + confidence per frame.
2. **Format conversion** (`lifter.coco_to_h36m`): uses the repo's official
   `lib/preprocess.h36m_coco_format` to map COCO-17 -> H36M-17. This is the
   same preprocessing used by the baseline, so the 2D input space is identical.
3. **Lifting** (`lifter.lift_sequence`): a dense **27-frame sliding window**
   feeds MotionAGFormer-XS. Each window's center frame becomes one 3D pose,
   giving one prediction per video frame. Horizontal-flip test-time averaging
   is applied, matching the official demo.
4. **Post-processing**: `camera_to_world` with the official fixed rotation,
   pelvis rooted at the origin, and scale normalization — identical to
   `demo/vis.py`.
5. **Visualization** (`visualize.py`): 2D overlay (COCO skeleton) on the left,
   3D matplotlib render (H36M skeleton) on the right.

## Files

| file                    | responsibility                              |
|-------------------------|---------------------------------------------|
| `pose_detector.py`      | 2D detector (RTMPose / YOLOv8-pose)         |
| `lifter.py`             | MotionAGFormer-XS load + 27-frame lifting   |
| `visualize.py`          | 2D overlay + 3D render, image/video saving  |
| `infer_image.py`        | image CLI entry point                       |
| `infer_video.py`        | video CLI entry point                       |

## Notes / limitations (be honest in the thesis)

- This is a **qualitative in-the-wild visualization**, not benchmark evidence.
  The 3D outputs on arbitrary images/videos should NOT be presented as
  metrically accurate reconstructions or used for quantitative evaluation.
- MotionAGFormer is a **2D-to-3D lifting** model trained on Human3.6M lab
  motions. When applied to in-the-wild images via YOLOv8-pose, there is a
  significant domain shift that affects output quality.
- For still images, the single frame is repeated 27 times, removing all
  temporal information that the model was designed to exploit.
- The demo lifts **single-person** sequences (the detector picks the highest
  confidence person per frame). Multi-person would need tracking + per-person
  windows.
- Absolute scale is lost (the demo normalizes to unit scale, like the official
  visualization). This is expected for monocular lifting without depth cues.
- The verified benchmark results (MPJPE 45.149 mm, P-MPJPE 36.892 mm) come
  exclusively from `train.py --eval-only` on the Human3.6M test set using
  ground-truth 2D detections — NOT from this demo pipeline.
