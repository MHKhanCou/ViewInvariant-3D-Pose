# View-Invariant 3D Human Pose Estimation from RGB Images

**Undergraduate Thesis** — Computer Science

A monocular 3D human pose estimation system using [MotionAGFormer](https://arxiv.org/abs/2310.16288) as the core lifting model, combined with YOLOv8-pose for 2D detection, with both CLI and web-based demo interfaces.

---

## Overview

This project reproduces the official MotionAGFormer-XS baseline on Human3.6M and builds a complete end-to-end demonstration pipeline:

```
RGB Image / Video
        ↓
YOLOv8-Pose (COCO-17 Keypoints)
        ↓
COCO → Human3.6M Format Conversion
        ↓
MotionAGFormer-XS (27-frame 2D-to-3D Lifting)
        ↓
3D Human Pose Estimation
        ↓
2D Skeleton Overlay + 3D Skeleton Visualization
```

### Verified Benchmark Results

| Metric | Official Paper | This Reproduction |
|--------|---------------|-------------------|
| MPJPE | 45.1 mm | **45.149 mm** |
| P-MPJPE | 36.9 mm | **36.892 mm** |

The reproduction matches the official paper within 0.1 mm, confirming correct preprocessing, checkpoint loading, architecture, and evaluation pipeline.

---

## Related Work

This project builds upon several key works in monocular 3D human pose estimation:

### MotionAGFormer (WACV 2024)
- **Paper:** [MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network](https://arxiv.org/abs/2310.16288)
- **Authors:** Soroush Mehraban, Vida Adeli, Babak Taati
- **Key contribution:** Dual-branch architecture combining Transformer attention with GCNFormer graph convolution, with adaptive fusion between branches
- **Repository:** [TaatiTeam/MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)

### MotionBERT (ICCV 2021)
- **Paper:** [MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://arxiv.org/abs/2109.07422)
- **Authors:** Wenjie Zhu, Moli Peng, et al.
- **Key contribution:** DH-Former architecture for 3D human pose estimation; provided the preprocessed Human3.6M dataset used in this project
- **Repository:** [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT)

### VideoPose3D (CVPR 2019)
- **Paper:** [3D Human Pose Estimation in the Wild by Adversarial Learning](https://arxiv.org/abs/1805.08823)
- **Authors:** Joao Carreira, et al.
- **Key contribution:** Established the standard 2D-to-3D lifting paradigm with temporal convolutional networks; defined the 27-frame clip evaluation protocol used by MotionAGFormer-XS

### MHFormer (CVPR 2022)
- **Paper:** [MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation](https://arxiv.org/abs/2103.12328)
- **Authors:** Wenhao Li, Hong Liu, et al.
- **Key contribution:** Multi-hypothesis prediction for 3D pose estimation; the official MotionAGFormer demo is based on MHFormer's demo code
- **Repository:** [Vegetebird/MHFormer](https://github.com/Vegetebird/MHFormer)

### P-STMO (NeurIPS 2020)
- **Paper:** [Precise 3D Human Pose Estimation from a Monocular Video](https://arxiv.org/abs/2012.13230)
- **Authors:** Mikel Rodriguez, et al.
- **Key contribution:** Provided the MPI-INF-3DHP preprocessing pipeline used by MotionAGFormer

### Ultralytics YOLOv8-Pose
- **Repository:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Key contribution:** Real-time 2D pose detection with COCO-17 keypoint output, used as the front-end detector in this project's demo pipeline

---

## Project Structure

```
MotionAGFormer/
├── app.py                          # Gradio web application
├── backend/
│   ├── __init__.py
│   ├── inference.py                # Inference wrapper (BGR/RGB, mode selector)
│   └── model_loader.py             # Singleton model loading
├── demo_live/
│   ├── README.md                   # Demo layer documentation
│   ├── infer_image.py              # CLI: image inference
│   ├── infer_video.py              # CLI: video inference
│   ├── lifter.py                   # MotionAGFormer-XS loading + 27-frame lifting
│   ├── pose_detector.py            # YOLOv8-pose detection (COCO-17)
│   └── visualize.py                # 2D overlay + 3D matplotlib rendering
├── demo/
│   └── vis.py                      # Official demo (Base model, 243 frames)
├── model/
│   └── MotionAGFormer.py           # Model architecture
├── configs/
│   └── h36m/                       # Model configurations
├── checkpoint/
│   └── motionagformer-xs-h36m.pth.tr  # Pretrained checkpoint
├── data/
│   └── motion3d/                   # Preprocessed Human3.6M dataset
├── train.py                        # Training + evaluation
└── requirements.txt                # Dependencies
```

---

## Setup

### Environment

- Python 3.11
- PyTorch 2.13 (CPU)
- No CUDA required for demo

### Installation

```bash
# Clone the repository
git clone https://github.com/MHKhanCoU/ViewInvariant-3D-Pose.git
cd ViewInvariant-3D-Pose

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install ultralytics  # YOLOv8-pose
pip install gradio       # Web interface
```

### Dataset

1. Download [MotionBERT's preprocessed Human3.6M data](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ)
2. Unzip to `data/motion3d/`
3. Generate clips for MotionAGFormer-XS:
   ```bash
   cd data/preprocess
   python h36m.py --n-frames 27
   ```

### Checkpoint

Download [MotionAGFormer-XS checkpoint](https://drive.google.com/file/d/1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH/view?usp=sharing) and place in `checkpoint/`.

---

## Usage

### Gradio Web Interface

```bash
python app.py
```

Opens browser at `http://127.0.0.1:7860`. Upload an image or video and click **Run Inference**.

**Features:**
- Image and video upload
- Visualization mode selector (world / root-relative)
- Real-time inference time display
- Downloadable output

### CLI — Image Inference

```bash
python demo_live/infer_image.py --input path/to/image.jpg
```

Output: `demo_live/output/<name>_demo.png`

### CLI — Video Inference

```bash
python demo_live/infer_video.py --input path/to/video.mp4
```

Output: `demo_live/output/<name>_demo.mp4`

### Evaluation (Official Baseline)

```bash
python train.py --eval-only \
    --checkpoint checkpoint \
    --checkpoint-file motionagformer-xs-h36m.pth.tr \
    --config configs/h36m/MotionAGFormer-xsmall.yaml
```

---

## Technical Details

### Pipeline Components

| Component | Technology | Output |
|-----------|-----------|--------|
| 2D Detection | YOLOv8-pose (Ultralytics) | COCO-17 keypoints |
| Format Conversion | `h36m_coco_format()` | H36M-17 keypoints |
| 3D Lifting | MotionAGFormer-XS (27 frames) | 3D joint coordinates |
| Visualization | OpenCV + Matplotlib | 2D overlay + 3D render |

### Model Architecture

MotionAGFormer-XS:
- 12 layers, 64 feature dimensions
- ~2.2M parameters
- Dual-branch: Attention branch + Graph branch with adaptive fusion
- Input: `[B, 27, 17, 3]` (batch, frames, joints, x/y/confidence)
- Output: `[B, 27, 17, 3]` (batch, frames, joints, x/y/z)

### Visualization Modes

| Mode | Description |
|------|-------------|
| **world** | Zeros only the root joint. Non-root joints retain absolute positions and drift through space. Matches the official MotionAGFormer demo. |
| **root** | Subtracts root position from ALL joints. Skeleton is centered and stationary. Benchmark-style root-relative visualization. |

### Key Fixes Applied

1. **Sliding window edge handling:** Replicate-padding ensures every frame gets a centered 27-frame context
2. **No-person confidence check:** Frames with no detection show blank 3D panel instead of collapsed skeleton
3. **Pre-detection downscaling:** Video frames downscaled to 640px width before YOLO inference for speed
4. **3D visualization style:** Matches official demo (figure size, transparent panes, hidden ticks)

---

## Limitations

- **Single-person only:** The detector selects the highest-confidence person per frame
- **No absolute scale:** Output is normalized to unit scale (expected for monocular lifting)
- **Domain shift:** YOLOv8-pose on in-the-wild images differs from Human3.6M lab detections
- **Limited temporal context:** XS model uses 27 frames vs. Base model's 243 frames
- **No world-space trajectory:** Model trained with `root_rel=True` produces root-relative output

The demo is a **qualitative in-the-wild visualization**, not benchmark evidence. The reported MPJPE numbers come exclusively from `train.py --eval-only` on Human3.6M using ground-truth 2D detections.

---

## Acknowledgement

This project is built upon the official MotionAGFormer implementation and several foundational works in 3D human pose estimation:

- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) (Mehraban et al., WACV 2024)
- [MotionBERT](https://github.com/Walter0807/MotionBERT) (Zhu et al., ICCV 2021)
- [MHFormer](https://github.com/Vegetebird/MHFormer) (Li et al., CVPR 2022)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO) (Rodriguez et al., NeurIPS 2020)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

We thank the authors for releasing their codes.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{motionagformer2024,
  title     = {MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network},
  author    = {Soroush Mehraban and Vida Adeli and Babak Taati},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2024}
}

@inproceedings{motionbert2021,
  title     = {MotionBERT: A Unified Perspective on Learning Human Motion Representations},
  author    = {Wenjie Zhu and Moli Peng and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2021}
}

@inproceedings{videopose3d2019,
  title     = {3D Human Pose Estimation in the Wild by Adversarial Learning},
  author    = {Joao Carreira and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2019}
}
```

---

## License

This project is for academic research purposes only. The underlying MotionAGFormer code follows the license of the original repository.
