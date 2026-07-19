"""
Singleton model loading for the Gradio web demo.

Loads YOLOv8-pose and MotionAGFormer-XS once, reuses across requests.
"""

import os
import sys

# Ensure demo_live/ and repo root are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo_live.pose_detector import PoseDetector
from demo_live.lifter import build_xs_model

_detector = None
_model = None


def get_detector(device: str = "cpu", conf: float = 0.4) -> PoseDetector:
    global _detector
    if _detector is None:
        weights = os.path.join(_REPO_ROOT, "yolov8n-pose.pt")
        if not os.path.exists(weights):
            weights = "yolov8n-pose.pt"
        _detector = PoseDetector(weights=weights, device=device, conf=conf)
    return _detector


def get_model(device: str = "cpu"):
    global _model
    if _model is None:
        _model = build_xs_model(device=device)
    return _model
