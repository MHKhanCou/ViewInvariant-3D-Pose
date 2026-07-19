"""
2D pose detector for the MotionAGFormer live demo.

Uses Ultralytics YOLOv8-pose / RTMPose, which outputs COCO-17 keypoints
(17 joints, x/y pixel coordinates + confidence). This is the exact 17-joint
layout that the official MotionAGFormer preprocessing (lib/preprocess.py ->
h36m_coco_format) expects, so no remapping is required before lifting.

The detector is intentionally the ONLY place that knows about pixels. Everything
downstream (lifter, visualizer) works in the H36M-17 normalized space.
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO


# COCO-17 keypoint names (Ultralytics default pose model order).
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17


class PoseDetector:
    """
    Wrap a Ultralytics pose model.

    Args:
        weights: path to a .pt pose model, or a hub name like 'yolov8n-pose.pt'.
                 RTMPose models (e.g. 'rtmpose-l.pt') work identically and also
                 output COCO-17.
        device:  'cpu' or '0' etc. Demo runs on CPU by default.
        conf:    detection confidence threshold.
    """

    def __init__(self, weights="yolov8n-pose.pt", device="cpu", conf=0.4):
        self.device = device
        self.conf = conf
        # Ultralytics auto-downloads official hub weights on first use.
        self.model = YOLO(weights)

    def detect(self, frame):
        """
        Detect a single person in one BGR frame.

        Returns:
            keypoints: np.ndarray (17, 2) float32 in pixel coords, or None.
            scores:    np.ndarray (17,)   float32 in [0, 1], or None.
        """
        results = self.model.predict(
            frame, device=self.device, conf=self.conf, verbose=False
        )
        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None, None

        kpts = result.keypoints
        xy = kpts.xy  # (N, 17, 2)
        conf = kpts.conf  # (N, 17)

        if xy is None or xy.shape[0] == 0:
            return None, None

        # Pick the person with the highest mean keypoint confidence.
        if xy.shape[0] > 1:
            mean_conf = conf.mean(dim=1)
            best = int(mean_conf.argmax())
        else:
            best = 0

        keypoints = xy[best].cpu().numpy().astype(np.float32)  # (17, 2)
        scores = conf[best].cpu().numpy().astype(np.float32)   # (17,)

        if np.all(scores < self.conf):
            return None, None

        return keypoints, scores


def detect_video(detector, video_path, out_kpts_npy=None, det_width=None):
    """
    Run detection over every frame of a video.

    Args:
        det_width: if set, downscale each frame to this width before detection
                   (much faster on 4K video); keypoints are scaled back to the
                   original resolution so overlays stay aligned.

    Returns:
        keypoints: np.ndarray (T, 17, 2) float32, pixel coords.
        scores:    np.ndarray (T, 17)    float32.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_kpts = []
    all_scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale = 1.0
        if det_width is not None and frame.shape[1] > det_width:
            scale = frame.shape[1] / float(det_width)
            small = cv2.resize(frame, (det_width, int(round(frame.shape[0] / scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        kpts, scores = detector.detect(small)
        if kpts is not None and scale != 1.0:
            kpts = kpts * scale
        if kpts is None:
            # Zero-filled frame so the temporal window stays aligned.
            kpts = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            scores = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
        all_kpts.append(kpts)
        all_scores.append(scores)

    cap.release()

    keypoints = np.array(all_kpts, dtype=np.float32)   # (T, 17, 2)
    scores = np.array(all_scores, dtype=np.float32)    # (T, 17)

    if out_kpts_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_kpts_npy)), exist_ok=True)
        np.save(out_kpts_npy, {"keypoints": keypoints, "scores": scores})

    return keypoints, scores
