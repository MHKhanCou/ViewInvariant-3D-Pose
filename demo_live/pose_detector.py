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


def _unrotate_kpts(kpts, angle, orig_W, orig_H):
    """
    Map keypoints detected on a cv2-rotated frame back to original-frame
    coordinates. Arithmetic extracted verbatim from detect_with_rotation so
    the default path is unchanged.

    Args:
        kpts: (17, 2) keypoints in the rotated frame.
        angle: one of 0, 90, 180, 270 (as applied by detect_with_rotation).
        orig_W, orig_H: dimensions of the ORIGINAL (unrotated) frame.

    Returns:
        (17, 2) keypoints in original-frame coordinates (a copy).
    """
    out = kpts.copy()
    if angle == 0:
        return out
    rx = out[:, 0].copy()
    ry = out[:, 1].copy()
    if angle == 90:
        # 90° CCW: rotated (rx, ry) -> original (W-1-ry, rx)
        out[:, 0] = orig_W - 1 - ry
        out[:, 1] = rx
    elif angle == 180:
        # 180°: rotated (rx, ry) -> original (W-1-rx, H-1-ry)
        out[:, 0] = orig_W - 1 - rx
        out[:, 1] = orig_H - 1 - ry
    elif angle == 270:
        # 270° CW: rotated (rx, ry) -> original (ry, H-1-rx)
        out[:, 0] = ry
        out[:, 1] = orig_H - 1 - rx
    return out


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

    def detect_with_rotation(self, frame, return_all=False):
        """
        Detect a person trying 0°, 90°, 180°, 270° rotations.
        Returns the detection with the highest mean keypoint confidence,
        remapped back to the original frame coordinates.

        This handles cases where YOLO fails on rotated images.

        Args:
            frame: (H, W, 3) RGB image.
            return_all: if True, additionally return every angle's detection,
                each already remapped to original-frame coordinates. All four
                detections are computed either way, so this costs nothing —
                by default three of them are discarded.

        Returns:
            (best_kpts, best_scores)                     when return_all=False
            (best_kpts, best_scores, per_angle)          when return_all=True,
            where per_angle maps each of 0/90/180/270 to (kpts, scores), or
            (None, None) if the detector failed at that angle.
        """
        best_kpts = None
        best_scores = None
        best_mean_conf = -1
        best_angle = 0
        per_angle = {}

        orig_H, orig_W = frame.shape[:2]

        rotations = [0, 90, 180, 270]
        for angle in rotations:
            if angle == 0:
                rotated = frame
            elif angle == 90:
                rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(frame, cv2.ROTATE_180)
            else:  # 270
                rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            kpts, scores = self.detect(rotated)
            if return_all:
                per_angle[angle] = (
                    (_unrotate_kpts(kpts, angle, orig_W, orig_H), scores.copy())
                    if kpts is not None else (None, None)
                )
            if kpts is not None:
                mean_conf = float(scores.mean())
                if mean_conf > best_mean_conf:
                    best_mean_conf = mean_conf
                    best_kpts = kpts.copy()
                    best_scores = scores.copy()
                    best_angle = angle

        # Remap keypoints back to original frame coordinates.
        if best_kpts is not None and best_angle != 0:
            best_kpts = _unrotate_kpts(best_kpts, best_angle, orig_W, orig_H)

        if return_all:
            return best_kpts, best_scores, per_angle
        return best_kpts, best_scores


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
