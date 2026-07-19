"""
Inference wrapper for the Gradio web demo.

Calls existing demo_live functions. Handles BGR/RGB conversion
and returns results in Gradio-friendly formats.
"""

import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo_live.lifter import coco_to_h36m, lift_sequence, N_FRAMES
from demo_live.visualize import draw_2d, render_3d, make_frame, save_video_demo
from demo_live.pose_detector import detect_video

from backend.model_loader import get_detector, get_model

# Max processing height — keeps memory and speed reasonable on CPU.
MAX_HEIGHT = 480
# Detection width — downscale before YOLO for speed on high-res video.
DET_WIDTH = 640


def predict_image(image_rgb: np.ndarray, mode="world"):
    """
    Run 3D pose estimation on a single RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB numpy array from Gradio.
        mode:      "world" — skeleton drifts through space (official demo).
                   "root"  — skeleton centered, root-relative (benchmark style).
    Returns:
        original_rgb, overlay_2d_rgb, combined_rgb: RGB numpy arrays for display.
        inference_time: seconds (float).
    """
    t0 = time.time()

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Downscale for detection + display if needed.
    H_orig, W_orig = frame_bgr.shape[:2]
    if H_orig > MAX_HEIGHT:
        scale = MAX_HEIGHT / H_orig
        new_w = int(round(W_orig * scale))
        frame_small = cv2.resize(frame_bgr, (new_w, MAX_HEIGHT))
    else:
        frame_small = frame_bgr

    detector = get_detector()
    model = get_model()

    # Detect 2D pose on downscaled frame.
    kpts, scores = detector.detect(frame_small)
    if kpts is None:
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    H, W = frame_small.shape[:2]

    # Repeat single frame to fill 27-frame window.
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)

    # Lift to 3D.
    h36m = coco_to_h36m(kpts_seq, scores_seq)
    pose3d = lift_sequence(model, h36m, img_size=(H, W), mode=mode)[-1]

    # Draw results (BGR).
    img_2d_bgr = draw_2d(kpts, frame_small, scores=scores)
    img_3d_bgr = render_3d(pose3d)
    combined_bgr = make_frame(img_2d_bgr, img_3d_bgr)

    # BGR -> RGB for Gradio.
    original_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    overlay_2d_rgb = cv2.cvtColor(img_2d_bgr, cv2.COLOR_BGR2RGB)
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

    inference_time = time.time() - t0
    return original_rgb, overlay_2d_rgb, combined_rgb, inference_time


def predict_video(video_path: str, mode="world"):
    """
    Run 3D pose estimation on a video file.

    Args:
        video_path: path to the uploaded video file.
        mode:       "world" — skeleton drifts (official demo).
                    "root"  — skeleton centered (benchmark style).
    Returns:
        output_path: path to the processed output video.
        inference_time: seconds (float).
    """
    t0 = time.time()

    detector = get_detector()
    model = get_model()

    # Read video metadata.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Detect 2D poses — downscale to DET_WIDTH before YOLO for speed.
    keypoints, scores = detect_video(detector, video_path, det_width=DET_WIDTH)
    T = keypoints.shape[0]

    # Scale keypoints to match our processing resolution (MAX_HEIGHT).
    if orig_H > MAX_HEIGHT:
        proc_scale = MAX_HEIGHT / orig_H
        keypoints = keypoints * proc_scale
    else:
        proc_scale = 1.0
    proc_H = min(orig_H, MAX_HEIGHT)
    proc_W = int(round(orig_W * proc_scale))

    # Lift to 3D.
    h36m = coco_to_h36m(keypoints, scores)
    poses3d = lift_sequence(model, h36m, img_size=(proc_H, proc_W), mode=mode)
    if len(poses3d) != T:
        poses3d = poses3d[-T:]

    # Render each frame at processing resolution.
    cap = cv2.VideoCapture(video_path)
    frames_out = []
    for i in range(T):
        ret, frame = cap.read()
        if not ret:
            break
        if proc_scale < 1.0:
            frame = cv2.resize(frame, (proc_W, proc_H))

        img_2d = draw_2d(keypoints[i], frame, scores=scores[i])

        # Skip 3D rendering when no person detected.
        mean_conf = float(scores[i].mean()) if scores[i] is not None else 0.0
        if mean_conf < 0.1:
            h = img_2d.shape[0]
            w = int(round(h * 9.6 / 5.4))
            img_3d = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
            img_3d = render_3d(poses3d[i])

        frames_out.append(make_frame(img_2d, img_3d))
    cap.release()

    # Save output video.
    out_dir = os.path.join(_REPO_ROOT, "demo_live", "output")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{mode}_demo.mp4")
    save_video_demo(frames_out, out_path, fps=fps)

    inference_time = time.time() - t0
    return out_path, inference_time
