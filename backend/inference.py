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
from canonical.canonicalizer import Canonicalizer
from canonical.visualization import render_canonical_3d
from presentation.avatar_renderer import render_stylized_avatar_3d
from evaluation.reliability import (
    compute_reliability_score, has_hard_geometric_failure, should_abstain,
)
from lib.utils import camera_to_world

from backend.model_loader import get_detector, get_model

# Max processing height — keeps memory and speed reasonable on CPU.
MAX_HEIGHT = 480
# Detection width — downscale before YOLO for speed on high-res video.
DET_WIDTH = 640

# Visualization rotation quaternion (matches official demo).
VIS_ROT = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]


# L/R limb chains for display-side symmetrization (H36M-17).
_LIMB_CHAINS = [[0, 1, 2, 3], [0, 4, 5, 6], [8, 11, 12, 13], [8, 14, 15, 16]]
_MIRROR_BONE = {  # bone (a,b) -> mirrored bone (a',b')
    (0, 1): (0, 4), (1, 2): (4, 5), (2, 3): (5, 6),
    (0, 4): (0, 1), (4, 5): (1, 2), (5, 6): (2, 3),
    (8, 11): (8, 14), (11, 12): (14, 15), (12, 13): (15, 16),
    (8, 14): (8, 11), (14, 15): (11, 12), (15, 16): (12, 13),
}


def equalize_limb_lengths(pose):
    """
    DISPLAY-ONLY: rebuild each limb chain with left/right-averaged bone
    lengths, keeping every bone's direction. Removes the visual L/R length
    mismatch that monocular depth ambiguity produces. Never applied to
    evaluated or scored poses.
    """
    p = np.asarray(pose, dtype=np.float64)
    out = p.copy()
    for chain in _LIMB_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            v = p[b] - p[a]
            n = np.linalg.norm(v)
            ma, mb = _MIRROR_BONE[(a, b)]
            n_mirror = np.linalg.norm(p[mb] - p[ma])
            target = (n + n_mirror) / 2.0
            direction = v / n if n > 1e-8 else np.zeros(3)
            out[b] = out[a] + direction * target
    return out.astype(p.dtype)


def estimate_poses(image_rgb: np.ndarray):
    """
    Run 3D pose estimation and return raw poses in both coordinate spaces.

    Inference is separated from rendering. The renderer consumes whichever
    pose representation is selected by the user.

    Args:
        image_rgb: (H, W, 3) uint8 RGB numpy array from Gradio.

    Returns:
        dict with keys:
            camera_pose:      (17, 3) float64 — camera-coordinate pose
            view_invariant:   (17, 3) float64 — view-invariant (canonical) pose
            kpts_2d:          (17, 2) float32 — 2D COCO keypoints
            scores:           (17,) float32 — keypoint confidence scores
            frame_rgb:        (H, W, 3) uint8 — the (possibly downscaled) input
            inference_time:   float — seconds
        Returns None if detection fails completely.
    """
    t0 = time.time()

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    H_orig, W_orig = frame_bgr.shape[:2]
    if H_orig > MAX_HEIGHT:
        scale = MAX_HEIGHT / H_orig
        new_w = int(round(W_orig * scale))
        frame_small = cv2.resize(frame_bgr, (new_w, MAX_HEIGHT))
    else:
        frame_small = frame_bgr

    detector = get_detector()
    model = get_model()

    kpts, scores = detector.detect_with_rotation(frame_small)
    if kpts is None:
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    H, W = frame_small.shape[:2]

    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)

    h36m = coco_to_h36m(kpts_seq, scores_seq)

    # --- Get the RAW model prediction (before any post-processing) ---
    # Uses flip augmentation: average original + horizontally-flipped predictions.
    # This matches the official eval protocol and significantly improves accuracy.
    from demo_live.lifter import normalize_screen_coordinates
    import torch
    import copy

    LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
    RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]

    def _flip_data(data):
        flipped = copy.deepcopy(data)
        flipped[..., 0] *= -1
        flipped[..., LEFT_JOINTS + RIGHT_JOINTS, :] = flipped[..., RIGHT_JOINTS + LEFT_JOINTS, :]
        return flipped

    norm = normalize_screen_coordinates(h36m, w=W, h=H)
    flipped_norm = _flip_data(norm)

    # Get the device from the model (not from params which may be on different device)
    with torch.no_grad():
        device = next(model.parameters()).device
        inp = torch.from_numpy(norm.astype(np.float32))[None].to(device)
        flipped_inp = torch.from_numpy(flipped_norm.astype(np.float32))[None].to(device)

        out_nonflip = model(inp).cpu().numpy()
        out_flip = model(flipped_inp).cpu().numpy()
        out_flip_unflipped = _flip_data(out_flip)
        pred = (out_nonflip + out_flip_unflipped) / 2.0

    raw_pose = pred[0, 13].copy()  # Center frame, (17, 3)

    # --- Raw root-relative pose (no transforms applied) ---
    # This is the true model output, suitable for reliability scoring
    # and quantitative evaluation.
    pose_root_relative = raw_pose - raw_pose[0:1]

    # --- Display pose: same pipeline as lift_sequence(mode="world") ---
    # Root-zero + camera_to_world quaternion + z-shift + global normalize.
    # This is a DISPLAY convention, not a physical camera-coordinate pose.
    pose_display = raw_pose.copy()
    pose_display[0] = 0  # Root zero (matches mode="world")
    pose_display = camera_to_world(pose_display.astype(np.float64), R=VIS_ROT, t=0)
    pose_display[:, 2] -= np.min(pose_display[:, 2])
    max_val = np.max(pose_display)
    if max_val > 0:
        pose_display /= max_val

    # --- View-invariant pose: canonical transform on raw root-relative ---
    canonicalizer = Canonicalizer()
    pose_canonical, _ = canonicalizer(pose_root_relative)

    # --- Training-free reliability score (same code path as offline eval) ---
    reliability, rel_components, _ = compute_reliability_score(
        pose_root_relative, scores)
    hard_failure, failure_reason = has_hard_geometric_failure(pose_root_relative)
    abstain = should_abstain(reliability, hard_failure)

    frame_rgb_out = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    return {
        "raw_root_relative": pose_root_relative,
        "motionagformer_display_pose": pose_display,
        "view_invariant": pose_canonical,
        "kpts_2d": kpts,
        "scores": scores,
        "frame_rgb": frame_rgb_out,
        "inference_time": time.time() - t0,
        "reliability": float(reliability),
        "reliability_components": {k: float(v) for k, v in rel_components.items()},
        "hard_failure": bool(hard_failure),
        "failure_reason": failure_reason,
        "abstain": bool(abstain),
    }


def predict_image(image_rgb: np.ndarray, mode="motionagformer"):
    """
    Run 3D pose estimation on a single RGB image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB numpy array from Gradio.
        mode:      "camera"    — camera-relative root pose (qualitative).
                   "canonical" — canonical body-frame pose (qualitative).
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

    # Detect 2D pose — try multiple rotations for robustness.
    kpts, scores = detector.detect_with_rotation(frame_small)
    if kpts is None:
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    H, W = frame_small.shape[:2]

    # Repeat single frame to fill 27-frame window.
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)

    if mode not in {"motionagformer", "canonical", "avatar"}:
        raise ValueError(f"Unknown visualization mode: {mode}")

    # The default and avatar outputs share the official-style display path.
    # Canonicalization is applied only in the research-view branch below.
    h36m = coco_to_h36m(kpts_seq, scores_seq)
    pose3d = lift_sequence(model, h36m, img_size=(H, W),
                           mode="root" if mode == "canonical" else "world",
                           apply_display_postprocess=(mode != "canonical"))[-1]

    # Draw results (BGR).
    img_2d_bgr = draw_2d(kpts, frame_small, scores=scores)
    if mode == "canonical":
        can = Canonicalizer()
        pose_can, _ = can(pose3d)
        # NOTE: pose3d from lift_sequence is already normalized to [0,1].
        # Do NOT apply additional normalization — it would double-normalize.
        img_3d_bgr = render_canonical_3d(pose_can)
    elif mode == "avatar":
        img_3d_bgr = render_stylized_avatar_3d(pose3d)
    else:
        img_3d_bgr = render_3d(pose3d)
    combined_bgr = make_frame(img_2d_bgr, img_3d_bgr)

    # BGR -> RGB for Gradio.
    original_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    overlay_2d_rgb = cv2.cvtColor(img_2d_bgr, cv2.COLOR_BGR2RGB)
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

    inference_time = time.time() - t0
    return original_rgb, overlay_2d_rgb, combined_rgb, inference_time


def predict_video(video_path: str, mode="motionagformer", export_bvh=False):
    """
    Run 3D pose estimation on a video file.

    Args:
        video_path: path to the uploaded video file.
        mode:       "camera"    — camera-relative root pose (qualitative).
                    "canonical" — canonical body-frame pose (qualitative).
        export_bvh: additionally write the pose sequence as a BVH motion-capture
                    file and return its path. BVH stores joint rotations
                    relative to each parent and carries no camera, so canonical
                    mode is the one that maps onto it cleanly; camera mode is
                    exported too but will import with the body arbitrarily
                    oriented.
    Returns:
        output_path: path to the processed output video.
        inference_time: seconds (float).
        bvh_path: only when export_bvh is True; None if the sequence was empty.
    """
    t0 = time.time()

    if mode not in {"motionagformer", "canonical"}:
        raise ValueError("Avatar View is currently supported for images only.")

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
    poses3d = lift_sequence(model, h36m, img_size=(proc_H, proc_W),
                            mode="root" if mode == "canonical" else "world",
                            apply_display_postprocess=(mode != "canonical"))
    if len(poses3d) != T:
        poses3d = poses3d[-T:]

    # Canonical mode: batch-canonicalize with temporal consistency.
    if mode == "canonical":
        can = Canonicalizer()
        poses3d_render, _ = can(poses3d)
        # NOTE: lift_sequence already returns poses normalized to [0,1] range.
        # Do NOT apply additional normalization here — it would double-normalize
        # and make the pose look like a small scientific plot.
    else:
        poses3d_render = poses3d

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
        elif mode == "canonical":
            img_3d = render_canonical_3d(poses3d_render[i])
        else:
            img_3d = render_3d(poses3d_render[i])

        frames_out.append(make_frame(img_2d, img_3d))
    cap.release()

    # Save output video.
    out_dir = os.path.join(_REPO_ROOT, "demo_live", "output")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{mode}_demo.mp4")
    save_video_demo(frames_out, out_path, fps=fps)

    inference_time = time.time() - t0

    if export_bvh:
        from presentation.bvh_export import write_bvh
        bvh_path = None
        if len(poses3d_render) > 0:
            bvh_path = os.path.join(out_dir, f"{base}_{mode}.bvh")
            # Root translation is meaningless in the canonical frame, where the
            # body is re-anchored every frame, so it is pinned there.
            write_bvh(bvh_path, np.asarray(poses3d_render), fps=fps, scale=100.0,
                      root_motion=(mode != "canonical"))
        return out_path, inference_time, bvh_path

    return out_path, inference_time
