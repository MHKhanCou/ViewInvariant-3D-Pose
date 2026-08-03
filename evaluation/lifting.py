"""
Shared 2D -> 3D lifting path for evaluation.

Extracted verbatim from `run_eval.predict_single_frame` so that the cross-view
evaluation and the degradation sweep run byte-identical model code. The split
point is deliberate: everything *after* 2D detection lives here, which lets the
degradation sweep substitute corrupted keypoints without re-running YOLO
(the sweep re-lifts ~3000 times; re-detecting each time would dominate runtime
for no scientific gain, since the detector is not what we are perturbing).

`lift_from_coco` (legacy single-frame protocol) now delegates to
`lift_from_coco_window` with a repeated window, so the two paths share one
implementation by construction.
"""

import copy

import numpy as np
import torch

from demo_live.lifter import coco_to_h36m, N_FRAMES, normalize_screen_coordinates

# H36M-17 left/right joint indices, used for flip augmentation.
LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]


def flip_data(data):
    """Mirror a pose array about x and swap left/right joints."""
    f = copy.deepcopy(data)
    f[..., 0] *= -1
    f[..., LEFT_JOINTS + RIGHT_JOINTS, :] = f[..., RIGHT_JOINTS + LEFT_JOINTS, :]
    return f


def lift_from_coco(model, kpts, scores, width, height, device="cpu"):
    """
    Lift COCO-17 2D keypoints to a raw root-relative 3D pose.

    Replicates the single frame into a 27-frame window (matching the existing
    cross-view protocol) and applies flip augmentation, averaging the original
    and un-flipped mirrored predictions as in the official eval protocol.

    No display post-processing is applied: no camera_to_world, no z-shift, no
    normalization. The return value is the raw model output, root-centered.

    Args:
        model: loaded MotionAGFormer-XS
        kpts: (17, 2) COCO-17 keypoints in pixel coordinates
        scores: (17,) COCO-17 confidences
        width, height: source frame dimensions, for screen normalization
        device: torch device

    Returns:
        (17, 3) float64 raw root-relative pose (centre frame of the window)
    """
    kpts_seq = np.repeat(np.asarray(kpts, dtype=np.float32)[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(np.asarray(scores, dtype=np.float32)[None], N_FRAMES, axis=0)
    return lift_from_coco_window(model, kpts_seq, scores_seq, width, height, device)


def lift_from_coco_window(model, kpts_seq, scores_seq, width, height, device="cpu",
                          return_arms=False):
    """
    Lift a TRUE 27-frame window of COCO-17 keypoints to a raw root-relative
    3D pose for the window's centre frame.

    Identical to `lift_from_coco` except the temporal window is real detections
    from consecutive frames rather than one frame repeated 27x (the protocol
    flaw flagged by the examiner). Same flip augmentation, same raw output.

    Args:
        model: loaded MotionAGFormer-XS
        kpts_seq: (27, 17, 2) COCO-17 keypoints in pixel coordinates
        scores_seq: (27, 17) COCO-17 confidences
        width, height: source frame dimensions, for screen normalization
        device: torch device
        return_arms: if True, also return the two flip-augmentation branches
            that are otherwise averaged away. Both are computed either way,
            so their disagreement is free — it is simply discarded by default.

    Returns:
        (17, 3) float64 raw root-relative pose (centre frame of the window),
        or (pose, {"nonflip": (17,3), "flip": (17,3)}) when return_arms=True.
        Each arm is root-centred identically to the returned mean; the flip
        arm is already un-mirrored, so both live in the same camera frame.
    """
    kpts_seq = np.asarray(kpts_seq, dtype=np.float32)
    scores_seq = np.asarray(scores_seq, dtype=np.float32)
    assert kpts_seq.shape == (N_FRAMES, 17, 2), f"expected ({N_FRAMES},17,2), got {kpts_seq.shape}"

    h36m = coco_to_h36m(kpts_seq, scores_seq)

    norm = normalize_screen_coordinates(h36m, w=width, h=height)
    inp = torch.from_numpy(norm.astype("float32"))[None].to(device)
    inp_aug = torch.from_numpy(flip_data(norm).astype("float32"))[None].to(device)

    with torch.no_grad():
        out_nonflip = model(inp).cpu().numpy()
        out_flip = model(inp_aug).cpu().numpy()
        out_flip_unmirrored = flip_data(out_flip)
        pred = (out_nonflip + out_flip_unmirrored) / 2.0

    c = N_FRAMES // 2
    raw_pose = pred[0, c].copy()              # centre frame of the window
    mean_pose = raw_pose - raw_pose[0:1]      # root-centre

    if not return_arms:
        return mean_pose

    arms = {}
    for name, out in (("nonflip", out_nonflip), ("flip", out_flip_unmirrored)):
        arm = out[0, c].copy()
        arms[name] = arm - arm[0:1]           # root-centre, same as the mean
    return mean_pose, arms
