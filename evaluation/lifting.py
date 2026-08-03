"""
Shared 2D -> 3D lifting path for evaluation.

Extracted verbatim from `run_eval.predict_single_frame` so that the cross-view
evaluation and the degradation sweep run byte-identical model code. The split
point is deliberate: everything *after* 2D detection lives here, which lets the
degradation sweep substitute corrupted keypoints without re-running YOLO
(the sweep re-lifts ~3000 times; re-detecting each time would dominate runtime
for no scientific gain, since the detector is not what we are perturbing).

`tests/test_lifting_consistency.py` asserts this reproduces the original
inline implementation exactly.
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
    h36m = coco_to_h36m(kpts_seq, scores_seq)

    norm = normalize_screen_coordinates(h36m, w=width, h=height)
    inp = torch.from_numpy(norm.astype("float32"))[None].to(device)
    inp_aug = torch.from_numpy(flip_data(norm).astype("float32"))[None].to(device)

    with torch.no_grad():
        out_nonflip = model(inp).cpu().numpy()
        out_flip = model(inp_aug).cpu().numpy()
        pred = (out_nonflip + flip_data(out_flip)) / 2.0

    raw_pose = pred[0, 13].copy()          # centre frame of the 27-frame window
    return raw_pose - raw_pose[0:1]        # root-centre
