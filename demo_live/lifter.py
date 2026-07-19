"""
3D pose lifter for the MotionAGFormer live demo.

Loads the official MotionAGFormer-XS checkpoint (27 input frames, H36M-17) and
lifts detected 2D COCO-17 keypoints to 3D. This is the SAME architecture and
checkpoint used for the reproduced baseline (MPJPE 45.149 mm), so the lifting
path is identical to the verified evaluation setup.

The only differences from the official demo are:
  * we use the XSmall (27-frame) model instead of Base (243-frame),
  * we run on CPU (the demo machine has no CUDA),
  * we feed detector keypoints instead of HRNet+YOLOv3 keypoints.
The preprocessing (COCO->H36M) and post-processing (camera_to_world, root,
scale) are kept exactly as in demo/vis.py so behaviour is unchanged.
"""

import os
import sys
import glob
import copy
import numpy as np
import torch
import torch.nn as nn

# Allow importing the repo's model and the official demo utilities.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "demo"))

from model.MotionAGFormer import MotionAGFormer
from lib.preprocess import h36m_coco_format
from lib.utils import normalize_screen_coordinates, camera_to_world


N_FRAMES = 27  # MotionAGFormer-XS receptive field.

# Fixed camera rotation used in the official demo for visualization alignment.
VIS_ROT = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype="float32",
)

# Mirror augmentation joints (official demo flip_data defaults).
LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]


def build_xs_model(device="cpu"):
    """Build the MotionAGFormer-XS model and load the official checkpoint."""
    args = dict(
        n_layers=12,
        dim_in=3,
        dim_feat=64,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        act_layer=nn.GELU,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=True,
        num_heads=8,
        qkv_bias=False,
        qkv_scale=None,
        hierarchical=False,
        num_joints=17,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=2,
        n_frames=N_FRAMES,
    )
    model = MotionAGFormer(**args).to(device)
    model.eval()

    ckpt_dir = os.path.join(REPO_ROOT, "checkpoint")
    ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, "motionagformer-xs-h36m.pth.tr")))[0]
    # weights_only=False: the official checkpoint embeds numpy scalars and was
    # saved with nn.DataParallel (hence the 'module.' prefix). It is a trusted
    # local file, so this is safe.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def _flip_data(data, left=LEFT_JOINTS, right=RIGHT_JOINTS):
    flipped = copy.deepcopy(data)
    flipped[..., 0] *= -1
    flipped[..., left + right, :] = flipped[..., right + left, :]
    return flipped


def coco_to_h36m(keypoints, scores):
    """
    Convert per-frame COCO-17 detections to H36M-17 input.

    Args:
        keypoints: (T, 17, 2) pixel coords, float32.
        scores:    (T, 17)    confidence,      float32.
    Returns:
        h36m_kpts: (T, 17, 2) H36M-17 pixel coords, float32 (with conf appended
                   as a 3rd dim, matching the baseline's input convention).
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # lib expects (M, T, N, 2) and (M, T, N) for multiple persons.
    kpts_m = keypoints[None]        # (1, T, 17, 2)
    scores_m = scores[None]         # (1, T, 17)
    h36m_kpts, h36m_scores, _ = h36m_coco_format(kpts_m, scores_m)

    h36m_kpts = h36m_kpts[0]        # (T, 17, 2)
    h36m_scores = h36m_scores[0]    # (T, 17)

    # Append confidence as the 3rd coordinate, exactly like the baseline.
    h36m = np.concatenate((h36m_kpts, h36m_scores[..., None]), axis=-1)
    return h36m  # (T, 17, 3)


def lift_sequence(model, h36m_seq, img_size, device="cpu", use_flip=True,
                   mode="world"):
    """
    Lift a full sequence using a centered 27-frame sliding window.

    Every output frame gets a window centered on it.  Frames near the start
    or end of the video are handled by replicate-padding the sequence on both
    sides, so the model always sees a full 27-frame context with the target
    frame in the middle.

    Args:
        model:     MotionAGFormer-XS.
        h36m_seq:  (T, 17, 3) H36M-17 pixel coords + conf.
        img_size:  (H, W) of the source frames.
        use_flip:  average with horizontal flip augmentation (official demo).
        mode:      "world" — zero root only, non-root joints drift (official demo).
                   "root"  — subtract root from all joints, centered skeleton.
    Returns:
        poses3d:   (T, 17, 3) camera-to-world 3D poses, normalized to [0,1]-ish
                   scale.
    """
    T = h36m_seq.shape[0]
    H, W = img_size
    half = N_FRAMES // 2  # 13

    # Replicate-pad both sides so every original frame can be window-centered.
    pad_left = np.repeat(h36m_seq[:1], half, axis=0)
    pad_right = np.repeat(h36m_seq[-1:], half, axis=0)
    padded = np.concatenate([pad_left, h36m_seq, pad_right], axis=0)

    out = []
    with torch.no_grad():
        for i in range(T):
            lo = i
            clip = padded[lo:lo + N_FRAMES]

            # Normalize to [-1, 1]-ish screen coordinates (official preprocessing).
            norm = normalize_screen_coordinates(clip, w=W, h=H)  # (27, 17, 3)
            inp = torch.from_numpy(norm.astype("float32"))[None].to(device)

            if use_flip:
                flipped = _flip_data(norm)
                inp_aug = torch.from_numpy(flipped.astype("float32"))[None].to(device)
                out_nonflip = model(inp)
                out_flip = _flip_data(out_nonflip.cpu().numpy())
                pred = (out_nonflip.cpu().numpy() + out_flip) / 2.0
            else:
                pred = model(inp).cpu().numpy()

            # Take the center frame of the window (index half = 13).
            pose = pred[0, half]  # (17, 3)

            if mode == "world":
                # Official demo style: zero root only, non-root joints retain
                # absolute positions and drift through space across frames.
                pose[0] = 0
            else:
                # Root-relative: subtract root from ALL joints so the
                # skeleton is centered and stationary across frames.
                pose -= pose[0:1]

            # Post-process exactly like the official demo.
            pose = camera_to_world(pose, R=VIS_ROT, t=0)
            pose[:, 2] -= np.min(pose[:, 2])
            max_value = np.max(pose)
            if max_value > 0:
                pose /= max_value
            out.append(pose)

    return np.array(out, dtype=np.float32)  # (T, 17, 3)
