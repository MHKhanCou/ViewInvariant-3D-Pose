"""
MotionBERT as a second frozen backbone on Human3.6M.

Why this exists
---------------
The framework consumes any 17 by 3 pose, so model independence is structural.
It is not, until this module runs, an empirical claim, and the report lists its
absence as a limitation. Here the identical evaluation modules are applied to a
second backbone that shares no architecture with the first:

    MotionAGFormer-XS   2.2M params   transformer + graph-convolution hybrid
    MotionBERT/DSTformer 42.5M params  dual-stream spatio-temporal transformer

Not one line of `h36m_crossview.py`, `h36m_multiscale.py` or `h36m_fusion.py`
changes. They gain a `--preds` argument pointing at a different cache, and that
is the whole adaptation. A trained canonicalizer such as 3DPCNet would at
minimum require validation, and usually retraining, per output distribution.

Honest caveat
-------------
The released MotionBERT Human3.6M checkpoint is fine-tuned at 243 frames. We run
it on the 27-frame clips used by the first backbone, because holding the input
identical is what makes the backbone the only variable. MotionBERT is therefore
operating below its published accuracy, and we report the figure it actually
achieves here rather than the one from its paper. The claim under test concerns
our module, not the backbone's accuracy, but the caveat belongs in the open.

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_motionbert
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import ACTION_NAMES
from evaluation.h36m_replication import (BLOCK_LIST, CLIP_DIR, N_FRAMES,
                                         aggregate_by_video, parse_video)
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.lifting import flip_data

MOTIONBERT_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "MotionBERT")
CHECKPOINT = os.path.join(MOTIONBERT_ROOT, "checkpoints", "motionbert_ft_h36m.bin")
OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_motionbert")
PRED_PATH = os.path.join(OUT_DIR, "preds_motionbert.npz")


def build_model():
    """DSTformer with the released fine-tuned Human3.6M weights."""
    if MOTIONBERT_ROOT not in sys.path:
        sys.path.insert(0, MOTIONBERT_ROOT)
    from lib.model.DSTformer import DSTformer

    model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5,
                      num_heads=8, mlp_ratio=2, maxlen=243, num_joints=17)
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    key = "model_pos" if "model_pos" in sd else list(sd)[0]
    state = {k.replace("module.", ""): v for k, v in sd[key].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_inference(meta, path, limit=None, batch_size=8, threads=None):
    import pickle

    if threads:
        torch.set_num_threads(threads)
    model = build_model()
    print("MotionBERT loaded: %.1fM parameters"
          % (sum(p.numel() for p in model.parameters()) / 1e6))

    n_clips = len(meta["clip_source"]) if limit is None else min(limit, len(meta["clip_source"]))
    hw, factor = meta["hw"], meta["factor"]

    preds = np.zeros((n_clips, N_FRAMES, 17, 3), dtype=np.float32)
    conf = np.zeros((n_clips, N_FRAMES, 17), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_clips, batch_size):
        stop = min(start + batch_size, n_clips)
        raw = np.stack([
            pickle.load(open(os.path.join(CLIP_DIR, "%08d.pkl" % i), "rb"))["data_input"]
            for i in range(start, stop)
        ]).astype(np.float32)
        conf[start:stop] = raw[..., 2]

        with torch.no_grad():
            out = model(torch.from_numpy(raw)).numpy()
            out_flip = flip_data(model(torch.from_numpy(flip_data(raw))).numpy())
        p = (out + out_flip) / 2.0
        p[:, :, 0, :] = 0.0                      # root-relative, as in the official eval

        for k in range(start, stop):             # denormalize, inlined
            res_w, res_h = hw[k]
            p[k - start, :, :, :2] = (p[k - start, :, :, :2]
                                      + np.array([1, res_h / res_w])) * res_w / 2
            p[k - start, :, :, 2:] = p[k - start, :, :, 2:] * res_w / 2

        preds[start:stop] = p * factor[start:stop][:, :, None, None]

        done = stop
        rate = done / (time.time() - t0)
        print("\r  %6d/%d clips  %5.2f clips/s  eta %5.1f min"
              % (done, n_clips, rate, (n_clips - done) / rate / 60), end="", flush=True)
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, preds=preds, conf=conf, n_clips=n_clips)
    print("Saved %s" % path)


def accuracy_check(meta, path):
    """Action-balanced MPJPE, the convention the published figures use."""
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))
    by_action = {}
    for vid, v in videos.items():
        pr = v["pred"] - v["pred"][:, 0:1, :]
        gr = v["gt"] - v["gt"][:, 0:1, :]
        by_action.setdefault(parse_video(vid)[1], []).append(
            np.linalg.norm(pr - gr, axis=-1).mean(axis=-1))

    per_action = {ACTION_NAMES.get(a, a): float(np.concatenate(v).mean())
                  for a, v in sorted(by_action.items())}
    balanced = float(np.mean(list(per_action.values())))

    out = {
        "backbone": "MotionBERT / DSTformer",
        "parameters_millions": 42.5,
        "checkpoint": os.path.basename(CHECKPOINT),
        "mpjpe_action_balanced_mm": balanced,
        "published_mpjpe_at_243_frames_mm": 37.5,
        "window_frames": N_FRAMES,
        "caveat": "The released checkpoint is fine-tuned at 243 frames and is "
                  "evaluated here at 27, so it operates below its published "
                  "accuracy. The window is held identical to the first backbone "
                  "so that the backbone is the only variable.",
        "per_action_mpjpe_mm": per_action,
        "comparison_motionagformer_xs_mm": 45.149,
    }
    print("\n  MotionBERT action-balanced MPJPE  %.2f mm  (at %d frames)"
          % (balanced, N_FRAMES))
    print("  MotionAGFormer-XS, same clips     %.2f mm" % 45.149)
    print("  MotionBERT published, 243 frames  %.1f mm" % 37.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "accuracy.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    if not args.skip_inference and not os.path.exists(PRED_PATH):
        run_inference(meta, PRED_PATH, args.limit, args.batch_size, args.threads)
    accuracy_check(meta, PRED_PATH)
    print("\nNow run, with no change to any of them:")
    for m in ("h36m_crossview", "h36m_multiscale", "h36m_fusion"):
        print("  python -m evaluation.%s --preds %s" % (m, PRED_PATH))


if __name__ == "__main__":
    main()
