"""
Replication of the temporal bone-length inconsistency signal on Human3.6M.

Why this exists
---------------
`evaluation/bone_consistency.py` found that temporal bone-length inconsistency
predicts a frozen model's own error on MPI-INF-3DHP (rho = +0.492, n = 2460).
That result rests on ONE dataset, ONE 2D detector and ONE recording setup, so
the report states in writing that it "requires replication on a second dataset
before it carries weight". This module is that replication.

What changes relative to the MPI experiment
-------------------------------------------
Three things vary at once, which makes this a genuinely independent test rather
than a re-run:
  1. Dataset      MPI-INF-3DHP (green screen, 8 cams)  ->  Human3.6M (4 cams)
  2. 2D front-end YOLOv8-pose detections               ->  Stacked Hourglass
  3. Domain       out-of-domain for an H36M-trained model -> in-domain test split

Point 3 matters and cuts against us: MotionAGFormer-XS was trained on H36M
subjects S1/S5/S6/S7/S8, so S9/S11 is a held-out but IN-DOMAIN test set where
the model is much more accurate. If the signal is real it should survive the
easier regime with a smaller but same-signed correlation; if it was an artefact
of large out-of-domain errors it should collapse. Either outcome is informative
and both are reported.

Protocol
--------
Predictions follow the official MotionAGFormer H36M evaluation path exactly
(`train.py:96-185`): flip test-time augmentation, root-relative output,
`DataReaderH36M.denormalize`, then per-frame scaling by `2.5d_factor` into
camera-space millimetres, with the same three corrupted videos excluded.
Reproducing the published MPJPE from this path is used as a pipeline sanity
check before any correlation is computed.

Frames are then subsampled to 5 Hz (stride 10 of the native 50 Hz). Adjacent
H36M frames are near-duplicates; keeping all of them would inflate the sample
count without adding independent evidence and would make the cluster bootstrap
intractable. The per-subject reference skeleton is still estimated from EVERY
frame of the video, because that estimate is free.

The five criteria are the same ones `bone_consistency.py` applied, with the
clustering unit changed from camera to video and the strata from
subject x condition to subject x camera (8 strata rather than 4, so criterion
(b) is strictly harder to pass here).

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_replication
      ./venv/Scripts/python.exe -m evaluation.h36m_replication --stage meta
      ./venv/Scripts/python.exe -m evaluation.h36m_replication --limit 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation.bone_consistency import bone_ratios, deviation, partial_spearman
from evaluation.gt_eval import similarity_align_error
from evaluation.lifting import flip_data
from evaluation.reliability import compute_reliability_score

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_replication")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "motion3d")
CLIP_DIR = os.path.join(DATA_ROOT, "H36M-27", "test")
SOURCE_PKL = "h36m_sh_conf_cam_source_final.pkl"

SEED = 12345
RHO_FLOOR = 0.30
BOOTSTRAP_DRAWS = 10000
N_FRAMES = 27
EVAL_STRIDE = 10  # 50 Hz -> 5 Hz; see module docstring

# Videos excluded by the official evaluation (corrupted mocap), train.py:159-161.
BLOCK_LIST = ["s_09_act_05_subact_02", "s_09_act_10_subact_02", "s_09_act_13_subact_01"]

# The reliability score reads 2D confidences at COCO-17 key-joint slots and
# takes their minimum. H36M confidences are in H36M-17 order, so map the seven
# joints it actually reads. Unread slots are filled with 1.0 and never used.
#            COCO slot : H36M-17 source
COCO_FROM_H36M = {0: 9, 5: 11, 6: 14, 11: 4, 12: 1, 13: 5, 14: 2}


# --------------------------------------------------------------------------
# Stage 1 — clip metadata (requires the 1 GB source pickle, run once)
# --------------------------------------------------------------------------

def build_meta(path):
    """Recover per-clip source video, 2.5D factor, GT and resolution."""
    from data.reader.h36m import DataReaderH36M

    print("Loading %s (this takes a minute and ~1 GB of RAM)..." % SOURCE_PKL)
    reader = DataReaderH36M(n_frames=N_FRAMES, sample_stride=1,
                            data_stride_train=N_FRAMES // 3, data_stride_test=N_FRAMES,
                            dt_file=SOURCE_PKL, dt_root=DATA_ROOT)
    _, split_id_test = reader.get_split_id()
    split_id_test = np.array([list(r) for r in split_id_test], dtype=np.int64)

    sources = np.array(reader.dt_dataset["test"]["source"])
    factors = np.array(reader.dt_dataset["test"]["2.5d_factor"], dtype=np.float32)
    gts = np.array(reader.dt_dataset["test"]["joints_2.5d_image"], dtype=np.float32)
    hw = reader.get_hw().astype(np.float32)

    np.savez_compressed(
        path,
        frame_idx=split_id_test,                  # (N, 27) global frame indices
        clip_source=sources[split_id_test[:, 0]],  # (N,) e.g. s_09_act_02_subact_01_ca_01
        factor=factors[split_id_test],             # (N, 27)
        gt=gts[split_id_test],                     # (N, 27, 17, 3) mm, camera space
        hw=hw,                                     # (N, 2) res_w, res_h
        n_test_frames=len(sources),
    )
    print("Saved %s  (%d clips, %d test frames)" % (path, len(split_id_test), len(sources)))


# --------------------------------------------------------------------------
# Stage 2 — inference over the pre-sliced clips
# --------------------------------------------------------------------------

def run_inference(meta, path, limit=None, batch_size=32, threads=None):
    """Official eval path: flip TTA -> root-relative -> denormalize -> x factor."""
    import pickle

    from demo_live.lifter import build_xs_model

    if threads:
        torch.set_num_threads(threads)
    model = build_xs_model("cpu")

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

        x = torch.from_numpy(raw)
        with torch.no_grad():
            out = model(x).numpy()
            out_flip = flip_data(model(torch.from_numpy(flip_data(raw))).numpy())
        p = (out + out_flip) / 2.0
        p[:, :, 0, :] = 0.0  # args.root_rel, train.py:113

        # DataReaderH36M.denormalize, inlined so the 1 GB pickle is not reloaded
        for k in range(start, stop):
            res_w, res_h = hw[k]
            p[k - start, :, :, :2] = (p[k - start, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            p[k - start, :, :, 2:] = p[k - start, :, :, 2:] * res_w / 2

        preds[start:stop] = p * factor[start:stop][:, :, None, None]

        done = stop
        rate = done / (time.time() - t0)
        print("\r  %6d/%d clips  %5.1f clips/s  eta %5.1f min"
              % (done, n_clips, rate, (n_clips - done) / rate / 60), end="", flush=True)
    print()

    np.savez_compressed(path, preds=preds, conf=conf, n_clips=n_clips)
    print("Saved %s" % path)


# --------------------------------------------------------------------------
# Stage 3 — analysis
# --------------------------------------------------------------------------

def h36m_conf_to_coco(scores_h36m):
    """(17,) H36M-ordered confidences -> (17,) COCO-ordered, key slots only."""
    out = np.ones(17, dtype=np.float64)
    for coco_i, h36m_i in COCO_FROM_H36M.items():
        out[coco_i] = scores_h36m[h36m_i]
    return out


def aggregate_by_video(meta, pred_npz, n_clips):
    """
    Regroup non-overlapping clips into per-video frame sequences.

    `split_clips` resamples a video's tail when its length does not divide
    evenly by 27, so a handful of frames appear in two clips. Those are
    averaged over their occurrence count, exactly as the official evaluation
    averages their errors (train.py:186-188). Scatter-adds run over flat
    per-frame accumulators rather than a dict so that memory stays bounded by
    the test set (~270 MB) instead of growing per frame.
    """
    frame_idx = meta["frame_idx"][:n_clips]
    source = meta["clip_source"][:n_clips]
    gt = meta["gt"][:n_clips]
    preds, conf = pred_npz["preds"], pred_npz["conf"]

    keep = np.array([str(s)[:-6] not in BLOCK_LIST for s in source])
    flat = frame_idx[keep].ravel()
    n_frames = int(meta["n_test_frames"])

    pred_sum = np.zeros((n_frames, 17, 3), dtype=np.float32)
    gt_sum = np.zeros((n_frames, 17, 3), dtype=np.float32)
    conf_sum = np.zeros((n_frames, 17), dtype=np.float32)
    np.add.at(pred_sum, flat, preds[keep].reshape(-1, 17, 3))
    np.add.at(gt_sum, flat, gt[keep].reshape(-1, 17, 3))
    np.add.at(conf_sum, flat, conf[keep].reshape(-1, 17))
    count = np.bincount(flat, minlength=n_frames).astype(np.float32)

    # Which video each retained frame belongs to (clips never straddle videos).
    vid_of_frame = np.zeros(n_frames, dtype=np.int32) - 1
    vids = sorted({str(s) for s, k in zip(source, keep) if k})
    vid_id = {v: i for i, v in enumerate(vids)}
    vid_of_frame[flat] = np.repeat([vid_id[str(s)] for s in source[keep]], N_FRAMES)

    out = {}
    for vid in vids:
        f = np.flatnonzero(vid_of_frame == vid_id[vid])
        n = count[f][:, None, None]
        out[vid] = {
            "frames": f,
            "pred": (pred_sum[f] / n).astype(np.float64),
            "gt": (gt_sum[f] / n).astype(np.float64),
            "conf": (conf_sum[f] / n[:, :, 0]).astype(np.float64),
        }
    return out


def parse_video(vid):
    """'s_09_act_02_subact_01_ca_01' -> ('S9', 'act_02', 'ca_01')."""
    p = vid.split("_")
    return "S%d" % int(p[1]), "%s_%s" % (p[2], p[3]), "%s_%s" % (p[6], p[7])


def analyse(videos, stride=EVAL_STRIDE, draws=BOOTSTRAP_DRAWS):
    per_video, mpjpe_by_action = {}, {}

    for vid, v in sorted(videos.items()):
        pred, gt, conf = v["pred"], v["gt"], v["conf"]

        # Pipeline sanity check: unaligned root-relative MPJPE, the published
        # metric. The official evaluation averages within each action and then
        # across the fifteen actions, which are of unequal length, so a flat
        # mean over frames does NOT reproduce the published figure (it gives
        # 45.59 mm). Grouping by action first is what makes the two comparable.
        pr = pred - pred[:, 0:1, :]
        gr = gt - gt[:, 0:1, :]
        mpjpe_by_action.setdefault(parse_video(vid)[1], []).append(
            np.linalg.norm(pr - gr, axis=-1).mean(axis=-1))

        # Reference skeleton from EVERY frame of the video (labels never used).
        R_full = bone_ratios(pred)
        sel = np.arange(0, len(pred), stride)
        half = len(pred) // 2
        med_all = np.median(R_full, axis=0)
        med_ref = np.median(R_full[:half], axis=0)

        err = np.array([similarity_align_error(pred[i], gt[i]) for i in sel])
        dev = deviation(R_full[sel], med_all)

        # Causal variant: reference from the first half, scored on the second only.
        sel_c = sel[sel >= half]
        dev_c = deviation(R_full[sel_c], med_ref)
        err_c = np.array([similarity_align_error(pred[i], gt[i]) for i in sel_c])

        # Incumbent reliability score, prev_rotation threaded across evaluated frames.
        rel, prev_R = [], None
        for i in sel:
            r, _, _ = compute_reliability_score(pred[i], h36m_conf_to_coco(conf[i]), prev_R)
            rel.append(r)
            _, R_cur, meta_c = canonicalize_single(pred[i].astype(np.float32),
                                                   prev_z=None if prev_R is None else prev_R[:, 2])
            prev_R = R_cur.astype(np.float64) if meta_c["valid"] else None
        subj, action, cam = parse_video(vid)

        per_video[vid] = {
            "subject": subj, "action": action, "camera": cam,
            "dev": dev, "err": err, "rel": np.array(rel),
            "conf": conf[sel].min(axis=1),
            "dev_causal": dev_c, "err_causal": err_c,
        }

    cat = lambda f: np.concatenate([v[f] for v in per_video.values()])
    dev, err, rel, conf = cat("dev"), cat("err"), cat("rel"), cat("conf")
    dev_c, err_c = cat("dev_causal"), cat("err_causal")

    rho_dev = float(spearmanr(dev, err)[0])
    rho_rel = float(spearmanr(rel, err)[0])
    rho_causal = float(spearmanr(dev_c, err_c)[0])
    rho_partial = partial_spearman(dev, err, conf)

    # Cluster bootstrap over VIDEOS (frames within a video share a skeleton estimate).
    keys, offs, o = list(per_video), [], 0
    for k in keys:
        m = len(per_video[k]["err"])
        offs.append(np.arange(o, o + m))
        o += m
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(draws):
        pick = rng.choice(len(offs), size=len(offs), replace=True)
        s = np.concatenate([offs[i] for i in pick])
        a, b = spearmanr(dev[s], err[s])[0], spearmanr(rel[s], err[s])[0]
        if not (np.isnan(a) or np.isnan(b)):
            deltas.append(abs(a) - abs(b))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    def strata(fields):
        g = {}
        for v in per_video.values():
            g.setdefault("_".join(v[f] for f in fields), {"dev": [], "err": []})
            g["_".join(v[f] for f in fields)]["dev"].extend(v["dev"].tolist())
            g["_".join(v[f] for f in fields)]["err"].extend(v["err"].tolist())
        return {k: float(spearmanr(d["dev"], d["err"])[0])
                for k, d in g.items() if len(d["err"]) > 10}

    by_subj_cam = strata(["subject", "camera"])

    # Does the signal work where the model actually fails? H36M is in-domain and
    # the model is accurate on most of it, so a weak pooled rho could mean either
    # "the signal is fake" or "there is little error here to predict". Correlating
    # each video's rho against that video's mean error separates the two. This is
    # a between-video comparison, so it does not suffer the range restriction that
    # would bias a within-video split on error terciles.
    vid_rho = np.array([spearmanr(v["dev"], v["err"])[0] for v in per_video.values()])
    vid_err = np.array([v["err"].mean() for v in per_video.values()])
    ok = ~np.isnan(vid_rho)
    difficulty = {
        "spearman_video_rho_vs_video_mean_error": float(spearmanr(vid_rho[ok], vid_err[ok])[0]),
        "n_videos": int(ok.sum()),
        "mean_rho_easiest_third": float(np.mean(vid_rho[ok][np.argsort(vid_err[ok])[:ok.sum() // 3]])),
        "mean_rho_hardest_third": float(np.mean(vid_rho[ok][np.argsort(vid_err[ok])[-(ok.sum() // 3):]])),
        "mean_error_easiest_third_mm": float(np.mean(np.sort(vid_err[ok])[:ok.sum() // 3])),
        "mean_error_hardest_third_mm": float(np.mean(np.sort(vid_err[ok])[-(ok.sum() // 3):])),
    }
    # Criterion (c) is recorded two ways on purpose. The pre-registered wording
    # was "excludes 0", which is two-sided and is therefore also satisfied by an
    # interval lying entirely BELOW zero -- that is, by the incumbent score
    # beating the bone signal. On MPI the interval was [+0.108, +0.515], so the
    # two readings agreed and the ambiguity never surfaced. It surfaces here, so
    # the literal reading is reported unchanged and the verdict uses the
    # directional one. Tightening it cannot flip the MPI verdict.
    crit = {
        "a_rho_floor": bool(rho_dev >= RHO_FLOOR),
        "b_sign_consistent": bool(len({np.sign(r) for r in by_subj_cam.values()}) == 1),
        "c_ci_favours_bone_signal": bool(lo > 0),
        "d_survives_confidence_control": bool(abs(rho_partial) >= RHO_FLOOR),
        "e_survives_causal_estimate": bool(rho_causal >= RHO_FLOOR),
    }

    return {
        "verdict_pass": all(crit.values()),
        "criteria": crit,
        "criterion_c_literal_two_sided": bool(lo > 0 or hi < 0),
        "n_frames_evaluated": int(len(err)),
        "n_videos": len(per_video),
        "eval_stride": stride,
        "sanity_check": {
            "mpjpe_action_balanced_mm": float(np.mean(
                [np.concatenate(v).mean() for v in mpjpe_by_action.values()])),
            "mpjpe_flat_over_frames_mm": float(np.concatenate(
                [e for v in mpjpe_by_action.values() for e in v]).mean()),
            "n_actions": len(mpjpe_by_action),
            "repo_official_eval_mpjpe_mm": 45.149,
            "published_motionagformer_xs_mpjpe_mm": 45.1,
            "note": "Action-balanced MPJPE is the published convention and is the "
                    "one to compare. The flat mean over frames is reported only to "
                    "show that the difference between them is the averaging rule.",
        },
        "pooled": {
            "spearman_bone_deviation_vs_error": rho_dev,
            "spearman_reliability_vs_error": rho_rel,
            "partial_given_detector_confidence": rho_partial,
            "causal_estimate_rho": rho_causal,
            "causal_n": int(len(err_c)),
            "bootstrap_ci_delta_abs_rho": [lo, hi],
            "bootstrap_draws": len(deltas),
            "mean_similarity_aligned_error_mm": float(err.mean()),
        },
        # How much bone-length violation is actually present to be read? The
        # signal can only rank frames by error if predicted skeletons deform
        # measurably. Comparing this magnitude against the MPI run separates
        # "the correlation was spurious there" from "there is nothing to read
        # here", which the pooled rho alone cannot distinguish.
        "signal_magnitude": {
            "bone_deviation_median": float(np.median(dev)),
            "bone_deviation_iqr": [float(np.percentile(dev, 25)), float(np.percentile(dev, 75))],
            "bone_deviation_mean": float(dev.mean()),
            "bone_deviation_std": float(dev.std()),
            "error_median_mm": float(np.median(err)),
            "error_iqr_mm": [float(np.percentile(err, 25)), float(np.percentile(err, 75))],
        },
        "difficulty_analysis": difficulty,
        "per_stratum_subject_camera": by_subj_cam,
        "per_stratum_subject": strata(["subject"]),
        "per_stratum_camera": strata(["camera"]),
        "per_stratum_action": strata(["action"]),
        "per_video": {k: float(spearmanr(v["dev"], v["err"])[0]) for k, v in per_video.items()},
    }


def report(res, mpi_rho=0.492):
    p = res["pooled"]
    print("=" * 74)
    print("HUMAN3.6M REPLICATION — temporal bone-length inconsistency")
    print("=" * 74)
    s = res["sanity_check"]
    print("  pipeline check: MPJPE %.3f mm action-balanced over %d actions"
          % (s["mpjpe_action_balanced_mm"], s["n_actions"]))
    print("                  repo official eval %.3f mm | paper %.1f mm | flat mean %.3f mm"
          % (s["repo_official_eval_mpjpe_mm"], s["published_motionagformer_xs_mpjpe_mm"],
             s["mpjpe_flat_over_frames_mm"]))
    print("  n = %d frames (5 Hz) over %d videos; mean aligned error %.1f mm\n"
          % (res["n_frames_evaluated"], res["n_videos"], p["mean_similarity_aligned_error_mm"]))
    print("  rho(bone deviation, error)            %+.3f   [MPI: %+.3f]"
          % (p["spearman_bone_deviation_vs_error"], mpi_rho))
    print("  rho(reliability, error)  [incumbent]  %+.3f" % p["spearman_reliability_vs_error"])
    print("  partial | detector confidence         %+.3f" % p["partial_given_detector_confidence"])
    print("  causal estimate (disjoint reference)  %+.3f  (n=%d)"
          % (p["causal_estimate_rho"], p["causal_n"]))
    print("  cluster bootstrap 95%% CI on |d|-|rel| [%+.3f, %+.3f]"
          % tuple(p["bootstrap_ci_delta_abs_rho"]))
    d = res["difficulty_analysis"]
    print("\n  does it work where the model fails?")
    print("    rho(per-video rho, per-video mean error)  %+.3f  (%d videos)"
          % (d["spearman_video_rho_vs_video_mean_error"], d["n_videos"]))
    print("    easiest third  mean rho %+.3f  at %5.1f mm"
          % (d["mean_rho_easiest_third"], d["mean_error_easiest_third_mm"]))
    print("    hardest third  mean rho %+.3f  at %5.1f mm"
          % (d["mean_rho_hardest_third"], d["mean_error_hardest_third_mm"]))
    print("\n  per subject x camera:")
    for k, v in sorted(res["per_stratum_subject_camera"].items()):
        print("    %-14s %+.3f" % (k, v))
    m = res["signal_magnitude"]
    print("\n  is there anything to read?")
    print("    bone deviation  median %.4f  IQR [%.4f, %.4f]"
          % (m["bone_deviation_median"], m["bone_deviation_iqr"][0], m["bone_deviation_iqr"][1]))
    print("    aligned error   median %.1f mm  IQR [%.1f, %.1f]"
          % (m["error_median_mm"], m["error_iqr_mm"][0], m["error_iqr_mm"][1]))
    print("\n  criteria:")
    for k, v in res["criteria"].items():
        print("    %-32s %s" % (k, v))
    print("    %-32s %s  (pre-registered wording, two-sided)"
          % ("c_ci_excludes_zero_literal", res["criterion_c_literal_two_sided"]))
    print("\n  VERDICT: %s" % ("PASS" if res["verdict_pass"] else "FAIL"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["meta", "infer", "analyze", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="clips, for smoke tests")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUT_DIR, "meta.npz")
    pred_path = os.path.join(OUT_DIR, "preds%s.npz" % ("" if args.limit is None else "_smoke"))

    if args.stage in ("meta", "all") and not os.path.exists(meta_path):
        build_meta(meta_path)
    meta = np.load(meta_path, allow_pickle=True)

    if args.stage in ("infer", "all") and not os.path.exists(pred_path):
        run_inference(meta, pred_path, args.limit, args.batch_size, args.threads)
    if args.stage in ("meta", "infer"):
        return

    pred_npz = np.load(pred_path)
    n_clips = int(pred_npz["n_clips"])
    videos = aggregate_by_video(meta, pred_npz, n_clips)
    res = analyse(videos, stride=args.stride, draws=args.draws)
    report(res)

    out = os.path.join(OUT_DIR, "h36m_replication%s.json"
                       % ("" if args.limit is None else "_smoke"))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved: %s" % out)


if __name__ == "__main__":
    main()
