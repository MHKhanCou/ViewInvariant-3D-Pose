"""
Cross-view canonicalization on Human3.6M: replication of the central claim.

Why this exists
---------------
Section 4.8 of the report replicated the bone-length signal on Human3.6M and it
failed. That failure raised a sharper question about everything else: the
canonicalization and fusion results also rest on a single dataset, and we now
have direct evidence that a result from that dataset can fail to transfer. This
module tests the CENTRAL claim, that a body-fixed frame makes predictions from
different viewpoints comparable, on the second dataset.

Human3.6M is well suited to the test. Its four cameras are hardware
synchronized, and every one of the thirty subject-action groups in the test
split has all four cameras with identical frame counts, so pairing needs no
interpolation and no nearest-timestamp matching. Four cameras give six pairs per
group, hence 180 camera pairs against the 29 evaluated on MPI-INF-3DHP.

Nothing here was tuned. The MPI-INF-3DHP protocol reserved one development pair
for tuning and held out the rest; Human3.6M played no part in developing the
method, so all 180 pairs are held out, and both test subjects are unseen by the
canonicalization design as well as by the backbone.

Predictions are reused from `evaluation.h36m_replication`, which puts them in
camera-space millimetres through the official evaluation path. No inference runs
here, so the experiment costs seconds rather than an hour.

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_crossview
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.oracle import procrustes_cross_view_distance

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_crossview")
EVAL_STRIDE = 10  # 50 Hz -> 5 Hz, as in the bone-length replication

# Human3.6M action indices as they appear in the source strings. Confirmed
# against per-action accuracy: act_10 is the least accurate at 66.4 mm and
# act_14 the most accurate at 32.4 mm, which is the ordering the backbone's
# paper reports for SittingDown and Walking respectively.
ACTION_NAMES = {
    "act_02": "Directions", "act_03": "Discussion", "act_04": "Eating",
    "act_05": "Greeting", "act_06": "Phoning", "act_07": "Posing",
    "act_08": "Purchases", "act_09": "Sitting", "act_10": "SittingDown",
    "act_11": "Smoking", "act_12": "Photo", "act_13": "Waiting",
    "act_14": "Walking", "act_15": "WalkDog", "act_16": "WalkTogether",
}


def canonicalize_stream(poses):
    """
    Canonicalize a camera stream, threading the previous z-axis for sign
    consistency exactly as `run_eval` does on MPI-INF-3DHP.

    Returns the canonical poses and a validity mask.
    """
    out = np.zeros_like(poses)
    valid = np.zeros(len(poses), dtype=bool)
    prev_z = None
    for i, p in enumerate(poses):
        can, _, meta = canonicalize_single(p.astype(np.float32), prev_z=prev_z)
        out[i] = can
        valid[i] = bool(meta["valid"])
        if meta["valid"]:
            prev_z = meta["z_axis"]
    return out, valid


def evaluate(videos, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    results = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        raw_streams, can_streams, valid = {}, {}, {}
        for cam, poses in cams.items():
            p = poses[sel]
            p = p - p[:, 0:1, :]  # root-centre, as every other metric path does
            raw_streams[cam] = p
            can_streams[cam], valid[cam] = canonicalize_stream(p)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            _, raw_mean = cross_view_joint_distance_sequence(
                raw_streams[a][keep], raw_streams[b][keep])
            _, can_mean = cross_view_joint_distance_sequence(
                can_streams[a][keep], can_streams[b][keep])

            # Oracle: best rigid alignment of B onto A per frame. It is the
            # floor a rotation-only method can reach, and it is not achievable
            # without knowing the other view.
            oracle = np.array([
                procrustes_cross_view_distance(pa, pb)
                for pa, pb in zip(raw_streams[a][keep], raw_streams[b][keep])
            ])

            results.append({
                "subject": subj, "action": action,
                "action_name": ACTION_NAMES.get(action, action),
                "cam_a": a, "cam_b": b,
                "n_frames": int(keep.sum()),
                "validity_pct": float(100.0 * keep.sum() / len(sel)),
                "raw_cross_view_distance": float(raw_mean),
                "canonical_cross_view_distance": float(can_mean),
                "oracle_cross_view_distance": float(oracle.mean()),
                "improvement_pct": float((1 - can_mean / max(raw_mean, 1e-8)) * 100),
                "oracle_gap_closed_pct": float(
                    (raw_mean - can_mean) / max(raw_mean - oracle.mean(), 1e-8) * 100),
            })
    return results


def summarise(results):
    imp = np.array([r["improvement_pct"] for r in results])
    by_subject = {}
    for s in sorted({r["subject"] for r in results}):
        v = np.array([r["improvement_pct"] for r in results if r["subject"] == s])
        by_subject[s] = {"n_pairs": len(v), "mean_improvement_pct": float(v.mean()),
                         "min_pct": float(v.min()), "max_pct": float(v.max()),
                         "pairs_improved": int((v > 0).sum())}
    by_pair = {}
    for a, b in sorted({(r["cam_a"], r["cam_b"]) for r in results}):
        v = np.array([r["improvement_pct"] for r in results
                      if r["cam_a"] == a and r["cam_b"] == b])
        by_pair["%s-%s" % (a, b)] = {"n": len(v), "mean_improvement_pct": float(v.mean())}

    by_action = {}
    for a in sorted({r["action"] for r in results}):
        v = np.array([r["improvement_pct"] for r in results if r["action"] == a])
        can = np.mean([r["canonical_cross_view_distance"] for r in results if r["action"] == a])
        by_action[ACTION_NAMES.get(a, a)] = {
            "n": len(v), "mean_improvement_pct": float(v.mean()),
            "mean_canonical_distance_mm": float(can),
            "pairs_improved": int((v > 0).sum()),
        }

    return {
        "n_pairs": len(results),
        "by_action": by_action,
        "n_pairs_improved": int((imp > 0).sum()),
        "all_pairs_improved": bool((imp > 0).all()),
        "mean_improvement_pct": float(imp.mean()),
        "median_improvement_pct": float(np.median(imp)),
        "std_improvement_pct": float(imp.std()),
        "worst_pair_pct": float(imp.min()),
        "best_pair_pct": float(imp.max()),
        "mean_raw_distance_mm": float(np.mean([r["raw_cross_view_distance"] for r in results])),
        "mean_canonical_distance_mm": float(
            np.mean([r["canonical_cross_view_distance"] for r in results])),
        "mean_oracle_distance_mm": float(
            np.mean([r["oracle_cross_view_distance"] for r in results])),
        "mean_oracle_gap_closed_pct": float(
            np.mean([r["oracle_gap_closed_pct"] for r in results])),
        "mean_validity_pct": float(np.mean([r["validity_pct"] for r in results])),
        "by_subject": by_subject,
        "by_camera_pair": by_pair,
        "mpi_reference": {
            "held_out_pairs_mean_improvement_pct": 32.4,
            "held_out_subject_improvement_pct": 13.4,
            "n_pairs": 29,
            "note": "MPI-INF-3DHP reserved one development pair for tuning. "
                    "Human3.6M was never used for tuning, so all pairs here are "
                    "held out.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pred_path = os.path.join(PRED_DIR, "preds.npz")
    if not os.path.exists(pred_path):
        print("ERROR: %s missing. Run evaluation.h36m_replication --stage infer first."
              % pred_path)
        return
    videos = aggregate_by_video(meta, np.load(pred_path), int(np.load(pred_path)["n_clips"]))

    results = evaluate(videos, stride=args.stride)
    summary = summarise(results)

    print("=" * 78)
    print("HUMAN3.6M CROSS-VIEW CANONICALIZATION — replication of the central claim")
    print("=" * 78)
    print("  %d camera pairs over %d subject-action groups, all held out\n"
          % (summary["n_pairs"], summary["n_pairs"] // 6))
    print("  raw cross-view distance        %7.1f mm" % summary["mean_raw_distance_mm"])
    print("  canonical cross-view distance  %7.1f mm" % summary["mean_canonical_distance_mm"])
    print("  oracle (per-frame Procrustes)  %7.1f mm" % summary["mean_oracle_distance_mm"])
    print("\n  improvement  mean %+.1f%%  median %+.1f%%  worst %+.1f%%  best %+.1f%%"
          % (summary["mean_improvement_pct"], summary["median_improvement_pct"],
             summary["worst_pair_pct"], summary["best_pair_pct"]))
    print("  pairs improved: %d / %d" % (summary["n_pairs_improved"], summary["n_pairs"]))
    print("  oracle gap closed: %.1f%%   canonicalization validity: %.1f%%"
          % (summary["mean_oracle_gap_closed_pct"], summary["mean_validity_pct"]))
    print("\n  MPI-INF-3DHP reference: %+.1f%% over 27 held-out pairs, %+.1f%% held-out subject"
          % (32.4, 13.4))
    print("\n  by subject:")
    for s, v in summary["by_subject"].items():
        print("    %-4s n=%3d  mean %+6.1f%%  range [%+.1f%%, %+.1f%%]  improved %d/%d"
              % (s, v["n_pairs"], v["mean_improvement_pct"], v["min_pct"], v["max_pct"],
                 v["pairs_improved"], v["n_pairs"]))
    print("\n  by camera pair:")
    for p, v in summary["by_camera_pair"].items():
        print("    %-14s n=%2d  mean %+6.1f%%" % (p, v["n"], v["mean_improvement_pct"]))
    print("\n  by action (worst first):")
    for a, v in sorted(summary["by_action"].items(), key=lambda kv: kv[1]["mean_improvement_pct"]):
        print("    %-14s n=%2d  mean %+6.1f%%  canonical %6.1f mm  improved %d/%d"
              % (a, v["n"], v["mean_improvement_pct"], v["mean_canonical_distance_mm"],
                 v["pairs_improved"], v["n"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "h36m_crossview.json"), "w") as fh:
        json.dump({"summary": summary, "per_pair": results}, fh, indent=2)
    print("\nSaved: %s" % os.path.join(OUT_DIR, "h36m_crossview.json"))


if __name__ == "__main__":
    main()
