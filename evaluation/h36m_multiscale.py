"""
Multi-scale per-limb canonicalization on Human3.6M.

The global body frame removes orientation variance aligned with the torso.
Limb orientation that differs from the torso survives it, so `canonical/
multiscale.py` builds one Gram-Schmidt frame per segment. On MPI-INF-3DHP that
extension improved all twenty-nine camera pairs. This module repeats the test on
the 180 held-out Human3.6M pairs, reusing the predictions already computed by
`evaluation.h36m_replication`.

Protocol notes
--------------
The comparison is internally consistent by construction: both the global and the
multi-scale distance come from `multiscale_canonicalize`, so the baseline is the
same code path as the treatment. That matters because the cross-view experiment
in `h36m_crossview.py` threads the previous frame's forward axis for temporal
sign consistency, which `multiscale_canonicalize` does not do. Mixing the two
would compare a threaded baseline against an unthreaded treatment and attribute
the difference to multi-scale. The absolute numbers here are therefore not
directly comparable with those of `h36m_crossview.py`; the ratio between them is
what this module reports, exactly as `multiscale_eval.py` does on MPI-INF-3DHP.

Each camera stream is canonicalized once and reused across the six pairs it
participates in, rather than recomputed per pair.

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_multiscale
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
from canonical.multiscale import (LEVELS, SEGMENTS, _gram_schmidt_frame,
                                  multiscale_canonicalize)
from evaluation.h36m_crossview import ACTION_NAMES, EVAL_STRIDE, cluster_bootstrap
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_multiscale")

TRUNK = [0, 7, 8, 9, 10]
LIMBS = ["left_arm", "right_arm", "left_leg", "right_leg"]

# The shipped definitions in canonical/multiscale.py are not bilaterally
# symmetric. Both arms take the thorax-to-shoulder vector as primary axis and
# the shoulder-to-elbow vector as secondary. The legs disagree with each other:
# the left leg takes root-to-hip as primary, while the right leg takes
# hip-to-knee. Since root-to-hip is roughly half a hip width and hip-to-knee is
# a full thigh, and frame sensitivity to joint error scales inversely with the
# length of the defining vector, the two legs are not measuring the same thing.
# We test the symmetric alternative rather than assume it is better.
#
# name -> (joint ids, segment root, (y_from, y_to), (x_from, x_to))
SEGMENTS_SYMMETRIC = dict(SEGMENTS)
SEGMENTS_SYMMETRIC["left_leg"] = ([1, 2, 3], 1, (1, 2), (2, 3))   # match right_leg
SEGMENTS_SYMMETRIC["right_leg"] = ([4, 5, 6], 4, (4, 5), (5, 6))  # unchanged

SEGMENT_SETS = {"shipped": SEGMENTS, "symmetric": SEGMENTS_SYMMETRIC}


def canonicalize_with(pose, segments):
    """
    `multiscale_canonicalize` with the segment table supplied rather than taken
    from module scope. Behaviour is identical when `segments is SEGMENTS`;
    canonical/multiscale.py is left untouched because the MPI-INF-3DHP numbers
    in this report were produced with it and are audited.
    """
    pose = np.asarray(pose, dtype=np.float32)
    can_global, _, meta_global = canonicalize_single(pose)
    out = {"global": {"joints": can_global, "joint_ids": list(range(17)),
                      "valid": bool(meta_global["valid"]), "fallback": False}}
    P = pose - pose[0:1]
    for name, (ids, root, (y0, y1), (x0, x1)) in segments.items():
        R = _gram_schmidt_frame(P[y1] - P[y0], P[x1] - P[x0])
        fallback = R is None
        seg = P[ids] - P[root:root + 1]
        if fallback:
            joints = (can_global[ids] - can_global[root:root + 1]
                      if meta_global["valid"] else np.zeros((len(ids), 3), np.float32))
        else:
            joints = seg @ R
        out[name] = {"joints": joints.astype(np.float32), "joint_ids": ids,
                     "valid": bool((not fallback) or meta_global["valid"]),
                     "fallback": fallback}
    return out


def combined_from_ms(ms_a, ms_b):
    """`combined_cross_view_distance` on already-canonicalized poses."""
    ga, gb = ms_a["global"], ms_b["global"]
    if not (ga["valid"] and gb["valid"]):
        return float("nan")
    total = np.linalg.norm(ga["joints"][TRUNK] - gb["joints"][TRUNK], axis=1).sum()
    n = len(TRUNK)
    for name in LIMBS:
        a, b = ms_a[name], ms_b[name]
        total += np.linalg.norm(a["joints"] - b["joints"], axis=1).sum()
        n += len(a["joint_ids"])
    return float(total / n)


def evaluate(videos, stride=EVAL_STRIDE, segments=SEGMENTS):
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    results, fallback_counts, n_poses = [], {s: 0 for s in SEGMENTS}, 0
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        # Canonicalize each camera once; every pair below reuses it.
        ms = {}
        for c in order:
            p = cams[c][sel]
            p = p - p[:, 0:1, :]
            ms[c] = [canonicalize_with(f, segments) for f in p]
            for m in ms[c]:
                for s in segments:
                    fallback_counts[s] += int(m[s]["fallback"])
            n_poses += len(ms[c])

        for a, b in itertools.combinations(order, 2):
            per_level = {lv: [] for lv in LEVELS}
            combined = []
            for ma, mb in zip(ms[a], ms[b]):
                for lv in LEVELS:
                    x, y = ma[lv], mb[lv]
                    if x["valid"] and y["valid"]:
                        per_level[lv].append(float(np.mean(
                            np.linalg.norm(x["joints"] - y["joints"], axis=1))))
                c = combined_from_ms(ma, mb)
                if not np.isnan(c):
                    combined.append(c)
            if not combined or not per_level["global"]:
                continue

            g = float(np.mean(per_level["global"]))
            cm = float(np.mean(combined))
            results.append({
                "subject": subj, "action": action,
                "action_name": ACTION_NAMES.get(action, action),
                "cam_a": a, "cam_b": b, "n_frames": len(combined),
                "global": g, "combined_multiscale": cm,
                "multiscale_vs_global_pct": float((1 - cm / max(g, 1e-8)) * 100),
                "per_level": {lv: (float(np.mean(v)) if v else None)
                              for lv, v in per_level.items()},
            })

    fallback_rate = {s: float(100.0 * c / max(n_poses, 1))
                     for s, c in fallback_counts.items()}
    return results, fallback_rate


def summarise(results, fallback_rate):
    d = np.array([r["multiscale_vs_global_pct"] for r in results])
    levels = {lv: float(np.mean([r["per_level"][lv] for r in results
                                 if r["per_level"][lv] is not None]))
              for lv in LEVELS}

    by_subject = {}
    for s in sorted({r["subject"] for r in results}):
        v = np.array([r["multiscale_vs_global_pct"] for r in results
                      if r["subject"] == s])
        by_subject[s] = {"n_pairs": len(v), "mean_pct": float(v.mean()),
                         "pairs_improved": int((v > 0).sum())}

    by_action = {}
    for a in sorted({r["action"] for r in results}):
        v = np.array([r["multiscale_vs_global_pct"] for r in results
                      if r["action"] == a])
        by_action[ACTION_NAMES.get(a, a)] = {
            "n": len(v), "mean_pct": float(v.mean()),
            "pairs_improved": int((v > 0).sum())}

    return {
        "n_pairs": len(results),
        "bootstrap": {
            "multiscale_vs_global_pct": cluster_bootstrap(
                results, "multiscale_vs_global_pct"),
            "combined_multiscale_mm": cluster_bootstrap(results, "combined_multiscale"),
            "unit": "subject-action group (6 pairs each)",
        },
        "n_pairs_improved": int((d > 0).sum()),
        "all_pairs_improved": bool((d > 0).all()),
        "mean_multiscale_vs_global_pct": float(d.mean()),
        "median_multiscale_vs_global_pct": float(np.median(d)),
        "worst_pair_pct": float(d.min()),
        "best_pair_pct": float(d.max()),
        "mean_global_distance_mm": float(np.mean([r["global"] for r in results])),
        "mean_multiscale_distance_mm": float(
            np.mean([r["combined_multiscale"] for r in results])),
        "per_level_distance_mm": levels,
        "segment_fallback_rate_pct": fallback_rate,
        "by_subject": by_subject,
        "by_action": by_action,
        "mpi_reference": {
            "pairs_improved": "29 of 29",
            "note": "On MPI-INF-3DHP one pair was the development pair. Here all "
                    "180 pairs are held out.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    ap.add_argument("--preds", default=None,
                    help="prediction cache to analyse; defaults to MotionAGFormer-XS")
    ap.add_argument("--tag", default=None, help="suffix for the output filename")
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pred_path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    if not os.path.exists(pred_path):
        print("ERROR: run evaluation.h36m_replication --stage infer first.")
        return
    pn = np.load(pred_path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    results, fallback = evaluate(videos, stride=args.stride, segments=SEGMENTS)
    s = summarise(results, fallback)

    sym_results, sym_fallback = evaluate(videos, stride=args.stride,
                                         segments=SEGMENTS_SYMMETRIC)
    sym = summarise(sym_results, sym_fallback)
    s["symmetric_leg_variant"] = {
        "mean_multiscale_vs_global_pct": sym["mean_multiscale_vs_global_pct"],
        "mean_multiscale_distance_mm": sym["mean_multiscale_distance_mm"],
        "n_pairs_improved": sym["n_pairs_improved"],
        "per_level_distance_mm": sym["per_level_distance_mm"],
        "note": "Both legs use hip-to-knee as primary axis and knee-to-ankle as "
                "secondary. The shipped table uses root-to-hip for the left leg "
                "only. canonical/multiscale.py is unmodified.",
    }

    print("=" * 78)
    print("HUMAN3.6M MULTI-SCALE PER-LIMB CANONICALIZATION")
    print("=" * 78)
    print("  %d held-out camera pairs\n" % s["n_pairs"])
    print("  global frame only          %7.1f mm" % s["mean_global_distance_mm"])
    print("  multi-scale combined       %7.1f mm" % s["mean_multiscale_distance_mm"])
    print("  improvement  mean %+.1f%%  median %+.1f%%  worst %+.1f%%  best %+.1f%%"
          % (s["mean_multiscale_vs_global_pct"], s["median_multiscale_vs_global_pct"],
             s["worst_pair_pct"], s["best_pair_pct"]))
    print("  pairs improved: %d / %d   (MPI-INF-3DHP: 29 of 29)"
          % (s["n_pairs_improved"], s["n_pairs"]))
    b = s["bootstrap"]["multiscale_vs_global_pct"]
    print("  cluster bootstrap over %d groups: %+.1f%%  95%% CI [%+.1f%%, %+.1f%%]"
          % (b["n_clusters"], b["mean"], *b["ci95"]))
    print("\n  per level (cross-view distance of that level's joints):")
    for lv in LEVELS:
        print("    %-11s %7.1f mm" % (lv, s["per_level_distance_mm"][lv]))
    v = s["symmetric_leg_variant"]
    print("\n  bilaterally symmetric leg axes (both legs hip->knee, knee->ankle):")
    print("    left_leg  %7.1f mm  (shipped %7.1f mm)"
          % (v["per_level_distance_mm"]["left_leg"], s["per_level_distance_mm"]["left_leg"]))
    print("    right_leg %7.1f mm  (shipped %7.1f mm)"
          % (v["per_level_distance_mm"]["right_leg"], s["per_level_distance_mm"]["right_leg"]))
    print("    combined  %7.1f mm  (shipped %7.1f mm)   improvement %+.1f%% vs %+.1f%%"
          % (v["mean_multiscale_distance_mm"], s["mean_multiscale_distance_mm"],
             v["mean_multiscale_vs_global_pct"], s["mean_multiscale_vs_global_pct"]))
    print("    pairs improved %d / %d" % (v["n_pairs_improved"], s["n_pairs"]))
    print("\n  segment fallback to global frame:")
    for seg, r in s["segment_fallback_rate_pct"].items():
        print("    %-11s %5.2f%%" % (seg, r))
    print("\n  by subject:")
    for k, v in s["by_subject"].items():
        print("    %-4s n=%3d  mean %+6.1f%%  improved %d/%d"
              % (k, v["n_pairs"], v["mean_pct"], v["pairs_improved"], v["n_pairs"]))
    print("\n  by action (worst first):")
    for a, v in sorted(s["by_action"].items(), key=lambda kv: kv[1]["mean_pct"]):
        print("    %-14s n=%2d  mean %+6.1f%%  improved %d/%d"
              % (a, v["n"], v["mean_pct"], v["pairs_improved"], v["n"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    name = "h36m_multiscale%s.json" % ("_" + args.tag if args.tag else "")
    s["prediction_cache"] = os.path.basename(pred_path)
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump({"summary": s, "per_pair": results}, fh, indent=2)
    print("\nSaved: %s" % os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
