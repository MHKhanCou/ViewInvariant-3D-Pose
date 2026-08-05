"""
Cross-view distance over the joints the frame is NOT built from.

Pre-registered in thesis_artifacts/noncon/PREREGISTRATION.md, committed before
this ran.

Section 5.16.2 shows the Gram-Schmidt frame pins the joints it is constructed
from: the thorax canonicalizes at 22.1 mm and the hips at 54.4 mm against
197.5 mm for articulated joints, because the construction fixes them rather than
because the method succeeds on them. The headline figures were nevertheless
seventeen-joint averages. This recomputes them over the thirteen joints that the
construction does not touch.

CONSTRUCTOR = {0, 1, 4, 8}: the frame is y = P[8] - P[0], x_raw = P[1] - P[4].

Joint 0 is inert for the improvement percentage and not for the oracle. It is
identically zero root-relative, so it contributes a zero term to the raw and
canonical distances, which are means over per-joint Euclidean distances;
dropping it rescales both by 17/16 and leaves their ratio unchanged. The oracle
is different: procrustes_align centres on the centroid of the point set, so
removing a point at the origin shifts the centroid and changes the fitted
rotation. The improvement therefore moves because of joints 1, 4 and 8; the
oracle gap moves because of all four.

Nothing here predicts a direction. See the pre-registration.

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_noncon
      ./venv/Scripts/python.exe -m evaluation.h36m_noncon --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import (ACTION_NAMES, EVAL_STRIDE,
                                       canonicalize_stream, cluster_bootstrap)
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.oracle import procrustes_cross_view_distance

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "noncon")

# The frame is built from these four; see equation 3.1.
CONSTRUCTOR_JOINTS = (0, 1, 4, 8)
RETAINED_JOINTS = tuple(j for j in range(17) if j not in CONSTRUCTOR_JOINTS)

assert len(CONSTRUCTOR_JOINTS) == 4, "constructor set must be exactly four"
assert len(RETAINED_JOINTS) == 13, "thirteen joints must remain"
assert set(CONSTRUCTOR_JOINTS) | set(RETAINED_JOINTS) == set(range(17))


def collect(videos, joints, stride=EVAL_STRIDE):
    """Per camera pair, scored on `joints` only. Mirrors h36m_crossview.collect."""
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    idx = list(joints)
    results = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        raw_streams, can_streams, valid = {}, {}, {}
        for cam, poses in cams.items():
            p = poses[sel]
            p = p - p[:, 0:1, :]
            raw_streams[cam] = p
            can_streams[cam], valid[cam] = canonicalize_stream(p)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            # Slice to the scored joints BEFORE measuring. Canonicalization is
            # unchanged and still uses the whole pose: only the scoring changes.
            ra, rb = raw_streams[a][keep][:, idx], raw_streams[b][keep][:, idx]
            ca, cb = can_streams[a][keep][:, idx], can_streams[b][keep][:, idx]

            _, raw_mean = cross_view_joint_distance_sequence(ra, rb)
            _, can_mean = cross_view_joint_distance_sequence(ca, cb)
            oracle = np.array([procrustes_cross_view_distance(pa, pb)
                               for pa, pb in zip(ra, rb)])

            results.append({
                "subject": subj, "action": action,
                "action_name": ACTION_NAMES.get(action, action),
                "cam_a": a, "cam_b": b,
                "n_frames": int(keep.sum()),
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
    return {
        "n_pairs": len(results),
        "n_pairs_improved": int((imp > 0).sum()),
        "mean_raw_distance_mm": float(np.mean(
            [r["raw_cross_view_distance"] for r in results])),
        "mean_canonical_distance_mm": float(np.mean(
            [r["canonical_cross_view_distance"] for r in results])),
        "mean_oracle_distance_mm": float(np.mean(
            [r["oracle_cross_view_distance"] for r in results])),
        "mean_improvement_pct": float(imp.mean()),
        "mean_oracle_gap_closed_pct": float(np.mean(
            [r["oracle_gap_closed_pct"] for r in results])),
        "bootstrap_improvement_pct": cluster_bootstrap(results, "improvement_pct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    all17 = summarise(collect(videos, tuple(range(17))))
    noncon = summarise(collect(videos, RETAINED_JOINTS))

    print("=" * 78)
    print("CROSS-VIEW DISTANCE, ALL JOINTS vs NON-CONSTRUCTOR JOINTS")
    print("=" * 78)
    print("  constructor {0,1,4,8}: root, right hip, left hip, thorax")
    print("  retained (13): %s\n" % (RETAINED_JOINTS,))
    print("  %-26s %14s %18s" % ("", "all 17 joints", "13 non-constructor"))
    for label, key, unit in (
            ("mean raw distance", "mean_raw_distance_mm", "mm"),
            ("mean canonical distance", "mean_canonical_distance_mm", "mm"),
            ("mean oracle distance", "mean_oracle_distance_mm", "mm"),
            ("mean improvement", "mean_improvement_pct", "%"),
            ("oracle gap closed", "mean_oracle_gap_closed_pct", "%")):
        print("  %-26s %13.2f%s %17.2f%s"
              % (label, all17[key], unit, noncon[key], unit))
    print("  %-26s %14d %18d" % ("pairs improved",
                                 all17["n_pairs_improved"], noncon["n_pairs_improved"]))
    b = noncon["bootstrap_improvement_pct"]
    print("\n  non-constructor improvement %.2f%%  95%% CI [%.2f, %.2f] over %d groups"
          % (b["mean"], b["ci95"][0], b["ci95"][1], b["n_clusters"]))

    out = {"all_17_joints": all17, "non_constructor": noncon,
           "constructor_joints": list(CONSTRUCTOR_JOINTS),
           "retained_joints": list(RETAINED_JOINTS),
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "noncon%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nSaved: %s" % p)


if __name__ == "__main__":
    main()
