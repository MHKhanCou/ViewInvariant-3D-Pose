"""
Does the axis-length principle generalise from limb frames to the global frame?

Section 5.12 established that a frame's cross-view consistency is governed by the
length of the axis it is built from, on per-limb frames, and Section 5.13 showed
that result holds across two unrelated backbones. The global frame still uses two
vectors and its lateral vector is the shorter of the two available. This runs the
pre-registered test in thesis_artifacts/multilandmark/PREREGISTRATION.md.

Isolated by design: imports helpers from the existing evaluation modules and
edits none of them, so no audited number can move.

Run:  ./venv/Scripts/python.exe -m evaluation.multilandmark_eval
      ./venv/Scripts/python.exe -m evaluation.multilandmark_eval --preds <cache>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.multilandmark_frame import VARIANTS, canonicalize_multilandmark
from evaluation.h36m_crossview import EVAL_STRIDE, cluster_bootstrap
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "multilandmark")
SEED = 12345


def canonicalize_stream(poses, variant):
    """Canonicalize a camera stream, threading prev_z as run_eval does."""
    out = np.zeros_like(poses)
    valid = np.zeros(len(poses), dtype=bool)
    prev_z = None
    for i, p in enumerate(poses):
        can, _, meta = canonicalize_multilandmark(p, variant, prev_z=prev_z)
        out[i] = can
        valid[i] = bool(meta["valid"])
        if meta["valid"]:
            prev_z = meta["z_axis"]
    return out, valid


def evaluate(videos, stride=EVAL_STRIDE):
    """Per-pair canonical distance for every variant, on identical frames."""
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    rows = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(cams[c]) for c in order)
        sel = np.arange(0, n, stride)

        raw = {c: cams[c][sel] - cams[c][sel][:, 0:1, :] for c in order}
        can, ok = {}, {}
        for variant in VARIANTS:
            can[variant], ok[variant] = {}, {}
            for c in order:
                can[variant][c], ok[variant][c] = canonicalize_stream(raw[c], variant)

        for a, b in itertools.combinations(order, 2):
            # One shared validity mask across variants, so every variant is
            # scored on exactly the same frames and the comparison is paired.
            keep = np.ones(len(sel), dtype=bool)
            for variant in VARIANTS:
                keep &= ok[variant][a] & ok[variant][b]
            if keep.sum() < 10:
                continue
            row = {"subject": subj, "action": action, "cam_a": a, "cam_b": b,
                   "n_frames": int(keep.sum())}
            _, row["raw"] = cross_view_joint_distance_sequence(
                raw[a][keep], raw[b][keep])
            for variant in VARIANTS:
                _, row[variant] = cross_view_joint_distance_sequence(
                    can[variant][a][keep], can[variant][b][keep])
            rows.append(row)
    return rows


def summarise(rows):
    base = float(np.mean([r["both"] for r in rows]))
    out = {"n_pairs": len(rows),
           "mean_raw_mm": float(np.mean([r["raw"] for r in rows])),
           "baseline_both_mm": base, "variants": {}}

    for variant in VARIANTS:
        v = np.array([r[variant] for r in rows])
        # Paired difference against the baseline on identical frames, which is
        # the quantity the pre-registration names.
        for r in rows:
            r["_d_" + variant] = r["both"] - r[variant]
        boot = cluster_bootstrap(rows, "_d_" + variant, seed=SEED)
        out["variants"][variant] = {
            "mean_canonical_mm": float(v.mean()),
            "improvement_vs_baseline_pct": float((1 - v.mean() / base) * 100),
            "paired_gain_mm": boot["mean"],
            "paired_gain_ci95_mm": boot["ci95"],
            "beats_baseline": bool(boot["ci95"][0] > 0),
            "pairs_better_than_baseline": int(sum(1 for r in rows
                                                  if r[variant] < r["both"])),
        }
    for r in rows:
        for variant in VARIANTS:
            r.pop("_d_" + variant, None)

    w, s = out["variants"]["weighted"], out["variants"]["svd"]
    out["prereg"] = {
        "1_weighted_or_svd_beats_baseline": bool(w["beats_baseline"] or s["beats_baseline"]),
        "2_shoulder_beats_hip": bool(out["variants"]["shoulder_only"]["mean_canonical_mm"]
                                     < out["variants"]["hip_only"]["mean_canonical_mm"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    rows = evaluate(videos, stride=args.stride)
    s = summarise(rows)
    s["prediction_cache"] = os.path.basename(path)

    print("=" * 78)
    print("MULTI-LANDMARK FRAME — does the axis-length principle generalise?")
    print("=" * 78)
    print("  %d pairs, cache %s" % (s["n_pairs"], s["prediction_cache"]))
    print("  raw cross-view distance      %6.1f mm" % s["mean_raw_mm"])
    print("  baseline (two vectors)       %6.1f mm\n" % s["baseline_both_mm"])
    print("  %-15s %10s %10s %22s %7s" %
          ("variant", "canon mm", "vs base", "paired gain 95% CI", "better"))
    for variant in VARIANTS:
        v = s["variants"][variant]
        print("  %-15s %10.1f %9.1f%% %8.2f [%+.2f, %+.2f] mm %5d/%d"
              % (variant, v["mean_canonical_mm"], v["improvement_vs_baseline_pct"],
                 v["paired_gain_mm"], v["paired_gain_ci95_mm"][0],
                 v["paired_gain_ci95_mm"][1],
                 v["pairs_better_than_baseline"], s["n_pairs"]))
    print("\n  pre-registered predictions:")
    print("    weighted or svd beats baseline   %s" % s["prereg"]["1_weighted_or_svd_beats_baseline"])
    print("    shoulder-only beats hip-only     %s" % s["prereg"]["2_shoulder_beats_hip"])

    os.makedirs(OUT_DIR, exist_ok=True)
    name = "results%s.json" % ("_" + args.tag if args.tag else "")
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump({"summary": s, "per_pair": rows}, fh, indent=2)
    print("\nSaved: %s" % os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
