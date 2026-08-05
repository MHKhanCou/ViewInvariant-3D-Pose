"""
Is the template baseline's advantage orientation, or translation normalisation?

Pre-registered in thesis_artifacts/translation_ablation/PREREGISTRATION.md.

The anatomical frame is root-centred: joint 0 sits at the origin. The template
baseline is centroid-centred, because procrustes_align subtracts the mean of the
point set. Centroid-centring averages seventeen joints where root-centring
relies on one, so part of the reported gap may be translation handling rather
than orientation. Section 5.10.1 says so and could not separate them.

This runs the two-by-two. Only the centring of the final output changes; the
Kabsch fit is translation-invariant and is untouched.

Run:  ./venv/Scripts/python.exe -m evaluation.translation_ablation
      ./venv/Scripts/python.exe -m evaluation.translation_ablation --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import (EVAL_STRIDE, canonicalize_stream,
                                       cluster_bootstrap)
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.oracle import procrustes_align
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "translation_ablation")


def recentre(poses, mode):
    """Re-origin a stack of poses. Rotation is already applied and unaffected."""
    if mode == "root":
        return poses - poses[:, 0:1, :]
    if mode == "centroid":
        return poses - poses.mean(axis=1, keepdims=True)
    raise ValueError(mode)


def run(videos, template, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        s, a, c = parse_video(vid)
        groups.setdefault((s, a), {})[c] = v["pred"]

    rows = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        anat, tmpl, valid = {}, {}, {}
        for cam, poses in cams.items():
            p = poses[sel]
            p = p - p[:, 0:1, :]
            can, ok = canonicalize_stream(p)
            anat[cam], valid[cam] = np.asarray(can, dtype=np.float64), ok
            rot = np.empty_like(p)
            for i, q in enumerate(p):
                R, _, _ = procrustes_align(q.astype(np.float32),
                                           template.astype(np.float32))
                rot[i] = q @ R.T          # rotation only; centring applied below
            tmpl[cam] = rot

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            row = {"subject": subj, "action": action}
            for method, store in (("anatomical", anat), ("template", tmpl)):
                for mode in ("root", "centroid"):
                    _, d = cross_view_joint_distance_sequence(
                        recentre(store[a][keep], mode),
                        recentre(store[b][keep], mode))
                    row["%s_%s_mm" % (method, mode)] = float(d)
            for mode in ("root", "centroid"):
                row["gap_%s_mm" % mode] = (row["template_%s_mm" % mode]
                                           - row["anatomical_%s_mm" % mode])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    template, _, _ = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    rows = run(videos, template)
    cell = {k: float(np.mean([r[k] for r in rows]))
            for k in ("anatomical_root_mm", "anatomical_centroid_mm",
                      "template_root_mm", "template_centroid_mm")}
    gaps = {m: cluster_bootstrap(rows, "gap_%s_mm" % m)
            for m in ("root", "centroid")}

    print("=" * 78)
    print("IS THE BASELINE'S ADVANTAGE ORIENTATION OR TRANSLATION?")
    print("=" * 78)
    print("  %d pairs. Negative gap means the template is closer.\n" % len(rows))
    print("  %-14s %14s %16s" % ("", "root-centred", "centroid-centred"))
    print("  %-14s %13.2f %15.2f"
          % ("anatomical", cell["anatomical_root_mm"],
             cell["anatomical_centroid_mm"]))
    print("  %-14s %13.2f %15.2f"
          % ("template", cell["template_root_mm"],
             cell["template_centroid_mm"]))
    print()
    for m in ("root", "centroid"):
        g = gaps[m]
        print("  gap, both %-9s %+7.2f mm   95%% CI [%+.2f, %+.2f]"
              % (m + "-centred", g["mean"], g["ci95"][0], g["ci95"][1]))
    reported = cell["template_centroid_mm"] - cell["anatomical_root_mm"]
    like = 0.5 * (gaps["root"]["mean"] + gaps["centroid"]["mean"])
    print("\n  as reported (mixed centring)  %+.2f mm" % reported)
    print("  like-for-like, averaged       %+.2f mm" % like)
    print("  attributable to translation   %+.2f mm  (%.0f%% of reported)"
          % (reported - like, 100.0 * (reported - like) / reported
             if reported else float("nan")))

    out = {"n_pairs": len(rows), "cells_mm": cell, "gaps": gaps,
           "reported_mixed_gap_mm": float(reported),
           "like_for_like_gap_mm": float(like),
           "translation_share_mm": float(reported - like),
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "translation%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nSaved: %s" % p)


if __name__ == "__main__":
    main()
