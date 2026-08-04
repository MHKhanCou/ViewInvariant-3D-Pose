"""
Calibration-free multi-view fusion on Human3.6M: replication of the third claim.

Canonicalization puts every camera's prediction in the same body-fixed frame, so
predictions from uncalibrated cameras can be averaged directly. This module tests
that on the four synchronized cameras of the Human3.6M test split, using the
predictions already computed by `evaluation.h36m_replication`.

Metric
------
Every pose is scored against ground truth with the same similarity-aligned
distance used throughout this project, which removes rotation, translation and
scale. That choice matters here: because the alignment removes rotation, the
frame a pose lives in cannot by itself flatter the score. What canonicalization
buys is the ability to AVERAGE at all. Averaging poses that sit in different
camera frames destroys them; averaging in a shared body frame does not.

Baseline
--------
`single_view_mean` is the expected error of an arbitrary camera, which is what a
deployment with one uncalibrated camera actually gets. `oracle_best_view` needs
ground truth to pick the best camera and is reported as an upper bound only.

Run:  ./venv/Scripts/python.exe -m evaluation.h36m_fusion
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.fusion import median_fuse, resolve_reflections, weighted_mean_fuse
from evaluation.gt_eval import similarity_align_error
from evaluation.h36m_crossview import ACTION_NAMES, canonicalize_stream
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, h36m_conf_to_coco, parse_video
from evaluation.reliability import compute_reliability_score

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_fusion")
EVAL_STRIDE = 25  # coarser than the cross-view run: fusion needs a GT error per frame


# H36M-17 bilateral joint indices, matching `evaluation.lifting`.
LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]


def resolve_reflections_anatomical(poses, weights=None):
    """
    Mirror resolution that also swaps left and right.

    `fusion.resolve_reflections` negates the x coordinate alone. That is a
    coordinate sign flip, not an anatomical mirror: it maps the left arm onto
    where the right arm should be while leaving the joint LABELS untouched, so
    averaging the result with an unflipped pose blends left limbs into right
    ones. A true mirror about the body-frame x axis negates x AND exchanges the
    bilateral joint indices, which is what `lifting.flip_data` does for the
    flip augmentation.

    The distinction is invisible on a symmetric standing pose and severe on a
    turning one, which is why it surfaces here on Discussion and WalkDog rather
    than on Walking. `fusion.py` is deliberately left unmodified, because the
    MPI-INF-3DHP numbers in this report were produced with it and are audited.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return poses, 0
    w = np.ones(len(poses)) if weights is None else np.asarray(weights)
    ref = poses[int(np.argmax(w))]

    aligned, n_flipped = [], 0
    for p in poses:
        mirrored = p.copy()
        mirrored[:, 0] *= -1
        mirrored[LEFT_JOINTS + RIGHT_JOINTS] = mirrored[RIGHT_JOINTS + LEFT_JOINTS]
        if np.linalg.norm(mirrored - ref) < np.linalg.norm(p - ref):
            aligned.append(mirrored)
            n_flipped += 1
        else:
            aligned.append(p)
    return np.array(aligned), n_flipped


def plain_mean_fuse(poses, _rel=None):
    """Unweighted mean after the original mirror resolution. The control."""
    aligned, _ = resolve_reflections(poses)
    return aligned.mean(axis=0)


def anatomical_mean_fuse(poses, _rel=None):
    """Unweighted mean after anatomical mirror resolution."""
    aligned, _ = resolve_reflections_anatomical(poses)
    return aligned.mean(axis=0)


def anatomical_median_fuse(poses, _rel=None):
    aligned, _ = resolve_reflections_anatomical(poses)
    return np.median(aligned, axis=0)


def naive_mean_fuse(poses, _rel=None):
    """Mean with NO reflection handling at all. Isolates what that step costs."""
    return np.asarray(poses, dtype=np.float64).mean(axis=0)


def naive_median_fuse(poses, _rel=None):
    return np.median(np.asarray(poses, dtype=np.float64), axis=0)


STRATEGIES = {
    "naive_mean": naive_mean_fuse,
    "naive_median": naive_median_fuse,
    "plain_mean": plain_mean_fuse,
    "median": lambda p, r: median_fuse(p, r),
    "reliability_weighted_mean": lambda p, r: weighted_mean_fuse(p, r),
    "anatomical_mean": anatomical_mean_fuse,
    "anatomical_median": anatomical_median_fuse,
}


def evaluate(videos, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v

    rows = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(cams[c]["pred"]) for c in order)
        sel = np.arange(0, n, stride)

        canon, rels, single_err = {}, {}, {}
        for c in order:
            p = cams[c]["pred"][sel]
            p = p - p[:, 0:1, :]
            canon[c], valid = canonicalize_stream(p)
            conf = cams[c]["conf"][sel]
            rels[c] = np.array([
                compute_reliability_score(p[i], h36m_conf_to_coco(conf[i]), None)[0]
                for i in range(len(p))
            ])
            gt = cams[c]["gt"][sel]
            gt = gt - gt[:, 0:1, :]
            single_err[c] = np.array([similarity_align_error(p[i], gt[i])
                                      for i in range(len(p))])
            if c == order[0]:
                gt_ref = gt
            canon[c] = np.where(valid[:, None, None], canon[c], np.nan)

        for i in range(len(sel)):
            poses = np.array([canon[c][i] for c in order])
            if np.isnan(poses).any():
                continue
            r = np.array([rels[c][i] for c in order])
            per_view = np.array([single_err[c][i] for c in order])

            row = {
                "subject": subj, "action": action,
                "single_view_mean": float(per_view.mean()),
                "oracle_best_view": float(per_view.min()),
                "worst_view": float(per_view.max()),
                "n_flipped": int(resolve_reflections(poses, r)[1]),
                "n_flipped_anatomical": int(resolve_reflections_anatomical(poses, r)[1]),
            }
            for name, fn in STRATEGIES.items():
                row[name] = float(similarity_align_error(fn(poses, r), gt_ref[i]))
            rows.append(row)
    return rows


def summarise(rows):
    keys = ["single_view_mean", "oracle_best_view", "worst_view"] + list(STRATEGIES)
    overall = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    base = overall["single_view_mean"]
    improvement = {k: float((1 - overall[k] / base) * 100)
                   for k in STRATEGIES}

    by_subject = {}
    for s in sorted({r["subject"] for r in rows}):
        sub = [r for r in rows if r["subject"] == s]
        b = float(np.mean([r["single_view_mean"] for r in sub]))
        by_subject[s] = {
            "n_frames": len(sub),
            "single_view_mean_mm": b,
            **{k: {"mm": float(np.mean([r[k] for r in sub])),
                   "improvement_pct": float((1 - np.mean([r[k] for r in sub]) / b) * 100)}
               for k in STRATEGIES},
        }

    by_action = {}
    for a in sorted({r["action"] for r in rows}):
        sub = [r for r in rows if r["action"] == a]
        b = float(np.mean([r["single_view_mean"] for r in sub]))
        f = float(np.mean([r["plain_mean"] for r in sub]))
        fa = float(np.mean([r["anatomical_median"] for r in sub]))
        by_action[ACTION_NAMES.get(a, a)] = {
            "n_frames": len(sub), "single_view_mm": b, "fused_mm": f,
            "improvement_pct": float((1 - f / b) * 100),
            "anatomical_median_mm": fa,
            "anatomical_median_improvement_pct": float((1 - fa / b) * 100),
        }

    best = max(STRATEGIES, key=lambda k: improvement[k])
    return {
        "n_frames": len(rows),
        "mean_views_flipped_of_4": float(np.mean([r["n_flipped"] for r in rows])),
        "mean_views_flipped_anatomical": float(
            np.mean([r["n_flipped_anatomical"] for r in rows])),
        # Is the mean's failure driven by a minority of catastrophic frames, or
        # is it uniformly bad? If the per-frame MEDIAN improvement is positive
        # while the mean improvement is negative, a few frames are doing the
        # damage, and a robust estimator is the right response.
        "per_frame_improvement_pct": {
            k: {
                "mean": float(np.mean([(1 - r[k] / max(r["single_view_mean"], 1e-8)) * 100
                                       for r in rows])),
                "median": float(np.median([(1 - r[k] / max(r["single_view_mean"], 1e-8)) * 100
                                           for r in rows])),
                "frames_improved_pct": float(100.0 * np.mean(
                    [r[k] < r["single_view_mean"] for r in rows])),
            } for k in ("naive_mean", "naive_median")
        },
        "overall_mm": overall,
        "improvement_vs_single_view_pct": improvement,
        "best_strategy": best,
        "weighting_gain_over_plain_mean_pct": float(
            improvement["reliability_weighted_mean"] - improvement["plain_mean"]),
        "actions_improved": int(sum(1 for v in by_action.values() if v["improvement_pct"] > 0)),
        "n_actions": len(by_action),
        "by_subject": by_subject,
        "by_action": by_action,
        "mpi_reference": {
            "improvement_pct_by_condition": [23.7, 10.6, 10.2, 8.2],
            "note": "MPI-INF-3DHP used up to 8 cameras; Human3.6M has 4, so a "
                    "smaller gain is expected from averaging alone.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pred_path = os.path.join(PRED_DIR, "preds.npz")
    if not os.path.exists(pred_path):
        print("ERROR: run evaluation.h36m_replication --stage infer first.")
        return
    pn = np.load(pred_path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    rows = evaluate(videos, stride=args.stride)
    s = summarise(rows)

    print("=" * 78)
    print("HUMAN3.6M CALIBRATION-FREE FUSION — four uncalibrated synchronized views")
    print("=" * 78)
    print("  %d fused frames\n" % s["n_frames"])
    o = s["overall_mm"]
    print("  single arbitrary view (baseline)  %6.1f mm" % o["single_view_mean"])
    print("  worst view                        %6.1f mm" % o["worst_view"])
    print("  oracle best view (needs GT)       %6.1f mm" % o["oracle_best_view"])
    print()
    for k in STRATEGIES:
        print("  %-28s  %6.1f mm   %+5.1f%%"
              % (k, o[k], s["improvement_vs_single_view_pct"][k]))
    print("\n  best strategy: %s" % s["best_strategy"])
    print("  reliability weighting buys %+.1f%% over a plain mean"
          % s["weighting_gain_over_plain_mean_pct"])
    print("  actions improved: %d / %d" % (s["actions_improved"], s["n_actions"]))
    print("\n  by subject:")
    for k, v in s["by_subject"].items():
        print("    %-4s baseline %6.1f mm -> plain mean %6.1f mm  (%+.1f%%)"
              % (k, v["single_view_mean_mm"], v["plain_mean"]["mm"],
                 v["plain_mean"]["improvement_pct"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "h36m_fusion.json"), "w") as fh:
        json.dump({"summary": s}, fh, indent=2)
    print("\nSaved: %s" % os.path.join(OUT_DIR, "h36m_fusion.json"))


if __name__ == "__main__":
    main()
