"""
Does the BEST frame construction beat Kabsch-to-template, or only the default?

Pre-registered in thesis_artifacts/bestframe/PREREGISTRATION.md, committed
before this ran (7e09462). Fifteenth pre-registration.

The template comparison that concludes a simpler baseline wins calls
`canonicalize_stream`, which is the default two-vector construction with the
hip axis primary. But this report's own confirmed level-one result is that the
longer shoulder axis makes a better frame -- 5.2 percent on MotionAGFormer-XS
and 4.4 on MotionBERT. The headline comparison was therefore run against a
construction the thesis itself predicts is suboptimal, and the best one has
never been put against Kabsch.

Scored on the nine joints that are a constructor for NO tested variant, so no
construction is flattered by joints its own frame pins.

Run:  ./venv/Scripts/python.exe -m evaluation.best_frame_baseline
      ./venv/Scripts/python.exe -m evaluation.best_frame_baseline --preds <cache> --tag <name>
      ./venv/Scripts/python.exe -m evaluation.best_frame_baseline --selfcheck
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
from evaluation.occlusion_robustness import align_to_template
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "bestframe")

# Union of constructors over all five variants: root/spine/thorax/neck chain
# (0,7,8,9), both hips (1,4), both shoulders (11,14). Anything a variant builds
# from is excluded so the scoring set is common and no frame is flattered.
UNION_CONSTRUCTORS = (0, 1, 4, 7, 8, 9, 11, 14)
SCORED = tuple(j for j in range(17) if j not in UNION_CONSTRUCTORS)
ALL17 = tuple(range(17))

assert SCORED == (2, 3, 5, 6, 10, 12, 13, 15, 16), SCORED


def canonicalize_stream_variant(poses, variant):
    """Mirror h36m_crossview.canonicalize_stream, with the variant selectable."""
    out = np.empty_like(poses)
    valid = np.zeros(len(poses), dtype=bool)
    prev_z = None
    for i, p in enumerate(poses):
        can, _, meta = canonicalize_multilandmark(p, variant=variant,
                                                  prev_z=prev_z)
        out[i] = can
        valid[i] = bool(meta["valid"])
        if meta["valid"]:
            prev_z = meta["z_axis"]
    return out, valid


def collect(videos, template, joints, stride=EVAL_STRIDE):
    """Every variant plus the template arm, on the same pairs and joints."""
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    idx = list(joints)
    rows = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        streams = {v: {} for v in VARIANTS}
        tmpl, valid = {}, {}
        for cam, poses in sorted(cams.items()):
            p = poses[sel]
            p = p - p[:, 0:1, :]
            ok = None
            for v in VARIANTS:
                streams[v][cam], vv = canonicalize_stream_variant(p, v)
                ok = vv if ok is None else (ok & vv)
            valid[cam] = ok
            tmpl[cam] = align_to_template(p, template)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            row = {"subject": subj, "action": action, "cam_a": a, "cam_b": b,
                   "n_frames": int(keep.sum())}
            _, dt = cross_view_joint_distance_sequence(
                tmpl[a][keep][:, idx], tmpl[b][keep][:, idx])
            row["template_mm"] = float(dt)
            for v in VARIANTS:
                _, d = cross_view_joint_distance_sequence(
                    streams[v][a][keep][:, idx], streams[v][b][keep][:, idx])
                row[v + "_mm"] = float(d)
                # positive => the frame is closer than the template, i.e. better
                row["gain_" + v] = float(dt - d)
            rows.append(row)
    return rows


def summarise(rows):
    out = {"n_pairs": len(rows),
           "mean_template_mm": float(np.mean([r["template_mm"] for r in rows]))}
    variants = {}
    for v in VARIANTS:
        boot = cluster_bootstrap(rows, "gain_" + v)
        lo, hi = boot["ci95"]
        variants[v] = {
            "mean_mm": float(np.mean([r[v + "_mm"] for r in rows])),
            "mean_gain_vs_template_mm": boot["mean"],
            "gain_ci95": boot["ci95"],
            "pairs_beating_template": int(sum(r["gain_" + v] > 0 for r in rows)),
            "verdict": ("frame_better" if lo > 0 else
                        "template_better" if hi < 0 else "not_established"),
        }
    out["variants"] = variants
    best = min(VARIANTS, key=lambda v: variants[v]["mean_mm"])
    out["best_variant"] = best
    out["best_minus_default_mm"] = float(variants["both"]["mean_mm"]
                                         - variants[best]["mean_mm"])
    return out


def selfcheck():
    """`both` must reproduce the published two-vector path exactly."""
    from evaluation.h36m_crossview import canonicalize_stream
    rng = np.random.default_rng(0)
    poses = rng.normal(0, 200, size=(6, 17, 3)).astype(np.float32)
    poses[:, 0] = 0.0
    poses[:, 8] = [0, 450, 0]
    poses[:, 1], poses[:, 4] = [-140, 0, 0], [140, 0, 0]
    a, va = canonicalize_stream(poses)
    b, vb = canonicalize_stream_variant(poses, "both")
    assert (va == vb).all(), "validity masks diverge"
    d = float(np.abs(a - b).max())
    assert d < 1e-6, "'both' must reproduce the baseline exactly, got %.3e" % d
    assert len(SCORED) == 9 and not set(SCORED) & set(UNION_CONSTRUCTORS)
    print("selfcheck OK: 'both' reproduces the published frame (%.1e), "
          "scored set is %s" % (d, (SCORED,)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    template, n_frames, n_streams = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    out = {}
    for label, joints in (("scored_9", SCORED), ("all_17_joints", ALL17)):
        s = summarise(collect(videos, template, joints))
        out[label] = s
        print("=" * 78)
        print("BEST FRAME CONSTRUCTION vs KABSCH-TO-TEMPLATE  [%s]" % label)
        print("=" * 78)
        print("  template          %7.2f mm" % s["mean_template_mm"])
        print("  %-14s %9s %12s %22s  %s"
              % ("variant", "mm", "gain vs t", "CI95", "verdict"))
        for v in VARIANTS:
            d = s["variants"][v]
            print("  %-14s %7.2f mm %+10.2f  [%+7.2f, %+7.2f]  %s"
                  % (v, d["mean_mm"], d["mean_gain_vs_template_mm"],
                     d["gain_ci95"][0], d["gain_ci95"][1], d["verdict"]))
        print("  best: %s, %.2f mm better than the published default\n"
              % (s["best_variant"], s["best_minus_default_mm"]))

    prim = out["scored_9"]["variants"]
    any_beats = [v for v in VARIANTS if prim[v]["verdict"] == "frame_better"]
    out["verdict"] = {
        "sanity_template_beats_default": prim["both"]["verdict"] == "template_better",
        "variants_beating_template": any_beats,
        "reading": ("1: a construction beats Kabsch (%s)" % any_beats) if any_beats
        else ("2/3: no construction beats Kabsch; best is %s, gap %.2f mm"
              % (out["scored_9"]["best_variant"],
                 -prim[out["scored_9"]["best_variant"]]["mean_gain_vs_template_mm"])),
    }
    out["union_constructors"] = list(UNION_CONSTRUCTORS)
    out["scored_joints"] = list(SCORED)
    out["template_frames"] = n_frames
    out["prediction_cache"] = os.path.basename(path)
    print("  READING %s" % out["verdict"]["reading"])

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "bestframe%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
