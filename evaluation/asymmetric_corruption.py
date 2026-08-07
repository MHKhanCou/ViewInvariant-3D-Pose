"""
Does ASYMMETRIC corruption move the Kabsch fit where symmetric corruption did not?

Pre-registered in thesis_artifacts/bestframe/PREREGISTRATION.md, committed
before this ran (7e09462). Sixteenth pre-registration.

Mechanism from the tenth and eleventh experiments: Kabsch fits a rotation, and
bilaterally symmetric perturbations leave the point cloud's principal axes
where they are, so the fit barely moves. Both previous corruptions were
symmetric -- all eight distal joints, or both limbs rescaled by one factor.

Corrupting only the LEFT distal joints displaces mass on one side, which should
rotate the fit, while the anatomical frame reads {0, 1, 4, 8} and is untouched.

The hypothesis has no literature support: a 104-agent sweep looking for a
documented regime where a landmark frame beats Procrustes under unilateral
failure returned zero surviving claims. It is run because the mechanism is
sound, and four previous searches in this family have already failed.

Everything except the corrupted set is identical to occlusion_robustness, so
the crossover severities are directly comparable.

Run:  ./venv/Scripts/python.exe -m evaluation.asymmetric_corruption
      ./venv/Scripts/python.exe -m evaluation.asymmetric_corruption --preds <cache> --tag <name>
      ./venv/Scripts/python.exe -m evaluation.asymmetric_corruption --selfcheck
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
from evaluation.h36m_noncon import CONSTRUCTOR_JOINTS
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.occlusion_robustness import (CROSSOVER_LIMIT_MM, SCORED_JOINTS,
                                             SEVERITIES, align_to_template,
                                             seed_for)
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "asymmetric")

# Left side only: knee, foot, elbow, wrist. Half of the tenth experiment's set.
CORRUPTED_JOINTS = (5, 6, 12, 13)
# The tenth experiment's bilateral set, for the comparison the pre-reg names.
SYMMETRIC_REFERENCE = (2, 3, 5, 6, 12, 13, 15, 16)

assert not set(CORRUPTED_JOINTS) & set(SCORED_JOINTS), "scored joints must be clean"
assert not set(CORRUPTED_JOINTS) & set(CONSTRUCTOR_JOINTS), "frame must be clean"
assert set(CORRUPTED_JOINTS) < set(SYMMETRIC_REFERENCE), "must be a strict subset"


def corrupt(poses, sigma, seed):
    if sigma <= 0:
        return poses
    out = poses.copy()
    rng = np.random.default_rng(seed)
    idx = list(CORRUPTED_JOINTS)
    out[:, idx, :] += rng.normal(0.0, sigma, size=(len(poses), len(idx), 3))
    return out


def collect(videos, template, sigma, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]

    idx = list(SCORED_JOINTS)
    rows = []
    for gi, ((subj, action), cams) in enumerate(sorted(groups.items())):
        if len(cams) < 4:
            continue
        n = min(len(p) for p in cams.values())
        sel = np.arange(0, n, stride)

        anat, t17, t4, valid = {}, {}, {}, {}
        for cam, poses in sorted(cams.items()):
            p = poses[sel]
            p = p - p[:, 0:1, :]
            p = corrupt(p, sigma, seed=seed_for(gi, cam, sigma))
            anat[cam], valid[cam] = canonicalize_stream(p)
            t17[cam] = align_to_template(p, template)
            t4[cam] = align_to_template(p, template, CONSTRUCTOR_JOINTS)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            row = {"subject": subj, "action": action, "cam_a": a, "cam_b": b,
                   "n_frames": int(keep.sum())}
            for name, st in (("anatomical", anat), ("template17", t17),
                             ("template4", t4)):
                _, d = cross_view_joint_distance_sequence(
                    st[a][keep][:, idx], st[b][keep][:, idx])
                row[name + "_mm"] = float(d)
            row["t17_minus_anatomical_mm"] = (row["template17_mm"]
                                              - row["anatomical_mm"])
            rows.append(row)
    return rows


def summarise(rows, sigma):
    boot = cluster_bootstrap(rows, "t17_minus_anatomical_mm")
    lo, hi = boot["ci95"]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "sigma_mm": sigma,
        "n_pairs": len(rows),
        "mean_anatomical_mm": mean("anatomical_mm"),
        "mean_template17_mm": mean("template17_mm"),
        "mean_template4_mm": mean("template4_mm"),
        "t17_minus_anatomical": boot,
        "verdict_vs_template17": ("anatomical_better" if lo > 0 else
                                  "template_better" if hi < 0 else
                                  "not_established"),
    }


def selfcheck():
    """One-sided corruption must be strictly smaller than the bilateral set."""
    assert len(CORRUPTED_JOINTS) == 4 and len(SYMMETRIC_REFERENCE) == 8
    rng = np.random.default_rng(0)
    p = rng.normal(0, 200, size=(4, 17, 3))
    c = corrupt(p.copy(), 100.0, 1)
    moved = np.linalg.norm(c - p, axis=2).mean(axis=0)
    for j in range(17):
        if j in CORRUPTED_JOINTS:
            assert moved[j] > 50.0, "joint %d should move, moved %.2f" % (j, moved[j])
        else:
            assert moved[j] < 1e-9, "joint %d must not move" % j
    right = [j for j in SYMMETRIC_REFERENCE if j not in CORRUPTED_JOINTS]
    assert all(moved[j] < 1e-9 for j in right), "right side must stay clean"
    print("selfcheck OK: only left distal joints %s move; right side %s clean"
          % (CORRUPTED_JOINTS, tuple(right)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    template, n_frames, _ = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    print("=" * 78)
    print("ASYMMETRIC (LEFT-SIDE) CORRUPTION: ANATOMICAL vs KABSCH")
    print("=" * 78)
    print("  corrupted %s  (the tenth experiment used %s)"
          % (CORRUPTED_JOINTS, SYMMETRIC_REFERENCE))
    print("  scored    %s  (clean, non-constructor)\n" % (SCORED_JOINTS,))
    print("  %8s %11s %11s %11s %11s  %s"
          % ("sigma", "anatomical", "template17", "template4", "t17 - anat",
             "verdict"))

    levels = []
    for sigma in SEVERITIES:
        s = summarise(collect(videos, template, sigma), sigma)
        levels.append(s)
        b = s["t17_minus_anatomical"]
        print("  %6.0fmm %9.2fmm %9.2fmm %9.2fmm %+9.2f  %s"
              % (sigma, s["mean_anatomical_mm"], s["mean_template17_mm"],
                 s["mean_template4_mm"], b["mean"], s["verdict_vs_template17"]))
        print("  %8s %11s %11s %11s [%+.2f, %+.2f]"
              % ("", "", "", "", b["ci95"][0], b["ci95"][1]))

    crossover = next((s["sigma_mm"] for s in levels
                      if s["sigma_mm"] <= CROSSOVER_LIMIT_MM
                      and s["verdict_vs_template17"] == "anatomical_better"), None)
    sanity = levels[0]["verdict_vs_template17"] == "template_better"
    if not sanity:
        reading = "void: template not better at sigma=0"
    elif crossover is not None:
        reading = ("1: crossover at %g mm, at or below the %g mm the tenth "
                   "pre-registration required and failed"
                   % (crossover, CROSSOVER_LIMIT_MM))
    else:
        first_any = next((s["sigma_mm"] for s in levels
                          if s["verdict_vs_template17"] == "anatomical_better"),
                         None)
        reading = ("2/3: no crossover at or below %g mm%s"
                   % (CROSSOVER_LIMIT_MM,
                      "" if first_any is None else "; first at %g mm" % first_any))
    print("\n  READING %s" % reading)

    out = {"verdict": {"sanity_template_better_at_zero": sanity,
                       "crossover_sigma_mm": crossover, "reading": reading},
           "levels": levels,
           "corrupted_joints": list(CORRUPTED_JOINTS),
           "symmetric_reference_joints": list(SYMMETRIC_REFERENCE),
           "scored_joints": list(SCORED_JOINTS),
           "crossover_limit_mm": CROSSOVER_LIMIT_MM,
           "template_frames": n_frames,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "asymmetric%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
