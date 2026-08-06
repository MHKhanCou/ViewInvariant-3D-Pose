"""
Does the Kabsch baseline's win depend on the template matching the subject?

Pre-registered in thesis_artifacts/mismatch/PREREGISTRATION.md, committed
before this ran (1ddcd31). Eleventh pre-registration.

Section 5.10's baseline needs a reference skeleton whose proportions resemble
the subject's. The anatomical frame needs no reference skeleton at all: it
reads a hip axis and a torso axis off the subject's own body. All three
published template ablations used adult templates of ordinary proportion, so
none of them varied this.

Here the template's limb bones are retargeted by a factor f, walking the H36M
kinematic tree so downstream joints follow their parents. Torso, hip, shoulder,
neck and head bones are untouched, so f changes the limb-to-torso ratio alone.

A uniformly scaled template would be a no-op: Kabsch here is rotation-only, and
scaling the cross-covariance by a positive constant leaves its SVD rotation
factors unchanged. `--selfcheck` asserts exactly that, so a null result cannot
be the retargeting silently doing nothing.

Run:  ./venv/Scripts/python.exe -m evaluation.template_mismatch
      ./venv/Scripts/python.exe -m evaluation.template_mismatch --preds <cache> --tag <name>
      ./venv/Scripts/python.exe -m evaluation.template_mismatch --selfcheck
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
from evaluation.h36m_noncon import CONSTRUCTOR_JOINTS, RETAINED_JOINTS
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.occlusion_robustness import align_to_template
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "mismatch")

# H36M-17 parent of each joint; -1 for the root.
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
# The eight bones past a hinge: to knees, feet, elbows, wrists. These are the
# ones a child-versus-adult proportion difference lands on.
LIMB_BONES = (2, 3, 5, 6, 12, 13, 15, 16)
# 1.0 is the identity control and must reproduce the published figures.
FACTORS = (0.6, 0.8, 1.0, 1.2, 1.4)
# A crossover only counts inside this mismatch.
MISMATCH_LIMIT = 0.4
CONTROL_MATCH_PCT = 10.0
# Section 5.10, thesis_artifacts/template/template.json, non_constructor, XS.
PUBLISHED_ANATOMICAL_MM = 93.35026396195683
PUBLISHED_TEMPLATE_MM = 57.470357333918216
SANITY_TOL_MM = 0.05

assert len(PARENTS) == 17
assert not set(LIMB_BONES) & set(CONSTRUCTOR_JOINTS), "frame joints must not move"


def retarget(template, factor):
    """
    Scale the limb bones by `factor`, walking the tree from the root.

    Joints are visited parent-before-child, which PARENTS guarantees since each
    parent index is smaller than its child's.
    """
    out = np.array(template, dtype=np.float64, copy=True)
    for j in range(1, 17):
        p = PARENTS[j]
        bone = np.asarray(template[j], dtype=np.float64) - np.asarray(
            template[p], dtype=np.float64)
        if j in LIMB_BONES:
            bone = bone * factor
        out[j] = out[p] + bone
    return out - out[0:1]


def collect(videos, template, joints, stride=EVAL_STRIDE):
    """All three arms at one template, scored on `joints`."""
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

        anat, t17, t4, valid = {}, {}, {}, {}
        for cam, poses in sorted(cams.items()):
            p = poses[sel]
            p = p - p[:, 0:1, :]
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
            row["t4_minus_anatomical_mm"] = (row["template4_mm"]
                                             - row["anatomical_mm"])
            rows.append(row)
    return rows


def summarise(rows, factor):
    boot = cluster_bootstrap(rows, "t17_minus_anatomical_mm")
    lo, hi = boot["ci95"]
    verdict = ("anatomical_better" if lo > 0 else
               "template_better" if hi < 0 else "not_established")
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "limb_factor": factor,
        "n_pairs": len(rows),
        "mean_anatomical_mm": mean("anatomical_mm"),
        "mean_template17_mm": mean("template17_mm"),
        "mean_template4_mm": mean("template4_mm"),
        "pairs_where_anatomical_better": int(sum(
            r["template17_mm"] > r["anatomical_mm"] for r in rows)),
        "t17_minus_anatomical": boot,
        "t4_minus_anatomical": cluster_bootstrap(rows, "t4_minus_anatomical_mm"),
        "verdict_vs_template17": verdict,
    }


def adjudicate(levels, check_sanity=True):
    """Apply the pre-registered criteria, and nothing beyond them."""
    by_f = {round(s["limb_factor"], 4): s for s in levels}
    ident = by_f[1.0]
    sanity = True
    if check_sanity:
        sanity = bool(
            abs(ident["mean_anatomical_mm"] - PUBLISHED_ANATOMICAL_MM) < SANITY_TOL_MM
            and abs(ident["mean_template17_mm"] - PUBLISHED_TEMPLATE_MM) < SANITY_TOL_MM)

    crossover = None
    for s in sorted(levels, key=lambda s: abs(s["limb_factor"] - 1.0)):
        if (abs(s["limb_factor"] - 1.0) <= MISMATCH_LIMIT + 1e-9
                and s["verdict_vs_template17"] == "anatomical_better"):
            crossover = s["limb_factor"]
            break

    control_matches = None
    if crossover is not None:
        s = by_f[round(crossover, 4)]
        gap = abs(s["mean_template4_mm"] - s["mean_anatomical_mm"])
        control_matches = bool(
            gap <= CONTROL_MATCH_PCT / 100.0 * s["mean_anatomical_mm"])

    if not sanity:
        reading = ("void: identity template did not reproduce the published "
                   "Section 5.10 figures")
    elif crossover is None:
        reading = ("3: no crossover within |f-1| <= %g; the baseline is robust "
                   "to body-proportion mismatch and its advantage is "
                   "unqualified on this data" % MISMATCH_LIMIT)
    elif control_matches:
        reading = ("2-control: crossover at f=%g but the four-joint template "
                   "control matches; the effect is the joint subset, not the "
                   "absence of a reference skeleton" % crossover)
    else:
        reading = ("1: crossover at f=%g and the control does not match; the "
                   "baseline's dominance is conditional on a template that "
                   "matches the subject's proportions" % crossover)

    return {"sanity_reproduces_published": sanity,
            "crossover_limb_factor": crossover,
            "control_matches_anatomical": control_matches,
            "reading": reading}


def selfcheck():
    """
    A uniform scale must be a no-op; a proportion change must not be.

    If retargeting silently did nothing, every level would return the identity
    numbers and reading 3 would fire for the wrong reason. This rules that out.
    """
    tmpl, _, _ = build_template()
    rng = np.random.default_rng(0)
    poses = rng.normal(0, 200, size=(8, 17, 3)).astype(np.float32)
    poses[:, 0] = 0.0

    uniform = (np.asarray(tmpl, dtype=np.float64) * 1.7)
    a = align_to_template(poses, np.asarray(tmpl, dtype=np.float64))
    b = align_to_template(poses, uniform)
    delta_uniform = float(np.abs(a - b).max())
    assert delta_uniform < 1e-4, \
        "uniform scale must not change the rotation, moved %.6f" % delta_uniform

    prop = retarget(tmpl, 0.6)
    c = align_to_template(poses, prop)
    delta_prop = float(np.abs(a - c).max())
    assert delta_prop > 1.0, \
        "proportion change must alter the fit, moved only %.6f" % delta_prop

    for f in FACTORS:
        r = retarget(tmpl, f)
        for j in CONSTRUCTOR_JOINTS:
            moved = float(np.abs(r[j] - (np.asarray(tmpl) - np.asarray(tmpl)[0:1])[j]).max())
            assert moved < 1e-9, \
                "constructor joint %d moved by %.6f at f=%g" % (j, moved, f)

    print("selfcheck OK: uniform scale is a no-op (%.2e), proportion change "
          "is not (%.2f mm), constructor joints never move" % (delta_uniform, delta_prop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    base, n_frames, n_streams = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    print("=" * 78)
    print("TEMPLATE PROPORTION MISMATCH: ANATOMICAL FRAME vs KABSCH")
    print("=" * 78)
    print("  limb bones scaled: %s" % (LIMB_BONES,))
    print("  scored on the 13 non-constructor joints, as Section 5.10")
    print("  positive difference means the anatomical frame is closer\n")
    print("  %8s %11s %11s %11s %11s  %s"
          % ("f", "anatomical", "template17", "template4", "t17 - anat",
             "verdict"))

    levels = []
    for f in FACTORS:
        tmpl = retarget(base, f)
        s = summarise(collect(videos, tmpl, RETAINED_JOINTS), f)
        levels.append(s)
        b = s["t17_minus_anatomical"]
        print("  %8.2f %9.2fmm %9.2fmm %9.2fmm %+9.2f  %s"
              % (f, s["mean_anatomical_mm"], s["mean_template17_mm"],
                 s["mean_template4_mm"], b["mean"], s["verdict_vs_template17"]))
        print("  %8s %11s %11s %11s [%+.2f, %+.2f]"
              % ("", "", "", "", b["ci95"][0], b["ci95"][1]))

    # The sanity criterion is stated against the XS figures, so it only applies
    # to the default cache. On another backbone the identity level is still run
    # and reported; it simply has no published pair to match.
    is_default = args.preds is None
    verdict = adjudicate(levels, check_sanity=is_default)
    print("\n  READING %s" % verdict["reading"])

    out = {"verdict": verdict, "levels": levels,
           "limb_bones": list(LIMB_BONES), "parents": list(PARENTS),
           "scored_joints": list(RETAINED_JOINTS),
           "constructor_joints": list(CONSTRUCTOR_JOINTS),
           "mismatch_limit": MISMATCH_LIMIT,
           "control_match_pct": CONTROL_MATCH_PCT,
           "sanity_checked": is_default,
           "template_frames": n_frames, "template_streams": n_streams,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "mismatch%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
