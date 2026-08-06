"""
Does the anatomical frame have a robustness regime the template baseline lacks?

Pre-registered in thesis_artifacts/occlusion/PREREGISTRATION.md, committed
before this ran (2014564, corrected in 9c8e787, both preceding any result).

Section 5.10 reports that Kabsch alignment onto a fixed template beats the
anatomical frame on every held-out pair. The report's answer to "then why keep
the method" is rhetorical: the baseline has no anatomical axis, so it cannot
pose the question the thesis asks. This looks for an experimental answer.

The one structural difference that is testable: the anatomical frame reads four
joints {0 root, 1 r_hip, 4 l_hip, 8 thorax}, while template Kabsch least-squares
fits all seventeen. Corrupt the eight joints past a hinge and the baseline's
alignment should rotate while ours does not move at all.

The regime is chosen because we expect to win in it. Arm C -- the same template
Kabsch restricted to the same four joints -- is what stops that being circular:
if C is also robust, the effect is joint subset, not anatomy.

Scoring is on {9 neck, 10 head, 11 l_shoulder, 14 r_shoulder}, which are
uncorrupted and non-constructor, so the metric measures damage to the alignment
and not the noise we injected.

Run:  ./venv/Scripts/python.exe -m evaluation.occlusion_robustness
      ./venv/Scripts/python.exe -m evaluation.occlusion_robustness --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys
import zlib

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import (EVAL_STRIDE, canonicalize_stream,
                                       cluster_bootstrap)
from evaluation.h36m_noncon import CONSTRUCTOR_JOINTS
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.oracle import procrustes_align
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "occlusion")

# The eight joints past a hinge: knees, feet, elbows, wrists.
CORRUPTED_JOINTS = (2, 3, 5, 6, 12, 13, 15, 16)
# Scored set: uncorrupted AND non-constructor. Neck, head, both shoulders.
SCORED_JOINTS = (9, 10, 11, 14)
# Millimetres of isotropic Gaussian noise. 0 is the identity control.
SEVERITIES = (0.0, 20.0, 40.0, 80.0, 160.0)
# A crossover only counts if it appears at or below this severity.
CROSSOVER_LIMIT_MM = 80.0
# Arm C counts as matching arm A within this relative margin.
CONTROL_MATCH_PCT = 10.0

assert not set(CORRUPTED_JOINTS) & set(SCORED_JOINTS), "scored joints must be clean"
assert not set(CORRUPTED_JOINTS) & set(CONSTRUCTOR_JOINTS), "frame must be clean"
assert not set(SCORED_JOINTS) & set(CONSTRUCTOR_JOINTS), "scored set must be free"


def seed_for(group_index, cam, sigma):
    """
    Deterministic across processes.

    Python's hash() is randomised per interpreter for strings, so it cannot be
    used here: the sweep has to reproduce exactly on a re-run.
    """
    return zlib.crc32(("%d|%s|%g" % (group_index, cam, sigma)).encode())


def corrupt(poses, sigma, seed):
    """Add isotropic Gaussian noise to the distal joints only."""
    if sigma <= 0:
        return poses
    out = poses.copy()
    rng = np.random.default_rng(seed)
    idx = list(CORRUPTED_JOINTS)
    out[:, idx, :] += rng.normal(0.0, sigma, size=(len(poses), len(idx), 3))
    return out


def align_to_template(poses, template, fit_joints=None):
    """
    Kabsch-rotate each pose onto the template, rotation only.

    `fit_joints` restricts which joints the rotation is *fitted* on; the whole
    pose is always transformed by the result. None means all seventeen, which
    is arm B and reproduces evaluation.template_baseline.align_to_template.
    """
    out = np.empty_like(poses)
    j = list(range(17)) if fit_joints is None else list(fit_joints)
    for i, p in enumerate(poses):
        R, _, _ = procrustes_align(p[j].astype(np.float32),
                                   template[j].astype(np.float32))
        out[i] = (p - p.mean(axis=0)) @ R.T
    return out


def collect(videos, template, sigma, stride=EVAL_STRIDE):
    """One severity level, all three arms, scored on SCORED_JOINTS."""
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
            # Independently per camera, so the noise does not cancel in the
            # comparison. Seed is a pure function of (group, camera, severity).
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
            # Positive means the anatomical frame is closer, i.e. better.
            row["t17_minus_anatomical_mm"] = (row["template17_mm"]
                                              - row["anatomical_mm"])
            row["t4_minus_anatomical_mm"] = (row["template4_mm"]
                                             - row["anatomical_mm"])
            rows.append(row)
    return rows


def summarise(rows, sigma):
    boot = cluster_bootstrap(rows, "t17_minus_anatomical_mm")
    lo, hi = boot["ci95"]
    if lo > 0:
        verdict = "anatomical_better"
    elif hi < 0:
        verdict = "template_better"
    else:
        verdict = "not_established"
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "sigma_mm": sigma,
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


def adjudicate(levels):
    """Apply the pre-registered criteria. No judgement beyond the criteria."""
    by_sigma = {s["sigma_mm"]: s for s in levels}
    clean = by_sigma[0.0]
    sanity = clean["verdict_vs_template17"] == "template_better"

    crossover = None
    for s in levels:
        if (s["sigma_mm"] <= CROSSOVER_LIMIT_MM
                and s["verdict_vs_template17"] == "anatomical_better"):
            crossover = s["sigma_mm"]
            break

    control_matches = None
    if crossover is not None:
        s = by_sigma[crossover]
        gap = abs(s["mean_template4_mm"] - s["mean_anatomical_mm"])
        control_matches = bool(
            gap <= CONTROL_MATCH_PCT / 100.0 * s["mean_anatomical_mm"])

    if not sanity:
        reading = "void: sanity check failed, template not better at sigma=0"
    elif crossover is None:
        reading = ("3: no crossover at or below %g mm; the template baseline "
                   "dominates including under distal corruption"
                   % CROSSOVER_LIMIT_MM)
    elif control_matches:
        reading = ("2: crossover at %g mm, but the four-joint template control "
                   "matches it; the effect is the joint subset, not anatomy"
                   % crossover)
    else:
        reading = ("1: crossover at %g mm and the control does not match; a "
                   "bounded robustness regime the baseline lacks" % crossover)

    return {"sanity_template_better_at_zero": sanity,
            "crossover_sigma_mm": crossover,
            "control_matches_anatomical": control_matches,
            "reading": reading}


def selfcheck():
    """
    Positive control on synthetic data, so a null result cannot be arithmetic.

    Two 'cameras' see the same pose under a pure rotation. The anatomical frame
    must cancel that rotation exactly, and must not move when distal joints are
    corrupted, because it never reads them. Arm B must move. Lives here rather
    than in tests/ so the audited test count stays at 76.

    Run:  ./venv/Scripts/python.exe -m evaluation.occlusion_robustness --selfcheck
    """
    rng = np.random.default_rng(0)
    pose = rng.normal(0, 200, size=(17, 3))
    pose[0] = 0.0                       # root at origin
    pose[8] = [0, 450, 0]               # a torso axis of usable length
    pose[1], pose[4] = [-140, 0, 0], [140, 0, 0]
    a = pose[None].repeat(4, axis=0)

    th = 0.7
    Q = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                  [-np.sin(th), 0, np.cos(th)]])
    b = a @ Q.T

    idx = list(SCORED_JOINTS)
    ca, _ = canonicalize_stream(a.astype(np.float32))
    cb, _ = canonicalize_stream(b.astype(np.float32))
    _, d = cross_view_joint_distance_sequence(ca[:, idx], cb[:, idx])
    assert d < 1e-3, "pure rotation must cancel exactly, got %.4f mm" % d

    a_c = corrupt(a, 80.0, seed_for(0, "0", 80.0))
    b_c = corrupt(b, 80.0, seed_for(0, "1", 80.0))
    ca2, _ = canonicalize_stream(a_c.astype(np.float32))
    cb2, _ = canonicalize_stream(b_c.astype(np.float32))
    _, d_anat = cross_view_joint_distance_sequence(ca2[:, idx], cb2[:, idx])
    assert d_anat < 1e-3, \
        "distal noise must not move the frame, got %.4f mm" % d_anat

    tmpl = pose - pose[0:1]
    ta = align_to_template(a_c, tmpl)
    tb = align_to_template(b_c, tmpl)
    _, d_t17 = cross_view_joint_distance_sequence(ta[:, idx], tb[:, idx])
    assert d_t17 > 1.0, \
        "arm B should be disturbed by distal noise, got %.4f mm" % d_t17

    print("selfcheck OK: rotation cancels (%.2e mm), anatomical unmoved by "
          "distal noise (%.2e mm), template17 moved (%.2f mm)"
          % (d, d_anat, d_t17))


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

    print("=" * 78)
    print("DISTAL CORRUPTION: ANATOMICAL FRAME vs TEMPLATE KABSCH")
    print("=" * 78)
    print("  corrupted %s" % (CORRUPTED_JOINTS,))
    print("  scored    %s  (clean, non-constructor)" % (SCORED_JOINTS,))
    print("  positive difference means the anatomical frame is closer\n")
    print("  %8s %11s %11s %11s %11s  %s"
          % ("sigma", "anatomical", "template17", "template4",
             "t17 - anat", "verdict"))

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

    verdict = adjudicate(levels)
    print("\n  READING %s" % verdict["reading"])

    out = {"verdict": verdict, "levels": levels,
           "corrupted_joints": list(CORRUPTED_JOINTS),
           "scored_joints": list(SCORED_JOINTS),
           "constructor_joints": list(CONSTRUCTOR_JOINTS),
           "crossover_limit_mm": CROSSOVER_LIMIT_MM,
           "control_match_pct": CONTROL_MATCH_PCT,
           "template_frames": n_frames, "template_streams": n_streams,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "occlusion%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
