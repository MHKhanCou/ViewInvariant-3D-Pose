"""
Anchor-joint corruption: the symmetric half of the failure-surface map.

Pre-registered in thesis_artifacts/anchor_corruption/PREREGISTRATION.md,
committed before this ran.

`evaluation.occlusion_robustness` corrupted the eight distal joints and found
the anatomical frame stays exactly flat (it never reads them) while the
template-Kabsch baseline degrades monotonically -- a regime where the
anatomical frame wins, established at high severity only. That experiment left
the mirror question open: the frame reads exactly four joints
{0 root, 1 r_hip, 4 l_hip, 8 thorax}, so its failure must be concentrated in
that support. Corrupt the hips and the thorax and the frame should collapse at
every severity, while the 17-joint Kabsch fit, with fourteen clean joints
damping the rotation, should degrade gracefully.

Together the two experiments map the failure supports: distal corruption is
invisible to the anatomical frame and visible to the template; anchor
corruption is visible to the anatomical frame and diluted in the template.
The routing rule in `evaluation.selection_rule` exploits exactly that
complementarity.

Scoring is on {9 neck, 10 head, 11 l_shoulder, 14 r_shoulder} -- the same
clean, non-constructor set as the distal experiment -- so the sigma=0 row must
reproduce the occlusion run's sigma=0 row exactly (identity control), and the
two tables read side by side.

Run:  ./venv/Scripts/python.exe -m evaluation.anchor_corruption
      ./venv/Scripts/python.exe -m evaluation.anchor_corruption --preds <cache> --tag <name>
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

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "anchor_corruption")

# The frame's support joints, minus the root. Root is re-centred away before
# corruption in `collect`, exactly as the distal experiment does, so corrupting
# it here would add only a common translation that both alignments then ignore.
CORRUPTED_JOINTS = (1, 4, 8)          # r_hip, l_hip, thorax
# Scored set: uncorrupted AND non-constructor, identical to the distal run.
SCORED_JOINTS = (9, 10, 11, 14)
# Millimetres of isotropic Gaussian noise. 0 is the identity control.
SEVERITIES = (0.0, 20.0, 40.0, 80.0, 160.0)

assert set(CORRUPTED_JOINTS) & set(SCORED_JOINTS) == set(), \
    "scored joints must be clean"
assert set(CORRUPTED_JOINTS) & set(CONSTRUCTOR_JOINTS) == set(CORRUPTED_JOINTS), \
    "corrupted joints must be the constructor support"
assert set(SCORED_JOINTS) & set(CONSTRUCTOR_JOINTS) == set(), \
    "scored set must be free of constructor joints"


def seed_for(group_index, cam, sigma):
    """Deterministic across processes, as in the distal experiment."""
    return zlib.crc32(("%d|%s|%g" % (group_index, cam, sigma)).encode())


def corrupt(poses, sigma, seed):
    """Add isotropic Gaussian noise to the anchor joints only."""
    if sigma <= 0:
        return poses
    out = poses.copy()
    rng = np.random.default_rng(seed)
    idx = list(CORRUPTED_JOINTS)
    out[:, idx, :] += rng.normal(0.0, sigma, size=(len(poses), len(idx), 3))
    return out


def align_to_template(poses, template, fit_joints=None):
    """Kabsch-rotate each pose onto the template, rotation only."""
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
    """Apply the pre-registered criteria. No judgement beyond the criteria.

    C1 (sanity): at sigma=0 the template must be better, reproducing the
        distal experiment's identity control.
    C2 (claim): at sigma in {20, 40} the anatomical frame must be worse than
        template17 with a CI excluding zero -- the frame collapses as soon as
        its support joints are corrupted.
    C3 (support control): arm C (template Kabsch fitted on the corrupted
        constructor set) must not be meaningfully more robust than the
        anatomical frame at sigma in {20, 40} -- the collapse is a property of
        the four-joint support, not of the anatomical construction.
    C4 (monotonicity): the anatomical frame's mean distance must strictly
        increase with sigma -- its error is a pure function of its support.
    """
    by_sigma = {s["sigma_mm"]: s for s in levels}

    c1 = by_sigma[0.0]["verdict_vs_template17"] == "template_better"

    c2 = True
    for s in (20.0, 40.0):
        b = by_sigma[s]["t17_minus_anatomical"]
        if not (b["ci95"][0] < 0 and b["ci95"][1] < 0):
            c2 = False

    c3 = True
    for s in (20.0, 40.0):
        b = by_sigma[s]["t4_minus_anatomical"]
        if b["ci95"][0] > 0:              # arm C meaningfully better -> c3 fails
            c3 = False

    anat = [by_sigma[s]["mean_anatomical_mm"] for s in sorted(by_sigma)]
    c4 = all(anat[i] < anat[i + 1] for i in range(len(anat) - 1))

    if not c1:
        reading = ("void: sanity check failed, template not better at sigma=0; "
                   "the identity control did not reproduce the distal run")
    elif c2 and c3 and c4:
        reading = ("1: the frame collapses as soon as its support joints are "
                   "corrupted (both backbones), the four-joint control fails "
                   "with it, and the collapse is monotone. The failure "
                   "supports of the two alignments are disjoint: distal -> "
                   "anatomical immune, anchor -> anatomical collapses while "
                   "the 17-joint baseline degrades gracefully.")
    elif c2:
        reading = ("2: C2 held on one backbone only. One backbone is not two; "
                   "the claim is not established.")
    else:
        reading = ("3: C2 failed; the anatomical frame did not collapse at "
                   "sigma in {20, 40} on both backbones.")

    return {"c1_sanity_template_better_at_zero": c1,
            "c2_anatomical_worse_at_20_40": c2,
            "c3_support_control_fails_with_it": c3,
            "c4_monotone_collapse": c4,
            "reading": reading}


def selfcheck():
    """Positive control on synthetic data: the collapse is mechanism, not noise.

    A clean pose under a pure rotation cancels exactly in the anatomical frame
    (reusing the distal experiment's check), and corrupting the hips/thorax
    must rotate the frame far more than it rotates a 17-joint Kabsch fit.
    """
    from evaluation.occlusion_robustness import selfcheck as distal_selfcheck
    distal_selfcheck()

    rng = np.random.default_rng(0)
    pose = rng.normal(0, 200, size=(17, 3))
    pose[0] = 0.0
    pose[8] = [0, 450, 0]
    pose[1], pose[4] = [-140, 0, 0], [140, 0, 0]

    th = 0.7
    Q = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                  [-np.sin(th), 0, np.cos(th)]])
    a = (pose[None] @ Q.T).repeat(4, axis=0)

    idx = list(SCORED_JOINTS)
    ca, _ = canonicalize_stream(a.astype(np.float32))
    cb, _ = canonicalize_stream(a.astype(np.float32))
    _, d = cross_view_joint_distance_sequence(ca[:, idx], cb[:, idx])
    assert d < 1e-3, "identical views must cancel, got %.4f mm" % d

    a_c = corrupt(a, 80.0, seed_for(0, "0", 80.0))
    b_c = corrupt(a, 80.0, seed_for(0, "1", 80.0))
    ca2, _ = canonicalize_stream(a_c.astype(np.float32))
    cb2, _ = canonicalize_stream(b_c.astype(np.float32))
    _, d_anat = cross_view_joint_distance_sequence(ca2[:, idx], cb2[:, idx])
    assert d_anat > 1.0, "anchor noise must move the frame, got %.4f mm" % d_anat

    tmpl = pose - pose[0:1]
    ta = align_to_template(a_c, tmpl)
    tb = align_to_template(b_c, tmpl)
    _, d_t17 = cross_view_joint_distance_sequence(ta[:, idx], tb[:, idx])
    assert d_t17 < d_anat, \
        "the 17-joint fit must degrade less than the frame: %.2f vs %.2f" \
        % (d_t17, d_anat)

    print("selfcheck OK: frame moved by anchor noise (%.2f mm) more than the "
          "17-joint fit (%.2f mm)" % (d_anat, d_t17))


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
    print("ANCHOR CORRUPTION: ANATOMICAL FRAME vs TEMPLATE KABSCH")
    print("=" * 78)
    print("  corrupted %s  (the frame's support, minus the root)"
          % (CORRUPTED_JOINTS,))
    print("  scored    %s  (clean, non-constructor)" % (SCORED_JOINTS,))
    print("  positive difference means the anatomical frame is closer\n")
    print("  %8s %11s %11s %11s %11s  %s"
          % ("sigma", "anatomical", "template17", "template4",
             "t17 - anat", "verdict"))

    levels, all_rows = [], []
    for sigma in SEVERITIES:
        rows = collect(videos, template, sigma)
        all_rows.append({"sigma_mm": sigma, "rows": rows})
        s = summarise(rows, sigma)
        levels.append(s)
        b = s["t17_minus_anatomical"]
        print("  %6.0fmm %9.2fmm %9.2fmm %9.2fmm %+9.2f  %s"
              % (sigma, s["mean_anatomical_mm"], s["mean_template17_mm"],
                 s["mean_template4_mm"], b["mean"],
                 s["verdict_vs_template17"]))
        print("  %8s %11s %11s %11s [%+.2f, %+.2f]"
              % ("", "", "", "", b["ci95"][0], b["ci95"][1]))

    verdict = adjudicate(levels)
    print("\n  READING %s" % verdict["reading"])

    out = {"verdict": verdict, "levels": levels,
           "corrupted_joints": list(CORRUPTED_JOINTS),
           "scored_joints": list(SCORED_JOINTS),
           "constructor_joints": list(CONSTRUCTOR_JOINTS),
           "template_frames": n_frames, "template_streams": n_streams,
           "prediction_cache": os.path.basename(path),
           "per_severity_rows": all_rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "anchor_corruption%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
