"""
Does laterality matter, at matched joint count?

Pre-registered in thesis_artifacts/laterality/PREREGISTRATION.md, committed
before this ran (ff5bd39). Seventeenth pre-registration.

The sixteenth compared four left-side joints against the tenth's eight
bilateral ones and found the one-sided set damaged Kabsch LESS. That comparison
is confounded: laterality and joint count moved together. This holds the count
and the anatomical types fixed and varies only the side.

    one-sided  {5 l_knee, 6 l_foot, 12 l_elbow, 13 l_wrist}   4 left, 0 right
    balanced   {5 l_knee, 3 r_foot, 12 l_elbow, 16 r_wrist}   2 left, 2 right

Both corrupt one knee, one foot, one elbow and one wrist, at the same radii.
The anatomical frame reads {0, 1, 4, 8} and is untouched by either, so it is
flat in both arms and serves as the internal control.

No reading of this can produce a regime where the frame beats Kabsch. It
explains why the sixteenth failed; it is not another attempt to win.

Run:  ./venv/Scripts/python.exe -m evaluation.laterality_control
      ./venv/Scripts/python.exe -m evaluation.laterality_control --preds <cache> --tag <name>
      ./venv/Scripts/python.exe -m evaluation.laterality_control --selfcheck
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
from evaluation.occlusion_robustness import (SCORED_JOINTS, SEVERITIES,
                                             align_to_template, seed_for)
from evaluation.template_baseline import build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "laterality")

ARMS = {
    "one_sided": (5, 6, 12, 13),    # l_knee, l_foot, l_elbow, l_wrist
    "balanced": (5, 3, 12, 16),     # l_knee, r_foot, l_elbow, r_wrist
}
# knee, foot, elbow, wrist -- one of each in both arms, so radii match.
TYPES = {5: "knee", 2: "knee", 6: "foot", 3: "foot",
         12: "elbow", 15: "elbow", 13: "wrist", 16: "wrist"}

for _n, _s in ARMS.items():
    assert len(_s) == 4, _n
    assert sorted(TYPES[j] for j in _s) == ["elbow", "foot", "knee", "wrist"], _n
    assert not set(_s) & set(SCORED_JOINTS), _n
    assert not set(_s) & set(CONSTRUCTOR_JOINTS), _n


def corrupt(poses, joints, sigma, seed):
    if sigma <= 0:
        return poses
    out = poses.copy()
    rng = np.random.default_rng(seed)
    idx = list(joints)
    out[:, idx, :] += rng.normal(0.0, sigma, size=(len(poses), len(idx), 3))
    return out


def collect(videos, template, sigma, stride=EVAL_STRIDE):
    """Both arms in one pass, so the pairing is exact."""
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

        tmpl = {a: {} for a in ARMS}
        anat, valid = {}, {}
        for cam, poses in sorted(cams.items()):
            p0 = poses[sel]
            p0 = p0 - p0[:, 0:1, :]
            # Same seed for both arms at a given (group, camera, severity), so
            # the two differ only in which joints receive the draw.
            sd = seed_for(gi, cam, sigma)
            for arm, joints in ARMS.items():
                p = corrupt(p0, joints, sigma, sd)
                tmpl[arm][cam] = align_to_template(p, template)
                if arm == "one_sided":
                    anat[cam], valid[cam] = canonicalize_stream(p)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            row = {"subject": subj, "action": action, "cam_a": a, "cam_b": b,
                   "n_frames": int(keep.sum())}
            _, da = cross_view_joint_distance_sequence(
                anat[a][keep][:, idx], anat[b][keep][:, idx])
            row["anatomical_mm"] = float(da)
            for arm in ARMS:
                _, d = cross_view_joint_distance_sequence(
                    tmpl[arm][a][keep][:, idx], tmpl[arm][b][keep][:, idx])
                row[arm + "_mm"] = float(d)
            # positive => one-sided corruption damages Kabsch MORE
            row["onesided_minus_balanced_mm"] = (row["one_sided_mm"]
                                                 - row["balanced_mm"])
            rows.append(row)
    return rows


def summarise(rows, sigma):
    boot = cluster_bootstrap(rows, "onesided_minus_balanced_mm")
    lo, hi = boot["ci95"]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "sigma_mm": sigma,
        "n_pairs": len(rows),
        "mean_anatomical_mm": mean("anatomical_mm"),
        "mean_one_sided_mm": mean("one_sided_mm"),
        "mean_balanced_mm": mean("balanced_mm"),
        "onesided_minus_balanced": boot,
        "verdict": ("one_sided_worse_for_kabsch" if lo > 0 else
                    "balanced_worse_for_kabsch" if hi < 0 else
                    "no_difference"),
    }


def selfcheck():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 200, size=(4, 17, 3))
    for arm, joints in ARMS.items():
        c = corrupt(p.copy(), joints, 100.0, 1)
        moved = np.linalg.norm(c - p, axis=2).mean(axis=0)
        for j in range(17):
            if j in joints:
                assert moved[j] > 50.0, "%s joint %d should move" % (arm, j)
            else:
                assert moved[j] < 1e-9, "%s joint %d must not move" % (arm, j)
    a = corrupt(p.copy(), ARMS["one_sided"], 0.0, 1)
    b = corrupt(p.copy(), ARMS["balanced"], 0.0, 1)
    assert np.abs(a - b).max() < 1e-12, "arms must be identical at sigma=0"
    left = sum(1 for j in ARMS["one_sided"] if j in (5, 6, 12, 13))
    bal_l = sum(1 for j in ARMS["balanced"] if j in (5, 6, 12, 13))
    assert left == 4 and bal_l == 2, "side split must be 4-0 and 2-2"
    print("selfcheck OK: both arms corrupt 4 joints of matched type; "
          "sides 4-0 vs 2-2; identical at sigma=0")


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
    print("LATERALITY AT MATCHED JOINT COUNT")
    print("=" * 78)
    print("  one-sided %s   balanced %s"
          % (ARMS["one_sided"], ARMS["balanced"]))
    print("  both corrupt one knee, foot, elbow and wrist; only the side differs\n")
    print("  %8s %11s %11s %11s %13s  %s"
          % ("sigma", "anatomical", "one-sided", "balanced", "1sided-bal",
             "verdict"))

    levels = []
    for sigma in SEVERITIES:
        s = summarise(collect(videos, template, sigma), sigma)
        levels.append(s)
        b = s["onesided_minus_balanced"]
        print("  %6.0fmm %9.2fmm %9.2fmm %9.2fmm %+11.2f  %s"
              % (sigma, s["mean_anatomical_mm"], s["mean_one_sided_mm"],
                 s["mean_balanced_mm"], b["mean"], s["verdict"]))
        print("  %8s %11s %11s %11s [%+.2f, %+.2f]"
              % ("", "", "", "", b["ci95"][0], b["ci95"][1]))

    zero = levels[0]
    sanity_zero = abs(zero["mean_one_sided_mm"] - zero["mean_balanced_mm"]) < 0.01
    anat = [l["mean_anatomical_mm"] for l in levels]
    sanity_flat = (max(anat) - min(anat)) < 0.01
    worse = [l["sigma_mm"] for l in levels
             if l["verdict"] == "one_sided_worse_for_kabsch"]
    if not (sanity_zero and sanity_flat):
        reading = "void: sanity failed (arms differ at zero, or frame not flat)"
    elif worse:
        reading = ("1: one-sided corruption damages Kabsch more, at sigma %s"
                   % worse)
    elif any(l["verdict"] == "balanced_worse_for_kabsch" for l in levels):
        reading = "3: balanced damages Kabsch more; the mechanism is backwards"
    else:
        reading = ("2: no difference at matched joint count; laterality is not "
                   "the variable, joint count is")
    print("\n  READING %s" % reading)

    out = {"verdict": {"sanity_identical_at_zero": bool(sanity_zero),
                       "sanity_anatomical_flat": bool(sanity_flat),
                       "sigmas_where_onesided_worse": worse,
                       "reading": reading},
           "levels": levels,
           "arms": {k: list(v) for k, v in ARMS.items()},
           "scored_joints": list(SCORED_JOINTS),
           "template_frames": n_frames,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "laterality%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
