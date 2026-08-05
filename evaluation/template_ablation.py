"""
Two ablations of the template baseline of Section 5.10.1.

Pre-registered in thesis_artifacts/template_ablation/PREREGISTRATION.md.

A. Does the baseline's win depend on our own canonicalization? Its template is
   the mean CANONICAL MPI pose, aligned by the very construction it beats.
   Rebuilt from an arbitrary raw prediction and from a synthetic figure.

B. Does Section 5.19's boundary predict a better method? Fit the rotation on the
   nine torso-rigid joints and score on the eight articulated ones. Disjoint
   sets: nothing is scored on a joint used to fit.

Run:  ./venv/Scripts/python.exe -m evaluation.template_ablation
      ./venv/Scripts/python.exe -m evaluation.template_ablation --preds <cache> --tag <name>
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
from evaluation.template_baseline import MPI_CACHE, build_template

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "template_ablation")

# Section 5.19: joints that move rigidly with the torso, including the four the
# frame is built from and the spine. Disjoint from ARTICULATED by construction.
TORSO_RIGID = (0, 1, 4, 7, 8, 9, 10, 11, 14)
ARTICULATED = (2, 3, 5, 6, 12, 13, 15, 16)
assert not (set(TORSO_RIGID) & set(ARTICULATED))
assert len(TORSO_RIGID) + len(ARTICULATED) == 17

H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def template_raw_first(path=MPI_CACHE):
    """One arbitrary raw MPI prediction, root-centred. No canonicalization."""
    z = np.load(path, allow_pickle=True)
    for k in sorted(f[:-len("__raw")] for f in z.files if f.endswith("__raw")):
        raw = np.asarray(z[k + "__raw"], dtype=np.float64)
        ok = z[k + "__valid"] if (k + "__valid") in z.files else np.ones(
            len(raw), bool)
        idx = np.flatnonzero(ok)
        if idx.size:
            p = raw[idx[0]]
            return p - p[0:1]
    raise RuntimeError("no valid raw MPI frame")


def template_synthetic(path=MPI_CACHE):
    """A neutral standing figure from median MPI bone lengths. Uses no pose."""
    z = np.load(path, allow_pickle=True)
    lens = []
    for k in sorted(f[:-len("__raw")] for f in z.files if f.endswith("__raw")):
        raw = np.asarray(z[k + "__raw"], dtype=np.float64)
        lens.append(np.linalg.norm(
            raw[:, 1:] - raw[:, [H36M_PARENTS[j] for j in range(1, 17)]], axis=2))
    L = np.median(np.concatenate(lens), axis=0)      # 16 bone lengths

    # Fixed anatomical directions: spine up, legs down, arms out. No data.
    d = {1: (1, 0, 0), 2: (0, -1, 0), 3: (0, -1, 0),
         4: (-1, 0, 0), 5: (0, -1, 0), 6: (0, -1, 0),
         7: (0, 1, 0), 8: (0, 1, 0), 9: (0, 1, 0), 10: (0, 1, 0),
         11: (-1, 0, 0), 12: (-1, 0, 0), 13: (-1, 0, 0),
         14: (1, 0, 0), 15: (1, 0, 0), 16: (1, 0, 0)}
    P = np.zeros((17, 3))
    for j in range(1, 17):
        P[j] = P[H36M_PARENTS[j]] + np.array(d[j], float) * L[j - 1]
    return P - P[0:1]


def align(poses, template, fit_joints):
    """Kabsch onto the template, fitting on `fit_joints`, applied to all."""
    f = list(fit_joints)
    out = np.empty_like(poses)
    for i, p in enumerate(poses):
        R, _, _ = procrustes_align(p[f].astype(np.float32),
                                   template[f].astype(np.float32))
        out[i] = (p - p[f].mean(axis=0)) @ R.T
    return out


def run(videos, template, fit_joints, score_joints, stride=EVAL_STRIDE):
    groups = {}
    for vid, v in videos.items():
        s, a, c = parse_video(vid)
        groups.setdefault((s, a), {})[c] = v["pred"]
    idx = list(score_joints)
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
            anat[cam], valid[cam] = canonicalize_stream(p)
            tmpl[cam] = align(p, template, fit_joints)
        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            _, da = cross_view_joint_distance_sequence(
                anat[a][keep][:, idx], anat[b][keep][:, idx])
            _, dt = cross_view_joint_distance_sequence(
                tmpl[a][keep][:, idx], tmpl[b][keep][:, idx])
            rows.append({"subject": subj, "action": action,
                         "anatomical_mm": float(da), "template_mm": float(dt),
                         "template_minus_anatomical_mm": float(dt - da)})
    b = cluster_bootstrap(rows, "template_minus_anatomical_mm")
    return {
        "n_pairs": len(rows),
        "mean_anatomical_mm": float(np.mean([r["anatomical_mm"] for r in rows])),
        "mean_template_mm": float(np.mean([r["template_mm"] for r in rows])),
        "paired_difference": b,
        "template_wins": bool(b["ci95"][1] < 0),
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

    templates = {
        "mpi_canonical": build_template()[0],
        "raw_first": template_raw_first(),
        "synthetic": template_synthetic(),
    }

    print("=" * 78)
    print("A. DOES THE BASELINE'S WIN DEPEND ON OUR CANONICALIZATION?")
    print("=" * 78)
    print("  fit and score on all 17 joints, as Section 5.10.1 does\n")
    print("  %-16s %11s %11s %12s %s"
          % ("template", "anatomical", "template", "difference", "wins"))
    a_res = {}
    for name, T in templates.items():
        r = run(videos, T, tuple(range(17)), tuple(range(17)))
        a_res[name] = r
        b = r["paired_difference"]
        print("  %-16s %10.2f %11.2f %+11.2f  [%+.1f,%+.1f] %s"
              % (name, r["mean_anatomical_mm"], r["mean_template_mm"],
                 b["mean"], b["ci95"][0], b["ci95"][1],
                 "template" if r["template_wins"] else "NOT ESTABLISHED"))

    print("\n" + "=" * 78)
    print("B. DOES THE BOUNDARY PREDICT A BETTER FIT SET?")
    print("=" * 78)
    print("  scored on the 8 articulated joints; fit sets disjoint from them\n")
    T = templates["mpi_canonical"]
    b_res = {}
    for name, fit in (("all_17", tuple(range(17))),
                      ("torso_rigid_9", TORSO_RIGID)):
        r = run(videos, T, fit, ARTICULATED)
        b_res[name] = r
        bb = r["paired_difference"]
        print("  fit on %-14s template %7.2f mm   (anatomical %7.2f)"
              % (name, r["mean_template_mm"], r["mean_anatomical_mm"]))
    gain = b_res["all_17"]["mean_template_mm"] - b_res["torso_rigid_9"]["mean_template_mm"]
    print("\n  torso-rigid fit is %+.2f mm better than all-17 fit  -> prediction %s"
          % (gain, "HOLDS" if gain > 0 else "FAILS"))

    out = {"A_template_independence": a_res, "B_fit_set": b_res,
           "B_torso_rigid_gain_mm": float(gain),
           "B_prediction_holds": bool(gain > 0),
           "torso_rigid_joints": list(TORSO_RIGID),
           "articulated_joints": list(ARTICULATED),
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "ablation%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nSaved: %s" % p)


if __name__ == "__main__":
    main()
