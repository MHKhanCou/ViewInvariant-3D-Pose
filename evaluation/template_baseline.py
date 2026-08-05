"""
Procrustes-to-template: the simplest alternative with the same requirements.

Pre-registered in thesis_artifacts/template/PREREGISTRATION.md, committed before
this ran, including all three readings of the outcome.

Every comparison elsewhere in this report is against raw camera-frame
predictions, which are the absence of a method, or against the per-frame
Procrustes oracle, which needs both views and is a floor rather than a
competitor. Kabsch-aligning each pose onto one fixed reference skeleton meets
the same requirement profile as our frame - no training, no labels, no
calibration, one view - and is what V-VIPE uses as preprocessing. It is the
obvious thing a reader will ask about and it has never been run here.

The template is the mean canonical pose over the MPI-INF-3DHP cache, not
Human3.6M: all 180 H36M pairs and both subjects are held out, so there is no
in-dataset non-test source, and a template built from S9/S11 would see the test
distribution our method never saw. Cross-dataset is disjoint and gives the
baseline a real skeleton rather than a synthetic one, which runs the competitor
at its strongest.

Run:  ./venv/Scripts/python.exe -m evaluation.template_baseline
      ./venv/Scripts/python.exe -m evaluation.template_baseline --preds <cache> --tag <name>
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
from evaluation.oracle import procrustes_align

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "template")
MPI_CACHE = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval",
                         "predictions_cache.npz")


def build_template(path=MPI_CACHE):
    """Mean canonical pose over every valid MPI-INF-3DHP frame."""
    z = np.load(path, allow_pickle=True)
    keys = [k[:-len("__canonical")] for k in z.files if k.endswith("__canonical")]
    poses = []
    for k in sorted(keys):
        can = z[k + "__canonical"]
        ok = z[k + "__valid"] if (k + "__valid") in z.files else np.ones(
            len(can), bool)
        if ok.any():
            poses.append(np.asarray(can, dtype=np.float64)[ok])
    stacked = np.concatenate(poses)
    tmpl = stacked.mean(axis=0)
    tmpl = tmpl - tmpl[0:1]                 # root-centre, as everything else is
    return tmpl, int(len(stacked)), len(keys)


def align_to_template(poses, template):
    """Kabsch-rotate each pose onto the template. Rotation only, no scaling."""
    out = np.empty_like(poses)
    for i, p in enumerate(poses):
        R, _, _ = procrustes_align(p.astype(np.float32),
                                   template.astype(np.float32))
        out[i] = (p - p.mean(axis=0)) @ R.T
    return out


def collect(videos, template, joints, stride=EVAL_STRIDE):
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

        anat, tmpl, valid = {}, {}, {}
        for cam, poses in cams.items():
            p = poses[sel]
            p = p - p[:, 0:1, :]
            anat[cam], valid[cam] = canonicalize_stream(p)
            tmpl[cam] = align_to_template(p, template)

        for a, b in itertools.combinations(sorted(cams), 2):
            keep = valid[a] & valid[b]
            if keep.sum() < 10:
                continue
            _, d_anat = cross_view_joint_distance_sequence(
                anat[a][keep][:, idx], anat[b][keep][:, idx])
            _, d_tmpl = cross_view_joint_distance_sequence(
                tmpl[a][keep][:, idx], tmpl[b][keep][:, idx])
            rows.append({
                "subject": subj, "action": action, "cam_a": a, "cam_b": b,
                "n_frames": int(keep.sum()),
                "anatomical_mm": float(d_anat),
                "template_mm": float(d_tmpl),
                # positive => the anatomical frame is closer, i.e. better
                "template_minus_anatomical_mm": float(d_tmpl - d_anat),
            })
    return rows


def summarise(rows):
    a = np.array([r["anatomical_mm"] for r in rows])
    t = np.array([r["template_mm"] for r in rows])
    boot = cluster_bootstrap(rows, "template_minus_anatomical_mm")
    lo, hi = boot["ci95"]
    if lo > 0:
        verdict = "anatomical_better"
    elif hi < 0:
        verdict = "template_better"
    else:
        verdict = "not_established"
    return {
        "n_pairs": len(rows),
        "mean_anatomical_mm": float(a.mean()),
        "mean_template_mm": float(t.mean()),
        "pairs_where_anatomical_better": int((t > a).sum()),
        "paired_difference": boot,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    template, n_frames, n_streams = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    non = summarise(collect(videos, template, RETAINED_JOINTS))
    all17 = summarise(collect(videos, template, tuple(range(17))))

    print("=" * 78)
    print("ANATOMICAL FRAME vs PROCRUSTES-TO-TEMPLATE")
    print("=" * 78)
    print("  template: mean of %d valid MPI-INF-3DHP canonical poses over %d "
          "streams" % (n_frames, n_streams))
    print("  positive difference means the anatomical frame is closer\n")
    for name, s in (("13 non-constructor joints", non), ("all 17 joints", all17)):
        b = s["paired_difference"]
        print("  --- %s ---" % name)
        print("    anatomical    %7.2f mm" % s["mean_anatomical_mm"])
        print("    template      %7.2f mm" % s["mean_template_mm"])
        print("    difference    %+7.2f mm   95%% CI [%+.2f, %+.2f]"
              % (b["mean"], b["ci95"][0], b["ci95"][1]))
        print("    anatomical better on %d of %d pairs"
              % (s["pairs_where_anatomical_better"], s["n_pairs"]))
        print("    VERDICT: %s\n" % s["verdict"])

    out = {"non_constructor": non, "all_17_joints": all17,
           "template_frames": n_frames, "template_streams": n_streams,
           "template_source": "MPI-INF-3DHP cached canonical poses",
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "template%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
