"""
Measure the hip and shoulder axis lengths the report quotes.

Section 5 states the hip axis at 271.6 mm against 295.2 mm for the shoulder
axis, and per action that SittingDown has the second-longest hip axis at
285.3 mm while WalkDog has the shortest at 268.4 mm. Those four figures were
computed at some point and typed into the text; no artifact held them, so
`verify_documents.py` flagged them and both reports' claim that no number is
typed by hand was false for exactly these four.

This recomputes them from the same cached predictions every other Human3.6M
number derives from and stores them, so the audit can check them.

    hip axis      = |P[1] - P[4]|   (right hip to left hip)
    shoulder axis = |P[14] - P[11]| (right shoulder to left shoulder)

Run:  ./venv/Scripts/python.exe -m evaluation.axis_lengths
      ./venv/Scripts/python.exe -m evaluation.axis_lengths --preds <cache> --tag <name>
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import ACTION_NAMES, EVAL_STRIDE
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "axis_lengths")

R_HIP, L_HIP, L_SHOULDER, R_SHOULDER = 1, 4, 11, 14


def axis_lengths(poses):
    """(hip, shoulder) axis length per frame, in mm."""
    hip = np.linalg.norm(poses[:, R_HIP] - poses[:, L_HIP], axis=1)
    sho = np.linalg.norm(poses[:, R_SHOULDER] - poses[:, L_SHOULDER], axis=1)
    return hip, sho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    per_action, all_hip, all_sho, n = {}, [], [], 0
    for vid, v in videos.items():
        _, action, _ = parse_video(vid)
        p = v["pred"][::args.stride]
        p = p - p[:, 0:1, :]
        hip, sho = axis_lengths(p)
        per_action.setdefault(action, {"hip": [], "sho": []})
        per_action[action]["hip"].append(hip)
        per_action[action]["sho"].append(sho)
        all_hip.append(hip)
        all_sho.append(sho)
        n += len(p)

    # The report quotes a second pair of figures "over 2672 poses", which is a
    # different selection: the one the cross-view experiment evaluates on,
    # where a subject-action group is used only if it has all four cameras and
    # every stream is truncated to the shortest. Computed the same way here so
    # both quoted pairs have an artifact.
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v["pred"]
    cv_hip, cv_sho, cv_n = [], [], 0
    for (_subj, _action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        m = min(len(p) for p in cams.values())
        sel = np.arange(0, m, args.stride)
        for cam in sorted(cams):
            p = cams[cam][sel]
            p = p - p[:, 0:1, :]
            h, s = axis_lengths(p)
            cv_hip.append(h)
            cv_sho.append(s)
            cv_n += len(p)
    cv_hip = np.concatenate(cv_hip) if cv_hip else np.zeros(0)
    cv_sho = np.concatenate(cv_sho) if cv_sho else np.zeros(0)

    hip = np.concatenate(all_hip)
    sho = np.concatenate(all_sho)

    actions = {}
    for a, d in per_action.items():
        h = np.concatenate(d["hip"])
        s = np.concatenate(d["sho"])
        actions[ACTION_NAMES.get(a, str(a))] = {
            "hip_axis_mm": float(h.mean()),
            "shoulder_axis_mm": float(s.mean()),
            "n_poses": int(len(h)),
        }

    by_hip = sorted(actions.items(), key=lambda kv: -kv[1]["hip_axis_mm"])
    out = {
        "n_poses": int(n),
        "stride": args.stride,
        "mean_hip_axis_mm": float(hip.mean()),
        "mean_shoulder_axis_mm": float(sho.mean()),
        "shoulder_minus_hip_mm": float(sho.mean() - hip.mean()),
        "per_action": actions,
        "longest_hip_actions": [k for k, _ in by_hip[:3]],
        "shortest_hip_action": by_hip[-1][0],
        "crossview_selection": {
            "n_poses": int(cv_n),
            "mean_hip_axis_mm": float(cv_hip.mean()) if cv_n else None,
            "mean_shoulder_axis_mm": float(cv_sho.mean()) if cv_n else None,
        },
        "prediction_cache": os.path.basename(path),
    }

    print("=" * 70)
    print("HIP AND SHOULDER AXIS LENGTHS  (%d poses, stride %d)"
          % (n, args.stride))
    print("=" * 70)
    print("  hip axis      %7.2f mm" % out["mean_hip_axis_mm"])
    print("  shoulder axis %7.2f mm   (+%.2f)"
          % (out["mean_shoulder_axis_mm"], out["shoulder_minus_hip_mm"]))
    print("\n  %-16s %10s %12s" % ("action", "hip mm", "shoulder mm"))
    for a, d in by_hip:
        print("  %-16s %10.2f %12.2f"
              % (a, d["hip_axis_mm"], d["shoulder_axis_mm"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "axis_lengths%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nSaved: %s" % p)


if __name__ == "__main__":
    main()
