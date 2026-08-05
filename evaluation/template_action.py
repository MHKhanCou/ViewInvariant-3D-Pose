"""
Per-action breakdown of the template baseline: does it have a failure mode?

Pre-registered in thesis_artifacts/template_action/PREREGISTRATION.md.

"All 180 pairs" is 180 pairs from fifteen largely upright actions, aligned onto
one fixed near-standing skeleton. If the baseline never loses, the anatomical
frame has no regime of its own and the report should say so. If it loses where
the pose is far from the template, that regime is worth naming.

Run:  ./venv/Scripts/python.exe -m evaluation.template_action
      ./venv/Scripts/python.exe -m evaluation.template_action --preds <cache> --tag <name>
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import ACTION_NAMES
from evaluation.h36m_noncon import RETAINED_JOINTS
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video
from evaluation.template_baseline import build_template, collect

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "template_action")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    template, _, _ = build_template()
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    rows = collect(videos, template, RETAINED_JOINTS)

    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(r)

    out_actions = []
    for act, rs in by.items():
        a = float(np.mean([r["anatomical_mm"] for r in rs]))
        t = float(np.mean([r["template_mm"] for r in rs]))
        out_actions.append({
            "action": act,
            "name": ACTION_NAMES.get(act, act),
            "n_pairs": len(rs),
            "anatomical_mm": a,
            "template_mm": t,
            "difference_mm": t - a,
            "anatomical_better": bool(t > a),
        })
    out_actions.sort(key=lambda d: d["difference_mm"])

    n_anat = sum(1 for d in out_actions if d["anatomical_better"])
    print("=" * 78)
    print("TEMPLATE BASELINE BY ACTION  (13 non-constructor joints)")
    print("=" * 78)
    print("  negative difference = template closer = baseline wins\n")
    print("  %-14s %6s %12s %11s %12s" % ("action", "pairs", "anatomical",
                                          "template", "difference"))
    for d in out_actions:
        flag = "   <- ours wins" if d["anatomical_better"] else ""
        print("  %-14s %6d %11.1f %11.1f %+11.1f%s"
              % (d["name"], d["n_pairs"], d["anatomical_mm"],
                 d["template_mm"], d["difference_mm"], flag))
    print("\n  anatomical frame better on %d of %d actions" % (n_anat, len(out_actions)))

    res = {"n_actions": len(out_actions),
           "n_actions_anatomical_better": n_anat,
           "by_action": out_actions,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "action%s.json" % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nSaved: %s" % p)


if __name__ == "__main__":
    main()
