"""
Confidence-gated routing between the anatomical frame and template Kabsch.

Pre-registered in thesis_artifacts/selection/PREREGISTRATION.md, committed
before this ran.

Section 5.6.1 shows template Kabsch beats the anatomical frame on the headline
metric, and the two post-freeze experiments map why. The two alignments read
different joint supports: the anatomical frame reads {0,1,4,8} and is exactly
flat under distal corruption while the 17-joint Kabsch fit degrades
(occlusion/), and it collapses as soon as those four joints are corrupted while
the 17-joint fit degrades gracefully (anchor_corruption/). The failure modes
are complementary and disjoint.

This module turns that complementarity into a decision rule. In deployment a
pose detector reports a per-joint confidence; the corrupted-joint experiments
have no such signal, so the rule's input is simulated as a calibrated function
of the injected noise -- a detector whose confidence drops linearly with
localization error, saturating at 80 mm. The oracle arm (route each pair to the
alignment with the lower true distance) is the ceiling the rule is compared
against. Both arms are labelled exactly that way; the simulation is stated, not
hidden.

Rule (pre-registered): route a frame to the anatomical frame iff

    min over {1 r_hip, 4 l_hip, 8 thorax} of c_j  >=  0.7     (core reliable)
    AND
    min over the eight distal joints of c_j       <  0.7     (periphery broken)

else route to template Kabsch. The threshold 0.7 means "confidence has dropped
to 70 %", i.e. noise of ~24 mm on the relevant joints -- the rule only commits
to the anatomical frame when the periphery is genuinely degraded and the core
is clean, which is exactly the region where the distal experiment says the
frame wins.

Run:  ./venv/Scripts/python.exe -m evaluation.selection_rule
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video
from evaluation.h36m_crossview import cluster_bootstrap
from evaluation.template_baseline import build_template
from evaluation.occlusion_robustness import (CORRUPTED_JOINTS as DISTAL,
                                             collect as collect_distal)
from evaluation.anchor_corruption import (CORRUPTED_JOINTS as ANCHOR,
                                          collect as collect_anchor)

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "selection")

ANCHOR_SET = (1, 4, 8)          # the frame's support minus the root
# note: imported DISTAL and ANCHOR above are the corrupted joint sets of the
# two regimes; ANCHOR (the variable) re-defines the core set identically to
# anchor_corruption.CORRUPTED_JOINTS, which is the frame's support.
CONFIDENCE_SATURATION_MM = 80.0 # confidence hits zero at this noise level
CONFIDENCE_THRESHOLD = 0.7      # "core reliable / periphery broken"
TRANSITION_TOLERANCE_MM = 7.0   # allowed shortfall in the transition band


def simulated_confidence(sigma_mm, corrupted_set):
    """Detector-confidence model: linear drop with injected noise, saturated."""
    conf = {}
    for j in range(17):
        s = sigma_mm if j in corrupted_set else 0.0
        conf[j] = float(np.clip(1.0 - s / CONFIDENCE_SATURATION_MM, 0.0, 1.0))
    return conf


def route_to_anatomical(conf):
    """The pre-registered rule. Returns True to use the anatomical frame."""
    core = min(conf[j] for j in ANCHOR_SET)
    periphery = min(conf[j] for j in DISTAL)
    return core >= CONFIDENCE_THRESHOLD and periphery < CONFIDENCE_THRESHOLD


def evaluate_regime(name, collect_fn, videos, template, severities):
    """Rows for one corruption regime, with routing and oracle columns."""
    out = []
    for sigma in severities:
        rows = collect_fn(videos, template, sigma)
        conf = simulated_confidence(sigma, (DISTAL if name == "distal"
                                            else ANCHOR))
        use_anat = route_to_anatomical(conf)
        for r in rows:
            r["regime"] = name
            r["sigma_mm"] = sigma
            r["routed_mm"] = (r["anatomical_mm"] if use_anat
                              else r["template17_mm"])
            r["oracle_mm"] = min(r["anatomical_mm"], r["template17_mm"])
        out.append({"sigma_mm": sigma, "route_anatomical": use_anat,
                    "rows": rows})
    return out


def selfcheck():
    """
    Verify the rule and its confidence model on hand-computed values, so a
    null result cannot be an implementation error.
    """
    # Confidence model: clean joint -> 1.0; 80 mm noise -> 0.0.
    c = simulated_confidence(0.0, DISTAL)
    assert all(v == 1.0 for v in c.values()), "clean confidences must be 1"
    c = simulated_confidence(80.0, DISTAL)
    assert c[DISTAL[0]] == 0.0 and c[ANCHOR_SET[0]] == 1.0, \
        "corrupted joints must read 0 at 80 mm, clean joints 1"
    c = simulated_confidence(40.0, DISTAL)
    assert c[DISTAL[0]] == 0.5, "linear drop expected at 40 mm"

    # Rule: anatomical iff core >= 0.7 AND periphery < 0.7.
    clean = simulated_confidence(0.0, DISTAL)
    assert route_to_anatomical(clean) is False, "clean must route to template"
    distal_bad = simulated_confidence(160.0, DISTAL)
    assert route_to_anatomical(distal_bad) is True, \
        "broken periphery + clean core must route to the frame"
    anchor_bad = simulated_confidence(160.0, ANCHOR)
    assert route_to_anatomical(anchor_bad) is False, \
        "broken core must never route to the frame"
    print("selfcheck OK: confidence model and routing rule behave as specified")


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

    severities = (0.0, 20.0, 40.0, 80.0, 160.0)
    distal = evaluate_regime("distal", collect_distal, videos, template,
                             severities)
    anchor = evaluate_regime("anchor", collect_anchor, videos, template,
                             severities)

    print("=" * 78)
    print("CONFIDENCE-GATED ROUTING: ANATOMICAL FRAME vs TEMPLATE KABSCH")
    print("=" * 78)
    print("  rule: core conf >= 0.7 AND periphery conf < 0.7 -> anatomical\n")

    cells = []
    for block in (distal, anchor):
        print("  -- %s corruption --" % block[0]["rows"][0]["regime"].upper())
        print("  %8s %5s %11s %11s %11s %11s %11s"
              % ("sigma", "route", "anatomical", "template17", "routed",
                 "oracle", "routed-min"))
        for lvl in block:
            rows = lvl["rows"]
            mean = lambda k: float(np.mean([r[k] for r in rows]))
            routed_min = mean("routed_mm") - min(mean("anatomical_mm"),
                                                 mean("template17_mm"))
            cells.append({"regime": lvl["rows"][0]["regime"],
                          "sigma_mm": lvl["sigma_mm"],
                          "route_anatomical": lvl["route_anatomical"],
                          "mean_anatomical_mm": mean("anatomical_mm"),
                          "mean_template17_mm": mean("template17_mm"),
                          "mean_routed_mm": mean("routed_mm"),
                          "mean_oracle_mm": mean("oracle_mm"),
                          "routed_minus_best_mm": routed_min})
            print("  %6.0fmm %5s %9.2fmm %9.2fmm %9.2fmm %9.2fmm %+9.2f"
                  % (lvl["sigma_mm"], "anat" if lvl["route_anatomical"] else "t17",
                     mean("anatomical_mm"), mean("template17_mm"),
                     mean("routed_mm"), mean("oracle_mm"), routed_min))

    # ---- pre-registered criteria ----
    worst = max(c["routed_minus_best_mm"] for c in cells)
    r1 = worst <= TRANSITION_TOLERANCE_MM

    # R2: at sigma=160 distal, routed (anatomical) is better than template17
    # with a bootstrap CI excluding zero.
    # R2: routed (anatomical) must beat template17 at sigma=160 distal, with a
    # cluster-bootstrap CI excluding zero on the per-pair difference.
    d160 = next(b for b in distal if b["sigma_mm"] == 160.0)
    rows_diff = [dict(r) for r in d160["rows"]]
    for r in rows_diff:
        r["t17_minus_routed_mm"] = r["template17_mm"] - r["routed_mm"]
    boot = cluster_bootstrap(rows_diff, "t17_minus_routed_mm")
    r2 = boot["ci95"][0] > 0

    # R3: at sigma>=40 anchor, routed == template17 (the rule never routes into
    # the collapsed arm).
    r3 = all(lvl["route_anatomical"] is False
             for lvl in anchor if lvl["sigma_mm"] >= 40.0)

    # R4: at sigma=0 routed == template17 on both regimes.
    r4 = all(not b["route_anatomical"] for b in (distal[0], anchor[0]))

    print("\n  worst routed-minus-best over all %d cells: %+.2f mm (tolerance %g)"
          % (len(cells), worst, TRANSITION_TOLERANCE_MM))
    print("  R1 routed never worse than best by > %g mm : %s"
          % (TRANSITION_TOLERANCE_MM, r1))
    print("  R2 sigma=160 distal beats template17, CI excl. zero: %s  [%+.2f, %+.2f]"
          % (r2, boot["ci95"][0], boot["ci95"][1]))
    print("  R3 anchor sigma>=40 never routes to the collapsed arm: %s" % r3)
    print("  R4 clean routed = template17 (best at clean): %s" % r4)
    reading = ("1: routing holds" if (r1 and r2 and r3 and r4)
               else "2: routing not established")
    print("  READING %s" % reading)

    out = {"verdict": {"r1_transition_tolerance": r1,
                       "r2_severe_distal_beats_template": r2,
                       "r3_never_routes_to_collapse": r3,
                       "r4_clean_uses_template": r4,
                       "reading": reading},
           "confidence_model": "c_j = clip(1 - sigma_j / 80, 0, 1)",
           "threshold": CONFIDENCE_THRESHOLD,
           "transition_tolerance_mm": TRANSITION_TOLERANCE_MM,
           "cells": cells,
           "distal_sigma160_bootstrap": boot,
           "template_frames": n_frames, "template_streams": n_streams,
           "prediction_cache": os.path.basename(path)}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "selection%s.json"
                     % (("_" + args.tag) if args.tag else ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved: %s" % p)


if __name__ == "__main__":
    main()
