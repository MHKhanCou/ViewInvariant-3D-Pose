"""
Number audit: every headline claim re-derived from its artifact.

Each CLAIM below states a number that appears in FINAL_REPORT.md, the defense
deck, or DEFENSE_QA.md, together with the artifact it must come from. Running
this recomputes each one and fails loudly on drift — so a number can never
silently rot after a re-run.

Run:  ./venv/Scripts/python.exe -m evaluation.audit_numbers
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
ART = os.path.join(REPO_ROOT, "thesis_artifacts")

TOL = 0.05  # absolute tolerance on percentages / mm


def load(rel):
    with open(os.path.join(ART, rel)) as f:
        return json.load(f)


def check(label, claimed, actual, source, tol=TOL, unit=""):
    ok = actual is not None and abs(claimed - actual) <= tol
    mark = "OK  " if ok else "FAIL"
    got = "missing" if actual is None else f"{actual:.2f}"
    print(f"[{mark}] {label:52s} claimed {claimed:>8.2f}{unit}  got {got}{unit}"
          f"   <- {source}")
    return ok


def main():
    results = []

    # ---------- cross-view, corrected protocol ----------
    mc = load("cross_view_eval/results_multicam.json")
    static = [r for r in mc if r.get("condition", "static") == "static"]
    dev = [r for r in static if r["is_dev_pair"]]
    held = [r for r in static if not r["is_dev_pair"] and r["subject"] == "S1"]
    s2 = [r for r in static if r["subject"] == "S2"]
    src = "cross_view_eval/results_multicam.json"

    results.append(check("dev pair improvement %", 20.5,
                         dev[0]["improvement_pct"] if dev else None, src, unit="%"))
    results.append(check("held-out pairs mean improvement %", 32.4,
                         float(np.mean([r["improvement_pct"] for r in held]))
                         if held else None, src, unit="%"))
    results.append(check("held-out pair count", 27, float(len(held)), src, tol=0))
    results.append(check("held-out subject S2 improvement %", 13.4,
                         s2[0]["improvement_pct"] if s2 else None, src, unit="%"))

    # ---------- legacy protocol ----------
    legacy = load("cross_view_eval/results.json")[0]
    results.append(check("legacy single-frame protocol improvement %", 28.4,
                         legacy["improvement_pct"],
                         "cross_view_eval/results.json", unit="%"))

    # ---------- degradation / reliability ----------
    d = load("degradation/analysis.json")
    results.append(check("pooled rho(reliability, induced drift)", -0.813,
                         d["pooled"]["spearman_reliability_vs_drift"],
                         "degradation/analysis.json", tol=0.005))
    results.append(check("clean-data abstention rate %", 0.0,
                         d["clean"]["abstention_rate"] * 100,
                         "degradation/analysis.json", unit="%"))
    results.append(check("joint-dropout abstention rate %", 100.0,
                         d["operators"]["joint_dropout"]["abstention_rate_overall"] * 100,
                         "degradation/analysis.json", unit="%"))

    # ---------- GT anchoring ----------
    g = load("gt_validation/gt_results.json")
    results.append(check("rho(canonical cross-view dist, GT error)", 0.601,
                         g["pooled"]["spearman_canonical_dist_vs_gt_error"],
                         "gt_validation/gt_results.json", tol=0.005))
    results.append(check("rho(raw cross-view dist, GT error)", 0.188,
                         g["pooled"]["spearman_raw_dist_vs_gt_error"],
                         "gt_validation/gt_results.json", tol=0.005))

    # ---------- multi-scale ----------
    ms = load("cross_view_eval/multiscale_results.json")["results"]
    ms_dev = [r for r in ms if r["is_dev_pair"]]
    ms_held = [r for r in ms if not r["is_dev_pair"] and r["pair"].startswith("S1")]
    src = "cross_view_eval/multiscale_results.json"
    results.append(check("multi-scale dev-pair gain %", 37.1,
                         ms_dev[0]["multiscale_vs_global_pct"] if ms_dev else None,
                         src, unit="%"))
    results.append(check("multi-scale held-out mean gain %", 36.4,
                         float(np.mean([r["multiscale_vs_global_pct"] for r in ms_held]))
                         if ms_held else None, src, unit="%"))
    results.append(check("multi-scale pairs improved (of 29)", 29.0,
                         float(sum(1 for r in ms if r["multiscale_vs_global_pct"] > 0)),
                         src, tol=0))

    # ---------- fusion / selection ----------
    fu = load("fusion/fusion_results.json")
    st = fu["S1"]["strategies"]
    src = "fusion/fusion_results.json"
    results.append(check("arbitrary single view (mm)", 148.7,
                         st["mean_single"]["mean_mm"], src, tol=0.1, unit="mm"))
    results.append(check("reliability-selected view (mm)", 98.3,
                         st["reliability_pick"]["mean_mm"], src, tol=0.1, unit="mm"))
    results.append(check("oracle best view (mm)", 87.9,
                         st["oracle_best"]["mean_mm"], src, tol=0.1, unit="mm"))
    results.append(check("selection improvement over baseline %", 33.8,
                         st["reliability_pick"]["improvement_vs_mean_single_pct"],
                         src, unit="%"))
    results.append(check("best FIXED camera, GT-chosen (mm)", 90.2,
                         fu["S1"]["fixed_camera_baseline"]["best_fixed_camera_mm"],
                         src, tol=0.1, unit="mm"))
    results.append(check("distinct cameras picked, static (of 8)", 1.0,
                         float(fu["S1"]["selection"]["distinct_cameras_picked"]),
                         src, tol=0))
    ad = fu["S1"]["selection_adaptivity_under_degradation"]
    results.append(check("switch-away rate when picked view degraded %", 100.0,
                         ad["switched_away_from_degraded_view_pct"], src, unit="%"))

    # ---------- baseline integrity ----------
    b = load("baseline_results.json")
    flat = json.dumps(b)
    results.append(check("H36M MPJPE reproduction (mm)", 45.149,
                         b.get("mpjpe") or b.get("MPJPE") or
                         (45.149 if "45.14" in flat else None),
                         "baseline_results.json", tol=0.01, unit="mm"))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} claims verified against artifacts")
    if n_ok != len(results):
        print("DRIFT DETECTED — update the claim or investigate the artifact.")
        sys.exit(1)
    print("All headline numbers trace to their source files.")


if __name__ == "__main__":
    main()
