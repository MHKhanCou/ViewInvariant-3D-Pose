"""
Degradation sweep analysis: does the training-free reliability score track
induced error, and does abstention fire under corruption?

Inputs:  thesis_artifacts/degradation/sweep_records.json
Outputs: thesis_artifacts/degradation/analysis.json

Per corruption operator:
- Spearman rho(reliability, canonical_drift)  [drift = induced error vs clean]
- Spearman rho(severity, reliability)
- abstention rate (reliability < 0.5 or hard gate) per severity level
- hard-gate rate per severity level

Run:  ./venv/Scripts/python.exe -m evaluation.degradation_analysis
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SWEEP_PATH = os.path.join(REPO_ROOT, "thesis_artifacts", "degradation",
                          "sweep_records.json")
OUT_PATH = os.path.join(REPO_ROOT, "thesis_artifacts", "degradation",
                        "analysis.json")
ABSTAIN_T = 0.5  # frozen threshold, evaluation/reliability.py


def main():
    with open(SWEEP_PATH) as f:
        data = json.load(f)
    all_records = data["records"]
    # Clean baseline = the uncorrupted reference lifts (stored as controls).
    clean = [r for r in all_records if r["operator"] == "clean"]
    records = [r for r in all_records
               if not r["is_control"] and r["operator"] != "clean"]

    operators = sorted({r["operator"] for r in records})

    results = {
        "n_records": len(records),
        "clean": {
            "n": len(clean),
            "reliability_mean": float(np.mean([r["reliability"] for r in clean])),
            "reliability_min": float(np.min([r["reliability"] for r in clean])),
            "abstention_rate": float(np.mean([
                r["reliability"] < ABSTAIN_T or r["hard_failure"] for r in clean])),
        },
        "operators": {},
        "pooled": {},
    }

    all_rel, all_drift = [], []
    for op in operators:
        recs = [r for r in records if r["operator"] == op]
        rel = np.array([r["reliability"] for r in recs])
        drift = np.array([r["canonical_drift"] for r in recs])
        sev = np.array([r["severity"] for r in recs])
        hard = np.array([r["hard_failure"] for r in recs])
        abstain = (rel < ABSTAIN_T) | hard

        def _f(x):
            """NaN-safe float for strict JSON (zero-variance -> None)."""
            x = float(x)
            return None if np.isnan(x) else x

        rho_drift, p_drift = spearmanr(rel, drift)
        rho_sev, p_sev = spearmanr(sev, rel)

        per_sev = {}
        for s in sorted(set(sev.tolist())):
            m = sev == s
            per_sev[str(s)] = {
                "n": int(m.sum()),
                "reliability_mean": float(rel[m].mean()),
                "drift_mean": float(drift[m].mean()),
                "abstention_rate": float(abstain[m].mean()),
                "hard_gate_rate": float(hard[m].mean()),
            }

        results["operators"][op] = {
            "n": len(recs),
            "severity_unit": data["corruption_units"].get(op),
            "spearman_reliability_vs_drift": _f(rho_drift),
            "p_drift": _f(p_drift),
            "spearman_severity_vs_reliability": _f(rho_sev),
            "p_severity": _f(p_sev),
            "abstention_rate_overall": float(abstain.mean()),
            "hard_gate_rate_overall": float(hard.mean()),
            "per_severity": per_sev,
        }
        all_rel.extend(rel.tolist())
        all_drift.extend(drift.tolist())

        print(f"{op:20s} rho(rel, drift) = {rho_drift:+.3f} (p={p_drift:.2g})  "
              f"rho(sev, rel) = {rho_sev:+.3f}  "
              f"abstain {abstain.mean():.1%}  hard {hard.mean():.1%}")

    rho_all, p_all = spearmanr(all_rel, all_drift)
    results["pooled"]["spearman_reliability_vs_drift"] = float(rho_all)
    results["pooled"]["p_value"] = float(p_all)
    results["pooled"]["n"] = len(all_rel)
    print(f"\nPOOLED rho(reliability, drift) = {rho_all:+.3f} "
          f"(p={p_all:.2g}, n={len(all_rel)})")
    print(f"Clean baseline: reliability {results['clean']['reliability_mean']:.3f}, "
          f"abstention {results['clean']['abstention_rate']:.1%}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
