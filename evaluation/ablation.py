"""
Reliability-aware ablation study — corrected protocol, computed from the
multi-camera results (which derive from the prediction cache).

Three conditions on every MPI-INF-3DHP camera pair:
1. Raw root-relative (no canonicalization)
2. Canonical (always canonicalize)
3. Reliability-aware canonical (keep only frames where BOTH views pass the
   reliability threshold and neither hard-fails) — reports the ACTUAL filtered
   cross-view distance alongside coverage, closing the prior stub.

Run:  ./venv/Scripts/python.exe -m evaluation.ablation
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_ablation(pair_results):
    """
    Ablation over all pairs. Condition 3 filters per-frame canonical distances
    by the joint (min of both views) reliability, using the per-frame arrays
    stored by run_eval.
    """
    out = []
    for r in pair_results:
        raw_d = np.array(r["per_frame_raw"])
        can_d = np.array(r["per_frame_canonical"])
        rel = np.minimum(np.array(r["per_frame_reliability_a"]),
                         np.array(r["per_frame_reliability_b"]))

        conditions = {
            "raw": float(raw_d.mean()),
            "canonical": float(can_d.mean()),
            "improvement_pct": float((1 - can_d.mean() / max(raw_d.mean(), 1e-8)) * 100),
        }

        ra = []
        for t in THRESHOLDS:
            mask = rel >= t
            entry = {
                "threshold": t,
                "coverage": float(mask.mean()),
                "n_kept": int(mask.sum()),
                "n_total": int(len(rel)),
            }
            if mask.any():
                entry["filtered_canonical_distance"] = float(can_d[mask].mean())
                entry["filtered_raw_distance"] = float(raw_d[mask].mean())
                entry["improvement_pct"] = float(
                    (1 - can_d[mask].mean() / max(raw_d[mask].mean(), 1e-8)) * 100)
            else:
                entry["filtered_canonical_distance"] = None
                entry["filtered_raw_distance"] = None
                entry["improvement_pct"] = None
            ra.append(entry)

        out.append({
            "pair": f"{r['subject']}/{r['sequence']} cam{r['cam_a']}-cam{r['cam_b']}",
            "is_dev_pair": r["is_dev_pair"],
            "conditions": conditions,
            "reliability_aware": ra,
        })
    return out


def main():
    results_path = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval",
                                "results_multicam.json")
    with open(results_path) as f:
        pair_results = json.load(f)

    ablation = run_ablation(pair_results)

    print("=" * 78)
    print("RELIABILITY-AWARE ABLATION (corrected protocol)")
    print("=" * 78)

    for a in ablation:
        tag = "  [DEV PAIR]" if a["is_dev_pair"] else ""
        c = a["conditions"]
        print(f"\n{a['pair']}{tag}")
        print(f"  1 Raw:        {c['raw']:.4f}")
        print(f"  2 Canonical:  {c['canonical']:.4f}  ({c['improvement_pct']:+.1f}%)")
        print(f"  3 Reliability-aware canonical:")
        print(f"    {'t':>5} {'cover':>7} {'kept':>5}  {'can-dist':>9}  {'impr':>7}")
        for e in a["reliability_aware"]:
            if e["filtered_canonical_distance"] is None:
                print(f"    {e['threshold']:>5.1f} {e['coverage']:>6.0%} "
                      f"{e['n_kept']:>5d}  {'—':>9}  {'—':>7}")
            else:
                print(f"    {e['threshold']:>5.1f} {e['coverage']:>6.0%} "
                      f"{e['n_kept']:>5d}  {e['filtered_canonical_distance']:>9.4f}  "
                      f"{e['improvement_pct']:>+6.1f}%")

    output_dir = os.path.join(REPO_ROOT, "thesis_artifacts", "coverage_error")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
