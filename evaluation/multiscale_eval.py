"""
Multi-scale canonicalization evaluation on cached predictions.

Gate declared in the plan: multi-scale must beat single-scale (global) on the
dev pair to be claimed as an improvement; otherwise it is reported as an
honest negative result with per-level numbers.

Run:  ./venv/Scripts/python.exe -m evaluation.multiscale_eval
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.multiscale import multiscale_cross_view_distance, combined_cross_view_distance, LEVELS
from evaluation.protocol import build_pairing_table
from evaluation.run_eval import load_cache, cam_key

OUT_PATH = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval",
                        "multiscale_results.json")


def main():
    cache = load_cache()
    table = build_pairing_table()

    results = []
    for pair in table:
        ka = cam_key({"subject": pair["subject"], "sequence": pair["sequence"],
                      "camera": pair["cam_a"]})
        kb = cam_key({"subject": pair["subject"], "sequence": pair["sequence"],
                      "camera": pair["cam_b"]})
        a, b = cache[ka], cache[kb]
        common, ia, ib = np.intersect1d(a["centers"], b["centers"],
                                        return_indices=True)

        level_sums = {lv: [] for lv in LEVELS}
        combined = []
        for i, j in zip(ia, ib):
            d = multiscale_cross_view_distance(a["raw"][i], b["raw"][j])
            for lv in LEVELS:
                if not np.isnan(d[lv]):
                    level_sums[lv].append(d[lv])
            c = combined_cross_view_distance(a["raw"][i], b["raw"][j])
            if not np.isnan(c):
                combined.append(c)

        entry = {
            "pair": f"{pair['subject']}/{pair['sequence']} "
                    f"cam{pair['cam_a']}-cam{pair['cam_b']}",
            "is_dev_pair": pair["is_dev_pair"],
            "global": float(np.mean(level_sums["global"])),
            "combined_multiscale": float(np.mean(combined)),
            "per_level": {lv: float(np.mean(v)) if v else None
                          for lv, v in level_sums.items()},
        }
        entry["multiscale_vs_global_pct"] = float(
            (1 - entry["combined_multiscale"] / max(entry["global"], 1e-8)) * 100)
        results.append(entry)

    dev = [r for r in results if r["is_dev_pair"]][0]
    held = [r for r in results if not r["is_dev_pair"]]
    print(f"{'pair':34s} {'global':>8} {'multi':>8} {'delta':>8}")
    for r in results:
        tag = " [DEV]" if r["is_dev_pair"] else ""
        print(f"{r['pair']:28s}{tag:6s} {r['global']:8.4f} "
              f"{r['combined_multiscale']:8.4f} "
              f"{r['multiscale_vs_global_pct']:+7.1f}%")

    deltas = [r["multiscale_vs_global_pct"] for r in held]
    print(f"\nDev pair: multiscale vs global = {dev['multiscale_vs_global_pct']:+.1f}%")
    print(f"Held-out mean: {np.mean(deltas):+.1f}% "
          f"(min {np.min(deltas):+.1f}%, max {np.max(deltas):+.1f}%)")
    gate = "PASS — multiscale beats global on dev pair" \
        if dev["multiscale_vs_global_pct"] > 0 else \
        "FAIL — report as honest negative result"
    print(f"GATE: {gate}")

    with open(OUT_PATH, "w") as f:
        json.dump({"gate": gate, "results": results}, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
