"""
Defense figures from frozen artifacts. Regenerable with one command:

  ./venv/Scripts/python.exe -m evaluation.make_figures

Outputs -> thesis_artifacts/figures/fig_*.png
Palette: validated categorical slots (blue/orange/aqua/yellow/magenta),
recessive grid, direct labels. Sources cited in each figure's footer.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ART = os.path.join(REPO_ROOT, "thesis_artifacts")
OUT = os.path.join(ART, "figures")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK2 = "#6b6a63"
GRID = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load(rel):
    with open(os.path.join(ART, rel)) as f:
        return json.load(f)


def _static29():
    """The 29 camera pairs these figures are captioned for.

    results_multicam.json holds 85 records across two conditions: the 29
    static pairs the cross-view section reports on, and 56 dynamic-window
    records used by the fusion and selection experiments. Plotting all 85
    against a per-pair label silently merges duplicates -- matplotlib treats
    the label as a category, so the dynamic record overwrites the static bar
    while both still draw a value annotation. That is what produced the
    phantom diagonal of labels disagreeing with the bars.

    Filtering to the static condition yields 29 records with 29 distinct
    labels, 27 held-out S1 pairs averaging +32.4%, one development pair and
    one held-out S2 pair -- exactly the set Section 5.3 describes.
    """
    rs = [r for r in _load("cross_view_eval/results_multicam.json")
          if r["condition"] == "static"]
    labels = {"%s cam%s-%s" % (r["subject"], r["cam_a"], r["cam_b"]) for r in rs}
    assert len(rs) == 29 and len(labels) == 29, (len(rs), len(labels))
    return rs


def _save(fig, name, source):
    fig.text(0.99, 0.01, f"source: {source}", ha="right", va="bottom",
             fontsize=7, color=INK2)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def fig_pairs():
    """29-pair improvement, sorted; dev pair and held-out subject highlighted."""
    rs = _static29()
    rows = sorted(rs, key=lambda r: r["improvement_pct"])
    labels = [f"{r['subject']} cam{r['cam_a']}-{r['cam_b']}" for r in rows]
    vals = [r["improvement_pct"] for r in rows]
    colors = [ORANGE if r["is_dev_pair"]
              else AQUA if r["subject"] == "S2" else BLUE for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    ax.axvline(0, color=INK2, linewidth=0.8)
    held = [r["improvement_pct"] for r in rs
            if not r["is_dev_pair"] and r["subject"] == "S1"]
    m = float(np.mean(held))
    ax.axvline(m, color=BLUE, linewidth=1.2, linestyle="--")
    ax.text(m + 0.7, 6.5, f"held-out mean {m:+.1f}%",
            color=BLUE, fontsize=9, va="center", rotation=90)
    for bar, v in zip(bars, vals):
        ax.text(v + (0.8 if v >= 0 else -0.8), bar.get_y() + bar.get_height() / 2,
                f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["held-out pair (S1)", "development pair", "held-out subject (S2)"],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_xlabel("Cross-view distance improvement, raw → canonical (%)")
    ax.set_title("Canonicalization improvement across all 29 camera pairs\n"
                 "(corrected protocol: true 27-frame windows)", fontsize=11)
    ax.tick_params(axis="y", labelsize=7.5)
    _save(fig, "fig_29pairs.png", "cross_view_eval/results_multicam.json")


def fig_oracle():
    """Dumbbell: raw / canonical / Procrustes oracle per pair, sorted by raw."""
    rs = sorted(_static29(), key=lambda r: r["raw_cross_view_distance"])
    labels = [f"{r['subject']} cam{r['cam_a']}-{r['cam_b']}" for r in rs]
    y = np.arange(len(rs))
    raw = [r["raw_cross_view_distance"] for r in rs]
    can = [r["canonical_cross_view_distance"] for r in rs]
    orc = [r["oracle_cross_view_distance"] for r in rs]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for i in range(len(rs)):
        ax.plot([orc[i], raw[i]], [y[i], y[i]], color=GRID,
                linewidth=1.4, zorder=1)
    ax.scatter(raw, y, s=26, color=BLUE, zorder=3, label="raw")
    ax.scatter(can, y, s=26, color=ORANGE, zorder=4, label="canonical")
    ax.scatter(orc, y, s=26, color=AQUA, zorder=2, marker="D",
               label="Procrustes oracle (lower bound)")
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("Cross-view joint distance")
    ax.set_title("Canonical distance approaches the per-pair rigid-alignment\n"
                 "oracle without training", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _save(fig, "fig_oracle.png", "cross_view_eval/results_multicam.json")


def fig_degradation():
    """Reliability vs induced drift, by corruption operator."""
    d = _load("degradation/sweep_records.json")
    recs = [r for r in d["records"]
            if not r["is_control"] and r["operator"] != "clean"]
    ops = ["jitter", "joint_dropout", "limb_foreshorten",
           "hip_axis_collapse", "torso_axis_collapse"]
    colors = dict(zip(ops, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
    a = _load("degradation/analysis.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for op in ops:
        pts = [(r["reliability"], r["canonical_drift"])
               for r in recs if r["operator"] == op]
        x, yv = zip(*pts)
        ax.scatter(x, yv, s=14, alpha=0.55, color=colors[op],
                   label=op.replace("_", " "), edgecolors="none")
    rho = a["pooled"]["spearman_reliability_vs_drift"]
    ax.text(0.22, 0.55, f"pooled Spearman ρ = {rho:+.3f}\n(n={a['pooled']['n']})",
            transform=ax.transAxes, fontsize=10, va="top", color=INK)
    ax.axvline(0.5, color=INK2, linewidth=0.9, linestyle=":")
    ax.text(0.5, 0.02, "abstention threshold  ", ha="right", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=8, color=INK2,
            rotation=90)
    ax.text(0.02, 0.13, "joint dropout →\n100% abstained", fontsize=8.5,
            color=ORANGE, transform=ax.transAxes)
    ax.set_xlabel("Training-free reliability score")
    ax.set_ylabel("Canonical drift vs clean baseline (induced error)")
    ax.set_title("Reliability score tracks induced error under controlled degradation",
                 fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, markerscale=1.6)
    _save(fig, "fig_degradation.png", "degradation/sweep_records.json + analysis.json")


def fig_coverage():
    """Coverage-error curve, dev pair: real error axis at each threshold."""
    ab = _load("coverage_error/ablation_results.json")
    dev = [a for a in ab if a["is_dev_pair"]][0]
    # Collapse thresholds that share the same operating point.
    grouped = {}
    for e in dev["reliability_aware"]:
        if e["filtered_canonical_distance"] is None:
            continue
        k = (round(e["coverage"], 4), round(e["filtered_canonical_distance"], 6))
        grouped.setdefault(k, []).append(e["threshold"])
    pts = sorted((c, dv, ts) for (c, dv), ts in grouped.items())
    cov, dist, _ = zip(*pts)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(cov, dist, color=BLUE, linewidth=2, marker="o", markersize=7)
    for c, dv, ts in pts:
        label = f"t={ts[0]}" if len(ts) == 1 else f"t≤{max(ts)}"
        ax.annotate(label, (c, dv), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color=INK2)
    base = dev["conditions"]["canonical"]
    ax.axhline(base, color=ORANGE, linewidth=1.2, linestyle="--")
    ax.text(min(cov), base, " canonical, no abstention ",
            fontsize=8.5, color=ORANGE, va="bottom")
    ax.set_xlabel("Coverage (fraction of frames kept)")
    ax.set_ylabel("Cross-view canonical distance on kept frames")
    ax.set_title("Reliability-aware abstention: error decreases as\n"
                 "low-reliability frames are dropped (dev pair)", fontsize=11)
    _save(fig, "fig_coverage_error.png", "coverage_error/ablation_results.json")


def fig_multiscale():
    """Global vs multi-scale distance per pair (sorted), all-positive deltas."""
    ms = _load("cross_view_eval/multiscale_results.json")["results"]
    rows = sorted(ms, key=lambda r: r["multiscale_vs_global_pct"])
    labels = [r["pair"].replace("/Seq1", "") for r in rows]
    vals = [r["multiscale_vs_global_pct"] for r in rows]
    colors = [ORANGE if r["is_dev_pair"]
              else AQUA if r["pair"].startswith("S2") else BLUE for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"+{v:.0f}%",
                va="center", fontsize=7.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["held-out pair (S1)", "development pair",
                        "held-out subject (S2)"],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_xlabel("Additional improvement of multi-scale over global canonical (%)")
    ax.set_title("Per-limb multi-scale frames improve every pair\n"
                 "(GT-error correlation preserved: ρ +0.610 vs +0.601)", fontsize=11)
    ax.tick_params(axis="y", labelsize=7.5)
    _save(fig, "fig_multiscale.png", "cross_view_eval/multiscale_results.json")


def fig_fusion():
    """Error vs number of available cameras, per training-free strategy."""
    d = _load("fusion/fusion_results.json")["S1"]
    curve = d["view_count_curve"]
    ks = sorted(curve, key=int)
    x = [int(k) for k in ks]

    series = [
        ("mean_single", "arbitrary single view (deployment baseline)", ORANGE, "-"),
        ("weighted_mean", "reliability-weighted fusion", AQUA, "-"),
        ("median", "median fusion", YELLOW, "-"),
        ("reliability_pick", "reliability-selected view", BLUE, "-"),
        ("oracle_best", "best view (oracle — needs ground truth)", INK2, "--"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5))
    for key, label, color, ls in series:
        y = [curve[k][key] for k in ks]
        ax.plot(x, y, color=color, linewidth=2.2, linestyle=ls,
                marker="o", markersize=6)
        ax.annotate(label, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8.5, color=color, va="center")

    ax.set_xlabel("Number of available cameras")
    ax.set_ylabel("Error vs ground truth (mm, similarity-aligned)")
    ax.set_title("Training-free view selection and fusion\n"
                 "(no calibration, no correspondence, no training)", fontsize=11)
    ax.set_xlim(1.8, 11.6)
    ax.set_xticks(x)
    _save(fig, "fig_fusion.png", "fusion/fusion_results.json")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_pairs()
    fig_oracle()
    fig_degradation()
    fig_coverage()
    fig_multiscale()
    fig_fusion()
