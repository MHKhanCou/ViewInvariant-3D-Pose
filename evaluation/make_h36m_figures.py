"""
Figures for the Human3.6M replication chapter, from frozen artifacts.

  ./venv/Scripts/python.exe -m evaluation.make_h36m_figures

Outputs -> thesis_artifacts/figures/fig_h36m_*.png and copies into
thesis_report/images/. Palette and rcParams are shared with make_figures.py so
the two sets look like one document.
"""

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.make_figures import (AQUA, BLUE, GRID, INK, INK2, MAGENTA,
                                     ORANGE, SURFACE, YELLOW)

ART = os.path.join(REPO_ROOT, "thesis_artifacts")
OUT = os.path.join(ART, "figures")
REPORT_IMG = os.path.join(REPO_ROOT, "thesis_report", "images")


def _load(rel):
    with open(os.path.join(ART, rel)) as f:
        return json.load(f)


def _finish(fig, name, source):
    fig.text(0.01, 0.01, "Source: %s" % source, fontsize=7, color=INK2)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    os.makedirs(REPORT_IMG, exist_ok=True)
    shutil.copy(p, os.path.join(REPORT_IMG, name))
    print("  %s" % name)


def fig_crossview_actions(cv):
    """Per-action improvement, with SittingDown called out as the exception."""
    s = cv["summary"]
    items = sorted(s["by_action"].items(), key=lambda kv: kv[1]["mean_improvement_pct"])
    names = [k for k, _ in items]
    vals = [v["mean_improvement_pct"] for _, v in items]
    colors = [ORANGE if v < 50 else BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(names, vals, color=colors, height=0.68)
    ax.axvline(32.4, color=INK2, ls="--", lw=1.1)
    ax.text(33.5, len(names) - 0.4, "MPI-INF-3DHP mean (+32.4%)", fontsize=8,
            color=INK2, ha="left", va="center")
    for n, v in zip(names, vals):
        ax.text(v + 1.2, n, "%+.1f%%" % v, va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Reduction in cross-view joint distance (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Canonicalization improves 14 of 15 actions by more than 70%",
                 fontsize=11, loc="left", pad=10)
    ax.text(0, 1.005, "Human3.6M, 180 held-out camera pairs, 12 per action",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    _finish(fig, "fig_h36m_actions.png", "h36m_crossview/h36m_crossview.json")


def fig_crossview_distances(cv):
    """Raw vs canonical vs oracle, and where SittingDown sits."""
    s = cv["summary"]
    per = cv["per_pair"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    bars = ["Raw", "Canonical", "Procrustes\noracle"]
    vals = [s["mean_raw_distance_mm"], s["mean_canonical_distance_mm"],
            s["mean_oracle_distance_mm"]]
    ax1.bar(bars, vals, color=[ORANGE, BLUE, AQUA], width=0.62)
    for i, v in enumerate(vals):
        ax1.text(i, v + 7, "%.1f" % v, ha="center", fontsize=9.5, color=INK)
    ax1.set_ylabel("Cross-view joint distance (mm)")
    ax1.set_ylim(0, max(vals) * 1.18)
    ax1.set_title("Mean over 180 pairs", fontsize=10, loc="left")

    ok = [r for r in per if r["action_name"] != "SittingDown"]
    bad = [r for r in per if r["action_name"] == "SittingDown"]
    ax2.scatter([r["raw_cross_view_distance"] for r in ok],
                [r["canonical_cross_view_distance"] for r in ok],
                s=16, color=BLUE, alpha=0.65, label="14 actions", zorder=3)
    ax2.scatter([r["raw_cross_view_distance"] for r in bad],
                [r["canonical_cross_view_distance"] for r in bad],
                s=26, color=ORANGE, alpha=0.9, label="SittingDown", zorder=4)
    lim = max(max(r["raw_cross_view_distance"] for r in per),
              max(r["canonical_cross_view_distance"] for r in per)) * 1.05
    ax2.plot([0, lim], [0, lim], color=INK2, ls="--", lw=1.0, zorder=2)
    ax2.text(lim * 0.52, lim * 0.60, "no change", fontsize=8, color=INK2, rotation=38)
    ax2.set_xlabel("Raw distance (mm)")
    ax2.set_ylabel("Canonical distance (mm)")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_title("Points below the line improved (179 of 180)",
                  fontsize=10, loc="left")
    fig.tight_layout()
    _finish(fig, "fig_h36m_distances.png", "h36m_crossview/h36m_crossview.json")


def fig_fusion(fz):
    """Mean fails, median works; the baseline is one arbitrary view."""
    s = fz["summary"]
    order = ["naive_median", "median", "reliability_weighted_mean",
             "naive_mean", "plain_mean"]
    pretty = {"naive_median": "Median", "median": "Median\n(+ reflection)",
              "reliability_weighted_mean": "Reliability-\nweighted mean",
              "naive_mean": "Mean", "plain_mean": "Mean\n(+ reflection)"}
    vals = [s["overall_mm"][k] for k in order]
    base = s["overall_mm"]["single_view_mean"]
    colors = [AQUA if v < base else ORANGE for v in vals]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar([pretty[k] for k in order], vals, color=colors, width=0.6, zorder=3)
    box = dict(facecolor=SURFACE, edgecolor="none", pad=1.6)
    ax.axhline(base, color=INK2, ls="--", lw=1.2, zorder=4)
    ax.text(len(order) - 0.42, base + 0.30,
            "single arbitrary view (%.1f mm)" % base,
            fontsize=8.5, color=INK2, ha="right", zorder=5, bbox=box)
    ax.axhline(s["overall_mm"]["oracle_best_view"], color=YELLOW, ls=":", lw=1.4, zorder=4)
    ax.text(-0.42, s["overall_mm"]["oracle_best_view"] + 0.30,
            "oracle best view (needs ground truth)", fontsize=8.5, color=INK2,
            zorder=5, bbox=box)
    for i, (k, v) in enumerate(zip(order, vals)):
        ax.text(i, v + 0.35, "%.1f  (%+.1f%%)"
                % (v, s["improvement_vs_single_view_pct"][k]),
                ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("Similarity-aligned error (mm)")
    ax.set_ylim(25, max(vals) * 1.10)
    ax.set_title("At four views the estimator must be robust", fontsize=11,
                 loc="left", pad=10)
    ax.text(0, 1.005,
            "Human3.6M, %d fused frames. A mean is worse than one view; a median is better."
            % s["n_frames"],
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    _finish(fig, "fig_h36m_fusion.png", "h36m_fusion/h36m_fusion.json")


def fig_bone_retraction(bc, hr):
    """The retraction, side by side."""
    labels = ["Bone deviation\nvs error", "Partial, given\nconfidence",
              "Causal reference\nwindow", "Incumbent\nreliability score"]
    mpi = [bc["pooled"]["spearman_bone_deviation_vs_error"],
           bc["pooled"]["partial_given_detector_confidence"],
           bc["pooled"]["causal_estimate_rho"],
           bc["pooled"]["spearman_reliability_vs_error"]]
    h36 = [hr["pooled"]["spearman_bone_deviation_vs_error"],
           hr["pooled"]["partial_given_detector_confidence"],
           hr["pooled"]["causal_estimate_rho"],
           hr["pooled"]["spearman_reliability_vs_error"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.bar(x - 0.19, mpi, 0.36, label="MPI-INF-3DHP", color=BLUE, zorder=3)
    ax.bar(x + 0.19, h36, 0.36, label="Human3.6M", color=ORANGE, zorder=3)
    ax.axhline(0.30, color=MAGENTA, ls="--", lw=1.2, zorder=4)
    ax.text(len(labels) - 0.5, 0.315, "pass threshold (+0.30)",
            fontsize=8.5, color=MAGENTA, ha="right")
    ax.axhline(0, color=INK2, lw=0.9)
    for xi, (a, b) in enumerate(zip(mpi, h36)):
        ax.text(xi - 0.19, a + (0.02 if a >= 0 else -0.05), "%+.3f" % a,
                ha="center", fontsize=8, color=INK)
        ax.text(xi + 0.19, b + (0.02 if b >= 0 else -0.05), "%+.3f" % b,
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"Spearman $\rho$ against true error")
    ax.set_ylim(-0.32, 0.62)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("The bone-length signal does not replicate; the score it beat does",
                 fontsize=11, loc="left", pad=10)
    _finish(fig, "fig_h36m_bone.png",
            "bone_consistency/*.json and h36m_replication/*.json")


def fig_multiscale(ms):
    """Per-level distances, and the bilateral asymmetry the fix removes."""
    s = ms["summary"]
    v = s["long_axis_variant"]
    levels = ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    pretty = ["Torso", "Left arm", "Right arm", "Left leg", "Right leg"]
    shipped = [s["per_level_distance_mm"][k] for k in levels]
    fixed = [v["per_level_distance_mm"][k] for k in levels]

    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - 0.19, shipped, 0.36, label="As implemented", color=ORANGE, zorder=3)
    ax.bar(x + 0.19, fixed, 0.36, label="Every frame from its own long axis",
           color=BLUE, zorder=3)
    ax.axhline(s["per_level_distance_mm"]["global"], color=INK2, ls="--", lw=1.2, zorder=4)
    ax.text(len(levels) - 0.5, s["per_level_distance_mm"]["global"] + 1.4,
            "global frame only (%.1f mm)" % s["per_level_distance_mm"]["global"],
            fontsize=8.5, color=INK2, ha="right",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6), zorder=5)
    for xi, (a, b) in enumerate(zip(shipped, fixed)):
        ax.text(xi - 0.19, a + 1.2, "%.1f" % a, ha="center", fontsize=8, color=INK)
        ax.text(xi + 0.19, b + 1.2, "%.1f" % b, ha="center", fontsize=8, color=INK)
    ax.annotate("", xy=(0.81, 30), xytext=(0.81, 55),
                arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.5))
    ax.annotate("", xy=(3.19, 32), xytext=(2.86, 66),
                arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.5))
    ax.text(1.30, 46, "both arms and the left leg\nused a short primary axis",
            fontsize=8.5, color=MAGENTA, va="center",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=9.5)
    ax.set_ylabel("Cross-view distance of the level's joints (mm)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Frames built from short axes were the limiting factor",
                 fontsize=11, loc="left", pad=22)
    ax.text(0, 1.012,
            "Human3.6M, 180 held-out pairs. Applying one rule to all five levels "
            "lifts the gain from %+.1f%% to %+.1f%%."
            % (s["mean_multiscale_vs_global_pct"], v["mean_multiscale_vs_global_pct"]),
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    _finish(fig, "fig_h36m_multiscale.png", "h36m_multiscale/h36m_multiscale.json")


def main():
    cv = _load("h36m_crossview/h36m_crossview.json")
    fz = _load("h36m_fusion/h36m_fusion.json")
    bc = _load("bone_consistency/bone_consistency.json")
    hr = _load("h36m_replication/h36m_replication.json")

    print("Writing figures:")
    fig_crossview_actions(cv)
    fig_crossview_distances(cv)
    fig_fusion(fz)
    fig_bone_retraction(bc, hr)
    fig_multiscale(_load("h36m_multiscale/h36m_multiscale.json"))
    print("Copied into %s" % REPORT_IMG)


if __name__ == "__main__":
    main()
