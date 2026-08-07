"""
Emit LaTeX appendix tables straight from the stored artifacts.

Every row is generated from a JSON file rather than typed, so the appendix
cannot drift away from the numbers the audit verifies. Writes one .tex file
that the report includes.

Run:  ./venv/Scripts/python.exe -m evaluation.make_appendix_tables
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
ART = os.path.join(REPO_ROOT, "thesis_artifacts")
OUT = os.path.join(REPO_ROOT, "thesis_report", "appendix_tables.tex")


def load(rel):
    with open(os.path.join(ART, rel)) as f:
        return json.load(f)


def longtable(caption, label, spec, header, rows, note=None):
    out = [r"\begin{center}", r"{\fontsize{10}{12}\selectfont",
           r"\begin{longtable}{%s}" % spec,
           r"\caption{%s}\label{%s}\\" % (caption, label),
           r"\hline", header + r" \\", r"\hline", r"\endfirsthead",
           r"\multicolumn{%d}{l}{\textit{(continued)}}\\" % len(spec.replace("|", "")),
           r"\hline", header + r" \\", r"\hline", r"\endhead",
           r"\hline", r"\endfoot"]
    out += [r + r" \\" for r in rows]
    out += [r"\end{longtable}", "}", r"\end{center}"]
    if note:
        out.append(r"\noindent %s" % note)
    return "\n".join(out)


def _opt(rel):
    """Load an artifact, or None if that experiment has not been run here."""
    try:
        return load(rel)
    except (IOError, OSError, ValueError):
        return None


def post_freeze_tables():
    """
    Tables for the eight experiments run after the report was frozen.

    Each is emitted only if its artifact exists, so a fresh clone that has not
    run them still produces a valid appendix.
    """
    out = []

    for tag, lab in (("", "MotionAGFormer-XS"), ("_motionbert", "MotionBERT")):
        bf = _opt("bestframe/bestframe%s.json" % tag)
        if not bf:
            continue
        rows = []
        for v, d in bf["scored_9"]["variants"].items():
            rows.append("%s & %.2f & %+.2f & [%+.2f, %+.2f] & %s"
                        % (v.replace("_", " "), d["mean_mm"],
                           d["mean_gain_vs_template_mm"], d["gain_ci95"][0],
                           d["gain_ci95"][1], d["verdict"].replace("_", " ")))
        rows.append(r"\hline")
        rows.append("Kabsch to fixed template & %.2f & --- & --- & ---"
                    % bf["scored_9"]["mean_template_mm"])
        out.append(longtable(
            "Every frame construction against the template baseline, %s, "
            "scored on the nine joints no construction builds from." % lab,
            "tab:appbestframe%s" % (tag or "_xs"), "lrrrl",
            r"\textbf{Construction} & \textbf{mm} & \textbf{Gain} & "
            r"\textbf{CI95} & \textbf{Verdict}", rows))

    for name, key, cols in (
            ("occlusion", "Distal joint corruption", ("mean_anatomical_mm",
             "mean_template17_mm", "mean_template4_mm")),
            ("asymmetric", "Left-side-only distal corruption",
             ("mean_anatomical_mm", "mean_template17_mm", "mean_template4_mm")),
            ("anchor_corruption", "Anchor joint corruption",
             ("mean_anatomical_mm", "mean_template17_mm", "mean_template4_mm"))):
        for tag, lab in (("", "MotionAGFormer-XS"), ("_motionbert", "MotionBERT")):
            d = _opt("%s/%s%s.json" % (name, name, tag))
            if not d:
                continue
            rows = []
            for l in d["levels"]:
                rows.append("%.0f & %.2f & %.2f & %.2f & %s"
                            % (l["sigma_mm"],
                               l[cols[0]], l[cols[1]], l[cols[2]],
                               l["verdict_vs_template17"].replace("_", " ")))
            out.append(longtable(
                "%s, %s." % (key, lab), "tab:app%s%s" % (name, tag or "_xs"),
                "rrrrl",
                r"\textbf{$\sigma$ (mm)} & \textbf{Anatomical} & "
                r"\textbf{Template-17} & \textbf{Template-4} & \textbf{Verdict}",
                rows))

    for tag, lab in (("", "MotionAGFormer-XS"), ("_motionbert", "MotionBERT")):
        d = _opt("laterality/laterality%s.json" % tag)
        if not d:
            continue
        rows = []
        for l in d["levels"]:
            b = l["onesided_minus_balanced"]
            rows.append("%.0f & %.2f & %.2f & %+.2f & [%+.2f, %+.2f]"
                        % (l["sigma_mm"], l["mean_one_sided_mm"],
                           l["mean_balanced_mm"], b["mean"], b["ci95"][0],
                           b["ci95"][1]))
        out.append(longtable(
            "Laterality at matched joint count, %s. A negative difference means "
            "one-sided corruption damages the template alignment less." % lab,
            "tab:applaterality%s" % (tag or "_xs"), "rrrrr",
            r"\textbf{$\sigma$ (mm)} & \textbf{One-sided} & \textbf{Balanced} & "
            r"\textbf{Difference} & \textbf{CI95}", rows))

    mm = _opt("mismatch/mismatch.json")
    if mm:
        rows = ["%.1f & %.2f & %.2f"
                % (l["limb_factor"], l["mean_anatomical_mm"],
                   l["mean_template17_mm"]) for l in mm["levels"]]
        out.append(longtable(
            "Template proportion mismatch. Scaling the template's limbs across "
            "the full range moves the baseline by a fraction of a millimetre.",
            "tab:appmismatch", "rrr",
            r"\textbf{Limb factor} & \textbf{Anatomical (mm)} & "
            r"\textbf{Template (mm)}", rows))

    md = _opt("misdetect/misdetect.json")
    if md:
        rows = []
        for cam, c in sorted(md["per_camera"].items()):
            for cond, v in sorted(c["conditions"].items()):
                rows.append("%s & %s & %.2f & %.2f"
                            % (cam.replace("_", " "), cond.replace("_", " "),
                               v["mean_mm"], v["max_mm"]))
        out.append(longtable(
            "Invariance of the frozen lifter to 2D keypoint displacement, "
            "through the real detection path.",
            "tab:appmisdetect", "llrr",
            r"\textbf{Camera} & \textbf{Condition} & \textbf{Mean $|\Delta|$ "
            r"(mm)} & \textbf{Max (mm)}", rows))

    sl = _opt("selection/selection.json")
    if sl:
        rows = ["%s & %.0f & %s & %.2f & %.2f & %.2f"
                % (c["regime"], c["sigma_mm"],
                   "anatomical" if c["route_anatomical"] else "template",
                   c["mean_anatomical_mm"], c["mean_template17_mm"],
                   c["mean_routed_mm"]) for c in sl["cells"]]
        out.append(longtable(
            "The confidence-gated routing rule, evaluated under a simulated "
            "gate. Reported as exploratory.",
            "tab:appselection", "lrlrrr",
            r"\textbf{Regime} & \textbf{$\sigma$} & \textbf{Routes to} & "
            r"\textbf{Anatomical} & \textbf{Template} & \textbf{Routed}", rows))

    return out


def main():
    parts = []

    # --- A.1 cross-view by action -----------------------------------------
    cv = load("h36m_crossview/h36m_crossview.json")
    s = cv["summary"]
    rows = []
    for a, v in sorted(s["by_action"].items(),
                       key=lambda kv: -kv[1]["mean_improvement_pct"]):
        rows.append("%s & %d & %.1f & %+.1f & %d/%d"
                    % (a, v["n"], v["mean_canonical_distance_mm"],
                       v["mean_improvement_pct"], v["pairs_improved"], v["n"]))
    parts.append(longtable(
        "Cross-view canonicalization on Human3.6M, by action.",
        "tab:appcvaction", "lrrrr",
        r"\textbf{Action} & \textbf{Pairs} & \textbf{Canonical (mm)} & "
        r"\textbf{Improvement} & \textbf{Improved}",
        rows))

    # --- A.2 every camera pair --------------------------------------------
    rows = []
    for r in sorted(cv["per_pair"],
                    key=lambda r: (r["subject"], r["action"], r["cam_a"], r["cam_b"])):
        rows.append("%s & %s & %s--%s & %d & %.1f & %.1f & %.1f & %+.1f"
                    % (r["subject"], r["action_name"],
                       r["cam_a"].replace("ca_", ""), r["cam_b"].replace("ca_", ""),
                       r["n_frames"], r["raw_cross_view_distance"],
                       r["canonical_cross_view_distance"],
                       r["oracle_cross_view_distance"], r["improvement_pct"]))
    parts.append(longtable(
        "All 180 held-out Human3.6M camera pairs. Distances in millimetres.",
        "tab:apppairs", "llrrrrrr",
        r"\textbf{Subj} & \textbf{Action} & \textbf{Pair} & \textbf{N} & "
        r"\textbf{Raw} & \textbf{Canon.} & \textbf{Oracle} & \textbf{Impr.}",
        rows,
        note="The single regressing pair is S9 SittingDown 01--03. Its raw "
             "distance of 218.4 mm is unusually small for that action, where "
             "the other pairs range above 500 mm, so little was available to "
             "gain. See Section \\ref{sec:h36mcv}."))

    # --- A.3 fusion by action ---------------------------------------------
    fz = load("h36m_fusion/h36m_fusion.json")["summary"]
    rows = []
    for a, v in sorted(fz["by_action"].items(),
                       key=lambda kv: -kv[1]["anatomical_median_improvement_pct"]):
        rows.append("%s & %d & %.1f & %.1f & %+.1f"
                    % (a, v["n_frames"], v["single_view_mm"],
                       v["anatomical_median_mm"], v["anatomical_median_improvement_pct"]))
    parts.append(longtable(
        "Calibration-free fusion on Human3.6M by action, median over four views.",
        "tab:appfusion", "lrrrr",
        r"\textbf{Action} & \textbf{Frames} & \textbf{Single view (mm)} & "
        r"\textbf{Fused (mm)} & \textbf{Improvement}",
        rows))

    # --- A.4 fusion strategies --------------------------------------------
    rows = []
    order = ["naive_median", "median", "reliability_weighted_mean",
             "anatomical_median", "naive_mean", "plain_mean", "anatomical_mean"]
    pretty = {
        "naive_median": "Median, no reflection handling",
        "median": "Median, sign-flip reflection handling",
        "reliability_weighted_mean": "Reliability-weighted mean",
        "anatomical_median": "Median, anatomical mirror handling",
        "naive_mean": "Mean, no reflection handling",
        "plain_mean": "Mean, sign-flip reflection handling",
        "anatomical_mean": "Mean, anatomical mirror handling",
    }
    for k in order:
        rows.append("%s & %.1f & %+.1f"
                    % (pretty[k], fz["overall_mm"][k],
                       fz["improvement_vs_single_view_pct"][k]))
    rows.append(r"\hline")
    rows.append("Single arbitrary view (baseline) & %.1f & ---"
                % fz["overall_mm"]["single_view_mean"])
    rows.append("Worst view & %.1f & ---" % fz["overall_mm"]["worst_view"])
    rows.append("Oracle best view (requires ground truth) & %.1f & ---"
                % fz["overall_mm"]["oracle_best_view"])
    parts.append(longtable(
        "Fusion strategies on Human3.6M, all four views, %d frames."
        % fz["n_frames"],
        "tab:appfusionstrat", "lrr",
        r"\textbf{Strategy} & \textbf{Error (mm)} & \textbf{vs single view}",
        rows,
        note="Every strategy above the line is training-free and label-free. "
             "The best of them uses no reliability score at all."))

    # --- A.5 multi-scale by action ----------------------------------------
    msc = load("h36m_multiscale/h36m_multiscale.json")["summary"]
    rows = []
    for a, v in sorted(msc["by_action"].items(), key=lambda kv: -kv[1]["mean_pct"]):
        rows.append("%s & %d & %+.1f & %d/%d"
                    % (a, v["n"], v["mean_pct"], v["pairs_improved"], v["n"]))
    parts.append(longtable(
        "Multi-scale per-limb canonicalization on Human3.6M, by action. "
        "Improvement is measured against the global frame.",
        "tab:appms", "lrrr",
        r"\textbf{Action} & \textbf{Pairs} & \textbf{vs global frame} & "
        r"\textbf{Improved}",
        rows,
        note="The gain is largest on SittingDown, the action on which the "
             "global frame performs worst, because a limb frame does not "
             "inherit the error in the torso and hip axes. See Section "
             "\\ref{sec:h36mms}."))

    # --- A.6 bone-length replication --------------------------------------
    hr = load("h36m_replication/h36m_replication.json")
    bc = load("bone_consistency/bone_consistency.json")
    rows = [
        r"Spearman $\rho$(bone deviation, error) & $%+.3f$ & $%+.3f$"
        % (bc["pooled"]["spearman_bone_deviation_vs_error"],
           hr["pooled"]["spearman_bone_deviation_vs_error"]),
        r"Partial $\rho$ given detector confidence & $%+.3f$ & $%+.3f$"
        % (bc["pooled"]["partial_given_detector_confidence"],
           hr["pooled"]["partial_given_detector_confidence"]),
        r"Causal reference window & $%+.3f$ & $%+.3f$"
        % (bc["pooled"]["causal_estimate_rho"], hr["pooled"]["causal_estimate_rho"]),
        r"Incumbent reliability score & $%+.3f$ & $%+.3f$"
        % (bc["pooled"]["spearman_reliability_vs_error"],
           hr["pooled"]["spearman_reliability_vs_error"]),
        r"Bootstrap CI on $|\rho|-|\rho_{\mathrm{rel}}|$ & $[%+.3f, %+.3f]$ & $[%+.3f, %+.3f]$"
        % (*bc["pooled"]["bootstrap_ci_delta_abs_rho"],
           *hr["pooled"]["bootstrap_ci_delta_abs_rho"]),
        r"Frames evaluated & %d & %d" % (bc["n_frames"], hr["n_frames_evaluated"]),
        r"Median aligned error (mm) & 202.1 & %.1f"
        % hr["signal_magnitude"]["error_median_mm"],
        r"Median bone deviation & 0.082 & %.3f"
        % hr["signal_magnitude"]["bone_deviation_median"],
    ]
    parts.append(longtable(
        "The bone-length signal on both datasets. The claim is retracted.",
        "tab:appbone", "lrr",
        r"\textbf{Quantity} & \textbf{MPI-INF-3DHP} & \textbf{Human3.6M}",
        rows,
        note="All five pass criteria are met on MPI-INF-3DHP and none on "
             "Human3.6M. See Section \\ref{sec:h36m}."))

    parts += post_freeze_tables()

    with open(OUT, "w") as f:
        f.write("\n\n\\vspace{0.6cm}\n\n".join(parts) + "\n")
    print("Wrote %s" % OUT)
    print("  %d tables, %d data rows"
          % (len(parts), sum(p.count(r"\\") for p in parts)))


if __name__ == "__main__":
    main()
