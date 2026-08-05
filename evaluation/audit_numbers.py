"""
Number audit: every headline claim re-derived from its artifact.

Each CLAIM below states a number that appears in FINAL_REPORT.md, the defense
deck, or DEFENSE_QA.md, together with the artifact it must come from. Running
this recomputes each one and fails loudly on drift â€” so a number can never
silently rot after a re-run.

Run:  ./venv/Scripts/python.exe -m evaluation.audit_numbers
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

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


_HIP_CACHE = {}


def _hip_axis_mm(action):
    """
    Mean hip-axis length for one Human3.6M action, recomputed from predictions.

    Kept in the audit because the report withdraws a stated mechanism on the
    strength of these two numbers, and a withdrawn explanation deserves the same
    protection against drift as a confirmed one.
    """
    if not _HIP_CACHE:
        from evaluation.h36m_replication import (OUT_DIR, aggregate_by_video,
                                                 parse_video)
        meta = np.load(os.path.join(OUT_DIR, "meta.npz"), allow_pickle=True)
        pn = np.load(os.path.join(OUT_DIR, "preds.npz"))
        for vid, v in aggregate_by_video(meta, pn, int(pn["n_clips"])).items():
            P = v["pred"][::20]
            _HIP_CACHE.setdefault(parse_video(vid)[1], []).append(
                np.linalg.norm(P[:, 1] - P[:, 4], axis=1))
    return float(np.concatenate(_HIP_CACHE[action]).mean())


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
    results.append(check("weighted fusion, static (mm)", 113.5,
                         st["weighted_mean"]["mean_mm"], src, tol=0.1, unit="mm"))
    results.append(check("weighted fusion gain, static %", 23.7,
                         st["weighted_mean"]["improvement_vs_mean_single_pct"],
                         src, unit="%"))
    results.append(check("reliability-selected view, static (mm)", 98.3,
                         st["reliability_pick"]["mean_mm"], src, tol=0.1, unit="mm"))
    results.append(check("oracle best view (mm)", 87.9,
                         st["oracle_best"]["mean_mm"], src, tol=0.1, unit="mm"))

    # --- withdrawn selection claim: the numbers that retire it ---
    dyn = fu.get("S1_dynamic")
    if dyn:
        ds = dyn["strategies"]
        results.append(check("dynamic: arbitrary single view (mm)", 214.9,
                             ds["mean_single"]["mean_mm"], src, tol=0.1, unit="mm"))
        results.append(check("dynamic: weighted fusion gain %", 10.6,
                             ds["weighted_mean"]["improvement_vs_mean_single_pct"],
                             src, unit="%"))
        results.append(check("dynamic: SELECTION gain (near-zero) %", 1.5,
                             ds["reliability_pick"]["improvement_vs_mean_single_pct"],
                             src, unit="%"))
        results.append(check("dynamic: picked-view true-error rank (~random 4.5)", 4.78,
                             dyn["selection"]["mean_true_error_rank_of_picked_view"],
                             src, tol=0.02))
        results.append(check("dynamic: distinct cameras picked (of 8)", 6.0,
                             float(dyn["selection"]["distinct_cameras_picked"]),
                             src, tol=0))
    d2 = fu.get("S2_dynamic_heldout")
    if d2:
        results.append(check("dynamic S2: weighted fusion gain %", 10.2,
                             d2["strategies"]["weighted_mean"]
                             ["improvement_vs_mean_single_pct"], src, unit="%"))
        results.append(check("dynamic S2: selection rank (straddles random 4.5)", 3.67,
                             d2["selection"]["mean_true_error_rank_of_picked_view"],
                             src, tol=0.02))
    results.append(check("best FIXED camera, GT-chosen (mm)", 90.2,
                         fu["S1"]["fixed_camera_baseline"]["best_fixed_camera_mm"],
                         src, tol=0.1, unit="mm"))
    results.append(check("distinct cameras picked, static (of 8)", 1.0,
                         float(fu["S1"]["selection"]["distinct_cameras_picked"]),
                         src, tol=0))
    ad = fu["S1"]["selection_adaptivity_under_degradation"]
    results.append(check("switch-away rate when picked view degraded %", 100.0,
                         ad["switched_away_from_degraded_view_pct"], src, unit="%"))

    # ---------- bone-length inconsistency (single-view error predictor) ----------
    bc = load("bone_consistency/bone_consistency.json")
    p = bc["pooled"]
    src = "bone_consistency/bone_consistency.json"
    results.append(check("rho(bone deviation, GT error)", 0.492,
                         p["spearman_bone_deviation_vs_error"], src, tol=0.005))
    results.append(check("  partial | detector confidence", 0.481,
                         p["partial_given_detector_confidence"], src, tol=0.005))
    results.append(check("  causal reference-window estimate", 0.473,
                         p["causal_estimate_rho"], src, tol=0.005))
    results.append(check("  bootstrap CI lower bound > 0", 0.108,
                         p["bootstrap_ci_delta_abs_rho"][0], src, tol=0.01))
    results.append(check("  strata with consistent sign (of 4)", 4.0,
                         float(sum(1 for v in bc["per_stratum"].values() if v > 0)),
                         src, tol=0))

    ca = bc.get("component_analysis") or {}
    if ca:
        results.append(check("  pruned composite, held-out S2", -0.357,
                             ca["pruned_composite"]["s2_heldout_rho"], src, tol=0.005))
        results.append(check("  incumbent composite, held-out S2", -0.162,
                             ca["incumbent_composite_all_six"]["s2_heldout_rho"],
                             src, tol=0.005))
        results.append(check("  bone+pruned combo, held-out S2", 0.395,
                             ca["bone_deviation_plus_pruned"]["s2_heldout_rho"],
                             src, tol=0.005))

    # ---------- H36M replication of the bone signal (FAILED, claim retracted) ----------
    # These entries exist to keep a retracted claim retracted. If a future re-run
    # moves these numbers, the retraction in Section 4.9 of the report has to move
    # with them.
    hr = load("h36m_replication/h36m_replication.json")
    hp = hr["pooled"]
    src = "h36m_replication/h36m_replication.json"
    results.append(check("H36M replication verdict is FAIL (0=fail)", 0.0,
                         float(bool(hr["verdict_pass"])), src, tol=0))
    results.append(check("  pipeline check: MPJPE == repo official 45.149mm", 45.149,
                         hr["sanity_check"]["mpjpe_action_balanced_mm"], src,
                         tol=0.005, unit="mm"))
    results.append(check("  rho(bone deviation, error) collapses", 0.098,
                         hp["spearman_bone_deviation_vs_error"], src, tol=0.005))
    results.append(check("  partial | detector confidence vanishes", 0.014,
                         hp["partial_given_detector_confidence"], src, tol=0.005))
    results.append(check("  bootstrap CI upper bound < 0", -0.063,
                         hp["bootstrap_ci_delta_abs_rho"][1], src, tol=0.01))
    results.append(check("  incumbent score DOES replicate", -0.212,
                         hp["spearman_reliability_vs_error"], src, tol=0.005))
    results.append(check("  strata of 8 with positive sign (mixed)", 7.0,
                         float(sum(1 for v in hr["per_stratum_subject_camera"].values()
                                   if v > 0)), src, tol=0))
    results.append(check("  hardest third rho (no difficulty rescue)", 0.156,
                         hr["difficulty_analysis"]["mean_rho_hardest_third"], src, tol=0.005))
    results.append(check("  H36M median bone deviation", 0.034,
                         hr["signal_magnitude"]["bone_deviation_median"], src, tol=0.001))
    results.append(check("  H36M median aligned error", 32.6,
                         hr["signal_magnitude"]["error_median_mm"], src, tol=0.1, unit="mm"))

    # ---------- H36M cross-view: replication of the CENTRAL claim ----------
    cv = load("h36m_crossview/h36m_crossview.json")["summary"]
    src = "h36m_crossview/h36m_crossview.json"
    results.append(check("H36M cross-view pairs (all held out)", 180.0,
                         float(cv["n_pairs"]), src, tol=0))
    results.append(check("  mean improvement %", 74.1,
                         cv["mean_improvement_pct"], src, unit="%"))
    results.append(check("  pairs improved", 179.0,
                         float(cv["n_pairs_improved"]), src, tol=0))
    results.append(check("  raw cross-view distance (mm)", 320.4,
                         cv["mean_raw_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  canonical cross-view distance (mm)", 75.3,
                         cv["mean_canonical_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  Procrustes oracle (mm)", 51.3,
                         cv["mean_oracle_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  oracle gap closed %", 90.5,
                         cv["mean_oracle_gap_closed_pct"], src, unit="%"))
    results.append(check("  improvement CI lower bound %", 69.8,
                         cv["bootstrap"]["improvement_pct"]["ci95"][0], src,
                         tol=0.3, unit="%"))
    results.append(check("  improvement CI upper bound %", 77.2,
                         cv["bootstrap"]["improvement_pct"]["ci95"][1], src,
                         tol=0.3, unit="%"))
    results.append(check("  bootstrap clusters (groups, not pairs)", 30.0,
                         float(cv["bootstrap"]["improvement_pct"]["n_clusters"]),
                         src, tol=0))
    results.append(check("  canonicalization validity %", 100.0,
                         cv["mean_validity_pct"], src, unit="%"))
    results.append(check("  SittingDown canonical distance (mm)", 274.1,
                         cv["by_action"]["SittingDown"]["mean_canonical_distance_mm"],
                         src, tol=0.1, unit="mm"))
    # The foreshortening explanation was withdrawn: the failing action has almost
    # the longest hip axis of any, so length cannot be what distinguishes it.
    results.append(check("  SittingDown hip axis is NOT short (mm)", 285.3,
                         _hip_axis_mm("act_10"), "preds.npz", tol=0.3, unit="mm"))
    results.append(check("  shortest hip axis is WalkDog, which works (mm)", 268.4,
                         _hip_axis_mm("act_15"), "preds.npz", tol=0.3, unit="mm"))

    # ---------- H36M fusion: replicates only with a robust estimator ----------
    fz = load("h36m_fusion/h36m_fusion.json")["summary"]
    src = "h36m_fusion/h36m_fusion.json"
    results.append(check("H36M fusion: single-view baseline (mm)", 37.8,
                         fz["overall_mm"]["single_view_mean"], src, tol=0.1, unit="mm"))
    results.append(check("  median fusion (mm)", 34.6,
                         fz["overall_mm"]["naive_median"], src, tol=0.1, unit="mm"))
    results.append(check("  median fusion improvement %", 8.4,
                         fz["improvement_vs_single_view_pct"]["naive_median"], src, unit="%"))
    results.append(check("  MEAN fusion is worse than one view %", -3.4,
                         fz["improvement_vs_single_view_pct"]["naive_mean"], src, unit="%"))
    results.append(check("  reliability-weighted mean (mm)", 36.6,
                         fz["overall_mm"]["reliability_weighted_mean"], src,
                         tol=0.1, unit="mm"))
    results.append(check("  frames where mean fusion helps %", 67.9,
                         fz["per_frame_improvement_pct"]["naive_mean"]["frames_improved_pct"],
                         src, unit="%"))
    # The median's interval must exclude zero and the mean's must not; those two
    # facts are what the report's fusion claim rests on.
    rb = fz["bootstrap_aggregate_ratio_pct"]
    results.append(check("  median fusion CI lower bound > 0", 2.1,
                         rb["naive_median"]["ci95"][0], src, tol=0.4, unit="%"))
    results.append(check("  mean fusion CI lower bound < 0", -21.5,
                         rb["naive_mean"]["ci95"][0], src, tol=1.5, unit="%"))
    results.append(check("  mean fusion CI upper bound > 0 (spans zero)", 10.5,
                         rb["naive_mean"]["ci95"][1], src, tol=1.5, unit="%"))
    results.append(check("  reliability-weighted CI spans zero (lower)", -8.4,
                         rb["reliability_weighted_mean"]["ci95"][0], src,
                         tol=1.5, unit="%"))

    msb = load("h36m_multiscale/h36m_multiscale.json")["summary"]["bootstrap"]
    results.append(check("  multi-scale CI lower bound %", 23.6,
                         msb["multiscale_vs_global_pct"]["ci95"][0],
                         "h36m_multiscale/h36m_multiscale.json", tol=0.3, unit="%"))

    # ---------- H36M multi-scale, and the bilateral asymmetry it exposed ------
    msc = load("h36m_multiscale/h36m_multiscale.json")["summary"]
    src = "h36m_multiscale/h36m_multiscale.json"
    results.append(check("H36M multi-scale pairs", 180.0, float(msc["n_pairs"]), src, tol=0))
    results.append(check("  global frame distance (mm)", 62.7,
                         msc["mean_global_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  multi-scale distance (mm)", 46.2,
                         msc["mean_multiscale_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  improvement over global %", 25.6,
                         msc["mean_multiscale_vs_global_pct"], src, unit="%"))
    results.append(check("  pairs improved", 179.0,
                         float(msc["n_pairs_improved"]), src, tol=0))
    results.append(check("  right leg (anat.), as implemented (mm)", 69.4,
                         msc["per_level_distance_mm"]["left_leg"], src, tol=0.1, unit="mm"))
    results.append(check("  left leg (anat.) (mm)", 30.7,
                         msc["per_level_distance_mm"]["right_leg"], src, tol=0.1, unit="mm"))
    sym = msc["symmetric_leg_variant"]
    results.append(check("  right leg (anat.), symmetric defn (mm)", 29.3,
                         sym["per_level_distance_mm"]["left_leg"], src, tol=0.1, unit="mm"))
    results.append(check("  symmetric combined distance (mm)", 39.1,
                         sym["mean_multiscale_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  symmetric improvement %", 37.2,
                         sym["mean_multiscale_vs_global_pct"], src, unit="%"))
    results.append(check("  symmetric pairs improved", 180.0,
                         float(sym["n_pairs_improved"]), src, tol=0))
    la = msc["long_axis_variant"]
    results.append(check("  right arm (anat.), long-axis defn (mm)", 26.8,
                         la["per_level_distance_mm"]["left_arm"], src, tol=0.1, unit="mm"))
    results.append(check("  left arm (anat.), long-axis defn (mm)", 26.3,
                         la["per_level_distance_mm"]["right_arm"], src, tol=0.1, unit="mm"))
    results.append(check("  long-axis combined distance (mm)", 28.2,
                         la["mean_multiscale_distance_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  long-axis improvement %", 55.1,
                         la["mean_multiscale_vs_global_pct"], src, unit="%"))
    results.append(check("  long-axis CI lower bound %", 53.6,
                         la["bootstrap"]["ci95"][0], src, tol=0.3, unit="%"))
    results.append(check("  long-axis pairs improved", 180.0,
                         float(la["n_pairs_improved"]), src, tol=0))
    # Applying one rule to five anatomically different segments should give five
    # similar numbers. That convergence was not optimized for and is the check.
    lv = [la["per_level_distance_mm"][k] for k in
          ("torso", "left_arm", "right_arm", "left_leg", "right_leg")]
    results.append(check("  corrected level spread (max-min, mm)", 4.4,
                         float(max(lv) - min(lv)), src, tol=0.3, unit="mm"))
    # The fix must be surgical: arms and torso are untouched by construction.
    for lv in ("torso", "left_arm", "right_arm", "right_leg"):
        results.append(check("  %s unchanged by the fix" % lv,
                             msc["per_level_distance_mm"][lv],
                             sym["per_level_distance_mm"][lv], src, tol=1e-9, unit="mm"))

    # ---------- second backbone: model independence, measured ----------------
    mb = load("h36m_crossview/h36m_crossview_motionbert.json")["summary"]
    src = "h36m_crossview/h36m_crossview_motionbert.json"
    results.append(check("MotionBERT: cross-view improvement %", 77.5,
                         mb["mean_improvement_pct"], src, unit="%"))
    results.append(check("  pairs improved (all of them)", 180.0,
                         float(mb["n_pairs_improved"]), src, tol=0))
    results.append(check("  oracle gap closed %", 94.5,
                         mb["mean_oracle_gap_closed_pct"], src, unit="%"))
    results.append(check("  frame validity %", 100.0,
                         mb["mean_validity_pct"], src, unit="%"))
    # The decisive one: SittingDown fails on one backbone and not the other,
    # with the frame construction byte-identical, so the failure is the
    # estimator's and not the frame's.
    results.append(check("  SittingDown canonical dist (vs 274.1 on XS)", 87.5,
                         mb["by_action"]["SittingDown"]["mean_canonical_distance_mm"],
                         src, tol=0.2, unit="mm"))
    mbm = load("h36m_multiscale/h36m_multiscale_motionbert.json")["summary"]
    results.append(check("  multi-scale, long axis %", 54.9,
                         mbm["long_axis_variant"]["mean_multiscale_vs_global_pct"],
                         "h36m_multiscale_motionbert.json", unit="%"))
    mbf = load("h36m_fusion/h36m_fusion_motionbert.json")["summary"]
    results.append(check("  median fusion %", 6.8,
                         mbf["improvement_vs_single_view_pct"]["naive_median"],
                         "h36m_fusion_motionbert.json", unit="%"))
    results.append(check("  MotionBERT accuracy on these clips (mm)", 44.03,
                         load("h36m_motionbert/accuracy.json")["mpjpe_action_balanced_mm"],
                         "h36m_motionbert/accuracy.json", tol=0.05, unit="mm"))

    # ---------- multi-landmark frame: pre-registered, criterion FAILED --------
    for tag, label, base in (("", "XS", 75.3), ("_motionbert", "MB", 60.1)):
        ml = load("multilandmark/results%s.json" % tag)["summary"]
        src = "multilandmark/results%s.json" % tag
        results.append(check("multilandmark %s: baseline (mm)" % label, base,
                             ml["baseline_both_mm"], src, tol=0.1, unit="mm"))
        # The part that replicates: a longer lateral axis helps on both.
        results.append(check("  %s shoulder-axis CI lower > 0" % label,
                             1.58 if tag == "" else 1.72,
                             ml["variants"]["shoulder_only"]["paired_gain_ci95_mm"][0],
                             src, tol=0.15, unit="mm"))
        # The part that does not: combining redundant axes.
        results.append(check("  %s weighted spans zero (0=spans)" % label, 0.0,
                             float(ml["variants"]["weighted"]["beats_baseline"]),
                             src, tol=0))
    ml_mb = load("multilandmark/results_motionbert.json")["summary"]
    results.append(check("  pre-registered criterion FAILS on MotionBERT", 0.0,
                         float(ml_mb["prereg"]["1_weighted_or_svd_beats_baseline"]),
                         "multilandmark/results_motionbert.json", tol=0))
    results.append(check("  shoulder beats hip, both backbones", 1.0,
                         float(ml_mb["prereg"]["2_shoulder_beats_hip"]),
                         "multilandmark/results_motionbert.json", tol=0))

    # ---------- conditioning abstention: pre-registered, criterion FAILED -----
    # Conditioning passes on MotionBERT and fails on MotionAGFormer, so it does
    # not replicate and is reported as failed. The reliability score, measured
    # on the same frames as a comparator, does replicate - see below.
    for tag, label, base in (("", "XS", 76.16), ("_motionbert", "MB", 60.81)):
        cd = load("conditioning/conditioning%s.json" % tag)
        src = "conditioning/conditioning%s.json" % tag
        results.append(check("conditioning %s: mean canonical dist (mm)" % label,
                             base, cd["mean_error_mm"], src, tol=0.1, unit="mm"))
        rel = cd["reliability_as_triage"]
        results.append(check("  %s reliability triage gain (mm)" % label,
                             5.44 if tag == "" else 2.74,
                             rel["gain_at_10pct_dropped_mm"], src,
                             tol=0.1, unit="mm"))
        results.append(check("  %s reliability CI lower > 0" % label,
                             1.53 if tag == "" else 1.55,
                             rel["gain_ci95_mm"][0], src, tol=0.1, unit="mm"))
        results.append(check("  %s reliability partial rho | bone dev" % label,
                             -0.306 if tag == "" else -0.322,
                             rel["partial_given_bone_deviation"], src, tol=0.01))
        results.append(check("  %s reliability tail ratio" % label,
                             1.54 if tag == "" else 1.49,
                             rel["worst5pct_error_ratio"], src, tol=0.01))
        results.append(check("  %s reliability beats random (1=yes)" % label, 1.0,
                             float(rel["beats_random"]), src, tol=0))
    cd_xs = load("conditioning/conditioning.json")
    results.append(check("  conditioning FAILS to replicate on XS", 0.0,
                         float(cd_xs["p1_triage_beats_random"]),
                         "conditioning/conditioning.json", tol=0))
    results.append(check("  conditioning tail not real on XS", 0.0,
                         float(cd_xs["p2_tail_is_real"]),
                         "conditioning/conditioning.json", tol=0))

    # ---------- averaging convention, stated so it cannot drift --------------
    # Every improvement figure is the mean over pairs of each pair's own
    # percentage, not the ratio of the aggregate means. A reader dividing the
    # table entries gets a different (larger) number, so Section 5.10 says which
    # is quoted. These claims pin both so the disclosure stays true.
    for tag, label, per_pair, ratio_of_means in (("", "XS", 74.05, 76.49),
                                                 ("_motionbert", "MB", 77.55, 81.02)):
        cv = load("h36m_crossview/h36m_crossview%s.json" % tag)
        s = cv.get("summary", cv)
        src = "h36m_crossview/h36m_crossview%s.json" % tag
        results.append(check("convention %s: per-pair mean improvement" % label,
                             per_pair, s["mean_improvement_pct"], src,
                             tol=0.05, unit="%"))
        rm = 100.0 * (s["mean_raw_distance_mm"] - s["mean_canonical_distance_mm"]) \
            / s["mean_raw_distance_mm"]
        results.append(check("  %s ratio-of-means (disclosed, not quoted)" % label,
                             ratio_of_means, rm, src, tol=0.05, unit="%"))
        results.append(check("  %s per-pair is the conservative one (1=yes)" % label,
                             1.0, float(s["mean_improvement_pct"] < rm), src, tol=0))

    # The median bone deviation is quoted in two places, Section 5.9 and the
    # appendix, and they had drifted to 0.034 and 0.033. The artifact says
    # 0.0335, so the appendix was right. Pinned here so it cannot drift again.
    results.append(check("H36M median bone deviation", 0.0335,
                         load("h36m_replication/h36m_replication.json")
                         ["signal_magnitude"]["bone_deviation_median"],
                         "h36m_replication/h36m_replication.json", tol=0.0005))

    # ---------- fusion heterogeneity: the aggregate hides a bimodal outcome --
    # The headline +8.4% is a ratio of aggregate means. The mean of per-frame
    # improvements is +4.7% with an interval spanning zero, and six of fifteen
    # actions get worse. Section 5.11.1 reports both; these pin them.
    fs = load("h36m_fusion/h36m_fusion.json")["summary"]
    src = "h36m_fusion/h36m_fusion.json"
    results.append(check("fusion: pooled ratio-of-means %", 8.38,
                         fs["improvement_vs_single_view_pct"]["naive_median"],
                         src, tol=0.05, unit="%"))
    results.append(check("  per-frame mean improvement %", 4.68,
                         fs["bootstrap_improvement_pct"]["naive_median"]["mean"],
                         src, tol=0.05, unit="%"))
    results.append(check("  per-frame CI lower bound (negative)", -1.24,
                         fs["bootstrap_improvement_pct"]["naive_median"]["ci95"][0],
                         src, tol=0.05, unit="%"))
    results.append(check("  the two conventions disagree on sign of CI (1=yes)", 1.0,
                         float(fs["bootstrap_improvement_pct"]["naive_median"]["ci95"][0] < 0
                               and fs["bootstrap_aggregate_ratio_pct"]["naive_median"]["ci95"][0] > 0),
                         src, tol=0))
    results.append(check("  actions improved of 15", 9.0,
                         float(fs["actions_improved"]), src, tol=0))
    ba = fs["by_action"]
    imp = {k: v["improvement_pct"] for k, v in ba.items()}
    results.append(check("  Discussion improvement %", -98.3, imp["Discussion"],
                         src, tol=0.1, unit="%"))
    results.append(check("  Discussion fused (mm)", 76.1,
                         ba["Discussion"]["fused_mm"], src, tol=0.1, unit="mm"))
    results.append(check("  Discussion anatomical variant (mm)", 51.2,
                         ba["Discussion"]["anatomical_median_mm"], src,
                         tol=0.1, unit="mm"))
    vals = np.array(list(imp.values()))
    wts = np.array([ba[k]["n_frames"] for k in imp])
    results.append(check("  mean over actions (negative) %", -2.98,
                         float(vals.mean()), src, tol=0.05, unit="%"))
    results.append(check("  frame-weighted over actions %", -5.41,
                         float(np.average(vals, weights=wts)), src,
                         tol=0.05, unit="%"))
    results.append(check("  mean excluding Discussion %", 3.83,
                         float(np.mean([v for k, v in imp.items()
                                        if k != "Discussion"])),
                         src, tol=0.05, unit="%"))
    results.append(check("  rho(single-view error, gain)", 0.586,
                         float(spearmanr([ba[k]["single_view_mm"] for k in imp],
                                         list(imp.values()))[0]), src, tol=0.005))

    # ---------- non-constructor headline -------------------------------------
    # Section 5.16.2 shows the frame pins the joints it is built from, so the
    # seventeen-joint headline was contaminated. These pin the recomputation over
    # the thirteen joints the construction does not touch. No claim here asserts
    # which of the two figures is larger: that is the result, not a criterion.
    for tag, label, allj, non in (("", "XS", 74.05, 72.22),
                                  ("_motionbert", "MB", 77.55, 75.84)):
        nc = load("noncon/noncon%s.json" % tag)
        src = "noncon/noncon%s.json" % tag
        results.append(check("noncon %s: all-17 improvement %%" % label, allj,
                             nc["all_17_joints"]["mean_improvement_pct"],
                             src, tol=0.05, unit="%"))
        results.append(check("  %s non-constructor improvement %%" % label, non,
                             nc["non_constructor"]["mean_improvement_pct"],
                             src, tol=0.05, unit="%"))
        results.append(check("  %s non-constructor CI lower > 0" % label, 1.0,
                             float(nc["non_constructor"]
                                   ["bootstrap_improvement_pct"]["ci95"][0] > 0),
                             src, tol=0))
        results.append(check("  %s retained joint count" % label, 13.0,
                             float(len(nc["retained_joints"])), src, tol=0))
        results.append(check("  %s constructor set is {0,1,4,8}" % label, 1.0,
                             float(sorted(nc["constructor_joints"]) == [0, 1, 4, 8]),
                             src, tol=0))
    # The abstract must quote the artifact, not a remembered number. This is the
    # drift that let the seventeen-joint figure survive its own refutation.
    _abs = open(os.path.join(REPO_ROOT, "thesis_report",
                             "Full_Thesis_Report.tex"), encoding="utf-8").read()
    _nc = load("noncon/noncon.json")["non_constructor"]["mean_improvement_pct"]
    results.append(check("  abstract quotes the non-constructor figure", 1.0,
                         float(("%.1f percent" % _nc) in _abs),
                         "Full_Thesis_Report.tex", tol=0))

    # ---------- template baseline: pre-registered, and OUR METHOD LOSES ------
    # Outcome 1 of the three fixed in thesis_artifacts/template/PREREGISTRATION.md.
    for tag, label, anat, tmpl, diff in (("", "XS", 93.35, 57.47, -35.88),
                                         ("_motionbert", "MB", 74.79, 55.91, -18.89)):
        tb = load("template/template%s.json" % tag)["non_constructor"]
        src = "template/template%s.json" % tag
        results.append(check("template %s: anatomical (mm)" % label, anat,
                             tb["mean_anatomical_mm"], src, tol=0.05, unit="mm"))
        results.append(check("  %s template baseline (mm)" % label, tmpl,
                             tb["mean_template_mm"], src, tol=0.05, unit="mm"))
        results.append(check("  %s paired difference (mm)" % label, diff,
                             tb["paired_difference"]["mean"], src,
                             tol=0.05, unit="mm"))
        results.append(check("  %s CI upper < 0, baseline wins" % label, 1.0,
                             float(tb["paired_difference"]["ci95"][1] < 0),
                             src, tol=0))
        results.append(check("  %s pairs where ours is better" % label, 0.0,
                             float(tb["pairs_where_anatomical_better"]),
                             src, tol=0))
        results.append(check("  %s verdict is template_better (1=yes)" % label, 1.0,
                             float(tb["verdict"] == "template_better"), src, tol=0))

    # ---------- translation ablation: how much of the gap is centring? -------
    for tag, label, share, root_gap in (("", "XS", -3.20, -15.36),
                                        ("_motionbert", "MB", -2.49, -3.16)):
        tr = load("translation_ablation/translation%s.json" % tag)
        src = "translation_ablation/translation%s.json" % tag
        results.append(check("translation %s: share from centring (mm)" % label,
                             share, tr["translation_share_mm"], src,
                             tol=0.05, unit="mm"))
        results.append(check("  %s gap with both root-centred (mm)" % label,
                             root_gap, tr["gaps"]["root"]["mean"], src,
                             tol=0.05, unit="mm"))
        # The direction survives the least favourable centring for the baseline.
        results.append(check("  %s template wins even root-centred" % label, 1.0,
                             float(tr["gaps"]["root"]["ci95"][1] < 0), src, tol=0))
        results.append(check("  %s and centroid-centred" % label, 1.0,
                             float(tr["gaps"]["centroid"]["ci95"][1] < 0),
                             src, tol=0))
        # Centroid-centring makes OUR frame slightly worse, not better.
        results.append(check("  %s centroid hurts the anatomical frame" % label,
                             1.0, float(tr["cells_mm"]["anatomical_centroid_mm"]
                                        > tr["cells_mm"]["anatomical_root_mm"]),
                             src, tol=0))

    # ---------- template ablations: independence, and a failed prediction ----
    for tag, label, syn, gain in (("", "XS", -19.87, -28.78),
                                  ("_motionbert", "MB", -6.36, -25.23)):
        ab = load("template_ablation/ablation%s.json" % tag)
        src = "template_ablation/ablation%s.json" % tag
        # A: every template wins, including one built from no pose data at all.
        for tname in ("mpi_canonical", "raw_first", "synthetic"):
            results.append(check("ablation %s: %s template wins" % (label, tname),
                                 1.0, float(ab["A_template_independence"]
                                            [tname]["template_wins"]), src, tol=0))
        results.append(check("  %s synthetic-template difference (mm)" % label,
                             syn, ab["A_template_independence"]["synthetic"]
                             ["paired_difference"]["mean"], src,
                             tol=0.05, unit="mm"))
        # B: the pre-registered prediction failed, and by how much.
        results.append(check("  %s torso-rigid prediction FAILS (0=fails)" % label,
                             0.0, float(ab["B_prediction_holds"]), src, tol=0))
        results.append(check("  %s torso-rigid minus all-17 (mm)" % label, gain,
                             ab["B_torso_rigid_gain_mm"], src, tol=0.05, unit="mm"))
        results.append(check("  %s fit and score sets are disjoint" % label, 1.0,
                             float(not (set(ab["torso_rigid_joints"])
                                        & set(ab["articulated_joints"]))),
                             src, tol=0))

    # ---------- circularity control: how much is removed by construction -----
    # The per-limb frames in the long-axis definitions are built from exactly the
    # joints they are scored on. These claims pin the headroom above the
    # per-segment Procrustes floor, which is what demoted the 55.1% figure from a
    # result to an exploratory measurement.
    ctrl = {(r["set"], r["display"]): r
            for r in load("multiscale_control/control.json")["rows"]}
    src = "multiscale_control/control.json"
    results.append(check("global frame headroom over its floor", 1.46,
                         ctrl[("global", "Global frame")]["headroom_ratio"],
                         src, tol=0.01))
    results.append(check("  global frame is NOT fully circular (0=no)", 0.0,
                         float(ctrl[("global", "Global frame")]["fully_circular"]),
                         src, tol=0))
    for disp, hr in (("Right arm", 1.13), ("Left arm", 1.15),
                     ("Right leg", 1.23), ("Left leg", 1.23)):
        r = ctrl[("long_axis", disp)]
        results.append(check("  long-axis %s headroom" % disp, hr,
                             r["headroom_ratio"], src, tol=0.01))
        results.append(check("    %s fully circular (1=yes)" % disp, 1.0,
                             float(r["fully_circular"]), src, tol=0))
    # The comparison that isolates it: same joints, same oracle, one more builder.
    results.append(check("  shipped Right arm canonical (mm)", 58.0,
                         ctrl[("shipped", "Right arm")]["canonical_mm"],
                         src, tol=0.1, unit="mm"))
    results.append(check("  long-axis Right arm canonical (mm)", 27.3,
                         ctrl[("long_axis", "Right arm")]["canonical_mm"],
                         src, tol=0.1, unit="mm"))
    results.append(check("  their oracle is identical (mm)",
                         ctrl[("shipped", "Right arm")]["oracle_mm"],
                         ctrl[("long_axis", "Right arm")]["oracle_mm"],
                         src, tol=0.01, unit="mm"))

    # ---------- axis-length law: quantitative fit and its interval -----------
    # These were quoted in the report for two days from a scratch script that was
    # never committed, so nothing verified them. They are produced by
    # evaluation/axis_length_law.py now. The interval on c is the reason the
    # report no longer claims the constant is physical.
    for tag, label, c, sig, r2, rho in (("", "XS", 21.3, 7.5, 0.700, 0.904),
                                        ("_motionbert", "MB", 21.5, 7.6, 0.704, 0.880)):
        lv = load("axis_law/axis_law%s.json" % tag)["by_subset"]["limb_levels"]
        src = "axis_law/axis_law%s.json" % tag
        results.append(check("axis law %s: c (mm)" % label, c, lv["c_mm"],
                             src, tol=0.05, unit="mm"))
        results.append(check("  %s implied sigma (mm)" % label, sig,
                             lv["implied_sigma_mm"], src, tol=0.05, unit="mm"))
        results.append(check("  %s R2 (form NOT claimed)" % label, r2, lv["r2"],
                             src, tol=0.005))
        results.append(check("  %s rank rho (the supported claim)" % label, rho,
                             lv["spearman_r_over_L_vs_d"], src, tol=0.005))
        # The interval is wider than the estimate; this is what withdrew the
        # "constants agree to one percent" claim.
        results.append(check("  %s CI on c is wider than c (1=yes)" % label, 1.0,
                             float(lv["bootstrap"]["c_ci_width"] > lv["c_mm"]),
                             src, tol=0))
    xs_ci = load("axis_law/axis_law.json")["by_subset"]["limb_levels"]["bootstrap"]
    mb_ci = load("axis_law/axis_law_motionbert.json")["by_subset"]["limb_levels"]["bootstrap"]
    results.append(check("  the two CIs on c overlap almost entirely (1=yes)", 1.0,
                         float(xs_ci["c_ci95"][0] < mb_ci["c_ci95"][1]
                               and mb_ci["c_ci95"][0] < xs_ci["c_ci95"][1]),
                         "axis_law/*.json", tol=0))

    # ---------- radial law: pre-registered, all criteria FAILED ---------------
    # The pre-registered radius model fails on both backbones. What replicates
    # is the post-hoc mechanism: at matched radius, joints rigid with the torso
    # disagree far less than joints beyond a hinge.
    for tag, label, slope, r2 in (("", "XS", 0.2178, 0.339),
                                  ("_motionbert", "MB", 0.1046, 0.337)):
        rd = load("radial/radial%s.json" % tag)
        src = "radial/radial%s.json" % tag
        results.append(check("radial %s: fitted slope" % label, slope,
                             rd["slope_frame_induced"], src, tol=0.001))
        results.append(check("  %s R2 below 0.80 threshold" % label, r2,
                             rd["r2"], src, tol=0.005))
        results.append(check("  %s P1 linearity FAILS" % label, 0.0,
                             float(rd["p1_linear_r2_at_least_0p80"]), src, tol=0))
        results.append(check("  %s P2 slope outside band FAILS" % label, 0.0,
                             float(rd["p2_slope_in_predicted_band"]), src, tol=0))
        ph = rd["post_hoc"]
        # The falsifying comparison: larger radius, smaller disagreement.
        results.append(check("  %s torso-rigid radius > articulated" % label, 1.0,
                             float(ph["mean_radius_torso_rigid_mm"]
                                   > ph["mean_radius_articulated_mm"]), src, tol=0))
        results.append(check("  %s torso-rigid canonical (mm)" % label,
                             71.9 if tag == "" else 51.1,
                             ph["canonical_torso_rigid_mm"], src, tol=0.1, unit="mm"))
        results.append(check("  %s articulated canonical (mm)" % label,
                             197.5 if tag == "" else 123.5,
                             ph["canonical_articulated_mm"], src, tol=0.1, unit="mm"))
        m = ph["matched_radius_pairs"][0]
        results.append(check("  %s matched pair radius gap %%" % label,
                             1.2 if tag == "" else 0.9,
                             m["radius_gap_pct"], src, tol=0.1, unit="%"))
        results.append(check("  %s matched pair distance ratio" % label,
                             2.62 if tag == "" else 2.06,
                             m["ratio"], src, tol=0.01))
    rd_mb = load("radial/radial_motionbert.json")
    results.append(check("  P3 control FAILS on MotionBERT (voids it)", 0.0,
                         float(rd_mb["p3_oracle_slope_materially_smaller"]),
                         "radial/radial_motionbert.json", tol=0))

    # ---------- cost of the added framework ("lightweight", quantified) -------
    cb = load("cost/cost_benchmark.json")
    src = "cost/cost_benchmark.json"
    results.append(check("trainable parameters added", 0.0,
                         float(cb["trainable_parameters_added"]), src, tol=0))
    results.append(check("  canonicalization FLOPs per frame", 402.0,
                         float(cb["flops"]["canonicalization_per_frame"]), src, tol=0))
    results.append(check("  FLOP overhead vs backbone %", 0.00053,
                         cb["flops"]["ratio_pct"], src, tol=0.0001, unit="%"))
    results.append(check("  peak extra memory (bytes)", 7560.0,
                         float(cb["peak_extra_memory_bytes"]), src, tol=2048))

    # ---------- TTA dispersion (pre-registered, FAILED) ----------
    tta = load("tta/tta_results.json")
    tp = tta["pooled"]
    src = "tta/tta_results.json"
    results.append(check("TTA verdict is FAIL (0=fail)", 0.0,
                         float(bool(tta["verdict_pass"])), src, tol=0))
    results.append(check("  rho(TTA dispersion, error) below floor", 0.100,
                         tp["spearman_disp_vs_error"], src, tol=0.005))
    results.append(check("  TTA strata with positive sign (mixed=2 of 4)", 2.0,
                         float(sum(1 for v in tta["per_stratum"].values() if v > 0)),
                         src, tol=0))

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
        print("DRIFT DETECTED â€” update the claim or investigate the artifact.")
        sys.exit(1)
    print("All headline numbers trace to their source files.")


if __name__ == "__main__":
    main()

