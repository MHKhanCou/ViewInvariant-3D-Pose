"""
Multi-view selection/fusion experiment (training-free, calibration-free).

Question: canonicalization makes N uncalibrated views comparable — can the
training-free reliability score then SELECT or FUSE them into an estimate
better than an arbitrary single view, and how close does it get to an oracle
that needs ground truth?

Protocol
- Reference: ground truth canonicalized. Validity check first — GT
  canonicalized from different cameras must agree, otherwise the reference is
  not view-independent and the comparison is meaningless.
- Error: similarity-aligned (Umeyama) per-joint distance in mm. Never MPJPE.
- Baselines: mean single view (what you get picking arbitrarily = the
  deployment baseline) and oracle-best view (needs GT = upper bound).
- View-count curve: for k = 2..8, random camera subsets per frame.
- Held-out subject S2 evaluated separately (2 cameras only).
- Significance: Wilcoxon signed-rank, paired per frame, vs mean single view.

Run:  ./venv/Scripts/python.exe -m evaluation.fusion_eval
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation.fusion import STRATEGIES, resolve_reflections
from evaluation.gt_eval import load_gt17, similarity_align_error
from evaluation.run_eval import load_cache

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "fusion")
SEED = 12345
N_SUBSETS = 20  # random camera subsets per frame per view-count


def gt_canonical(gt_cam_frames, cam, frame):
    """Canonicalized, root-centred GT for one camera/frame (None if degenerate)."""
    g = gt_cam_frames[cam][frame]
    g = g - g[0:1]
    c, _, meta = canonicalize_single(g.astype(np.float32))
    return c.astype(np.float64) if meta["valid"] else None


def check_reference_validity(gt, cams, frames):
    """
    GT canonicalized from different cameras must agree — otherwise the
    reference frame is view-dependent and every number below is unfounded.
    """
    diffs = []
    for f in frames[:20]:
        cs = [gt_canonical(gt, c, f) for c in cams]
        cs = [c for c in cs if c is not None]
        if len(cs) < 2:
            continue
        aligned, _ = resolve_reflections(np.array(cs))
        ref = aligned[0]
        for other in aligned[1:]:
            diffs.append(similarity_align_error(other, ref))
    return float(np.mean(diffs)) if diffs else None


def collect(subject, sequence, cache, gt, condition="static"):
    """Per-frame stacked canonical poses, reliabilities, and per-view errors."""
    prefix = f"{subject}_{sequence}_cam"
    suffix = "" if condition == "static" else f"_{condition}"
    keys = [k for k in cache
            if k.startswith(prefix) and k[len(prefix):].lstrip("0123456789") == suffix]
    cams = sorted(int(k[len(prefix):].split("_")[0]) for k in keys)
    per_cam = {c: cache[f"{prefix}{c}{suffix}"] for c in cams}
    frames = sorted(set.intersection(
        *[set(per_cam[c]["centers"].tolist()) for c in cams]))

    rows = []
    for f in frames:
        ref = gt_canonical(gt, cams[0], f)
        if ref is None:
            continue
        poses, rels, errs, used = [], [], [], []
        for c in cams:
            i = int(np.where(per_cam[c]["centers"] == f)[0][0])
            if not per_cam[c]["valid"][i]:
                continue
            p = per_cam[c]["canonical"][i].astype(np.float64)
            poses.append(p)
            rels.append(float(per_cam[c]["reliability"][i]))
            errs.append(similarity_align_error(p, ref))
            used.append(c)
        if len(poses) >= 2:
            rows.append({"frame": f, "ref": ref, "poses": np.array(poses),
                         "rels": np.array(rels), "errs": np.array(errs),
                         "cams": used})
    return cams, frames, rows


def evaluate(rows, strategies=STRATEGIES):
    """Per-frame error for every strategy plus both baselines."""
    out = {k: [] for k in list(strategies) + ["mean_single", "oracle_best", "worst_single"]}
    ranks = []
    for r in rows:
        out["mean_single"].append(float(r["errs"].mean()))
        out["oracle_best"].append(float(r["errs"].min()))
        out["worst_single"].append(float(r["errs"].max()))
        for name, fn in strategies.items():
            out[name].append(float(similarity_align_error(
                fn(r["poses"], r["rels"]), r["ref"])))
        picked = int(np.argmax(r["rels"]))
        ranks.append(int(np.where(np.argsort(r["errs"]) == picked)[0][0]) + 1)
    return out, ranks


def summarize(out, baseline="mean_single"):
    base = np.array(out[baseline])
    summary = {}
    for name, vals in out.items():
        v = np.array(vals)
        entry = {
            "mean_mm": float(v.mean()),
            "median_mm": float(np.median(v)),
            "std_mm": float(v.std()),
            "improvement_vs_mean_single_pct": float((1 - v.mean() / base.mean()) * 100),
        }
        if name != baseline and len(v) > 10:
            try:
                stat, p = wilcoxon(v, base)
                entry["wilcoxon_p_vs_mean_single"] = float(p)
            except ValueError:
                entry["wilcoxon_p_vs_mean_single"] = None
        summary[name] = entry
    return summary


def view_count_curve(rows, rng):
    """Error vs number of available cameras, random subsets per frame."""
    curve = {}
    max_v = max(len(r["cams"]) for r in rows)
    for k in range(2, max_v + 1):
        acc = {name: [] for name in list(STRATEGIES) + ["mean_single", "oracle_best"]}
        for r in rows:
            n = len(r["cams"])
            if n < k:
                continue
            for _ in range(N_SUBSETS):
                sel = rng.choice(n, size=k, replace=False)
                poses, rels, errs = r["poses"][sel], r["rels"][sel], r["errs"][sel]
                acc["mean_single"].append(float(errs.mean()))
                acc["oracle_best"].append(float(errs.min()))
                for name, fn in STRATEGIES.items():
                    acc[name].append(float(similarity_align_error(
                        fn(poses, rels), r["ref"])))
        if acc["mean_single"]:
            curve[str(k)] = {name: float(np.mean(v)) for name, v in acc.items()}
    return curve


def selection_adaptivity(rows, cams, rng):
    """
    Does the score SELECT per frame, or just prefer one good camera?

    On clean synchronized studio footage the answer is the latter (see
    'fixed_camera' below), so the meaningful test is whether the choice MOVES
    when the preferred view degrades. Each frame's currently-picked view has a
    geometric corruption applied to its 3D pose (torso-axis collapse, the
    operator the reliability score is designed to catch); we then ask whether
    selection switches away from it and whether the switch avoids error.
    """
    from evaluation.reliability import compute_reliability_score

    switched, improved, deltas = 0, 0, []
    n = 0
    for r in rows:
        picked = int(np.argmax(r["rels"]))
        corrupted = r["poses"].copy()
        p = corrupted[picked].copy()
        # Collapse the torso axis: scale joints toward the hip plane.
        p[:, 1] *= 0.25
        corrupted[picked] = p

        new_rels = r["rels"].copy()
        new_rels[picked] = compute_reliability_score(p, None)[0]
        new_pick = int(np.argmax(new_rels))

        n += 1
        if new_pick != picked:
            switched += 1
            err_corrupted_pick = similarity_align_error(corrupted[picked], r["ref"])
            err_new_pick = similarity_align_error(corrupted[new_pick], r["ref"])
            deltas.append(float(err_corrupted_pick - err_new_pick))
            if err_new_pick < err_corrupted_pick:
                improved += 1

    return {
        "n_frames": n,
        "switched_away_from_degraded_view_pct": float(100 * switched / max(n, 1)),
        "switch_reduced_error_pct": float(100 * improved / max(switched, 1)),
        "mean_error_avoided_mm": float(np.mean(deltas)) if deltas else None,
    }


def fixed_camera_baseline(rows, cams):
    """Best single camera held fixed for the whole sequence (needs GT to choose)."""
    errs = {c: [] for c in cams}
    for r in rows:
        for c, e in zip(r["cams"], r["errs"]):
            errs[c].append(e)
    means = {c: float(np.mean(v)) for c, v in errs.items() if v}
    best_cam = min(means, key=means.get)
    return {"best_fixed_camera": int(best_cam),
            "best_fixed_camera_mm": means[best_cam],
            "per_camera_mm": means}


def main():
    cache = load_cache()
    if not cache:
        print("ERROR: prediction cache missing — run evaluation.run_eval first.")
        return
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "n_subsets_per_frame": N_SUBSETS}

    # ---------- S1: 8 cameras ----------
    gt_s1 = load_gt17("S1", "Seq1")
    cams, frames, rows = collect("S1", "Seq1", cache, gt_s1)
    ref_err = check_reference_validity(gt_s1, cams, frames)
    results["reference_validity_mm"] = ref_err
    print(f"Reference validity: GT canonicalized from different cameras agrees "
          f"to {ref_err:.2f} mm  (must be small)")
    print(f"S1/Seq1: {len(cams)} cameras, {len(rows)} frames with >=2 valid views\n")

    out, ranks = evaluate(rows)
    summary = summarize(out)
    results["S1"] = {"n_frames": len(rows), "n_cameras": len(cams),
                     "strategies": summary}
    r = np.array(ranks)
    results["S1"]["selection"] = {
        "mean_true_error_rank_of_picked_view": float(r.mean()),
        "random_expectation": float((len(cams) + 1) / 2),
        "picked_best_view_pct": float(100 * (r == 1).mean()),
        "picked_top2_pct": float(100 * (r <= 2).mean()),
    }

    order = sorted(summary, key=lambda k: summary[k]["mean_mm"], reverse=True)
    print(f"{'strategy':18s} {'mean':>8s} {'median':>8s} {'vs single':>10s} {'p':>10s}")
    for name in order:
        s = summary[name]
        p = s.get("wilcoxon_p_vs_mean_single")
        p_s = "—" if p is None else f"{p:.2e}"
        print(f"{name:18s} {s['mean_mm']:7.1f}mm {s['median_mm']:7.1f}mm "
              f"{s['improvement_vs_mean_single_pct']:+9.1f}% {p_s:>10s}")

    sel = results["S1"]["selection"]
    print(f"\nSelection quality: reliability-picked view ranks "
          f"{sel['mean_true_error_rank_of_picked_view']:.2f} of {len(cams)} by true error "
          f"(random {sel['random_expectation']:.1f}); "
          f"best {sel['picked_best_view_pct']:.0f}%, top-2 {sel['picked_top2_pct']:.0f}%")

    # --- Honesty checks: is selection adaptive, and is a fixed camera better? ---
    picked_cams = [r["cams"][int(np.argmax(r["rels"]))] for r in rows]
    distinct = sorted(set(picked_cams))
    fixed = fixed_camera_baseline(rows, cams)
    results["S1"]["selection"]["distinct_cameras_picked"] = len(distinct)
    results["S1"]["selection"]["picked_cameras"] = [int(c) for c in distinct]
    results["S1"]["fixed_camera_baseline"] = fixed
    print(f"  distinct cameras ever picked: {len(distinct)} of {len(cams)} {distinct}")
    print(f"  best FIXED camera (needs GT to choose): cam{fixed['best_fixed_camera']} "
          f"at {fixed['best_fixed_camera_mm']:.1f}mm")
    if len(distinct) == 1:
        print("  -> selection is NOT per-frame adaptive on clean footage; it "
              "identifies one consistently good viewpoint.")

    adapt = selection_adaptivity(rows, cams, rng)
    results["S1"]["selection_adaptivity_under_degradation"] = adapt
    print(f"\nAdaptivity under degradation (corrupt the picked view):")
    print(f"  switched away from degraded view: "
          f"{adapt['switched_away_from_degraded_view_pct']:.0f}% of frames")
    if adapt["mean_error_avoided_mm"] is not None:
        print(f"  switch reduced error in {adapt['switch_reduced_error_pct']:.0f}% of "
              f"switches, avoiding {adapt['mean_error_avoided_mm']:.1f}mm on average")

    # ---------- view-count curve ----------
    curve = view_count_curve(rows, rng)
    results["S1"]["view_count_curve"] = curve
    print(f"\n{'views':>5s} " + " ".join(f"{n:>16s}" for n in
                                          ["mean_single", "reliability_pick",
                                           "weighted_mean", "oracle_best"]))
    for k, vals in curve.items():
        print(f"{k:>5s} " + " ".join(
            f"{vals[n]:15.1f}mm" for n in
            ["mean_single", "reliability_pick", "weighted_mean", "oracle_best"]))

    # ---------- dynamic windows: does selection become per-frame adaptive? ----------
    for label, subj, seq in [("S1_dynamic", "S1", "Seq1"),
                             ("S2_dynamic_heldout", "S2", "Seq2")]:
        gt_d = load_gt17(subj, seq)
        cams_d, frames_d, rows_d = collect(subj, seq, cache, gt_d, "dynamic")
        if not rows_d:
            continue
        out_d, ranks_d = evaluate(rows_d)
        summary_d = summarize(out_d)
        picked_d = [r["cams"][int(np.argmax(r["rels"]))] for r in rows_d]
        distinct_d = sorted(set(picked_d))
        fixed_d = fixed_camera_baseline(rows_d, cams_d)
        rd = np.array(ranks_d)

        # Switch rate: how often does the chosen view change between frames?
        switches = sum(1 for a, b in zip(picked_d[:-1], picked_d[1:]) if a != b)

        results[label] = {
            "n_frames": len(rows_d), "n_cameras": len(cams_d),
            "strategies": summary_d,
            "selection": {
                "mean_true_error_rank_of_picked_view": float(rd.mean()),
                "random_expectation": float((len(cams_d) + 1) / 2),
                "picked_best_view_pct": float(100 * (rd == 1).mean()),
                "distinct_cameras_picked": len(distinct_d),
                "picked_cameras": [int(c) for c in distinct_d],
                "switch_rate_pct": float(100 * switches / max(len(picked_d) - 1, 1)),
            },
            "fixed_camera_baseline": fixed_d,
            "adaptive_beats_best_fixed": bool(
                summary_d["reliability_pick"]["mean_mm"] < fixed_d["best_fixed_camera_mm"]),
        }
        s = results[label]
        print(f"\n=== {label}: {s['n_cameras']} cams, {s['n_frames']} frames ===")
        for name in ["mean_single", "weighted_mean", "median",
                     "reliability_pick", "oracle_best"]:
            e = summary_d[name]
            print(f"  {name:18s} {e['mean_mm']:7.1f}mm "
                  f"({e['improvement_vs_mean_single_pct']:+5.1f}%)")
        print(f"  best FIXED camera (needs GT): cam{fixed_d['best_fixed_camera']} "
              f"at {fixed_d['best_fixed_camera_mm']:.1f}mm")
        print(f"  distinct cameras picked: {s['selection']['distinct_cameras_picked']}"
              f" of {s['n_cameras']} {s['selection']['picked_cameras']}"
              f" | switch rate {s['selection']['switch_rate_pct']:.0f}%")
        print(f"  picked-view true-error rank "
              f"{s['selection']['mean_true_error_rank_of_picked_view']:.2f} "
              f"(random {s['selection']['random_expectation']:.1f})")
        print(f"  ADAPTIVE BEATS BEST-FIXED (GT-chosen): {s['adaptive_beats_best_fixed']}")

    # ---------- held-out subject ----------
    gt_s2 = load_gt17("S2", "Seq1")
    cams2, _, rows2 = collect("S2", "Seq1", cache, gt_s2)
    if rows2:
        out2, ranks2 = evaluate(rows2)
        summary2 = summarize(out2)
        results["S2_heldout"] = {"n_frames": len(rows2), "n_cameras": len(cams2),
                                 "strategies": summary2}
        print(f"\nHeld-out subject S2 ({len(cams2)} cameras, {len(rows2)} frames):")
        for name in ["mean_single", "reliability_pick", "weighted_mean", "oracle_best"]:
            s = summary2[name]
            print(f"  {name:18s} {s['mean_mm']:7.1f}mm "
                  f"({s['improvement_vs_mean_single_pct']:+.1f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "fusion_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
