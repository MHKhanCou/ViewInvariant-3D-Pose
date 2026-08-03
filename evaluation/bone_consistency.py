"""
Temporal bone-length inconsistency as a single-view, training-free error predictor.

Idea
----
A person's bones do not change length. Any frame-to-frame variation in predicted
bone lengths is therefore evidence of prediction error — and monocular depth
ambiguity produces exactly that, because a limb angled toward the camera is
foreshortened differently as the depth estimate wanders.

This directly targets the failure mode the 6-component `reliability` score is
blind to. That score measures WITHIN-frame geometric plausibility, which a
coherent depth error leaves intact (see thesis_artifacts/FINAL_REPORT.md §7);
bone-length inconsistency is measured ACROSS time, so it sees what plausibility
cannot.

Signal (scale-free by construction)
-----------------------------------
1. Per frame, 16 bone lengths -> divide by that frame's mean bone length, giving
   bone RATIOS. This removes global scale, so the signal cannot be a restatement
   of "this skeleton is bigger".
2. Estimate the subject's own skeleton as the median ratio vector over a
   REFERENCE window (no labels, no ground truth).
3. Signal = mean relative deviation of the frame's ratios from that median.

Honesty note: this signal was found by exploratory probing, NOT pre-registered.
It is therefore held to the same three criteria that were pre-registered for the
TTA experiment (thesis_artifacts/tta/PREREGISTRATION.md), plus two robustness
checks the pre-registration did not require:
  (a) pooled rho >= +0.30
  (b) same sign across every stratum
  (c) cluster bootstrap over CAMERAS on |rho| - |rho_reliability| excludes 0
  (d) survives partial correlation controlling for detector confidence
  (e) survives a CAUSAL skeleton estimate (reference window disjoint from the
      evaluated frames), so it is deployable online rather than transductive

Run:  ./venv/Scripts/python.exe -m evaluation.bone_consistency
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import compute_bone_lengths
from evaluation.gt_eval import load_gt17, similarity_align_error
from evaluation.run_eval import load_cache

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "bone_consistency")
SEED = 12345
RHO_FLOOR = 0.30
BOOTSTRAP_DRAWS = 10000


def bone_ratios(poses):
    """(T,17,3) -> (T,16) scale-free bone-ratio vectors."""
    B = np.array([compute_bone_lengths(p) for p in poses], dtype=np.float64)
    return B / (B.mean(axis=1, keepdims=True) + 1e-9)


def deviation(ratios, reference):
    """Mean relative deviation of each frame's ratios from a reference skeleton."""
    return np.mean(np.abs(ratios - reference) / (reference + 1e-9), axis=1)


def partial_spearman(x, y, z):
    """rho(x, y | z) on ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.vstack([rz, np.ones_like(rz)]).T
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(spearmanr(res(rx), res(ry))[0])


def parse_key(key):
    p = key.split("_")
    return p[0], p[1], int(p[2].replace("cam", "")), (p[3] if len(p) > 3 else "static")


def main():
    cache = load_cache()
    if not cache:
        print("ERROR: prediction cache missing — run evaluation.run_eval first.")
        return

    gts, per_cam, strata = {}, {}, {}
    for key, f in sorted(cache.items()):
        subj, seq, cam, cond = parse_key(key)
        if (subj, seq) not in gts:
            gts[(subj, seq)] = load_gt17(subj, seq)
        gt = gts[(subj, seq)][cam]

        R = bone_ratios(f["raw"])
        n = len(R)
        half = n // 2
        err = np.array([similarity_align_error(f["raw"][i], gt[c])
                        for i, c in enumerate(f["centers"])])

        med_all = np.median(R, axis=0)                 # transductive
        med_ref = np.median(R[:half], axis=0)          # causal reference window

        per_cam[key] = {
            "condition": cond, "subject": subj, "camera": cam,
            "dev_transductive": deviation(R, med_all),
            "dev_causal_eval_half": deviation(R[half:], med_ref),
            "err": err,
            "err_eval_half": err[half:],
            "reliability": np.asarray(f["reliability"], dtype=float),
            "detector_conf": np.asarray(f["components"][:, 4], dtype=float),
        }
        st = f"{subj}_{cond}"
        strata.setdefault(st, {"dev": [], "err": []})
        strata[st]["dev"].extend(per_cam[key]["dev_transductive"].tolist())
        strata[st]["err"].extend(err.tolist())

    cat = lambda field: np.concatenate([v[field] for v in per_cam.values()])
    dev, err = cat("dev_transductive"), cat("err")
    rel, conf = cat("reliability"), cat("detector_conf")
    dev_c, err_c = cat("dev_causal_eval_half"), cat("err_eval_half")

    rho_dev = float(spearmanr(dev, err)[0])
    rho_rel = float(spearmanr(rel, err)[0])
    rho_causal = float(spearmanr(dev_c, err_c)[0])
    rho_partial = partial_spearman(dev, err, conf)

    # cluster bootstrap over cameras (frames within a camera share windows)
    keys = list(per_cam)
    offs, o = [], 0
    for k in keys:
        m = len(per_cam[k]["err"])
        offs.append(np.arange(o, o + m))
        o += m
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(len(offs), size=len(offs), replace=True)
        sel = np.concatenate([offs[i] for i in pick])
        a = spearmanr(dev[sel], err[sel])[0]
        b = spearmanr(rel[sel], err[sel])[0]
        if not (np.isnan(a) or np.isnan(b)):
            deltas.append(abs(a) - abs(b))
    lo, hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

    strat_rho = {s: float(spearmanr(v["dev"], v["err"])[0])
                 for s, v in strata.items() if len(v["err"]) > 10}
    signs = {np.sign(r) for r in strat_rho.values()}

    crit = {
        "a_rho_floor": bool(rho_dev >= RHO_FLOOR),
        "b_sign_consistent": bool(len(signs) == 1),
        "c_ci_excludes_zero": bool(lo > 0 or hi < 0),
        "d_survives_confidence_control": bool(abs(rho_partial) >= 0.30),
        "e_survives_causal_estimate": bool(rho_causal >= RHO_FLOOR),
    }
    passed = all(crit.values())

    print("=" * 74)
    print("TEMPORAL BONE-LENGTH INCONSISTENCY vs GT ERROR")
    print("=" * 74)
    print(f"n = {len(err)} frames over {len(per_cam)} camera-streams\n")
    print(f"  rho(bone deviation, error)              {rho_dev:+.3f}")
    print(f"  rho(reliability, error)   [incumbent]   {rho_rel:+.3f}")
    print(f"  partial | detector confidence           {rho_partial:+.3f}")
    print(f"  causal estimate (disjoint reference)    {rho_causal:+.3f}  "
          f"(n={len(err_c)})")
    print(f"  cluster bootstrap 95% CI on |d|-|rel|   [{lo:+.3f}, {hi:+.3f}]")
    print("\n  per stratum:")
    for s, r in sorted(strat_rho.items()):
        print(f"    {s:14s} {r:+.3f}")
    print("\n  criteria:")
    for k, v in crit.items():
        print(f"    {k:32s} {v}")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")

    out = {
        "note": "Signal found by exploratory probing, NOT pre-registered; held to "
                "the TTA pre-registration criteria plus two robustness checks.",
        "verdict_pass": passed,
        "criteria": crit,
        "n_frames": int(len(err)),
        "n_camera_streams": len(per_cam),
        "pooled": {
            "spearman_bone_deviation_vs_error": rho_dev,
            "spearman_reliability_vs_error": rho_rel,
            "partial_given_detector_confidence": rho_partial,
            "causal_estimate_rho": rho_causal,
            "causal_n": int(len(err_c)),
            "bootstrap_ci_delta_abs_rho": [lo, hi],
        },
        "per_stratum": strat_rho,
        "per_camera": {k: float(spearmanr(v["dev_transductive"], v["err"])[0])
                       for k, v in per_cam.items()},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "bone_consistency.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
