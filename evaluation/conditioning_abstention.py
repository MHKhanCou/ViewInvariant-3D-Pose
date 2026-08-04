"""
Can geometric conditioning tell us when to abstain from canonicalizing?

Pre-registered in thesis_artifacts/conditioning/PREREGISTRATION.md, committed
before this ran. That document records why the obvious version of this
experiment - does conditioning predict canonicalization error continuously - was
drafted and rejected: the predictor varies by only 1.29x per frame against 3.34x
between frame constructions, and it is confounded with what it predicts, since a
distorted pose both raises the conditioning number and canonicalizes worse.

So the question here is the narrower one the theory supports. As L goes to zero,
or as the two axes approach collinearity, the Gram-Schmidt construction is
ill-posed and the frame is meaningless rather than merely noisy. Triage of that
tail is a weaker claim than regression, and it is what an abstention rule needs.

Primary metric is a coverage-error curve against a random-ordering null. The
confound control is a partial correlation given bone-ratio deviation, which
measures how distorted the predicted skeleton is independently of its geometry;
without that control a positive result cannot be interpreted.

Run:  ./venv/Scripts/python.exe -m evaluation.conditioning_abstention
      ./venv/Scripts/python.exe -m evaluation.conditioning_abstention --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.bone_consistency import bone_ratios, deviation, partial_spearman
from evaluation.h36m_crossview import ACTION_NAMES, EVAL_STRIDE, canonicalize_stream
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, h36m_conf_to_coco, parse_video
from evaluation.metrics import cross_view_joint_distance
from evaluation.reliability import compute_reliability_score

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "conditioning")
SEED = 12345
N_SHUFFLES = 100
BOOTSTRAP_DRAWS = 10000
_EPS = 1e-8


def conditioning(pose):
    """
    Conditioning of the body frame, from one pose, with no labels.

    kappa is the lever arm over the baseline: a long reach from the root driven
    by a short lateral axis amplifies angular error into joint displacement.
    ortho is how far the two axes are from collinear, which is what the
    Gram-Schmidt cross product needs to be well posed.
    """
    P = np.asarray(pose, dtype=np.float64)
    P = P - P[0:1]
    y = P[8] - P[0]
    x = P[1] - P[4]
    ly, lx = np.linalg.norm(y), np.linalg.norm(x)
    r_bar = float(np.mean(np.linalg.norm(P, axis=1)))
    if ly < _EPS or lx < _EPS:
        return {"kappa": np.inf, "ortho": 0.0, "cond": np.inf,
                "L_hip": float(lx), "r_bar": r_bar}
    cos = float(np.dot(y, x) / (ly * lx))
    ortho = float(np.sqrt(max(0.0, 1.0 - cos ** 2)))
    kappa = r_bar / lx
    return {"kappa": float(kappa), "ortho": ortho,
            "cond": float(kappa / max(ortho, 1e-3)),
            "L_hip": float(lx), "r_bar": r_bar}


def collect(videos, stride=EVAL_STRIDE):
    """Per camera-pair frame: conditioning, incumbent score, and realised error."""
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v

    rows = []
    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(cams[c]["pred"]) for c in order)
        sel = np.arange(0, n, stride)

        per_cam = {}
        for c in order:
            P = cams[c]["pred"][sel]
            P = P - P[:, 0:1, :]
            can, ok = canonicalize_stream(P)
            cond = [conditioning(p) for p in P]
            rel = [compute_reliability_score(
                       P[i], h36m_conf_to_coco(cams[c]["conf"][sel][i]), None)[0]
                   for i in range(len(P))]
            R = bone_ratios(P)
            bdev = deviation(R, np.median(R, axis=0))
            per_cam[c] = {"can": can, "ok": ok, "cond": cond,
                          "rel": np.asarray(rel), "bdev": bdev}

        for a, b in itertools.combinations(order, 2):
            A, B = per_cam[a], per_cam[b]
            keep = A["ok"] & B["ok"]
            for i in np.flatnonzero(keep):
                ca, cb = A["cond"][i], B["cond"][i]
                if not np.isfinite(ca["cond"]) or not np.isfinite(cb["cond"]):
                    continue
                rows.append({
                    "subject": subj, "action": action, "pair": "%s-%s" % (a, b),
                    # The pair is only as well conditioned as its worse view, so
                    # the max is the quantity an abstention rule would threshold.
                    "cond": max(ca["cond"], cb["cond"]),
                    "kappa": max(ca["kappa"], cb["kappa"]),
                    "ortho": min(ca["ortho"], cb["ortho"]),
                    "reliability": float(min(A["rel"][i], B["rel"][i])),
                    "bone_dev": float(max(A["bdev"][i], B["bdev"][i])),
                    "d": float(cross_view_joint_distance(A["can"][i], B["can"][i])),
                })
    return rows


def coverage_curve(score, err, higher_is_worse, fractions):
    """Mean error after discarding the worst `frac` of frames by `score`."""
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s if higher_is_worse else s)   # worst first
    out = []
    for f in fractions:
        n_drop = int(round(f * len(s)))
        keep = order[n_drop:]
        out.append(float(err[keep].mean()) if len(keep) else float("nan"))
    return out


def random_curve(err, fractions, rng, n=N_SHUFFLES):
    """Null: discard the same number of frames at random."""
    curves = []
    for _ in range(n):
        perm = rng.permutation(len(err))
        curves.append([float(err[perm[int(round(f * len(err))):]].mean())
                       for f in fractions])
    return np.array(curves).mean(axis=0).tolist()


def analyse(rows, draws=BOOTSTRAP_DRAWS):
    err = np.array([r["d"] for r in rows])
    cond = np.array([r["cond"] for r in rows])
    rel = np.array([r["reliability"] for r in rows])
    bdev = np.array([r["bone_dev"] for r in rows])
    fracs = [0.0, 0.05, 0.10, 0.20, 0.30]
    rng = np.random.default_rng(SEED)

    curves = {
        "fractions_dropped": fracs,
        "by_conditioning": coverage_curve(cond, err, True, fracs),
        "by_reliability": coverage_curve(rel, err, False, fracs),
        "random": random_curve(err, fracs, rng),
    }

    # Prediction 1, at 10 percent dropped, bootstrapped over subject-action groups.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["subject"], r["action"]), []).append(i)
    keys = list(groups)
    gains = np.empty(draws)
    for t in range(draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[j]] for j in pick])
        e, c = err[idx], cond[idx]
        n_drop = int(round(0.10 * len(idx)))
        by_c = e[np.argsort(-c)[n_drop:]].mean()
        by_r = e[rng.permutation(len(idx))[n_drop:]].mean()
        gains[t] = by_r - by_c
    ci = [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]

    # Prediction 2: is there a genuinely bad tail?
    thr = np.percentile(cond, 95)
    tail = err[cond >= thr]
    tail_ratio = float(tail.mean() / err.mean())

    # Prediction 3, the confound control.
    rho = float(spearmanr(cond, err)[0])
    rho_partial = partial_spearman(cond, err, bdev)

    # The incumbent score is measured on the same footing. The report falsifies
    # it against GROUND-TRUTH POSE ERROR; this is a different target, and one its
    # own docstring names, so it gets the same triage test and the same confound
    # control rather than being assumed to fail here too.
    rel_gains = np.empty(draws)
    rng2 = np.random.default_rng(SEED + 1)
    for t in range(draws):
        pick = rng2.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[j]] for j in pick])
        e, rr = err[idx], rel[idx]
        n_drop = int(round(0.10 * len(idx)))
        by_r = e[np.argsort(rr)[n_drop:]].mean()          # low reliability = worst
        by_rand = e[rng2.permutation(len(idx))[n_drop:]].mean()
        rel_gains[t] = by_rand - by_r
    rel_ci = [float(np.percentile(rel_gains, 2.5)),
              float(np.percentile(rel_gains, 97.5))]
    rel_thr = np.percentile(rel, 5)
    rel_tail_ratio = float(err[rel <= rel_thr].mean() / err.mean())

    return {
        "n_frames": len(rows),
        "mean_error_mm": float(err.mean()),
        "coverage_curves": curves,
        "gain_at_10pct_dropped_mm": float(curves["random"][2] - curves["by_conditioning"][2]),
        "gain_ci95_mm": ci,
        "p1_triage_beats_random": bool(ci[0] > 0),
        "worst5pct_error_ratio": tail_ratio,
        "p2_tail_is_real": bool(tail_ratio >= 1.25),
        "spearman_cond_vs_error": rho,
        "partial_given_bone_deviation": rho_partial,
        "p3_survives_confound_control": bool(abs(rho_partial) >= 0.5 * abs(rho)
                                             and np.sign(rho_partial) == np.sign(rho)),
        "reliability_as_triage": {
            "spearman_vs_error": float(spearmanr(rel, err)[0]),
            "partial_given_bone_deviation": partial_spearman(rel, err, bdev),
            "gain_at_10pct_dropped_mm": float(curves["random"][2]
                                              - curves["by_reliability"][2]),
            "gain_ci95_mm": rel_ci,
            "beats_random": bool(rel_ci[0] > 0),
            "worst5pct_error_ratio": rel_tail_ratio,
            "note": "The report falsifies this score against ground-truth POSE "
                    "error. This is a different target - canonicalization "
                    "quality - which reliability.py's own docstring names as its "
                    "purpose. Same triage test and same confound control.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=EVAL_STRIDE)
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    print("Collecting per-frame conditioning...")
    rows = collect(videos, stride=args.stride)
    res = analyse(rows)
    res["prediction_cache"] = os.path.basename(path)

    c = res["coverage_curves"]
    print("=" * 78)
    print("CONDITIONING AS AN ABSTENTION CRITERION")
    print("=" * 78)
    print("  %d frame-pairs, mean canonical distance %.1f mm\n"
          % (res["n_frames"], res["mean_error_mm"]))
    print("  %-14s %9s %9s %9s" % ("dropped", "by cond", "by rel", "random"))
    for i, f in enumerate(c["fractions_dropped"]):
        print("  %-14s %9.1f %9.1f %9.1f"
              % ("%.0f%%" % (100 * f), c["by_conditioning"][i],
                 c["by_reliability"][i], c["random"][i]))
    print("\n  gain over random at 10%% dropped  %+.2f mm  95%% CI [%+.2f, %+.2f]"
          % (res["gain_at_10pct_dropped_mm"], *res["gain_ci95_mm"]))
    print("  worst 5%% by conditioning are %.2fx the pooled mean"
          % res["worst5pct_error_ratio"])
    print("  rho(cond, error) %+.3f   partial | bone deviation %+.3f"
          % (res["spearman_cond_vs_error"], res["partial_given_bone_deviation"]))
    rt = res["reliability_as_triage"]
    print("\n  the incumbent score, on the SAME triage test:")
    print("    rho(reliability, error)         %+.3f" % rt["spearman_vs_error"])
    print("    partial | bone deviation        %+.3f" % rt["partial_given_bone_deviation"])
    print("    gain at 10%% dropped            %+.2f mm  95%% CI [%+.2f, %+.2f]"
          % (rt["gain_at_10pct_dropped_mm"], *rt["gain_ci95_mm"]))
    print("    worst 5%% are                    %.2fx the pooled mean"
          % rt["worst5pct_error_ratio"])
    print("    beats random                    %s" % rt["beats_random"])
    print("\n  pre-registered predictions:")
    print("    1 triage beats random           %s" % res["p1_triage_beats_random"])
    print("    2 tail is real (>=1.25x)        %s" % res["p2_tail_is_real"])
    print("    3 survives confound control     %s" % res["p3_survives_confound_control"])

    os.makedirs(OUT_DIR, exist_ok=True)
    name = "conditioning%s.json" % ("_" + args.tag if args.tag else "")
    with open(os.path.join(OUT_DIR, name), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved: %s" % os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
