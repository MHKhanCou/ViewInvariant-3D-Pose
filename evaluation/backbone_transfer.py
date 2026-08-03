"""
Backbone transfer: the SAME zero-parameter module applied to predictions from
architecturally unrelated frozen predictors, with no adaptation of any kind.

Why this experiment exists
--------------------------
"Training-free" is a weak phrase: the backbone itself was of course trained.
The precise, checkable claim is that OUR module adds **zero learned
parameters** and contains **no model-specific constants fitted to any
predictor**. The way to demonstrate that is transfer: run the identical code
over outputs from different architectures and show it behaves consistently
without a single changed line or refitted threshold.

Contrast with the closest prior art: 3DPCNet is also estimator-agnostic, but
it *trains* a canonicalization network (GCN + transformer predicting a 6D
rotation). A trained module must at minimum be validated — usually retrained —
per output distribution. A zero-parameter module cannot be tuned to one, which
is precisely what this experiment tests.

Sources (all pre-existing on disk, no new inference):
  ground_truth.npy            MPI-INF-3DHP S1/Seq1 cam1 GT, 243 frames (mm)
  motionbert_predictions.npy  MotionBERT / DSTformer  (~42M params)
  mlp_predictions.npy         plain PoseMLP17 baseline (~1M params)
  + MotionAGFormer-XS from the cached cross-view predictions (2.2M params)

Note: the MotionBERT and MLP arrays were scale-normalized against GT by
compare_models.py, so absolute error magnitudes are not comparable across
backbones. Every statistic below is rank-based or within-backbone for that
reason.

Run:  ./venv/Scripts/python.exe -m evaluation.backbone_transfer
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation.gt_eval import similarity_align_error
from evaluation.reliability import (
    compute_reliability_score, has_hard_geometric_failure,
)
from evaluation.run_eval import load_cache

DATA = "E:/thesis"
OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "backbone_transfer")
ABSTAIN_T = 0.5  # frozen threshold — NOT refitted per backbone


def module_parameter_count():
    """Learned parameters introduced by this framework. Structurally zero."""
    return 0


def profile(name, preds, gts, backbone_params):
    """Run the unmodified framework over one backbone's predictions."""
    rels, errs, valid, hard = [], [], [], []
    for p, g in zip(preds, gts):
        p = np.asarray(p, dtype=np.float64)
        p = p - p[0:1]
        g = np.asarray(g, dtype=np.float64)
        g = g - g[0:1]

        _, _, meta = canonicalize_single(p.astype(np.float32))
        rel, _, _ = compute_reliability_score(p, None)
        hf, _ = has_hard_geometric_failure(p)

        rels.append(float(rel))
        errs.append(similarity_align_error(p, g))
        valid.append(bool(meta["valid"]))
        hard.append(bool(hf))

    rels, errs = np.array(rels), np.array(errs)
    valid, hard = np.array(valid), np.array(hard)
    rho, pval = spearmanr(rels, errs)

    abstain = (rels < ABSTAIN_T) | hard
    kept = ~abstain
    return {
        "backbone": name,
        "backbone_params_M": backbone_params,
        "module_params_added": module_parameter_count(),
        "n_frames": int(len(rels)),
        "canonicalization_valid_pct": float(100 * valid.mean()),
        "reliability_mean": float(rels.mean()),
        "reliability_std": float(rels.std()),
        "gt_error_mean_mm": float(errs.mean()),
        "spearman_reliability_vs_error": float(rho),
        "p_value": float(pval),
        "hard_gate_rate_pct": float(100 * hard.mean()),
        "abstention_rate_pct": float(100 * abstain.mean()),
        "error_on_kept_mm": float(errs[kept].mean()) if kept.any() else None,
    }


def main():
    gt = np.load(os.path.join(DATA, "ground_truth.npy"))
    sources = [
        ("MotionBERT (DSTformer)", np.load(os.path.join(DATA, "motionbert_predictions.npy")), 42.3),
        ("PoseMLP17 (plain MLP)", np.load(os.path.join(DATA, "mlp_predictions.npy")), 1.0),
    ]

    rows = [profile(n, p, gt, mp) for n, p, mp in sources]

    # MotionAGFormer-XS from the cached cross-view predictions (different
    # sequence window, so reported for behavioural comparison only).
    cache = load_cache()
    key = "S1_Seq1_cam1"
    if key in cache:
        f = cache[key]
        rels = f["reliability"]
        valid = f["valid"]
        hard = f["hard_failure"]
        rows.append({
            "backbone": "MotionAGFormer-XS",
            "backbone_params_M": 2.2,
            "module_params_added": module_parameter_count(),
            "n_frames": int(len(rels)),
            "canonicalization_valid_pct": float(100 * np.mean(valid)),
            "reliability_mean": float(np.mean(rels)),
            "reliability_std": float(np.std(rels)),
            "gt_error_mean_mm": None,   # different window; see fusion/gt results
            "spearman_reliability_vs_error": None,
            "p_value": None,
            "hard_gate_rate_pct": float(100 * np.mean(hard)),
            "abstention_rate_pct": float(100 * np.mean(
                (np.asarray(rels) < ABSTAIN_T) | np.asarray(hard))),
            "error_on_kept_mm": None,
            "note": "cached cross-view window; magnitudes not comparable",
        })

    print(f"{'backbone':26s} {'params':>8s} {'+ours':>6s} {'valid':>7s} "
          f"{'rel':>6s} {'rho(rel,err)':>13s} {'abstain':>8s}")
    for r in rows:
        rho = "—" if r["spearman_reliability_vs_error"] is None \
            else f"{r['spearman_reliability_vs_error']:+.3f}"
        print(f"{r['backbone']:26s} {r['backbone_params_M']:7.1f}M "
              f"{r['module_params_added']:6d} "
              f"{r['canonicalization_valid_pct']:6.1f}% "
              f"{r['reliability_mean']:6.3f} {rho:>13s} "
              f"{r['abstention_rate_pct']:7.1f}%")

    scored = [r for r in rows if r["spearman_reliability_vs_error"] is not None]
    print(f"\nSame code, no refitting, no per-backbone constants. "
          f"Module parameters added: {module_parameter_count()}.")
    if len(scored) >= 2:
        rhos = [r["spearman_reliability_vs_error"] for r in scored]
        agree = all(r < 0 for r in rhos) or all(r > 0 for r in rhos)
        print(f"rho(reliability, GT error) across backbones: "
              f"{', '.join(f'{r:+.3f}' for r in rhos)} "
              f"-> {'consistent sign' if agree else 'INCONSISTENT sign'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "backbone_transfer.json")
    with open(path, "w") as f:
        json.dump({"abstention_threshold": ABSTAIN_T,
                   "module_params_added": module_parameter_count(),
                   "backbones": rows}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
