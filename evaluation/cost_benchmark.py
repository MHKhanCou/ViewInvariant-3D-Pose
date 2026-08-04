"""
Measure what "lightweight" costs, since the title claims it and nothing measured it.

The report demonstrates "training-free" exhaustively and leaves "lightweight" as
an adjective. This turns it into a number: latency per frame for each component
we add, expressed both absolutely and as a fraction of the frozen backbone's own
forward pass, which is the comparison that decides whether the framework is
deployable.

Inputs are real cached poses rather than synthetic ones, so branch behaviour
(degeneracy checks, fallbacks) matches production.

Honesty note on timing: absolute latencies depend on the machine and on what else
it is doing. The ratio to the backbone is the robust quantity and is the one to
quote. If a competing job was running during the measurement it is recorded in
the artifact.

Run:  ./venv/Scripts/python.exe -m evaluation.cost_benchmark
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from canonical.multiscale import multiscale_canonicalize
from evaluation.fusion import median_fuse
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.reliability import compute_reliability_score

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "cost")

# Analytic FLOP counts for the added geometry. These are exact for the
# implementation, not estimates: a cross product is 6 multiplies and 3 adds, a
# 3-vector norm is 3 multiplies, 2 adds and a square root, and the projection is
# a (17,3) by (3,3) matrix product.
FLOPS = {
    "root_centre": 17 * 3,                    # subtract root from every joint
    "axis_vectors": 2 * 3,                    # two differences
    "normalise": 3 * 9,                       # three unit-vector normalisations
    "cross_products": 2 * 9,                  # z = x cross y, x = y cross z
    "projection": 17 * 3 * 3 + 17 * 3 * 2,    # P_rel @ R
    "orthonormality_check": 3 * 3 * 3 + 3 * 3 * 2,
}
CANON_FLOPS = sum(FLOPS.values())

# MotionAGFormer-XS: 2.2M parameters over 27 frames x 17 joints of tokens. A
# transformer forward costs roughly two FLOPs per parameter per token, which is
# the standard approximation and is labelled as such wherever it is reported.
BACKBONE_PARAMS = 2.24e6
BACKBONE_TOKENS = 27 * 17
BACKBONE_FLOPS_PER_CLIP = 2 * BACKBONE_PARAMS * BACKBONE_TOKENS
BACKBONE_FLOPS_PER_FRAME = BACKBONE_FLOPS_PER_CLIP / 27


def _time(fn, n, warmup=200):
    """Median and IQR of per-call wall time, in microseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return {
        "median_us": float(statistics.median(samples)),
        "p25_us": float(samples[len(samples) // 4]),
        "p75_us": float(samples[3 * len(samples) // 4]),
        "n_calls": n,
    }


def load_poses(n=512):
    """Real predicted poses from the Human3.6M cache."""
    path = os.path.join(PRED_DIR, "preds.npz")
    if not os.path.exists(path):
        raise SystemExit("preds.npz missing; run evaluation.h36m_replication first.")
    preds = np.load(path)["preds"][:n // 27 + 1].reshape(-1, 17, 3).astype(np.float64)
    return preds[:n] - preds[:n, 0:1, :]


def measure_backbone(n=30):
    """One forward pass of the frozen backbone, for the ratio that matters."""
    import torch

    from demo_live.lifter import build_xs_model
    model = build_xs_model("cpu")
    x = torch.randn(1, 27, 17, 3)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    per_clip = float(statistics.median(samples))
    return {
        "median_us_per_clip": per_clip,
        "median_us_per_frame": per_clip / 27.0,
        "parameters": int(BACKBONE_PARAMS),
        "n_calls": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=20000)
    ap.add_argument("--note", default="",
                    help="record competing load, e.g. 'MotionBERT inference running'")
    args = ap.parse_args()

    poses = load_poses()
    idx = {"i": 0}

    def nxt():
        p = poses[idx["i"] % len(poses)]
        idx["i"] += 1
        return p

    print("Measuring on %d real cached poses...\n" % len(poses))

    results = {
        "canonicalize_single": _time(lambda: canonicalize_single(nxt()), args.calls),
        "multiscale_canonicalize": _time(lambda: multiscale_canonicalize(nxt()),
                                         max(args.calls // 4, 1000)),
        "reliability_score": _time(lambda: compute_reliability_score(nxt()),
                                   max(args.calls // 4, 1000)),
        "median_fuse_4_views": _time(
            lambda: median_fuse(np.stack([nxt() for _ in range(4)])),
            max(args.calls // 10, 500)),
    }

    tracemalloc.start()
    canonicalize_single(poses[0])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    backbone = measure_backbone()
    per_frame = results["canonicalize_single"]["median_us"]
    ratio = 100.0 * per_frame / backbone["median_us_per_frame"]

    out = {
        "note": args.note,
        "platform": "%s %s, Python %s" % (platform.system(), platform.machine(),
                                          platform.python_version()),
        "trainable_parameters_added": 0,
        "components": results,
        "backbone": backbone,
        "canonicalization_overhead_pct_of_backbone": ratio,
        "peak_extra_memory_bytes": int(peak),
        "flops": {
            "canonicalization_per_frame": CANON_FLOPS,
            "breakdown": FLOPS,
            "backbone_per_frame_approx": BACKBONE_FLOPS_PER_FRAME,
            "ratio_pct": 100.0 * CANON_FLOPS / BACKBONE_FLOPS_PER_FRAME,
            "backbone_note": "two FLOPs per parameter per token, the standard "
                             "approximation for a transformer forward pass",
        },
    }

    print("=" * 74)
    print("COST OF THE ADDED FRAMEWORK")
    print("=" * 74)
    print("  trainable parameters added        %d" % 0)
    print("  peak extra memory                 %d bytes" % peak)
    print()
    for k, v in results.items():
        print("  %-32s %8.1f us  (IQR %.1f-%.1f)"
              % (k, v["median_us"], v["p25_us"], v["p75_us"]))
    print()
    print("  frozen backbone, per clip         %8.1f us" % backbone["median_us_per_clip"])
    print("  frozen backbone, per frame        %8.1f us" % backbone["median_us_per_frame"])
    print()
    print("  canonicalization overhead         %8.3f%% of the backbone" % ratio)
    print("  by analytic FLOPs                 %8.5f%% (%d vs %.2e)"
          % (out["flops"]["ratio_pct"], CANON_FLOPS, BACKBONE_FLOPS_PER_FRAME))
    if args.note:
        print("\n  note: %s" % args.note)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "cost_benchmark.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved: %s" % path)


if __name__ == "__main__":
    main()
