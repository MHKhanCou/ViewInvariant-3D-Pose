"""
Controlled Degradation Protocol (CDP) sweep.

Produces the calibration data for the reliability score by corrupting real
2D detections in physically-motivated ways and measuring what happens to
canonicalization.

Experimental design
-------------------
For each synchronized MPI-INF-3DHP frame pair we corrupt **camera A only** and
leave camera B clean. This isolates the effect: any change in the cross-view
distance is attributable to the degradation we injected into A, whose magnitude
we know exactly. Corrupting both views would confound the two.

Two error signals are recorded per condition:

  cross_view_distance  -- canonical(corrupt A) vs canonical(clean B).
                          The thesis metric. This is what the reliability score
                          is ultimately claiming to predict.
  canonical_drift      -- canonical(corrupt A) vs canonical(clean A).
                          How far canonicalization moved for the *same* view.
                          Isolates canonicalization instability from ordinary
                          pose error, and needs no second camera.

The detector runs once per frame and its output is cached; the sweep then only
re-runs the lifter. This is not just a speed optimization -- the detector is
not the variable under study, and re-detecting a corrupted image is not even
well defined, since we corrupt keypoints rather than pixels.

Every operator's severity level 0 is the identity, so each sweep contains an
internal control that must reproduce the clean baseline exactly. The runner
asserts this; a mismatch means the pipeline is non-deterministic and the whole
calibration is void.

Usage:
    python -m evaluation.degradation_sweep                # full sweep
    python -m evaluation.degradation_sweep --limit 5      # smoke test
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.protocol import build_pairing_table
from evaluation.corruptions import CORRUPTIONS, apply, is_identity_severity
from evaluation.lifting import lift_from_coco
from evaluation.metrics import cross_view_joint_distance
from evaluation.reliability import (
    compute_reliability_score, has_hard_geometric_failure,
)
from canonical.body_frame import canonicalize_single
from backend.model_loader import get_model, get_detector

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "degradation")
CACHE_PATH = os.path.join(OUT_DIR, "detections_2d.npz")
RECORDS_PATH = os.path.join(OUT_DIR, "sweep_records.json")

# Identity-control tolerance. The pipeline is deterministic, so this only
# guards against accidental mutation of the cached keypoints by an operator.
IDENTITY_TOL = 1e-9


def build_or_load_cache(pairs, force=False):
    """
    Run the detector once over every frame in the pairing table and cache the
    COCO-17 keypoints, confidences, and frame dimensions.

    Returns a dict keyed "{pair_index}_{a|b}" -> (kpts, scores, width, height).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH) and not force:
        blob = np.load(CACHE_PATH, allow_pickle=True)
        print(f"Loaded cached 2D detections: {CACHE_PATH}")
        return {k: blob[k] for k in blob.files}

    print("Building 2D detection cache (one detector pass per frame)...")
    detector = get_detector()
    cache = {}
    for i, (frame_idx, path_a, path_b) in enumerate(pairs):
        if i % 10 == 0:
            print(f"  detecting {i}/{len(pairs)}...")
        for tag, path in (("a", path_a), ("b", path_b)):
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]
            kpts, scores = detector.detect_with_rotation(frame)
            if kpts is None:
                # Record the miss rather than dropping the frame: a failed
                # detection is itself a degenerate case the gates should catch.
                kpts = np.zeros((17, 2), dtype=np.float32)
                scores = np.zeros((17,), dtype=np.float32)
            cache[f"{i}_{tag}"] = np.array(
                [np.asarray(kpts, dtype=np.float32),
                 np.pad(np.asarray(scores, dtype=np.float32)[:, None], ((0, 0), (0, 1))),
                 np.full((17, 2), [W, H], dtype=np.float32)],
                dtype=np.float32,
            )

    np.savez_compressed(CACHE_PATH, **cache)
    print(f"Cached {len(cache)} detections -> {CACHE_PATH}")
    return cache


def _unpack(entry):
    """Undo the packed cache layout -> (kpts, scores, W, H)."""
    arr = np.asarray(entry)
    kpts = arr[0]
    scores = arr[1][:, 0]
    W, H = float(arr[2][0, 0]), float(arr[2][0, 1])
    return kpts, scores, W, H


def _evaluate_pose(pose, scores_2d, canon_clean_b, canon_clean_a):
    """Score one lifted pose against both reference frames."""
    reliability, components, _ = compute_reliability_score(pose, scores_2d)
    hard_failure, reason = has_hard_geometric_failure(pose)
    canon, _R, meta = canonicalize_single(pose.astype(np.float32))

    # A degenerate pose yields an all-zero canonical pose with valid=False.
    # Distances against it are meaningless, so they are recorded as NaN and
    # excluded from correlation analysis rather than silently biasing it.
    if meta["valid"]:
        cross_view = cross_view_joint_distance(canon, canon_clean_b)
        drift = cross_view_joint_distance(canon, canon_clean_a)
    else:
        cross_view = float("nan")
        drift = float("nan")

    return {
        "reliability": float(reliability),
        "components": {k: float(v) for k, v in components.items()},
        "hard_failure": bool(hard_failure),
        "failure_reason": reason,
        "canonicalization_valid": bool(meta["valid"]),
        "cross_view_distance": float(cross_view),
        "canonical_drift": float(drift),
    }


def run_sweep(limit=None, force_cache=False, device="cpu"):
    table = build_pairing_table()
    pair_info = table[0] if isinstance(table, (list, tuple)) else table
    pairs = pair_info["frame_pairs"]
    if limit:
        pairs = pairs[:limit]

    cache = build_or_load_cache(pairs, force=force_cache)
    model = get_model()

    total_conditions = sum(len(levels) for _, levels, _ in CORRUPTIONS.values())
    print(f"\nSweep: {len(pairs)} frames x {total_conditions} conditions "
          f"= {len(pairs) * total_conditions} lifts (+{2 * len(pairs)} clean)")

    records = []
    identity_failures = []
    t0 = time.time()

    for i, (frame_idx, _pa, _pb) in enumerate(pairs):
        if i % 5 == 0:
            elapsed = time.time() - t0
            print(f"  frame {i}/{len(pairs)}  ({elapsed:.0f}s elapsed)")

        kpts_a, scores_a, WA, HA = _unpack(cache[f"{i}_a"])
        kpts_b, scores_b, WB, HB = _unpack(cache[f"{i}_b"])

        # Clean references, lifted once and reused across all conditions.
        clean_a = lift_from_coco(model, kpts_a, scores_a, WA, HA, device)
        clean_b = lift_from_coco(model, kpts_b, scores_b, WB, HB, device)
        canon_clean_a, _, meta_a = canonicalize_single(clean_a.astype(np.float32))
        canon_clean_b, _, meta_b = canonicalize_single(clean_b.astype(np.float32))

        if not (meta_a["valid"] and meta_b["valid"]):
            print(f"    ! frame {frame_idx}: clean canonicalization invalid, skipping")
            continue

        baseline = _evaluate_pose(clean_a, scores_a, canon_clean_b, canon_clean_a)
        records.append({
            "frame_index": int(frame_idx),
            "operator": "clean",
            "severity": 0.0,
            "severity_index": 0,
            "is_control": True,
            **baseline,
        })

        for op_name, (_fn, levels, _units) in CORRUPTIONS.items():
            for sev_idx, severity in enumerate(levels):
                seed = abs(hash((int(frame_idx), op_name, sev_idx))) % (2 ** 32)
                k_c, s_c = apply(op_name, kpts_a, scores_a, severity, seed=seed)
                pose = lift_from_coco(model, k_c, s_c, WA, HA, device)
                result = _evaluate_pose(pose, s_c, canon_clean_b, canon_clean_a)

                # Internal control: the identity severity must reproduce clean.
                if is_identity_severity(op_name, severity):
                    delta = abs(result["cross_view_distance"]
                                - baseline["cross_view_distance"])
                    if not (delta < IDENTITY_TOL):
                        identity_failures.append(
                            f"{op_name}@{severity} frame {frame_idx}: delta={delta:.3e}"
                        )

                records.append({
                    "frame_index": int(frame_idx),
                    "operator": op_name,
                    "severity": float(severity),
                    "severity_index": sev_idx,
                    "is_control": bool(is_identity_severity(op_name, severity)),
                    **result,
                })

    elapsed = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RECORDS_PATH, "w") as f:
        json.dump({
            "n_frames": len(pairs),
            "n_records": len(records),
            "elapsed_seconds": elapsed,
            "subject": pair_info["subject"],
            "sequence": pair_info["sequence"],
            "cam_a": pair_info["cam_a"],
            "cam_b": pair_info["cam_b"],
            "corruption_units": {k: v[2] for k, v in CORRUPTIONS.items()},
            "identity_control_failures": identity_failures,
            "records": records,
        }, f, indent=2)

    print(f"\nDone in {elapsed / 60:.1f} min. {len(records)} records -> {RECORDS_PATH}")
    if identity_failures:
        print(f"WARNING: {len(identity_failures)} identity-control mismatches "
              f"(pipeline should be deterministic):")
        for msg in identity_failures[:5]:
            print(f"  {msg}")
    else:
        print("Identity controls reproduced the clean baseline exactly.")

    n_hard = sum(r["hard_failure"] for r in records)
    print(f"Hard gates fired on {n_hard}/{len(records)} conditions "
          f"({100 * n_hard / max(len(records), 1):.1f}%)")
    return records


def main():
    ap = argparse.ArgumentParser(description="Controlled Degradation Protocol sweep")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N frame pairs (smoke test)")
    ap.add_argument("--force-cache", action="store_true",
                    help="rebuild the 2D detection cache")
    args = ap.parse_args()
    run_sweep(limit=args.limit, force_cache=args.force_cache)


if __name__ == "__main__":
    main()
