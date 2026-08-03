"""
Cross-view evaluation on MPI-INF-3DHP synchronized cameras — corrected protocol.

Protocol (corrects the examiner-flagged single-frame-repeated-27x flaw):
1. Per camera: run YOLO once per physical frame, then lift each fully-windowed
   center frame (13..N-14) with a TRUE 27-frame detection window.
2. Canonicalize each frame INDEPENDENTLY (no temporal state enters the metric).
3. Reliability per frame; the temporal_stability component receives the
   previous frame's rotation from the SAME camera stream (legitimate video
   continuity — never crosses cameras).
4. All predictions cached to predictions_cache.npz; every pair metric,
   ablation, and GT analysis derives from the cache without re-inference.
5. Pair metrics for ALL camera pairs per sequence: raw / canonical /
   Procrustes-oracle distances, coverage-error curves WITH an error axis.

Constraint checklist:
- Hard gates + 0.5 threshold frozen at implementation time (no tuning here)
- Raw root-relative predictions only (no display transforms)
- Dev pair (S1/Seq1 cam0-1) labeled; all other pairs held-out
"""

import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.protocol import (
    build_pairing_table, discover_cameras, evaluated_centers, HALF, WINDOW,
)
from evaluation.reliability import (
    compute_reliability_score, has_hard_geometric_failure,
)
from evaluation.metrics import (
    cross_view_joint_distance_sequence, bone_length_deviation,
    joint_angle_consistency,
)
from evaluation.lifting import lift_from_coco_window
from evaluation.oracle import procrustes_cross_view_distance
from canonical.body_frame import canonicalize_single

OUTPUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval")
CACHE_PATH = os.path.join(OUTPUT_DIR, "predictions_cache.npz")

# Fixed order for the cached per-component reliability breakdown.
COMPONENT_KEYS = [
    "axis_conditioning", "torso_hip_angle", "bilateral_symmetry",
    "abnormal_bone_ratio", "detector_confidence", "temporal_stability",
]

COVERAGE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def cam_key(cam_info):
    """
    Cache key. The condition suffix keeps the static and dynamic windows of the
    same camera in separate entries ('' for static preserves existing keys).
    """
    cond = cam_info.get("condition", "static")
    suffix = "" if cond == "static" else f"_{cond}"
    return (f"{cam_info['subject']}_{cam_info['sequence'].replace('/', '_')}"
            f"_cam{cam_info['camera']}{suffix}")


def predict_camera(model, detector, cam_info, device="cpu"):
    """
    Predict every fully-windowed frame of one camera with true temporal windows.

    Returns dict of arrays: centers, raw, canonical, R, valid, reliability,
    components, hard_failure.
    """
    paths = cam_info["frame_paths"]
    n = len(paths)
    centers = evaluated_centers(n)

    # --- One detector pass per physical frame ---
    kpts_all = np.zeros((n, 17, 2), dtype=np.float32)
    scores_all = np.zeros((n, 17), dtype=np.float32)
    wh = None
    for i, path in enumerate(paths):
        if i % 20 == 0:
            print(f"    detect {i}/{n}...")
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if wh is None:
            wh = (frame.shape[1], frame.shape[0])
        kpts, scores = detector.detect_with_rotation(frame)
        if kpts is not None:
            kpts_all[i] = kpts
            scores_all[i] = scores
    W, H = wh

    # --- Lift each center with its true window; sequential reliability chain ---
    C = len(centers)
    raw = np.zeros((C, 17, 3), dtype=np.float64)
    canonical = np.zeros((C, 17, 3), dtype=np.float32)
    R_all = np.zeros((C, 3, 3), dtype=np.float32)
    valid = np.zeros(C, dtype=bool)
    reliability = np.zeros(C, dtype=np.float64)
    components = np.zeros((C, len(COMPONENT_KEYS)), dtype=np.float64)
    hard_failure = np.zeros(C, dtype=bool)

    prev_R = None
    for j, c in enumerate(centers):
        if j % 20 == 0:
            print(f"    lift {j}/{C}...")
        lo = c - HALF
        window_kpts = kpts_all[lo:lo + WINDOW]
        window_scores = scores_all[lo:lo + WINDOW]
        pose = lift_from_coco_window(model, window_kpts, window_scores, W, H, device)

        # Metric canonicalization: independent per frame, no temporal state.
        can, R, meta = canonicalize_single(pose.astype(np.float32))

        # Reliability: prev_R from the same camera stream only.
        rel, comps, _ = compute_reliability_score(
            pose, scores_all[c], prev_rotation=prev_R)
        hard, _ = has_hard_geometric_failure(pose)

        raw[j] = pose
        canonical[j] = can
        R_all[j] = R
        valid[j] = meta["valid"]
        reliability[j] = rel
        components[j] = [comps.get(k, np.nan) for k in COMPONENT_KEYS]
        hard_failure[j] = hard
        if meta["valid"]:
            prev_R = R

    # `centers` are positions in this camera's extracted list; `frame_ids` are
    # the true source-video indices used for GT lookup and cross-camera pairing
    # (they differ whenever an extraction does not start at frame 0).
    frame_ids = np.array([cam_info["frame_ids"][c] for c in centers], dtype=np.int32)

    return {
        "centers": frame_ids,
        "positions": np.array(centers, dtype=np.int32),
        "raw": raw,
        "canonical": canonical,
        "R": R_all,
        "valid": valid,
        "reliability": reliability,
        "components": components,
        "hard_failure": hard_failure,
    }


def load_cache():
    """Load the prediction cache as {cam_key: {field: array}}."""
    if not os.path.exists(CACHE_PATH):
        return {}
    data = np.load(CACHE_PATH)
    cams = {}
    for full_key in data.files:
        key, field = full_key.rsplit("__", 1)
        cams.setdefault(key, {})[field] = data[full_key]
    return cams


def save_cache(cams):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flat = {}
    for key, fields in cams.items():
        for field, arr in fields.items():
            flat[f"{key}__{field}"] = arr
    np.savez_compressed(CACHE_PATH, **flat)


def evaluate_pair_from_cache(pair_info, cache):
    """Compute all pair metrics from cached per-camera predictions."""
    cond = pair_info.get("condition", "static")
    key_a = cam_key({"subject": pair_info["subject"],
                     "sequence": pair_info["sequence"],
                     "camera": pair_info["cam_a"], "condition": cond})
    key_b = cam_key({"subject": pair_info["subject"],
                     "sequence": pair_info["sequence"],
                     "camera": pair_info["cam_b"], "condition": cond})
    a, b = cache[key_a], cache[key_b]

    # Match frames by center index (cameras are synchronized).
    common, ia, ib = np.intersect1d(a["centers"], b["centers"],
                                    return_indices=True)
    raw_a, raw_b = a["raw"][ia], b["raw"][ib]
    can_a, can_b = a["canonical"][ia], b["canonical"][ib]
    rel_a, rel_b = a["reliability"][ia], b["reliability"][ib]
    hard_a, hard_b = a["hard_failure"][ia], b["hard_failure"][ib]

    raw_per_frame, raw_mean = cross_view_joint_distance_sequence(raw_a, raw_b)
    can_per_frame, can_mean = cross_view_joint_distance_sequence(can_a, can_b)

    oracle_per_frame = np.array([
        procrustes_cross_view_distance(raw_a[i], raw_b[i])
        for i in range(len(common))
    ])

    bl_devs = [bone_length_deviation(raw_a[i], raw_b[i])[0]
               for i in range(len(common))]
    ja_diffs = [joint_angle_consistency(raw_a[i], raw_b[i])[0]
                for i in range(len(common))]

    # Coverage-error: joint reliability gate (both views must pass).
    pair_rel = np.minimum(rel_a, rel_b)
    pair_ok = ~(hard_a | hard_b)
    coverage_results = {}
    for t in COVERAGE_THRESHOLDS:
        mask = (pair_rel >= t) & pair_ok
        entry = {"coverage": float(mask.mean()), "n_frames": int(mask.sum())}
        if mask.any():
            entry["filtered_raw_distance"] = float(raw_per_frame[mask].mean())
            entry["filtered_canonical_distance"] = float(can_per_frame[mask].mean())
        else:
            entry["filtered_raw_distance"] = None
            entry["filtered_canonical_distance"] = None
        coverage_results[str(t)] = entry

    rels = np.concatenate([rel_a, rel_b])
    n_hard = int(hard_a.sum() + hard_b.sum())

    return {
        "subject": pair_info["subject"],
        "sequence": pair_info["sequence"],
        "condition": cond,
        "cam_a": pair_info["cam_a"],
        "cam_b": pair_info["cam_b"],
        "is_dev_pair": pair_info["is_dev_pair"],
        "n_frames": int(len(common)),
        "raw_cross_view_distance": float(raw_mean),
        "canonical_cross_view_distance": float(can_mean),
        "oracle_cross_view_distance": float(oracle_per_frame.mean()),
        "improvement_pct": float((1 - can_mean / max(raw_mean, 1e-8)) * 100),
        "oracle_gap_closed_pct": float(
            (raw_mean - can_mean) / max(raw_mean - oracle_per_frame.mean(), 1e-8) * 100),
        "bone_length_deviation": float(np.mean(bl_devs)),
        "joint_angle_diff": float(np.mean(ja_diffs)),
        "reliability_mean": float(rels.mean()),
        "reliability_min": float(rels.min()),
        "hard_failure_count": n_hard,
        "hard_failure_rate": float(n_hard / max(len(common) * 2, 1)),
        "coverage_at_threshold": coverage_results,
        "per_frame_raw": raw_per_frame.tolist(),
        "per_frame_canonical": can_per_frame.tolist(),
        "per_frame_oracle": oracle_per_frame.tolist(),
        "per_frame_reliability_a": rel_a.tolist(),
        "per_frame_reliability_b": rel_b.tolist(),
        "frame_centers": common.tolist(),
    }


def main():
    print("Loading models...")
    from backend.model_loader import get_model, get_detector
    model = get_model()
    detector = get_detector()

    cache = load_cache()
    cameras = discover_cameras()

    for cam_info in cameras:
        key = cam_key(cam_info)
        if key in cache:
            print(f"  {key}: cached ({len(cache[key]['centers'])} frames), skip")
            continue
        print(f"  Predicting {key} ({cam_info['n_frames']} frames, "
              f"{len(evaluated_centers(cam_info['n_frames']))} windowed centers)")
        cache[key] = predict_camera(model, detector, cam_info)
        save_cache(cache)  # checkpoint after each camera

    print("\nBuilding pairing table...")
    table = build_pairing_table()

    all_results = [evaluate_pair_from_cache(pair, cache) for pair in table]

    # --- Summary ---
    print("\n" + "=" * 80)
    print("CROSS-VIEW EVALUATION SUMMARY (corrected protocol: true 27-frame windows)")
    print("=" * 80)
    for r in all_results:
        tag = "  [DEV PAIR]" if r["is_dev_pair"] else ""
        print(f"{r['subject']}/{r['sequence']} cam{r['cam_a']}-cam{r['cam_b']}{tag}: "
              f"raw {r['raw_cross_view_distance']:.4f}  "
              f"can {r['canonical_cross_view_distance']:.4f}  "
              f"oracle {r['oracle_cross_view_distance']:.4f}  "
              f"impr {r['improvement_pct']:+.1f}%  "
              f"rel {r['reliability_mean']:.3f}")

    held_out = [r for r in all_results
                if not r["is_dev_pair"] and r["subject"] == "S1"]
    s2 = [r for r in all_results if r["subject"] == "S2"]
    dev = [r for r in all_results if r["is_dev_pair"]]

    def agg(rs, name):
        if not rs:
            return
        imp = np.array([r["improvement_pct"] for r in rs])
        print(f"\n{name}: n={len(rs)} pairs, "
              f"improvement mean {imp.mean():+.1f}% "
              f"(min {imp.min():+.1f}%, max {imp.max():+.1f}%)")

    agg(dev, "Dev pair (S1/Seq1 cam0-1)")
    agg(held_out, "Held-out pairs (S1/Seq1, 27 pairs)")
    agg(s2, "Held-out subject (S2/Seq1)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "results_multicam.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Prediction cache:  {CACHE_PATH}")


if __name__ == "__main__":
    main()
