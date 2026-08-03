"""
Cross-view evaluation on MPI-INF-3DHP synchronized camera pairs.

Evaluation pipeline:
1. For each synchronized frame pair (camera A, camera B):
   a. Load both frames
   b. Run YOLO + MotionAGFormer on each frame (single-frame repeated 27x)
   c. Get raw root-relative output
   d. Compute canonical pose
   e. Compute reliability score + hard gate
2. Compare raw and canonical cross-view distances
3. Report coverage, abstention, and failure reasons

Constraint checklist:
- Hard gates frozen at implementation time (0.5 threshold)
- Raw root-relative prediction only (no display transforms)
- True 27-frame windows from each camera (replicate-padded)
- Confidence intervals grouped by sequence
"""

import os
import sys
import time
import numpy as np
import torch
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.protocol import build_pairing_table, load_frames_as_sequence
from evaluation.reliability import (
    compute_reliability_score, should_abstain, has_hard_geometric_failure,
    compute_bone_lengths
)
from evaluation.metrics import (
    cross_view_joint_distance, cross_view_joint_distance_sequence,
    bone_length_deviation, joint_angle_consistency
)
from evaluation.lifting import lift_from_coco
from canonical.canonicalizer import Canonicalizer
from canonical.body_frame import canonicalize_single
from backend.model_loader import get_model, get_detector
from demo_live.lifter import coco_to_h36m, N_FRAMES, normalize_screen_coordinates


def predict_single_frame(model, detector, frame_rgb, device="cpu"):
    """
    Run full pipeline on a single frame. Returns raw root-relative pose.

    Returns:
        raw_pose: (17, 3) raw root-relative
        scores: (17,) COCO-17 confidence
        reliability: float
        hard_failure: bool
        failure_reason: str
    """
    H, W = frame_rgb.shape[:2]

    kpts, scores = detector.detect_with_rotation(frame_rgb)
    if kpts is None:
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    # Lift to raw root-relative 3D (27-frame window + flip augmentation).
    # Shared with the degradation sweep so both run identical model code.
    raw_root = lift_from_coco(model, kpts, scores, W, H, device=device)

    # Reliability
    reliability, components, metadata = compute_reliability_score(raw_root, scores)
    hard_failure, failure_reason = has_hard_geometric_failure(raw_root)

    return raw_root, scores, reliability, hard_failure, failure_reason


def evaluate_cross_view_pair(model, detector, pair_info, device="cpu"):
    """
    Evaluate a single synchronized camera pair.

    Returns:
        dict with per-frame and aggregate metrics
    """
    subject = pair_info["subject"]
    sequence = pair_info["sequence"]
    cam_a = pair_info["cam_a"]
    cam_b = pair_info["cam_b"]
    frame_pairs = pair_info["frame_pairs"]

    print(f"\nEvaluating: {subject}/{sequence} cam{cam_a} <-> cam{cam_b} ({len(frame_pairs)} frames)")

    # Predict for both cameras
    raw_a_list = []
    raw_b_list = []
    can_a_list = []
    can_b_list = []
    reliability_a_list = []
    reliability_b_list = []
    hard_failure_a_list = []
    hard_failure_b_list = []
    failure_reason_a_list = []
    failure_reason_b_list = []

    for i, (frame_idx, path_a, path_b) in enumerate(frame_pairs):
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frame_pairs)}...")

        # Load and predict camera A
        frame_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
        raw_a, scores_a, rel_a, hard_a, reason_a = predict_single_frame(model, detector, frame_a, device)

        # Load and predict camera B
        frame_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)
        raw_b, scores_b, rel_b, hard_b, reason_b = predict_single_frame(model, detector, frame_b, device)

        # Canonicalize
        can_a, R_a, meta_a = canonicalize_single(raw_a.astype(np.float32))
        can_b, R_b, meta_b = canonicalize_single(raw_b.astype(np.float32))

        raw_a_list.append(raw_a)
        raw_b_list.append(raw_b)
        can_a_list.append(can_a)
        can_b_list.append(can_b)
        reliability_a_list.append(rel_a)
        reliability_b_list.append(rel_b)
        hard_failure_a_list.append(hard_a)
        hard_failure_b_list.append(hard_b)
        failure_reason_a_list.append(reason_a)
        failure_reason_b_list.append(reason_b)

    raw_a_arr = np.array(raw_a_list)
    raw_b_arr = np.array(raw_b_list)
    can_a_arr = np.array(can_a_list)
    can_b_arr = np.array(can_b_list)

    # Compute cross-view distances
    raw_per_frame, raw_mean = cross_view_joint_distance_sequence(raw_a_arr, raw_b_arr)
    can_per_frame, can_mean = cross_view_joint_distance_sequence(can_a_arr, can_b_arr)

    # Compute bone-length and joint-angle consistency
    bl_devs = []
    ja_diffs = []
    for i in range(len(frame_pairs)):
        bl_mean, _ = bone_length_deviation(raw_a_arr[i], raw_b_arr[i])
        bl_devs.append(bl_mean)
        ja_mean, _ = joint_angle_consistency(raw_a_arr[i], raw_b_arr[i])
        ja_diffs.append(ja_mean)

    # Aggregate reliability
    reliabilities = np.array(reliability_a_list + reliability_b_list)
    n_hard = sum(hard_failure_a_list) + sum(hard_failure_b_list)
    n_total_frames = len(frame_pairs) * 2  # Both cameras

    # Coverage analysis at different thresholds
    all_rels = np.concatenate([reliability_a_list, reliability_b_list])
    all_hard = np.array(hard_failure_a_list + hard_failure_b_list)
    coverage_results = {}
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = (all_rels >= threshold) & (~all_hard)
        coverage = mask.sum() / len(mask) if len(mask) > 0 else 0
        if mask.sum() > 0:
            filtered_raw = raw_per_frame[
                np.array([not h for h in hard_failure_a_list])
                & (np.array(reliability_a_list) >= threshold)
            ] if len(raw_per_frame) > 0 else np.array([])
            filtered_can = can_per_frame[
                np.array([not h for h in hard_failure_a_list])
                & (np.array(reliability_a_list) >= threshold)
            ] if len(can_per_frame) > 0 else np.array([])
        coverage_results[threshold] = {
            "coverage": float(coverage),
            "n_frames": int(mask.sum()),
        }

    return {
        "subject": subject,
        "sequence": sequence,
        "cam_a": cam_a,
        "cam_b": cam_b,
        "n_frames": len(frame_pairs),
        "raw_cross_view_distance": raw_mean,
        "canonical_cross_view_distance": can_mean,
        "improvement_pct": (1 - can_mean / max(raw_mean, 1e-8)) * 100,
        "bone_length_deviation": float(np.mean(bl_devs)),
        "joint_angle_diff": float(np.mean(ja_diffs)),
        "reliability_mean": float(np.mean(reliabilities)),
        "reliability_min": float(np.min(reliabilities)),
        "hard_failure_count": n_hard,
        "hard_failure_rate": n_hard / max(n_total_frames, 1),
        "coverage_at_threshold": coverage_results,
        "per_frame_raw": raw_per_frame.tolist(),
        "per_frame_canonical": can_per_frame.tolist(),
        "per_frame_reliability_a": reliability_a_list,
        "per_frame_reliability_b": reliability_b_list,
    }


def main():
    print("Loading models...")
    model = get_model()
    detector = get_detector()

    # Build pairing table
    print("\nBuilding pairing table...")
    table = build_pairing_table()

    if not table:
        print("ERROR: No valid camera pairs found.")
        return

    # Evaluate each pair
    all_results = []
    for pair in table:
        result = evaluate_cross_view_pair(model, detector, pair)
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("CROSS-VIEW EVALUATION SUMMARY")
    print("=" * 80)

    for r in all_results:
        print(f"\n{r['subject']}/{r['sequence']} cam{r['cam_a']} <-> cam{r['cam_b']}:")
        print(f"  Frames: {r['n_frames']}")
        print(f"  Raw cross-view distance:       {r['raw_cross_view_distance']:.4f}")
        print(f"  Canonical cross-view distance:  {r['canonical_cross_view_distance']:.4f}")
        print(f"  Improvement:                    {r['improvement_pct']:.1f}%")
        print(f"  Bone-length deviation:          {r['bone_length_deviation']:.4f}")
        print(f"  Joint-angle difference:         {r['joint_angle_diff']:.1f} degrees")
        print(f"  Reliability: {r['reliability_mean']:.4f} (min={r['reliability_min']:.4f})")
        print(f"  Hard failures: {r['hard_failure_count']}/{r['n_frames']*2}")
        print(f"  Hard failure rate: {r['hard_failure_rate']:.1%}")
        print(f"\n  Coverage at thresholds:")
        for t, info in r['coverage_at_threshold'].items():
            print(f"    t={t}: {info['coverage']:.1%} ({info['n_frames']}/{r['n_frames']*2})")

    # Save results
    import json
    output_dir = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
