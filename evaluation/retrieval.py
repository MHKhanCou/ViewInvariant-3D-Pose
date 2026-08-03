"""
Cross-view pose retrieval experiment.

Treats canonical poses as representations. For each query pose from camera B,
retrieves the matching pose from camera A by distance.

Compares retrieval accuracy between:
1. Raw root-relative representation
2. Canonical representation

If canonical retrieval outperforms raw retrieval, it provides strong evidence
that the canonical representation is genuinely view-invariant.
"""

import os
import sys
import json
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.metrics import recall_at_k, mean_reciprocal_rank


def compute_distance_matrix(poses_a, poses_b):
    """
    Compute pairwise L2 distances between two sets of poses.

    Args:
        poses_a: (N, 17, 3) gallery poses (camera A)
        poses_b: (M, 17, 3) query poses (camera B)
    Returns:
        (M, N) distance matrix where dist[i,j] = L2 distance between query i and gallery j
    """
    # Flatten each pose to a vector
    A = poses_a.reshape(len(poses_a), -1)  # (N, 51)
    B = poses_b.reshape(len(poses_b), -1)  # (M, 51)

    # L2 distance: ||b - a||^2 = ||b||^2 + ||a||^2 - 2*b@a^T
    dist_sq = np.sum(B**2, axis=1, keepdims=True) + np.sum(A**2, axis=1) - 2 * B @ A.T
    dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
    return np.sqrt(dist_sq)


def compute_pairwise_distances(poses):
    """Compute pairwise L2 distances within a single set of poses."""
    return compute_distance_matrix(poses, poses)


def retrieval_experiment(poses_a, poses_b, label=""):
    """
    Run retrieval experiment: for each query in B, retrieve from A.
    Ground truth: frame i in A matches frame i in B.

    Args:
        poses_a: (N, 17, 3) gallery (camera A)
        poses_b: (M, 17, 3) queries (camera B)
        label: name for this representation
    Returns:
        dict with metrics and per-query ranks
    """
    assert len(poses_a) == len(poses_b), "Must have same number of frames from both cameras"
    N = len(poses_a)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(poses_a, poses_b)  # (N, N)

    # For each query i, rank gallery by distance
    # Ground truth: query i should match gallery i
    ranks = []
    for i in range(N):
        distances = dist_matrix[i]
        # Sort gallery indices by distance (ascending)
        sorted_indices = np.argsort(distances)
        # Find rank of the correct match (index i)
        rank = int(np.where(sorted_indices == i)[0][0]) + 1  # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)

    # Compute metrics
    r1 = recall_at_k(ranks, 1)
    r5 = recall_at_k(ranks, 5)
    r10 = recall_at_k(ranks, 10)
    mrr = mean_reciprocal_rank(ranks.tolist())
    median_rank = float(np.median(ranks))
    mean_rank = float(np.mean(ranks))

    return {
        "label": label,
        "n_queries": N,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "mrr": mrr,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "ranks": ranks.tolist(),
        "distance_matrix_shape": dist_matrix.shape,
    }


def main():
    output_dir = os.path.join(REPO_ROOT, "thesis_artifacts", "retrieval")
    os.makedirs(output_dir, exist_ok=True)

    # Load cross-view evaluation results
    results_path = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval", "results.json")
    with open(results_path) as f:
        data = json.load(f)

    r = data[0]
    n_frames = r["n_frames"]

    # The current run_eval.py stores per-frame distances, not per-camera poses.
    # We need to re-run the evaluation to get per-camera raw and canonical poses.
    # For now, let's re-run the inference to get the actual poses.

    print("Loading models and running inference...")
    from backend.model_loader import get_model, get_detector
    from evaluation.protocol import build_pairing_table, get_frame_paths
    from evaluation.reliability import compute_reliability_score
    from canonical.body_frame import canonicalize_single
    from demo_live.lifter import coco_to_h36m, N_FRAMES, normalize_screen_coordinates
    import torch
    import copy

    model = get_model()
    detector = get_detector()

    table = build_pairing_table()
    pair = table[0]  # S1/Seq1 cam0 <-> cam1

    print("Processing %d frame pairs..." % pair["n_frames"])

    raw_a_list = []
    raw_b_list = []
    can_a_list = []
    can_b_list = []
    rel_a_list = []
    rel_b_list = []

    for i, (frame_idx, path_a, path_b) in enumerate(pair["frame_pairs"]):
        if i % 10 == 0:
            print("  Frame %d/%d..." % (i, pair["n_frames"]))

        for path, raw_list, can_list, rel_list in [
            (path_a, raw_a_list, can_a_list, rel_a_list),
            (path_b, raw_b_list, can_b_list, rel_b_list),
        ]:
            frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]

            kpts, scores = detector.detect_with_rotation(frame)
            if kpts is None:
                kpts = np.zeros((17, 2), dtype=np.float32)
                scores = np.zeros((17,), dtype=np.float32)

            kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
            scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)
            h36m = coco_to_h36m(kpts_seq, scores_seq)

            norm = normalize_screen_coordinates(h36m, w=W, h=H)
            inp = torch.from_numpy(norm.astype("float32"))[None]

            LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
            RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]
            def _flip_data(data):
                f = copy.deepcopy(data)
                f[..., 0] *= -1
                f[..., LEFT_JOINTS + RIGHT_JOINTS, :] = f[..., RIGHT_JOINTS + LEFT_JOINTS, :]
                return f

            flipped = _flip_data(norm)
            inp_aug = torch.from_numpy(flipped.astype("float32"))[None]

            with torch.no_grad():
                pred = (model(inp).cpu().numpy() + _flip_data(model(inp_aug).cpu().numpy())) / 2.0

            raw_pose = pred[0, 13].copy()
            raw_root = raw_pose - raw_pose[0:1]

            can_pose, _, _ = canonicalize_single(raw_root.astype(np.float32))
            rel, _, _ = compute_reliability_score(raw_root, scores)

            raw_list.append(raw_root)
            can_list.append(can_pose)
            rel_list.append(rel)

    raw_a = np.array(raw_a_list)
    raw_b = np.array(raw_b_list)
    can_a = np.array(can_a_list)
    can_b = np.array(can_b_list)
    rel_a = np.array(rel_a_list)
    rel_b = np.array(rel_b_list)

    print("\nInference complete. Running retrieval experiments...")

    # === Experiment 1: Raw retrieval ===
    raw_result = retrieval_experiment(raw_a, raw_b, label="Raw Root-Relative")

    # === Experiment 2: Canonical retrieval ===
    can_result = retrieval_experiment(can_a, can_b, label="Canonical Body-Frame")

    # === Experiment 3: Reliability-stratified retrieval ===
    # Split into high-reliability and low-reliability queries
    rel_avg = (rel_a + rel_b) / 2.0
    high_mask = rel_avg >= 0.85
    low_mask = rel_avg < 0.85

    if high_mask.sum() > 0:
        raw_high = retrieval_experiment(raw_a[high_mask], raw_b[high_mask],
                                         label="Raw (high reliability)")
        can_high = retrieval_experiment(can_a[high_mask], can_b[high_mask],
                                         label="Canonical (high reliability)")
    else:
        raw_high = can_high = None

    if low_mask.sum() > 2:
        raw_low = retrieval_experiment(raw_a[low_mask], raw_b[low_mask],
                                        label="Raw (low reliability)")
        can_low = retrieval_experiment(can_a[low_mask], can_b[low_mask],
                                        label="Canonical (low reliability)")
    else:
        raw_low = can_low = None

    # === Print results ===
    print("\n" + "=" * 70)
    print("CROSS-VIEW POSE RETRIEVAL RESULTS")
    print("=" * 70)

    print("\n%-40s %8s %8s %8s %8s %8s %8s" % (
        "Representation", "R@1", "R@5", "R@10", "MRR", "MedRank", "MeanRank"))
    print("-" * 90)

    for result in [raw_result, can_result]:
        print("%-40s %.4f %8.4f %8.4f %8.4f %8.1f %8.1f" % (
            result["label"],
            result["recall_at_1"],
            result["recall_at_5"],
            result["recall_at_10"],
            result["mrr"],
            result["median_rank"],
            result["mean_rank"],
        ))

    # Improvement
    improvement_r1 = (can_result["recall_at_1"] - raw_result["recall_at_1"]) / max(raw_result["recall_at_1"], 1e-8) * 100
    improvement_mrr = (can_result["mrr"] - raw_result["mrr"]) / max(raw_result["mrr"], 1e-8) * 100

    print("\nRetrieval improvement (canonical vs raw):")
    print("  Recall@1: %.1f%% improvement" % improvement_r1)
    print("  MRR: %.1f%% improvement" % improvement_mrr)

    if raw_high and can_high:
        print("\nHigh-reliability subset (rel >= 0.85):")
        print("  Raw R@1: %.4f" % raw_high["recall_at_1"])
        print("  Canonical R@1: %.4f" % can_high["recall_at_1"])
        print("  n_frames: %d" % raw_high["n_queries"])

    if raw_low and can_low:
        print("\nLow-reliability subset (rel < 0.85):")
        print("  Raw R@1: %.4f" % raw_low["recall_at_1"])
        print("  Canonical R@1: %.4f" % can_low["recall_at_1"])
        print("  n_frames: %d" % raw_low["n_queries"])
    else:
        print("\nLow-reliability subset: insufficient frames")

    # === Save results ===
    all_results = {
        "raw": raw_result,
        "canonical": can_result,
        "improvement_recall_at_1_pct": improvement_r1,
        "improvement_mrr_pct": improvement_mrr,
    }
    if raw_high:
        all_results["raw_high_reliability"] = raw_high
        all_results["canonical_high_reliability"] = can_high
    if raw_low:
        all_results["raw_low_reliability"] = raw_low
        all_results["canonical_low_reliability"] = can_low

    with open(os.path.join(output_dir, "retrieval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to: %s" % output_dir)


if __name__ == "__main__":
    import cv2
    main()
