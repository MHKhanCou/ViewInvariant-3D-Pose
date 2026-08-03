"""
Cross-view evaluation metrics.

All metrics operate on raw root-relative or canonical poses.
None use display transforms (camera_to_world, z-shift, scaling).

Naming: NO metric is called "MPJPE" unless it is prediction-vs-ground-truth
with Procrustes alignment on actual H36M test data.
"""

import numpy as np


def cross_view_joint_distance(pred_a, pred_b):
    """
    Mean Euclidean distance between two root-relative predictions.
    Both must be root-centered (joint 0 at origin).

    Args:
        pred_a: (17, 3) prediction from camera A
        pred_b: (17, 3) prediction from camera B
    Returns:
        float: mean joint distance across all 17 joints
    """
    distances = np.linalg.norm(pred_a - pred_b, axis=1)
    return float(np.mean(distances))


def cross_view_joint_distance_sequence(preds_a, preds_b):
    """
    Cross-view joint distance for a sequence of frame pairs.

    Args:
        preds_a: (T, 17, 3) predictions from camera A
        preds_b: (T, 17, 3) predictions from camera B
    Returns:
        per_frame: (T,) array of per-frame joint distances
        mean: float mean over all frames
    """
    per_frame = np.array([
        cross_view_joint_distance(preds_a[i], preds_b[i])
        for i in range(len(preds_a))
    ])
    return per_frame, float(np.mean(per_frame))


def bone_length_deviation(pred_a, pred_b):
    """
    Mean absolute difference in bone lengths between two predictions.

    Args:
        pred_a, pred_b: (17, 3) root-relative poses
    Returns:
        float: mean per-bone length difference
        (16,) array: per-bone differences
    """
    from evaluation.reliability import BONE_CONNECTIONS
    lengths_a = np.array([np.linalg.norm(pred_a[i] - pred_a[j]) for i, j in BONE_CONNECTIONS])
    lengths_b = np.array([np.linalg.norm(pred_b[i] - pred_b[j]) for i, j in BONE_CONNECTIONS])
    per_bone = np.abs(lengths_a - lengths_b)
    return float(np.mean(per_bone)), per_bone


def joint_angle_consistency(pred_a, pred_b):
    """
    Mean absolute difference in joint angles between two predictions.

    Computes angles at 5 key joints: both knees, both elbows, torso.
    """
    def angle_at(p, joint, prev_j, next_j):
        v1 = p[prev_j] - p[joint]
        v2 = p[next_j] - p[joint]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    key_joints = [
        ("R_knee", 2, 1, 3),
        ("L_knee", 5, 4, 6),
        ("R_elbow", 12, 11, 13),
        ("L_elbow", 15, 14, 16),
        ("torso", 8, 0, 9),
    ]

    angles_a = [angle_at(pred_a, j, p, n) for _, j, p, n in key_joints]
    angles_b = [angle_at(pred_b, j, p, n) for _, j, p, n in key_joints]

    diffs = [abs(a - b) for a, b in zip(angles_a, angles_b)]
    return float(np.mean(diffs)), diffs


def recall_at_k(retrieval_ranks, k):
    """Recall@k: fraction of queries where the correct match is in top-k."""
    hits = sum(1 for rank in retrieval_ranks if rank <= k)
    return hits / max(len(retrieval_ranks), 1)


def mean_reciprocal_rank(retrieval_ranks):
    """Mean Reciprocal Rank."""
    return float(np.mean([1.0 / r for r in retrieval_ranks if r > 0])) if retrieval_ranks else 0.0
