"""
Evaluation metrics for canonical pose representation.

These metrics are separate from the official H36M benchmark and should
NOT be confused with the frozen MPJPE/P-MPJPE baseline results.
"""

import numpy as np
from canonical.body_frame import canonicalize_single, compute_bone_lengths


def canonical_mpjpe(pred, gt):
    """
    Mean Joint Position Error after independent canonicalization.

    Both pred and gt are canonicalized separately (each gets its own
    body-frame), then the mean Euclidean distance across all 17 joints
    is computed.

    Args:
        pred: (17, 3) or (T, 17, 3) predicted root-relative 3D pose(s).
        gt:   (17, 3) or (T, 17, 3) ground-truth root-relative 3D pose(s).

    Returns:
        error: float, mean joint distance in the same units as input (mm
               if input is in mm).

    Note:
        This is NOT directly comparable to official H36M P1 MPJPE, because
        canonicalization applies independent rotations to pred and gt.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if pred.ndim == 2:
        pred_can, _, _ = canonicalize_single(pred)
        gt_can, _, _ = canonicalize_single(gt)
        return float(np.mean(np.linalg.norm(pred_can - gt_can, axis=-1)))

    elif pred.ndim == 3:
        errors = []
        for t in range(pred.shape[0]):
            pred_can, _, _ = canonicalize_single(pred[t])
            gt_can, _, _ = canonicalize_single(gt[t])
            errors.append(np.mean(np.linalg.norm(pred_can - gt_can, axis=-1)))
        return float(np.mean(errors))

    else:
        raise ValueError(f"Expected (17,3) or (T,17,3), got {pred.shape}")


def cross_view_consistency_error(pred_a, pred_b):
    """
    Cross-view consistency error after canonicalization.

    Canonicalizes two predictions independently and computes mean
    joint distance.  Intended for matched frames from the same
    subject/action across two different H36M cameras.

    Each frame pair is canonicalized independently (no temporal
    state is carried between pairs).

    Args:
        pred_a: (17, 3) or (T, 17, 3) prediction from camera A.
        pred_b: (17, 3) or (T, 17, 3) prediction from camera B.

    Returns:
        error: float, mean joint distance between canonicalized predictions.
    """
    pred_a = np.asarray(pred_a, dtype=np.float32)
    pred_b = np.asarray(pred_b, dtype=np.float32)

    if pred_a.ndim == 2:
        can_a, _, _ = canonicalize_single(pred_a)
        can_b, _, _ = canonicalize_single(pred_b)
        return float(np.mean(np.linalg.norm(can_a - can_b, axis=-1)))

    elif pred_a.ndim == 3:
        errors = []
        for t in range(pred_a.shape[0]):
            can_a, _, _ = canonicalize_single(pred_a[t])
            can_b, _, _ = canonicalize_single(pred_b[t])
            errors.append(np.mean(np.linalg.norm(can_a - can_b, axis=-1)))
        return float(np.mean(errors))

    else:
        raise ValueError(f"Expected (17,3) or (T,17,3), got {pred_a.shape}")


def bone_length_stability(pred_a, pred_b):
    """
    Compare bone lengths between two cross-view predictions.

    Bone lengths are invariant to rotation and translation, so this
    metric measures whether the two views produce consistent skeletal
    proportions.

    Args:
        pred_a: (17, 3) prediction from camera A.
        pred_b: (17, 3) prediction from camera B.

    Returns:
        mean_diff: float, mean absolute bone-length difference.
        per_bone: (16,) array of per-bone absolute differences.
    """
    pred_a = np.asarray(pred_a, dtype=np.float32)
    pred_b = np.asarray(pred_b, dtype=np.float32)

    lengths_a = compute_bone_lengths(pred_a)
    lengths_b = compute_bone_lengths(pred_b)
    per_bone = np.abs(lengths_a - lengths_b)
    return float(np.mean(per_bone)), per_bone
