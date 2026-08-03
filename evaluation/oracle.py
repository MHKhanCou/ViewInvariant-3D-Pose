"""
Procrustes (Kabsch) alignment — oracle baseline for cross-view evaluation.

The oracle answers: "if we were allowed the optimal rigid rotation per frame
pair (which requires knowing both poses jointly — information the training-free
canonicalizer does NOT have), how small could the cross-view distance get?"

Canonical distance is lower-bounded by this oracle. Reporting
raw / canonical / oracle situates the 28.4%-style reduction against the
best-achievable rigid alignment.

Ported from the retired synthetic evaluation package (E:/thesis/evaluation/oracle.py).
"""

import numpy as np


def procrustes_align(source, target):
    """
    Optimal rigid alignment (rotation + translation, no scaling) of source
    onto target via the Kabsch algorithm.

    Args:
        source: (N, 3) point cloud.
        target: (N, 3) point cloud.

    Returns:
        R: (3, 3) rotation matrix (proper, det=+1).
        t: (3,) translation.
        error: float, mean squared error after alignment.
    """
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T).astype(np.float32)

    # Reflection guard: force a proper rotation.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = (Vt.T @ U.T).astype(np.float32)

    t = (target_mean - source_mean @ R.T).astype(np.float32)

    aligned = source_centered @ R.T + target_mean
    error = float(np.mean((aligned - target) ** 2))

    return R, t, error


def procrustes_cross_view_distance(pose_a, pose_b):
    """
    Oracle cross-view joint distance: align pose_a onto pose_b with the
    optimal rigid transform, then measure mean per-joint distance.

    This is the floor for any rotation-based normalization of the pair.

    Args:
        pose_a, pose_b: (17, 3) root-relative poses from two cameras.

    Returns:
        distance: float, mean per-joint L2 after optimal alignment.
    """
    pose_a = np.asarray(pose_a, dtype=np.float32)
    pose_b = np.asarray(pose_b, dtype=np.float32)

    R, _, _ = procrustes_align(pose_a, pose_b)
    aligned_a = (pose_a - pose_a.mean(axis=0)) @ R.T + pose_b.mean(axis=0)
    return float(np.mean(np.linalg.norm(aligned_a - pose_b, axis=1)))


if __name__ == "__main__":
    # Self-check: identity, known rotation, reflection guard.
    rng = np.random.default_rng(0)
    P = rng.standard_normal((17, 3)).astype(np.float32)

    R, t, err = procrustes_align(P, P)
    assert np.allclose(R, np.eye(3), atol=1e-4), "identity alignment failed"
    assert err < 1e-8, "identity error nonzero"

    angle = np.pi / 5
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    Q = P @ R_true.T
    assert procrustes_cross_view_distance(P, Q) < 1e-4, "rotation recovery failed"
    R2, _, _ = procrustes_align(P, Q)
    assert np.linalg.det(R2) > 0.99, "improper rotation"

    print("oracle.py self-checks passed")
