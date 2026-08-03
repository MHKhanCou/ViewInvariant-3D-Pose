"""
Multi-scale (per-segment) canonicalization — training-free extension.

Motivation (thesis_artifacts/multiscale_canonical_proposal.md): the global
body frame removes torso-aligned orientation variance; limb orientation
variance that differs from the torso remains. Each body segment gets its own
Gram-Schmidt frame, built the same way as the global frame, so the whole
extension stays training-free.

Levels (H36M-17 indices, see canonical/body_frame.py):
  global     all 17 joints          existing canonicalize_single
  torso      [7,8,9,10,11,14]       y: head-center_torso, x: lshoulder-rshoulder
  left_arm   [14,15,16]             y: shoulder-upper_torso, x: elbow-shoulder
  right_arm  [11,12,13]             symmetric
  left_leg   [1,2,3]                y: hip-root, x: knee-hip
  right_leg  [4,5,6]                symmetric

Degenerate segments (near-collinear axes) fall back to the GLOBAL frame for
that segment, so multi-scale can never do worse than global on a degenerate
limb — it degrades to the existing behavior.
"""

import numpy as np

from .body_frame import canonicalize_single

_EPS = 1e-8

# name -> (segment joint ids, segment root id, (y_from, y_to), (x_from, x_to))
SEGMENTS = {
    "torso":     ([7, 8, 9, 10, 11, 14], 7,  (7, 10),  (11, 14)),
    "left_arm":  ([14, 15, 16],          14, (8, 14),  (14, 15)),
    "right_arm": ([11, 12, 13],          11, (8, 11),  (11, 12)),
    "left_leg":  ([1, 2, 3],             1,  (0, 1),   (1, 2)),
    "right_leg": ([4, 5, 6],             4,  (4, 5),   (5, 6)),
}

LEVELS = ["global"] + list(SEGMENTS.keys())


def _gram_schmidt_frame(y_raw, x_raw):
    """Build orthonormal R from a vertical-ish and horizontal-ish axis; None if degenerate."""
    ny = np.linalg.norm(y_raw)
    nx = np.linalg.norm(x_raw)
    if ny < _EPS or nx < _EPS:
        return None
    y = y_raw / ny
    z = np.cross(x_raw, y)
    nz = np.linalg.norm(z)
    if nz < _EPS:
        return None
    z = z / nz
    x = np.cross(y, z)
    R = np.column_stack([x, y, z]).astype(np.float32)
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-4:
        return None
    return R


def multiscale_canonicalize(pose):
    """
    Canonicalize one (17, 3) root-relative pose at every scale level.

    Returns:
        dict level -> {
            "joints": (K, 3) canonical coordinates of the level's joints,
            "joint_ids": list of H36M joint ids,
            "valid": bool,
            "fallback": bool (True if segment frame degenerate -> global used),
        }
    """
    pose = np.asarray(pose, dtype=np.float32)
    can_global, R_global, meta_global = canonicalize_single(pose)

    out = {
        "global": {
            "joints": can_global,
            "joint_ids": list(range(17)),
            "valid": bool(meta_global["valid"]),
            "fallback": False,
        }
    }

    P = pose - pose[0:1]
    for name, (ids, root, (y0, y1), (x0, x1)) in SEGMENTS.items():
        R = _gram_schmidt_frame(P[y1] - P[y0], P[x1] - P[x0])
        fallback = R is None
        seg = P[ids] - P[root:root + 1]
        if fallback:
            joints = can_global[ids] - can_global[root:root + 1] \
                if meta_global["valid"] else np.zeros((len(ids), 3), np.float32)
        else:
            joints = seg @ R
        out[name] = {
            "joints": joints.astype(np.float32),
            "joint_ids": ids,
            "valid": bool((not fallback) or meta_global["valid"]),
            "fallback": fallback,
        }
    return out


def multiscale_cross_view_distance(pose_a, pose_b):
    """
    Per-level mean joint distance between two independently multi-scale
    canonicalized poses.

    Returns:
        dict level -> float distance (np.nan when a level is invalid in
        either view).
    """
    ms_a = multiscale_canonicalize(pose_a)
    ms_b = multiscale_canonicalize(pose_b)
    dists = {}
    for level in LEVELS:
        a, b = ms_a[level], ms_b[level]
        if a["valid"] and b["valid"]:
            dists[level] = float(np.mean(
                np.linalg.norm(a["joints"] - b["joints"], axis=1)))
        else:
            dists[level] = float("nan")
    return dists


def combined_cross_view_distance(pose_a, pose_b):
    """
    Single-number multi-scale distance: global torso frame for the trunk +
    per-limb frames for the limbs, joint-count-weighted so it is comparable
    to the 17-joint global canonical distance.
    """
    ms_a = multiscale_canonicalize(pose_a)
    ms_b = multiscale_canonicalize(pose_b)

    total, n = 0.0, 0
    # Trunk (+ root) from the global frame.
    trunk = [0, 7, 8, 9, 10]
    ga, gb = ms_a["global"], ms_b["global"]
    if not (ga["valid"] and gb["valid"]):
        return float("nan")
    d = np.linalg.norm(ga["joints"][trunk] - gb["joints"][trunk], axis=1)
    total += d.sum()
    n += len(trunk)
    # Limbs from their own frames.
    for name in ["left_arm", "right_arm", "left_leg", "right_leg"]:
        a, b = ms_a[name], ms_b[name]
        d = np.linalg.norm(a["joints"] - b["joints"], axis=1)
        total += d.sum()
        n += len(a["joint_ids"])
    return float(total / n)


if __name__ == "__main__":
    # Self-check: rigid-rotation invariance per level.
    rng = np.random.default_rng(7)
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[1] = [0.13, -0.02, 0.01];  pose[2] = [0.15, -0.45, 0.03]
    pose[3] = [0.16, -0.90, 0.02];  pose[4] = [-0.13, -0.02, -0.01]
    pose[5] = [-0.15, -0.44, 0.02]; pose[6] = [-0.17, -0.89, 0.01]
    pose[7] = [0.01, 0.25, 0.0];    pose[8] = [0.02, 0.50, -0.02]
    pose[9] = [0.02, 0.62, 0.0];    pose[10] = [0.03, 0.75, 0.02]
    pose[11] = [-0.18, 0.48, 0.0];  pose[12] = [-0.30, 0.25, 0.05]
    pose[13] = [-0.35, 0.05, 0.1];  pose[14] = [0.19, 0.49, 0.0]
    pose[15] = [0.32, 0.27, 0.04];  pose[16] = [0.37, 0.06, 0.09]

    A = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(A)
    R_rand = (U @ Vt).astype(np.float32)
    if np.linalg.det(R_rand) < 0:
        R_rand[:, 0] *= -1
    rotated = pose @ R_rand.T

    d = multiscale_cross_view_distance(pose, rotated)
    for level, dist in d.items():
        assert dist < 1e-5, f"{level} not rotation-invariant: {dist}"
    c = combined_cross_view_distance(pose, rotated)
    assert c < 1e-5, f"combined not rotation-invariant: {c}"
    print("multiscale.py self-checks passed:",
          {k: f"{v:.2e}" for k, v in d.items()})
