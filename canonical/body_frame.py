"""
Body-frame canonicalization for 3D human poses.

Constructs a body-fixed orthonormal coordinate system from the predicted
root-relative 3D joints and re-expresses all joints in that frame.

H36M-17 joint layout. The left/right assignment below follows the COCO-to-H36M
mapping in demo/lib/preprocess.py:5-6, which is authoritative for this repository.
Earlier versions of this docstring had the sides inverted; the arithmetic was
never affected, since every use is either side-symmetric or pairs the two sides
consistently, but the labels were wrong.

    0  root/pelvis
    1  right hip      2  right knee      3  right foot
    4  left hip       5  left knee       6  left foot
    7  center torso   8  upper torso     9  neck
    10 head           11 left shoulder   12 left elbow    13 left hand
    14 right shoulder 15 right elbow     16 right hand
"""

import numpy as np

_EPS = 1e-8

# H36M-17 bone connections for bone-length computation.
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8),                # spine
    (8, 9), (9, 10),               # neck/head
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
]


def _normalize(v):
    """Normalize a vector; return zero vector if degenerate."""
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.zeros_like(v)
    return v / n


def canonicalize_single(pose, prev_z=None):
    """
    Canonicalize a single root-relative 3D pose into a body-fixed frame.

    Args:
        pose: (17, 3) float32 array of root-relative 3D joint positions.
              Root (joint 0) should be at or near the origin.
        prev_z: (3,) optional previous frame's z-axis for temporal sign
                consistency.  If None, no flip is applied.

    Returns:
        canonical_pose: (17, 3) float32 array in the canonical frame.
        R: (3, 3) float32 rotation matrix (columns = body-frame axes in
           original coordinates).
        metadata: dict with keys:
            - "valid" (bool): False if degenerate input.
            - "y_axis" (3,): body vertical axis.
            - "z_axis" (3,): body forward axis.
            - "flipped" (bool): whether temporal flip was applied.
    """
    pose = np.asarray(pose, dtype=np.float32)
    assert pose.shape == (17, 3), f"Expected (17, 3), got {pose.shape}"

    # --- Defensive root subtraction ---
    P_rel = pose - pose[0:1]

    # --- Body vertical axis: upper torso - root ---
    y_raw = P_rel[8] - P_rel[0]
    y_body = _normalize(y_raw)
    if np.linalg.norm(y_raw) < _EPS:
        # Fallback: root -> neck
        y_raw = P_rel[9] - P_rel[0]
        y_body = _normalize(y_raw)

    # --- Horizontal axis: hip-to-hip ---
    x_raw = P_rel[1] - P_rel[4]
    if np.linalg.norm(x_raw) < _EPS:
        # Fallback: left shoulder - right shoulder
        x_raw = P_rel[14] - P_rel[11]

    # --- Degenerate check: axes too small ---
    if np.linalg.norm(y_body) < _EPS or np.linalg.norm(x_raw) < _EPS:
        return (
            np.zeros((17, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            {"valid": False, "y_axis": np.zeros(3), "z_axis": np.zeros(3),
             "flipped": False},
        )

    # --- Gram-Schmidt orthonormalization ---
    z_body = _normalize(np.cross(x_raw, y_body))
    x_body = _normalize(np.cross(y_body, z_body))

    # --- Degeneracy check: collinear axes produce zero cross products ---
    # If torso and hip axes are collinear (or nearly so), z_body and/or x_body
    # will be near-zero after normalization, and R will not be orthonormal.
    if np.linalg.norm(z_body) < _EPS or np.linalg.norm(x_body) < _EPS:
        return (
            np.zeros((17, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            {"valid": False, "y_axis": y_body.astype(np.float32),
             "z_axis": np.zeros(3), "flipped": False},
        )

    # --- Verify orthonormality of R ---
    R = np.column_stack([x_body, y_body, z_body]).astype(np.float32)
    ortho_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if ortho_error > 1e-4:
        return (
            np.zeros((17, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            {"valid": False, "y_axis": y_body.astype(np.float32),
             "z_axis": z_body.astype(np.float32), "flipped": False},
        )

    # --- Temporal sign consistency ---
    flipped = False
    if prev_z is not None:
        if np.dot(z_body, prev_z) < 0:
            z_body = -z_body
            x_body = -x_body
            flipped = True

    # --- Rotation matrix: columns are body-frame axes ---
    R = np.column_stack([x_body, y_body, z_body]).astype(np.float32)

    # --- Canonical coordinates ---
    canonical_pose = P_rel @ R

    metadata = {
        "valid": True,
        "y_axis": y_body.astype(np.float32),
        "z_axis": z_body.astype(np.float32),
        "flipped": flipped,
    }
    return canonical_pose, R, metadata


def canonicalize_batch(poses, prev_z=None):
    """
    Canonicalize a sequence of root-relative 3D poses.

    Args:
        poses: (T, 17, 3) float32 array.
        prev_z: (3,) optional initial z-axis for temporal consistency.

    Returns:
        canonical_poses: (T, 17, 3) float32 array.
        R_matrices: (T, 3, 3) float32 array of rotation matrices.
        metadata_list: list of T metadata dicts.
    """
    poses = np.asarray(poses, dtype=np.float32)
    assert poses.ndim == 3 and poses.shape[1] == 17 and poses.shape[2] == 3, \
        f"Expected (T, 17, 3), got {poses.shape}"

    T = poses.shape[0]
    canonical_poses = np.zeros_like(poses)
    R_matrices = np.zeros((T, 3, 3), dtype=np.float32)
    metadata_list = []
    current_z = prev_z

    for t in range(T):
        can, R, meta = canonicalize_single(poses[t], prev_z=current_z)
        canonical_poses[t] = can
        R_matrices[t] = R
        metadata_list.append(meta)
        if meta["valid"]:
            current_z = meta["z_axis"]

    return canonical_poses, R_matrices, metadata_list


def compute_bone_lengths(pose):
    """
    Compute bone lengths for a single (17, 3) pose.

    Returns:
        bone_lengths: (16,) array of Euclidean distances for each bone.
    """
    pose = np.asarray(pose, dtype=np.float32)
    lengths = np.zeros(len(BONE_CONNECTIONS), dtype=np.float32)
    for i, (a, b) in enumerate(BONE_CONNECTIONS):
        lengths[i] = np.linalg.norm(pose[b] - pose[a])
    return lengths
