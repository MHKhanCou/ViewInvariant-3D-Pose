"""
Reliability score for geometric canonical body-frame normalization.

Estimates whether the body frame axes (torso, hip) are sufficiently
reliable for canonicalization. When reliability is low, canonicalization
is likely to produce incorrect results.

The score has two layers:
1. Hard geometric gates â€” explicit rejection for degenerate configurations.
2. Continuous reliability score â€” geometric mean of normalized components
   for ranking and coverage-error analysis.

This is a deterministic, training-free, post-processing metric.
"""

import numpy as np
from canonical.body_frame import canonicalize_single


# H36M-17 skeleton connections (16 bones) with correct parent hierarchy.
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),   # Root -> R_Hip -> R_Knee -> R_Ankle
    (0, 4), (4, 5), (5, 6),   # Root -> L_Hip -> L_Knee -> L_Ankle
    (0, 7), (7, 8),            # Root -> Spine -> Thorax
    (8, 9), (9, 10),           # Thorax -> Neck -> Head
    (8, 11), (11, 12), (12, 13),  # Thorax -> L_Shoulder -> L_Elbow -> L_Wrist
    (8, 14), (14, 15), (15, 16),  # Thorax -> R_Shoulder -> R_Elbow -> R_Wrist
]

# Bilateral bone pairs: (right_bone, left_bone) as parent->child.
BONE_SYMMETRY_PAIRS = [
    ((0, 1), (0, 4)),   # R_Hip vs L_Hip
    ((1, 2), (4, 5)),   # R_Knee vs L_Knee
    ((2, 3), (5, 6)),   # R_Ankle vs L_Ankle
    ((8, 11), (8, 14)), # R_Shoulder vs L_Shoulder
    ((11, 12), (14, 15)), # R_Elbow vs L_Elbow
    ((12, 13), (15, 16)), # R_Wrist vs L_Wrist
]

# COCO-17 indices for key structural joints (used for detector confidence).
# COCO order: nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4,
#   l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
#   l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
#   l_knee=13, r_knee=14, l_ankle=15, r_ankle=16
COCO_KEY_JOINTS = [0, 5, 6, 11, 12, 13, 14]  # nose, shoulders, hips, knees


def compute_bone_lengths(pose3d):
    """Compute 16 bone lengths from H36M-17 skeleton."""
    P = pose3d.astype(np.float64)
    return np.array([np.linalg.norm(P[i] - P[j]) for i, j in BONE_CONNECTIONS])


def compute_reliability_score(pose3d, scores_2d=None, prev_rotation=None):
    """
    Compute a reliability score for geometric canonicalization of a single pose.

    Args:
        pose3d: (17, 3) raw root-relative 3D pose from the model.
        scores_2d: (17,) 2D confidence scores in COCO-17 order (from YOLOv8).
                    Key structural joints are indexed as:
                    nose(0), l_shoulder(5), r_shoulder(6), l_hip(11),
                    r_hip(12), l_knee(13), r_knee(14).
                    Do NOT pass H36M-17 ordered scores.
        prev_rotation: (3, 3) previous frame's orthonormal rotation matrix
                       (from canonicalize_single()). Video only.

    Returns:
        reliability: float in [0, 1]
        components: dict of component scores
        metadata: dict with body frame info
    """
    pose = pose3d.astype(np.float64)
    P_rel = pose - pose[0:1]  # Root-center

    # --- Compute body frame axes ---
    y_raw = P_rel[8] - P_rel[0]   # Torso: thorax - root
    x_raw = P_rel[1] - P_rel[4]   # Hip: R_Hip - L_Hip
    y_norm = np.linalg.norm(y_raw)
    x_norm = np.linalg.norm(x_raw)

    # Reference scale from median bone length
    bone_lengths = compute_bone_lengths(P_rel)
    median_bl = float(np.median(bone_lengths)) if len(bone_lengths) > 0 else 1.0
    if median_bl < 1e-8:
        median_bl = 1.0

    components = {}

    # 1. Axis conditioning
    y_rel = y_norm / median_bl
    x_rel = x_norm / median_bl
    y_score = max(0, 1.0 - max(0, 0.1 - y_rel) / 0.1) * max(0, 1.0 - max(0, y_rel - 3.0) / 3.0)
    x_score = max(0, 1.0 - max(0, 0.1 - x_rel) / 0.1) * max(0, 1.0 - max(0, x_rel - 3.0) / 3.0)
    components['axis_conditioning'] = float(np.clip(y_score * x_score, 0, 1))

    # 2. Torso-hip orthogonality
    if y_norm > 1e-8 and x_norm > 1e-8:
        cos_angle = np.dot(y_raw, x_raw) / (y_norm * x_norm)
        sin_angle = np.sqrt(max(0, 1 - cos_angle ** 2))
        components['torso_hip_angle'] = float(np.clip(sin_angle, 0, 1))
    else:
        components['torso_hip_angle'] = 0.0

    # 3. Bilateral bone-length symmetry
    symmetry_scores = []
    for (ri, rj), (li, lj) in BONE_SYMMETRY_PAIRS:
        r_len = np.linalg.norm(P_rel[ri] - P_rel[rj])
        l_len = np.linalg.norm(P_rel[li] - P_rel[lj])
        if max(r_len, l_len) > 1e-8:
            ratio = min(r_len, l_len) / max(r_len, l_len)
            symmetry_scores.append(ratio)

    if symmetry_scores:
        mean_sym = float(np.mean(symmetry_scores))
        min_sym = float(np.min(symmetry_scores))
        components['bilateral_symmetry'] = mean_sym * (0.5 + 0.5 * min_sym)
    else:
        components['bilateral_symmetry'] = 0.0

    # 4. Abnormal bone-length ratio (tighter bounds)
    abnormal_count = 0
    for bl in bone_lengths:
        if bl < median_bl * 0.3 or bl > median_bl * 2.5:
            abnormal_count += 1
    components['abnormal_bone_ratio'] = float(1.0 - abnormal_count / len(bone_lengths))

    # 5. Detector confidence â€” COCO-17 key joints
    if scores_2d is not None and len(scores_2d) >= 17:
        # Use minimum confidence among structural joints (COCO-17 indices)
        key_scores = scores_2d[COCO_KEY_JOINTS]
        components['detector_confidence'] = float(np.min(key_scores))
    else:
        components['detector_confidence'] = 0.8

    # 6. Temporal stability (video only) â€” uses canonicalize_single() for proper SO(3)
    if prev_rotation is not None:
        try:
            pose_f32 = pose3d.astype(np.float32)
            # Use previous z-axis for temporal sign consistency
            prev_z = prev_rotation[:, 2] if prev_rotation is not None else None
            _, R_current, meta = canonicalize_single(pose_f32, prev_z=prev_z)

            if meta['valid']:
                # Geodesic angle on SO(3)
                R_rel = R_current.astype(np.float64) @ prev_rotation.T
                cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                components['temporal_stability'] = float(np.clip(1.0 - angle / 0.52, 0, 1))
            else:
                components['temporal_stability'] = 0.0
        except Exception:
            components['temporal_stability'] = 0.5
    else:
        components['temporal_stability'] = 1.0  # First frame

    # --- Final score: geometric mean ---
    values = list(components.values())
    if all(v > 0 for v in values):
        reliability = float(np.exp(np.mean(np.log(values))))
    else:
        reliability = 0.0

    metadata = {
        'y_norm': float(y_norm),
        'x_norm': float(x_norm),
        'median_bone_length': median_bl,
        'axes_valid': y_norm > 1e-8 and x_norm > 1e-8,
        'components': components,
    }

    return reliability, components, metadata


def has_hard_geometric_failure(pose3d):
    """
    Check for hard geometric failures that should always trigger abstention.
    These are structural degeneracies that no continuous score can compensate.

    Returns:
        (abstain: bool, reason: str)
    """
    pose = pose3d.astype(np.float64)
    P_rel = pose - pose[0:1]

    # Check 1: Body axes are too short
    y_norm = np.linalg.norm(P_rel[8] - P_rel[0])
    x_norm = np.linalg.norm(P_rel[1] - P_rel[4])

    bone_lengths = compute_bone_lengths(P_rel)
    median_bl = float(np.median(bone_lengths)) if len(bone_lengths) > 0 else 1.0
    if median_bl < 1e-8:
        median_bl = 1.0

    if y_norm < median_bl * 0.05:
        return True, "torso_axis_too_short"
    if x_norm < median_bl * 0.05:
        return True, "hip_axis_too_short"

    # Check 2: Axes are nearly collinear
    if y_norm > 1e-8 and x_norm > 1e-8:
        cos_angle = abs(np.dot(P_rel[8] - P_rel[0], P_rel[1] - P_rel[4]) / (y_norm * x_norm))
        if cos_angle > 0.95:
            return True, "axes_nearly_collinear"

    # Check 3: More than half of bones are degenerate
    degenerate_bones = np.sum(bone_lengths < median_bl * 0.05)
    if degenerate_bones > len(bone_lengths) // 2:
        return True, "majority_bones_degenerate"

    return False, ""


def should_abstain(reliability, hard_failure=False):
    """
    Determine if canonicalization should be skipped.

    Uses hard geometric gates first, then the continuous score.
    """
    if hard_failure:
        return True
    return reliability < 0.5

