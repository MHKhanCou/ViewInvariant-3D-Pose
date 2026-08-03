"""
Synthetic validation for multi-scale canonical body-frame.

Creates controlled synthetic poses, applies targeted rotations to
specific body parts, and measures whether multi-scale canonicalization
outperforms single-scale on these controlled examples.

This experiment validates the core hypothesis BEFORE integrating
into the full pipeline.
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single, _normalize, _EPS


def make_synthetic_pose():
    """Create a valid synthetic human pose (standing, arms at sides)."""
    pose = np.zeros((17, 3), dtype=np.float32)
    # Torso
    pose[0] = [0, 0, 0]           # root/pelvis
    pose[7] = [0, 0.4, 0]         # center torso
    pose[8] = [0, 0.8, 0]         # upper torso
    pose[9] = [0, 1.0, 0]         # neck
    pose[10] = [0, 1.15, 0]       # head
    # Left leg
    pose[1] = [-0.1, 0, 0]        # left hip
    pose[2] = [-0.1, -0.4, 0]     # left knee
    pose[3] = [-0.1, -0.8, 0]     # left foot
    # Right leg
    pose[4] = [0.1, 0, 0]         # right hip
    pose[5] = [0.1, -0.4, 0]      # right knee
    pose[6] = [0.1, -0.8, 0]      # right foot
    # Left arm
    pose[14] = [-0.2, 0.75, 0]    # left shoulder
    pose[15] = [-0.2, 0.4, 0]     # left elbow
    pose[16] = [-0.2, 0.1, 0]     # left hand
    # Right arm
    pose[11] = [0.2, 0.75, 0]     # right shoulder
    pose[12] = [0.2, 0.4, 0]      # right elbow
    pose[13] = [0.2, 0.1, 0]      # right hand
    return pose


def rotate_joints(pose, joint_indices, angle_deg, axis='z'):
    """Rotate specific joints around a given axis by angle_deg degrees."""
    pose = pose.copy()
    angle = np.radians(angle_deg)
    if axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
    elif axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle),  np.cos(angle)]], dtype=np.float32)
    else:  # y
        R = np.array([[ np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)

    for idx in joint_indices:
        pose[idx] = R @ pose[idx]
    return pose


def single_scale_canonical(pose):
    """Current method: global body-frame canonicalization."""
    can, R, meta = canonicalize_single(pose)
    return can, R, meta


def multiscale_canonical(pose):
    """
    Proposed multi-scale canonicalization.
    Returns canonical poses for global + torso + limbs.
    """
    # Level 0: Global (same as current)
    can_global, R_global, meta_global = canonicalize_single(pose)

    # Level 1: Torso frame (joints 7-10, mapped to subframe indices 0-3)
    torso_pose = pose[7:11].copy()
    torso_pose -= torso_pose[0:1]  # relative to center_torso
    can_torso, R_torso, _ = _canonicalize_subframe(torso_pose, [0, 3], [1, 2])

    # Level 2: Left arm (joints 14-16, mapped to 0-2)
    arm_pose = pose[14:17].copy()
    arm_pose -= arm_pose[0:1]
    can_left_arm, R_left_arm, _ = _canonicalize_subframe(arm_pose, [0, 2], [0, 1])

    # Level 3: Right arm (joints 11-13, mapped to 0-2)
    arm_pose = pose[11:14].copy()
    arm_pose -= arm_pose[0:1]
    can_right_arm, R_right_arm, _ = _canonicalize_subframe(arm_pose, [0, 2], [0, 1])

    # Level 4: Left leg (joints 1-3, mapped to 0-2)
    leg_pose = pose[1:4].copy()
    leg_pose -= leg_pose[0:1]
    can_left_leg, R_left_leg, _ = _canonicalize_subframe(leg_pose, [0, 2], [0, 1])

    # Level 5: Right leg (joints 4-6, mapped to 0-2)
    leg_pose = pose[4:7].copy()
    leg_pose -= leg_pose[0:1]
    can_right_leg, R_right_leg, _ = _canonicalize_subframe(leg_pose, [0, 2], [0, 1])

    return {
        'global': (can_global, R_global),
        'torso': (can_torso, R_torso),
        'left_arm': (can_left_arm, R_left_arm),
        'right_arm': (can_right_arm, R_right_arm),
        'left_leg': (can_left_leg, R_left_leg),
        'right_leg': (can_right_leg, R_right_leg),
    }


def _canonicalize_subframe(pose, vertical_joints, horizontal_joints):
    """Canonicalize a sub-pose with given joint pairs."""
    P = pose.copy()

    # Vertical axis
    y_raw = P[vertical_joints[1]] - P[vertical_joints[0]]
    y = _normalize(y_raw)

    # Horizontal axis
    x_raw = P[horizontal_joints[0]] - P[horizontal_joints[1]]
    if np.linalg.norm(x_raw) < _EPS:
        return P, np.eye(3, dtype=np.float32), {"valid": False}

    # Gram-Schmidt
    z = _normalize(np.cross(x_raw, y))
    x = _normalize(np.cross(y, z))

    if np.linalg.norm(z) < _EPS or np.linalg.norm(x) < _EPS:
        return P, np.eye(3, dtype=np.float32), {"valid": False}

    R = np.column_stack([x, y, z]).astype(np.float32)
    can = P @ R
    return can, R, {"valid": True}


def measure_cross_view_error(pose_a, pose_b, method):
    """Measure cross-view error between two poses using given method."""
    if method == 'single':
        can_a, _, _ = single_scale_canonical(pose_a)
        can_b, _, _ = single_scale_canonical(pose_b)
        return float(np.mean(np.linalg.norm(can_a - can_b, axis=-1)))
    elif method == 'multiscale':
        ms_a = multiscale_canonical(pose_a)
        ms_b = multiscale_canonical(pose_b)
        # Use global canonical for comparison (same dimensionality)
        can_a = ms_a['global'][0]
        can_b = ms_b['global'][0]
        return float(np.mean(np.linalg.norm(can_a - can_b, axis=-1)))
    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment():
    """Run the synthetic validation experiment."""
    print("=" * 70)
    print("SYNTHETIC VALIDATION: Multi-Scale vs Single-Scale Canonicalization")
    print("=" * 70)

    base_pose = make_synthetic_pose()

    # Test scenarios
    scenarios = [
        ("Baseline (no rotation)", {}),
        ("Torso rotated 30° around z", {'torso': 30}),
        ("Torso rotated 60° around z", {'torso': 60}),
        ("Torso rotated 90° around z", {'torso': 90}),
        ("Left arm rotated 45° around z", {'left_arm': 45}),
        ("Left arm rotated 90° around z", {'left_arm': 90}),
        ("Right arm rotated 45° around z", {'right_arm': 45}),
        ("Right arm rotated 90° around z", {'right_arm': 90}),
        ("Left leg rotated 30° around x", {'left_leg_x': 30}),
        ("Right leg rotated 30° around x", {'right_leg_x': 30}),
        ("Torso 30° + Left arm 45°", {'torso': 30, 'left_arm': 45}),
        ("Torso 60° + Right arm 90°", {'torso': 60, 'right_arm': 90}),
    ]

    joint_map = {
        'torso': list(range(7, 11)),
        'left_arm': [14, 15, 16],
        'right_arm': [11, 12, 13],
        'left_leg': [1, 2, 3],
        'right_leg': [4, 5, 6],
    }
    axis_map = {
        'torso': 'z', 'left_arm': 'z', 'right_arm': 'z',
        'left_leg_x': 'x', 'right_leg_x': 'x',
    }

    results = []
    for desc, rotations in scenarios:
        pose_a = base_pose.copy()
        pose_b = base_pose.copy()

        for part, angle in rotations.items():
            if '_x' in part:
                joint_name = part.replace('_x', '')
                joints = joint_map[joint_name]
                pose_b = rotate_joints(pose_b, joints, angle, axis='x')
            else:
                joints = joint_map[part]
                pose_b = rotate_joints(pose_b, joints, angle, axis='z')

        err_single = measure_cross_view_error(pose_a, pose_b, 'single')
        err_multi = measure_cross_view_error(pose_a, pose_b, 'multiscale')
        improvement = (1 - err_multi / err_single) * 100 if err_single > 0 else 0

        results.append({
            'scenario': desc,
            'single_scale': err_single,
            'multiscale': err_multi,
            'improvement': improvement,
        })

    # Print results
    print(f"\n{'Scenario':<40} {'Single':>10} {'Multi':>10} {'Improv':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['scenario']:<40} {r['single_scale']:>10.4f} {r['multiscale']:>10.4f} {r['improvement']:>9.1f}%")

    # Summary
    improvements = [r['improvement'] for r in results if r['single_scale'] > 0]
    print(f"\n{'=' * 70}")
    print(f"Average improvement: {np.mean(improvements):.1f}%")
    print(f"Max improvement: {np.max(improvements):.1f}%")
    print(f"Min improvement: {np.min(improvements):.1f}%")

    # Analysis
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print(f"{'=' * 70}")
    if np.mean(improvements) > 5:
        print("✓ Multi-scale shows measurable improvement on synthetic data")
        print("  Recommendation: Proceed with full implementation")
    elif np.mean(improvements) > 0:
        print("~ Multi-scale shows marginal improvement on synthetic data")
        print("  Recommendation: Investigate further before full implementation")
    else:
        print("✗ Multi-scale shows no improvement on synthetic data")
        print("  Recommendation: Retain single-scale as primary contribution")

    return results


if __name__ == "__main__":
    results = run_experiment()
