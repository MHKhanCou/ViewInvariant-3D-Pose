"""
Unit tests for canonical pose representation.

Uses Python unittest (no pytest dependency).
Run: python -m unittest canonical.test_canonical -v
"""

import unittest
import numpy as np
from canonical.body_frame import canonicalize_single, canonicalize_batch, compute_bone_lengths
from canonical.canonicalizer import Canonicalizer
from canonical.metrics import canonical_mpjpe, cross_view_consistency_error, bone_length_stability


def _make_synthetic_pose():
    """Create a valid synthetic human pose (standing, arms at sides)."""
    pose = np.zeros((17, 3), dtype=np.float32)
    # Torso
    pose[0] = [0, 0, 0]       # root/pelvis
    pose[7] = [0, 0.4, 0]     # center torso
    pose[8] = [0, 0.8, 0]     # upper torso
    pose[9] = [0, 1.0, 0]     # neck
    pose[10] = [0, 1.15, 0]   # head
    # Left leg
    pose[1] = [-0.1, 0, 0]    # left hip
    pose[2] = [-0.1, -0.4, 0] # left knee
    pose[3] = [-0.1, -0.8, 0] # left foot
    # Right leg
    pose[4] = [0.1, 0, 0]     # right hip
    pose[5] = [0.1, -0.4, 0]  # right knee
    pose[6] = [0.1, -0.8, 0]  # right foot
    # Left arm
    pose[14] = [-0.2, 0.75, 0] # left shoulder
    pose[15] = [-0.2, 0.4, 0]  # left elbow
    pose[16] = [-0.2, 0.1, 0]  # left hand
    # Right arm
    pose[11] = [0.2, 0.75, 0]  # right shoulder
    pose[12] = [0.2, 0.4, 0]   # right elbow
    pose[13] = [0.2, 0.1, 0]   # right hand
    return pose


def _random_rotation():
    """Generate a random 3x3 rotation matrix via QR decomposition."""
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.astype(np.float32)


class TestShape(unittest.TestCase):
    def test_single_frame(self):
        pose = _make_synthetic_pose()
        can, R, meta = canonicalize_single(pose)
        self.assertEqual(can.shape, (17, 3))
        self.assertEqual(R.shape, (3, 3))
        self.assertIsInstance(meta, dict)

    def test_batch(self):
        T = 5
        poses = np.stack([_make_synthetic_pose() for _ in range(T)])
        can, R_metas, meta_list = canonicalize_batch(poses)
        self.assertEqual(can.shape, (T, 17, 3))
        self.assertEqual(R_metas.shape, (T, 3, 3))
        self.assertEqual(len(meta_list), T)


class TestRoot(unittest.TestCase):
    def test_root_is_zero(self):
        pose = _make_synthetic_pose()
        can, _, _ = canonicalize_single(pose)
        self.assertLess(np.linalg.norm(can[0]), 1e-6)


class TestOrthonormality(unittest.TestCase):
    def test_R_is_orthonormal(self):
        pose = _make_synthetic_pose()
        _, R, _ = canonicalize_single(pose)
        product = R.T @ R
        np.testing.assert_allclose(product, np.eye(3), atol=1e-6)

    def test_det_is_one(self):
        pose = _make_synthetic_pose()
        _, R, _ = canonicalize_single(pose)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=5)


class TestRigidRotationInvariance(unittest.TestCase):
    def test_rotated_poses_canonicalize_to_same(self):
        pose = _make_synthetic_pose()
        can_orig, _, _ = canonicalize_single(pose)

        # Compute pose spatial extent for normalization
        pose_extent = float(np.max(pose) - np.min(pose))
        self.assertGreater(pose_extent, 0, "Pose has zero extent")

        max_raw_error = 0.0
        for _ in range(10):
            R_rand = _random_rotation()
            rotated = (R_rand @ pose.T).T  # (17, 3)
            can_rot, _, meta = canonicalize_single(rotated)
            self.assertTrue(meta["valid"])
            raw_error = float(np.max(np.abs(can_rot - can_orig)))
            max_raw_error = max(max_raw_error, raw_error)

        # Relative error: error / pose extent (dimensionless ratio)
        relative_error = max_raw_error / pose_extent

        # Both should be very small
        self.assertLess(max_raw_error, 1e-4,
                        f"Raw canonical error too large: {max_raw_error}")
        self.assertLess(relative_error, 1e-4,
                        f"Relative canonical error too large: {relative_error}")


class TestDegenerateInput(unittest.TestCase):
    def test_all_zero_pose(self):
        pose = np.zeros((17, 3), dtype=np.float32)
        can, R, meta = canonicalize_single(pose)
        np.testing.assert_array_equal(can, np.zeros((17, 3)))
        np.testing.assert_array_equal(R, np.eye(3))
        self.assertFalse(meta["valid"])

    def test_nearly_zero_pose(self):
        pose = np.full((17, 3), 1e-12, dtype=np.float32)
        can, R, meta = canonicalize_single(pose)
        self.assertFalse(meta["valid"])
        self.assertFalse(np.any(np.isnan(can)))
        self.assertFalse(np.any(np.isinf(can)))

    def test_collinear_torso_hip_rejected(self):
        """When torso and hip axes are collinear, canonicalization should be rejected."""
        pose = _make_synthetic_pose()
        # Make hip axis parallel to torso axis by placing both hips on the
        # vertical line through the root (x=0 for all joints).
        pose[1] = [0, -0.1, 0]   # left hip on vertical axis
        pose[4] = [0, 0.1, 0]    # right hip on vertical axis
        # Now x_raw = P[1] - P[4] = [0, -0.2, 0] which is parallel to y_body
        can, R, meta = canonicalize_single(pose)
        self.assertFalse(meta["valid"],
                         "Collinear torso/hip axes should be rejected")
        np.testing.assert_array_equal(can, np.zeros((17, 3)))
        np.testing.assert_array_equal(R, np.eye(3))


class TestTemporalConsistency(unittest.TestCase):
    def test_repeated_frames_no_flip(self):
        pose = _make_synthetic_pose()
        can1, _, meta1 = canonicalize_single(pose, prev_z=None)
        z1 = meta1["z_axis"]

        can2, _, meta2 = canonicalize_single(pose, prev_z=z1)
        self.assertFalse(meta2["flipped"])
        np.testing.assert_allclose(can1, can2, atol=1e-6)


class TestCanonicalizer(unittest.TestCase):
    def test_single_frame(self):
        can = Canonicalizer()
        pose = _make_synthetic_pose()
        result, R = can(pose)
        self.assertEqual(result.shape, (17, 3))

    def test_batch(self):
        can = Canonicalizer()
        T = 5
        poses = np.stack([_make_synthetic_pose() for _ in range(T)])
        result, R_metas = can(poses)
        self.assertEqual(result.shape, (T, 17, 3))

    def test_reset(self):
        can = Canonicalizer()
        pose = _make_synthetic_pose()
        can(pose)
        self.assertIsNotNone(can._prev_z)
        can.reset()
        self.assertIsNone(can._prev_z)


class TestMetrics(unittest.TestCase):
    def test_canonical_mpjpe_identical(self):
        pose = _make_synthetic_pose()
        err = canonical_mpjpe(pose, pose)
        self.assertLess(err, 1e-5)

    def test_canonical_mpjpe_batch(self):
        T = 3
        poses = np.stack([_make_synthetic_pose() for _ in range(T)])
        err = canonical_mpjpe(poses, poses)
        self.assertLess(err, 1e-5)

    def test_cross_view_consistency(self):
        pose = _make_synthetic_pose()
        err = cross_view_consistency_error(pose, pose)
        self.assertLess(err, 1e-5)

    def test_bone_length_stability_identical(self):
        pose = _make_synthetic_pose()
        mean_diff, per_bone = bone_length_stability(pose, pose)
        self.assertLess(mean_diff, 1e-5)
        self.assertEqual(per_bone.shape, (16,))


class TestBoneLengths(unittest.TestCase):
    def test_bone_lengths(self):
        pose = _make_synthetic_pose()
        lengths = compute_bone_lengths(pose)
        self.assertEqual(lengths.shape, (16,))
        self.assertTrue(np.all(lengths >= 0))
        # Left leg should have non-zero length
        self.assertGreater(lengths[0], 0)


if __name__ == "__main__":
    unittest.main()
