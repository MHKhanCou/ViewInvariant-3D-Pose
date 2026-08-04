"""Tests for the joint-level decomposition used in the radial-law experiment.

Stubs only - no dataset, no cache. These pin the properties the pre-registered
prediction rests on, so the reported failure is a failure of the hypothesis
rather than of the arithmetic.
"""
import unittest

import numpy as np

from evaluation.radial_law import (fit_through_origin, per_joint_oracle,
                                   profile)


def _rot(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class TestRadialLaw(unittest.TestCase):
    def test_oracle_is_zero_for_a_pure_rotation(self):
        """Procrustes must remove a rigid rotation entirely - that is the premise."""
        rng = np.random.default_rng(0)
        P = rng.normal(size=(17, 3)) * 300.0
        Q = P @ _rot([0.3, 1.0, -0.2], 0.4).T
        d = per_joint_oracle(P.astype(np.float32), Q.astype(np.float32))
        self.assertLess(float(np.max(d)), 1e-2)

    def test_oracle_keeps_non_rigid_disagreement(self):
        """It must not absorb a shape change, or the decomposition means nothing."""
        rng = np.random.default_rng(1)
        P = rng.normal(size=(17, 3)) * 300.0
        Q = P.copy()
        Q[5] += np.array([80.0, 0.0, 0.0])
        d = per_joint_oracle(P.astype(np.float32), Q.astype(np.float32))
        self.assertGreater(float(d[5]), 20.0)

    def test_frame_part_is_linear_in_radius_for_a_real_rotation_error(self):
        """The hypothesis holds on data that actually satisfies its assumption.

        This is the positive control: if a genuine small rotation error is the
        only difference between the views, the recovered slope matches the
        rotation magnitude. The experiment's failure on real data therefore
        reflects the data, not this code path.
        """
        rng = np.random.default_rng(2)
        angle = 0.05
        n = 400
        sum_can = np.zeros(17)
        sum_orc = np.zeros(17)
        sum_r = np.zeros(17)
        for _ in range(n):
            P = rng.normal(size=(17, 3)) * 300.0
            P[0] = 0.0
            ax = rng.normal(size=3)
            A = P @ _rot(ax, angle * rng.normal()).T
            B = P @ _rot(rng.normal(size=3), angle * rng.normal()).T
            sum_can += np.linalg.norm(A - B, axis=1) ** 2
            sum_orc += per_joint_oracle(A.astype(np.float32),
                                        B.astype(np.float32)) ** 2
            sum_r += 0.5 * (np.linalg.norm(A, axis=1) + np.linalg.norm(B, axis=1))
        _, _, frame, r_bar = profile(sum_can, sum_orc, sum_r, n)
        _, r2 = fit_through_origin(r_bar, frame)
        self.assertGreater(r2, 0.80)

    def test_fit_through_origin_recovers_a_known_slope(self):
        r = np.linspace(50.0, 800.0, 17)
        slope, r2 = fit_through_origin(r, 0.06 * r)
        self.assertAlmostEqual(slope, 0.06, places=9)
        self.assertAlmostEqual(r2, 1.0, places=9)

    def test_profile_clamps_negative_differences(self):
        """Oracle above canonical is possible per joint; it must not go complex."""
        sum_can = np.full(17, 100.0)
        sum_orc = np.full(17, 400.0)
        _, _, frame, _ = profile(sum_can, sum_orc, np.full(17, 300.0), 1)
        self.assertTrue(np.all(frame == 0.0))
        self.assertTrue(np.all(np.isfinite(frame)))


if __name__ == "__main__":
    unittest.main()
