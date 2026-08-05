"""Tests for the non-constructor joint masking.

Stubs only. These pin the mask itself and the one metric property the
pre-registration relies on. They deliberately do NOT test whether the
recomputed improvement is smaller than the seventeen-joint figure: that is the
result, not a specification, and asserting it would make a disconfirming outcome
look like a broken mask.
"""
import unittest

import numpy as np

from evaluation.h36m_noncon import CONSTRUCTOR_JOINTS, RETAINED_JOINTS
from evaluation.metrics import cross_view_joint_distance_sequence
from evaluation.oracle import procrustes_cross_view_distance


class TestMask(unittest.TestCase):
    def test_constructor_set_is_the_frame_equation(self):
        """{0,1,4,8} are exactly the joints in y = P[8]-P[0], x = P[1]-P[4]."""
        self.assertEqual(set(CONSTRUCTOR_JOINTS), {0, 1, 4, 8})

    def test_thirteen_retained_and_disjoint(self):
        self.assertEqual(len(RETAINED_JOINTS), 13)
        self.assertEqual(set(RETAINED_JOINTS) & set(CONSTRUCTOR_JOINTS), set())
        self.assertEqual(set(RETAINED_JOINTS) | set(CONSTRUCTOR_JOINTS),
                         set(range(17)))


class TestJointZeroNeutrality(unittest.TestCase):
    """Joint 0 is inert for the improvement ratio, and not for the oracle.

    Root-relative, P[0] is identically zero, so it adds a zero term to the raw
    and canonical distances, which are means over per-joint Euclidean distances.
    Dropping it rescales both by 17/16 and leaves their ratio unchanged.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.a = rng.normal(size=(24, 17, 3)) * 200.0
        self.b = rng.normal(size=(24, 17, 3)) * 200.0
        self.a[:, 0] = 0.0                       # root-relative by construction
        self.b[:, 0] = 0.0

    def test_improvement_ratio_is_unchanged_by_joint_zero(self):
        keep = [j for j in range(17) if j != 0]
        _, with0 = cross_view_joint_distance_sequence(self.a, self.b)
        _, without0 = cross_view_joint_distance_sequence(
            self.a[:, keep], self.b[:, keep])
        # dropping a zero term from a mean over 17 rescales by 17/16
        self.assertAlmostEqual(with0 * 17.0 / 16.0, without0, places=6)

        # and therefore any ratio of two such means is untouched
        _, can_with0 = cross_view_joint_distance_sequence(self.a, self.b * 0.5)
        _, can_without0 = cross_view_joint_distance_sequence(
            self.a[:, keep], self.b[:, keep] * 0.5)
        self.assertAlmostEqual(can_with0 / with0, can_without0 / without0,
                               places=9)

    def test_oracle_is_NOT_neutral_to_joint_zero(self):
        """procrustes_align centres on the centroid, so removing the origin moves it.

        This is why the neutrality test above is scoped to raw and canonical.
        An assertion covering the oracle would be false.
        """
        keep = [j for j in range(17) if j != 0]
        with0 = procrustes_cross_view_distance(self.a[0], self.b[0])
        without0 = procrustes_cross_view_distance(self.a[0][keep],
                                                  self.b[0][keep])
        self.assertNotAlmostEqual(with0 * 17.0 / 16.0, without0, places=3)


if __name__ == "__main__":
    unittest.main()
