"""Tests for the conditioning index used in the abstention experiment.

Stubs only - no dataset, no network, no prediction cache. These pin the
properties the pre-registration relies on, so a later edit that breaks them
fails here rather than silently changing a reported number.
"""
import unittest

import numpy as np

from evaluation.conditioning_abstention import conditioning


def _pose(hip_width=300.0, torso_height=500.0):
    """H36M-17 skeleton, T-pose, only the joints the index reads matter."""
    p = np.zeros((17, 3), dtype=np.float64)
    p[0] = [0.0, 0.0, 0.0]                  # root
    p[1] = [hip_width / 2, 0.0, 0.0]        # right hip
    p[4] = [-hip_width / 2, 0.0, 0.0]       # left hip
    p[8] = [0.0, torso_height, 0.0]         # thorax
    p[10] = [0.0, torso_height + 150.0, 0.0]
    p[11] = [-200.0, torso_height, 0.0]
    p[14] = [200.0, torso_height, 0.0]
    return p


class TestConditioning(unittest.TestCase):
    def test_scale_invariant(self):
        """kappa is a ratio of lengths, so uniform scaling must not change it."""
        a = conditioning(_pose())
        b = conditioning(_pose() * 2.5)
        self.assertAlmostEqual(a["kappa"], b["kappa"], places=9)
        self.assertAlmostEqual(a["cond"], b["cond"], places=9)

    def test_shorter_hip_axis_raises_kappa(self):
        """The derivation says error scales as 1/L, so short L must score worse."""
        wide = conditioning(_pose(hip_width=300.0))
        narrow = conditioning(_pose(hip_width=100.0))
        self.assertGreater(narrow["kappa"], wide["kappa"])
        self.assertGreater(narrow["cond"], wide["cond"])

    def test_rotation_invariant(self):
        """Both axes rotate together, so a rigid rotation must leave cond fixed."""
        t = 0.7
        R = np.array([[np.cos(t), 0.0, np.sin(t)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(t), 0.0, np.cos(t)]])
        a = conditioning(_pose())
        b = conditioning(_pose() @ R.T)
        self.assertAlmostEqual(a["cond"], b["cond"], places=9)

    def test_collinear_axes_are_ill_conditioned(self):
        """Hip axis parallel to torso axis: Gram-Schmidt has nothing to work with."""
        p = _pose()
        p[1] = [0.0, 100.0, 0.0]
        p[4] = [0.0, -100.0, 0.0]
        c = conditioning(p)
        self.assertLess(c["ortho"], 1e-6)
        self.assertGreater(c["cond"], conditioning(_pose())["cond"])

    def test_degenerate_pose_flagged_not_scored(self):
        """A collapsed skeleton returns inf rather than a plausible-looking number."""
        c = conditioning(np.zeros((17, 3)))
        self.assertTrue(np.isinf(c["cond"]))
        self.assertTrue(np.isinf(c["kappa"]))

    def test_orthogonal_axes_give_unit_ortho(self):
        c = conditioning(_pose())
        self.assertAlmostEqual(c["ortho"], 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
