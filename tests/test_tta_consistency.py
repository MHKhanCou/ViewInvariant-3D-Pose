"""
Tests for TTA dispersion (5 tests). Stub model + stub detector: no weights,
no dataset, runs anywhere.

The first two are the behaviour-preservation gate for the additive flags on
`lift_from_coco_window` and `detect_with_rotation`.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.lifter import N_FRAMES
from evaluation.lifting import lift_from_coco_window
from evaluation.tta_consistency import (
    disp_axis_split, disp_procrustes, disp_scale, parse_key,
)


class _StubModel:
    """Deterministic stand-in for MotionAGFormer: (B,T,17,2) -> (B,T,17,3)."""

    def __init__(self, bias=0.0):
        self.bias = bias

    def __call__(self, x):
        import torch
        arr = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        b, t, j, _ = arr.shape          # input is (B,T,17,3): x, y, confidence
        out = np.zeros((b, t, j, 3), dtype=np.float32)
        out[..., :2] = arr[..., :2]
        # depth as a fixed function of the 2D input, plus a per-branch bias so
        # the flip and non-flip arms genuinely differ
        out[..., 2] = arr[..., :2].sum(axis=-1) * 0.1 + self.bias
        return torch.from_numpy(out)

    def parameters(self):
        import torch
        return iter([torch.zeros(1)])


def _window():
    rng = np.random.default_rng(0)
    kpts = rng.uniform(50, 400, size=(N_FRAMES, 17, 2)).astype(np.float32)
    scores = np.full((N_FRAMES, 17), 0.9, dtype=np.float32)
    return kpts, scores


class TestBehaviourPreservation(unittest.TestCase):
    def test_return_arms_mean_matches_default(self):
        """return_arms=True must not change the returned pose at all."""
        model = _StubModel()
        kpts, scores = _window()
        plain = lift_from_coco_window(model, kpts, scores, 640, 480)
        mean, arms = lift_from_coco_window(model, kpts, scores, 640, 480,
                                           return_arms=True)
        np.testing.assert_array_equal(plain, mean)
        self.assertEqual(set(arms), {"nonflip", "flip"})
        # the mean must be exactly the mean of the two arms (both root-centred)
        recon = (arms["nonflip"] + arms["flip"]) / 2.0
        np.testing.assert_allclose(recon, mean, atol=1e-6)

    def test_unrotate_round_trip(self):
        """_unrotate_kpts must invert the cv2 rotation it documents.

        Imported here rather than at module scope: demo_live.pose_detector pulls
        in Ultralytics, which writes a settings file on first import and fails on
        a machine without a writable user profile. This file promises to run
        anywhere with no weights and no dataset, and a module-level import broke
        that promise for the whole file rather than for this one test.
        """
        try:
            from demo_live.pose_detector import _unrotate_kpts
        except Exception as exc:            # pragma: no cover - environment only
            self.skipTest("detector unavailable in this environment: %s" % exc)
        W, H = 640, 480
        pts = np.array([[0.0, 0.0], [100.0, 50.0], [W - 1.0, H - 1.0]],
                       dtype=np.float32)
        # forward map for 180 is its own inverse
        back = _unrotate_kpts(_unrotate_kpts(pts, 180, W, H), 180, W, H)
        np.testing.assert_allclose(back, pts, atol=1e-4)
        # angle 0 is identity and must copy, not alias
        out = _unrotate_kpts(pts, 0, W, H)
        np.testing.assert_array_equal(out, pts)
        out[0, 0] = 999.0
        self.assertNotEqual(pts[0, 0], 999.0)


class TestDispersionMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.pose = rng.normal(0, 100, size=(17, 3))

    def test_identity_augmentation_gives_zero(self):
        """K identical arms must give exactly zero dispersion."""
        arms = [self.pose.copy() for _ in range(6)]
        self.assertAlmostEqual(disp_procrustes(arms), 0.0, places=6)
        depth, inplane = disp_axis_split(arms)
        self.assertAlmostEqual(depth, 0.0, places=9)
        self.assertAlmostEqual(inplane, 0.0, places=9)
        self.assertAlmostEqual(disp_scale(arms), 0.0, places=9)

    def test_invariance_and_monotonicity(self):
        """
        disp_procrustes is invariant to a global rotation+translation of every
        arm, EQUIVARIANT to a global scale (it reports an absolute distance in
        the poses' own units, exactly like the GT error it is compared
        against), and strictly increases when one arm is perturbed.
        """
        rng = np.random.default_rng(7)
        arms = [self.pose + rng.normal(0, 2, size=(17, 3)) for _ in range(6)]
        base = disp_procrustes(arms)

        theta = 0.4
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        rigid = [(a @ R.T) + np.array([10.0, -5.0, 3.0]) for a in arms]
        self.assertAlmostEqual(disp_procrustes(rigid) / base, 1.0, places=3)

        # scale equivariance — documented, not a bug: absolute units by design
        scaled = [a * 2.5 for a in arms]
        self.assertAlmostEqual(disp_procrustes(scaled) / base, 2.5, places=2)

        worse = [a.copy() for a in arms]
        worse[0] = worse[0] + rng.normal(0, 40, size=(17, 3))
        self.assertGreater(disp_procrustes(worse), base)

    def test_depth_vs_inplane_separates(self):
        """Depth-only disagreement must show up in disp_depth, not in-plane."""
        arms = []
        for i in range(6):
            a = self.pose.copy()
            a[:, 2] += i * 5.0          # perturb camera depth only
            arms.append(a)
        depth, inplane = disp_axis_split(arms)
        self.assertGreater(depth, 1.0)
        self.assertAlmostEqual(inplane, 0.0, places=9)


class TestKeyParsing(unittest.TestCase):
    def test_parses_static_and_dynamic_keys(self):
        """
        gt_eval's split('_', 2) mis-parses dynamic keys; parse_key must not.
        """
        self.assertEqual(parse_key("S1_Seq1_cam0"), ("S1", "Seq1", 0, "static"))
        self.assertEqual(parse_key("S1_Seq1_cam0_dynamic"),
                         ("S1", "Seq1", 0, "dynamic"))
        self.assertEqual(parse_key("S2_Seq2_cam8_dynamic"),
                         ("S2", "Seq2", 8, "dynamic"))


if __name__ == "__main__":
    unittest.main()
