"""
Tests for the controlled degradation operators.

These run without the model: they verify that each operator does exactly what
its documented failure mode claims, that severity is monotonic, that the
identity level is a true no-op, and that inputs are never mutated. If these
do not hold, the calibration derived from the sweep is meaningless.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.corruptions import (
    CORRUPTIONS, apply, is_identity_severity,
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER, L_KNEE, L_ANKLE, DROPOUT_ORDER,
)


def make_standing_pose():
    """A plausible upright COCO-17 detection in pixel coordinates."""
    kpts = np.array([
        [320, 100],  # nose
        [310, 92], [330, 92],    # eyes
        [300, 95], [340, 95],    # ears
        [285, 150], [355, 150],  # shoulders
        [270, 220], [370, 220],  # elbows
        [260, 290], [380, 290],  # wrists
        [300, 300], [340, 300],  # hips
        [295, 420], [345, 420],  # knees
        [292, 540], [348, 540],  # ankles
    ], dtype=np.float32)
    scores = np.full(17, 0.9, dtype=np.float32)
    return kpts, scores


class TestCorruptionContract(unittest.TestCase):
    """Properties every operator must satisfy."""

    def setUp(self):
        self.kpts, self.scores = make_standing_pose()

    def test_identity_severity_is_noop(self):
        for name, (_fn, levels, _u) in CORRUPTIONS.items():
            k_out, s_out = apply(name, self.kpts, self.scores, levels[0], seed=1)
            np.testing.assert_allclose(
                k_out, self.kpts, atol=1e-6,
                err_msg=f"{name} severity {levels[0]} altered keypoints")
            np.testing.assert_allclose(
                s_out, self.scores, atol=1e-6,
                err_msg=f"{name} severity {levels[0]} altered scores")
            self.assertTrue(is_identity_severity(name, levels[0]))

    def test_inputs_never_mutated(self):
        kpts_ref = self.kpts.copy()
        scores_ref = self.scores.copy()
        for name, (_fn, levels, _u) in CORRUPTIONS.items():
            for sev in levels:
                apply(name, self.kpts, self.scores, sev, seed=3)
        np.testing.assert_array_equal(self.kpts, kpts_ref)
        np.testing.assert_array_equal(self.scores, scores_ref)

    def test_deterministic_given_seed(self):
        for name, (_fn, levels, _u) in CORRUPTIONS.items():
            a = apply(name, self.kpts, self.scores, levels[-1], seed=42)
            b = apply(name, self.kpts, self.scores, levels[-1], seed=42)
            np.testing.assert_array_equal(a[0], b[0], f"{name} not deterministic")
            np.testing.assert_array_equal(a[1], b[1], f"{name} not deterministic")

    def test_output_shapes_preserved(self):
        for name, (_fn, levels, _u) in CORRUPTIONS.items():
            k_out, s_out = apply(name, self.kpts, self.scores, levels[-1], seed=7)
            self.assertEqual(k_out.shape, (17, 2), name)
            self.assertEqual(s_out.shape, (17,), name)

    def test_unknown_operator_raises(self):
        with self.assertRaises(KeyError):
            apply("does_not_exist", self.kpts, self.scores, 1.0)


class TestOperatorSemantics(unittest.TestCase):
    """Each operator must attack the structure it claims to attack."""

    def setUp(self):
        self.kpts, self.scores = make_standing_pose()

    def _hip_width(self, k):
        return float(np.linalg.norm(k[L_HIP] - k[R_HIP]))

    def _torso_height(self, k):
        shoulder_mid = 0.5 * (k[L_SHOULDER] + k[R_SHOULDER])
        hip_mid = 0.5 * (k[L_HIP] + k[R_HIP])
        return float(np.linalg.norm(shoulder_mid - hip_mid))

    def test_hip_collapse_shrinks_hip_axis_monotonically(self):
        levels = CORRUPTIONS["hip_axis_collapse"][1]
        widths = [self._hip_width(apply("hip_axis_collapse", self.kpts,
                                        self.scores, s, seed=0)[0])
                  for s in levels]
        self.assertAlmostEqual(widths[0], self._hip_width(self.kpts), places=4)
        for a, b in zip(widths, widths[1:]):
            self.assertLess(b, a, "hip width must decrease with severity")
        self.assertLess(widths[-1], 0.1 * widths[0])

    def test_hip_collapse_leaves_torso_intact(self):
        # The operator must be surgical: it attacks x_raw only, so any change
        # in reliability cannot be attributed to a confounded torso axis.
        k_out, _ = apply("hip_axis_collapse", self.kpts, self.scores, 0.05, seed=0)
        self.assertAlmostEqual(self._torso_height(k_out),
                               self._torso_height(self.kpts), places=3)

    def test_torso_collapse_shrinks_torso_axis_monotonically(self):
        levels = CORRUPTIONS["torso_axis_collapse"][1]
        heights = [self._torso_height(apply("torso_axis_collapse", self.kpts,
                                            self.scores, s, seed=0)[0])
                   for s in levels]
        self.assertAlmostEqual(heights[0], self._torso_height(self.kpts), places=4)
        for a, b in zip(heights, heights[1:]):
            self.assertLess(b, a, "torso height must decrease with severity")
        self.assertLess(heights[-1], 0.1 * heights[0])

    def test_torso_collapse_leaves_hip_axis_intact(self):
        k_out, _ = apply("torso_axis_collapse", self.kpts, self.scores, 0.05, seed=0)
        self.assertAlmostEqual(self._hip_width(k_out),
                               self._hip_width(self.kpts), places=3)

    def test_dropout_zeroes_structural_joints_in_order(self):
        for k in (1, 2, 3, 4, 6):
            k_out, s_out = apply("joint_dropout", self.kpts, self.scores, k, seed=0)
            dropped = DROPOUT_ORDER[:k]
            for idx in dropped:
                self.assertEqual(s_out[idx], 0.0, f"joint {idx} score not zeroed")
                np.testing.assert_array_equal(k_out[idx], np.zeros(2))
            kept = [i for i in range(17) if i not in dropped]
            for idx in kept:
                self.assertGreater(s_out[idx], 0.0, f"joint {idx} wrongly zeroed")

    def test_dropout_hits_hips_first(self):
        # Hips are the most load-bearing joints for the body frame, so k=2
        # must remove exactly those.
        _k, s_out = apply("joint_dropout", self.kpts, self.scores, 2, seed=0)
        self.assertEqual(s_out[L_HIP], 0.0)
        self.assertEqual(s_out[R_HIP], 0.0)

    def test_foreshorten_contracts_legs_toward_hips(self):
        levels = CORRUPTIONS["limb_foreshorten"][1]
        hip_mid = 0.5 * (self.kpts[L_HIP] + self.kpts[R_HIP])
        dists = []
        for s in levels:
            k_out, _ = apply("limb_foreshorten", self.kpts, self.scores, s, seed=0)
            dists.append(float(np.linalg.norm(k_out[L_ANKLE] - hip_mid)))
        for a, b in zip(dists, dists[1:]):
            self.assertLess(b, a, "leg extent must decrease with severity")

    def test_foreshorten_leaves_upper_body_intact(self):
        k_out, _ = apply("limb_foreshorten", self.kpts, self.scores, 0.1, seed=0)
        for idx in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            np.testing.assert_allclose(k_out[idx], self.kpts[idx], atol=1e-4)

    def test_jitter_magnitude_scales_with_severity(self):
        levels = CORRUPTIONS["jitter"][1]
        displacements = []
        for s in levels:
            # Average over seeds so the comparison is not one noisy draw.
            per_seed = []
            for seed in range(24):
                k_out, _ = apply("jitter", self.kpts, self.scores, s, seed=seed)
                per_seed.append(np.mean(np.linalg.norm(k_out - self.kpts, axis=1)))
            displacements.append(float(np.mean(per_seed)))
        self.assertAlmostEqual(displacements[0], 0.0, places=6)
        for a, b in zip(displacements[1:], displacements[2:]):
            self.assertLess(a, b, "jitter magnitude must grow with sigma")

    def test_jitter_is_scale_invariant(self):
        # The same sigma must produce the same *relative* displacement on a
        # person twice the size, otherwise severity is not comparable across
        # frames with different subject scales.
        big = self.kpts * 2.0
        sev = 0.04
        disp_small, disp_big = [], []
        for seed in range(24):
            ks, _ = apply("jitter", self.kpts, self.scores, sev, seed=seed)
            kb, _ = apply("jitter", big, self.scores, sev, seed=seed)
            disp_small.append(np.mean(np.linalg.norm(ks - self.kpts, axis=1)))
            disp_big.append(np.mean(np.linalg.norm(kb - big, axis=1)))
        ratio = float(np.mean(disp_big)) / float(np.mean(disp_small))
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)

    def test_jitter_does_not_touch_scores(self):
        # Confidence must stay clean so the detector-confidence component of
        # the reliability score gets no free signal from this operator.
        _k, s_out = apply("jitter", self.kpts, self.scores, 0.08, seed=0)
        np.testing.assert_array_equal(s_out, self.scores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
