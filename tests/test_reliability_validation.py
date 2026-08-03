"""
Validate reliability score against synthetic failure cases.
Each case has a known expected behavior.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.reliability import (
    compute_reliability_score, should_abstain, has_hard_geometric_failure
)


def make_good_pose():
    """Standing pose with proper proportions."""
    return np.array([
        [0, 0, 0], [0.05, -0.1, 0], [0.06, 0.1, 0], [0.07, 0.3, 0],
        [-0.05, -0.1, 0], [-0.06, 0.1, 0], [-0.07, 0.3, 0],
        [0, -0.15, 0], [0, -0.3, 0], [0, -0.4, 0], [0, -0.5, 0],
        [0.1, -0.28, 0], [0.2, -0.15, 0], [0.25, 0, 0],
        [-0.1, -0.28, 0], [-0.2, -0.15, 0], [-0.25, 0, 0],
    ], dtype=np.float64)


class TestReliabilitySyntheticCases(unittest.TestCase):
    """Synthetic failure cases with expected abstention behavior."""

    def test_good_pose_accepts(self):
        """Well-formed standing pose: no hard failure, high score."""
        pose = make_good_pose()
        hard, reason = has_hard_geometric_failure(pose)
        rel, comp, _ = compute_reliability_score(pose)
        self.assertFalse(hard, f"Good pose should not have hard failure: {reason}")
        self.assertGreater(rel, 0.5, f"Good pose score {rel:.4f} should be > 0.5")
        self.assertFalse(should_abstain(rel, hard))

    def test_collapsed_body_hard_gate(self):
        """All joints at origin: hard gate triggers."""
        pose = np.zeros((17, 3), dtype=np.float64)
        hard, reason = has_hard_geometric_failure(pose)
        self.assertTrue(hard, "Collapsed body should trigger hard gate")
        self.assertTrue(should_abstain(0.0, hard))

    def test_collinear_axes_hard_gate(self):
        """Torso and hip axes parallel: hard gate triggers."""
        pose = make_good_pose()
        # Make hip axis parallel to torso axis
        direction = pose[8] - pose[0]
        pose[1] = pose[0] + direction * 0.3
        pose[4] = pose[0] + direction * -0.3
        hard, reason = has_hard_geometric_failure(pose)
        self.assertTrue(hard, f"Collinear axes should trigger gate: {reason}")

    def test_near_zero_hip_hard_gate(self):
        """Near-zero hip width: hard gate triggers."""
        pose = make_good_pose()
        pose[1] = pose[0] + np.array([0.0001, -0.1, 0])
        pose[4] = pose[0] + np.array([-0.0001, -0.1, 0])
        hard, reason = has_hard_geometric_failure(pose)
        self.assertTrue(hard, f"Near-zero hip should trigger gate: {reason}")

    def test_near_zero_torso_hard_gate(self):
        """Near-zero torso axis: hard gate triggers."""
        pose = make_good_pose()
        pose[8] = pose[0] + np.array([0, 0.0001, 0])
        hard, reason = has_hard_geometric_failure(pose)
        self.assertTrue(hard, f"Near-zero torso should trigger gate: {reason}")

    def test_severe_bone_outlier_reduces_score(self):
        """One bone 5x longer reduces abnormal_bone_ratio below 1.0."""
        pose = make_good_pose()
        pose[13] = pose[8] + np.array([5.0, 0, 0])
        _, comp, _ = compute_reliability_score(pose)
        self.assertLess(comp['abnormal_bone_ratio'], 1.0,
                        "Outlier bone should reduce ratio")

    def test_left_right_asymmetry_reduces_score(self):
        """Severe L/R asymmetry reduces bilateral symmetry."""
        pose = make_good_pose()
        pose[13] = pose[12] + np.array([1.0, 0, 0])
        _, comp, _ = compute_reliability_score(pose)
        self.assertLess(comp['bilateral_symmetry'], 0.7,
                        "Severe asymmetry should reduce symmetry")

    def test_low_confidence_reduces_score(self):
        """Low detector confidence reduces score."""
        pose = make_good_pose()
        _, comp_low, _ = compute_reliability_score(pose, scores_2d=np.ones(17) * 0.1)
        _, comp_high, _ = compute_reliability_score(pose, scores_2d=np.ones(17) * 0.9)
        self.assertLess(comp_low['detector_confidence'], comp_high['detector_confidence'])

    def test_score_range(self):
        """Score always in [0, 1]."""
        for _ in range(10):
            pose = np.random.randn(17, 3).astype(np.float64) * 0.5
            rel, _, _ = compute_reliability_score(pose)
            self.assertGreaterEqual(rel, 0.0)
            self.assertLessEqual(rel, 1.0)

    def test_coco_confidence_key_joints(self):
        """Detector confidence should use COCO-17 key joints, not H36M indices."""
        pose = make_good_pose()
        # Create COCO-17 scores where key joints have low confidence
        scores = np.ones(17, dtype=np.float32) * 0.9
        scores[0] = 0.1   # nose (COCO 0) — key joint
        scores[5] = 0.1   # left_shoulder (COCO 5) — key joint
        _, comp_key, _ = compute_reliability_score(pose, scores)

        # vs. all high confidence
        _, comp_all, _ = compute_reliability_score(pose, scores_2d=np.ones(17) * 0.9)

        self.assertLess(comp_key['detector_confidence'], comp_all['detector_confidence'],
                        "Low key-joint confidence should reduce score")


class TestBenchmarkRouteUsesHardGate(unittest.TestCase):
    """Verify that the benchmark evaluation route calls has_hard_geometric_failure."""

    def test_benchmark_script_imports_and_calls_hard_gate(self):
        """The benchmark script must import and call has_hard_geometric_failure."""
        src_path = os.path.join(REPO_ROOT, "scripts", "run_example_benchmark.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("has_hard_geometric_failure", src,
                       "Benchmark script must import has_hard_geometric_failure")
        self.assertIn("has_hard_geometric_failure(raw_pose)", src,
                       "Benchmark must call has_hard_geometric_failure on raw_pose")

    def test_benchmark_abstain_uses_hard_failure(self):
        """The benchmark must call should_abstain(reliability, hard_failure), not just should_abstain(reliability)."""
        src_path = os.path.join(REPO_ROOT, "scripts", "run_example_benchmark.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("should_abstain(reliability, hard_failure)", src,
                       "Benchmark must pass hard_failure to should_abstain")

    def test_benchmark_saves_failure_reason(self):
        """The benchmark must save failure_reason in CSV and per-image report."""
        src_path = os.path.join(REPO_ROOT, "scripts", "run_example_benchmark.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("failure_reason", src)
        self.assertIn("hard_failure", src)


if __name__ == "__main__":
    unittest.main()
