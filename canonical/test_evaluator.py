"""
Evaluator capability-detection tests.

Tests are split into:
  1. Fixture tests with synthetic metadata (verify detection logic)
  2. Real pkl metadata tests (verify current data is insufficient)

All tests should PASS — they confirm the system correctly handles both
valid and invalid metadata scenarios.

Run: python -m unittest canonical.test_evaluator -v
"""

import unittest
import numpy as np
from canonical.metadata_capability import (
    inspect_prediction_metadata,
    inspect_gt_metadata,
    inspect_cross_view_metadata,
)


# ─── Fixture: valid synchronized camera metadata ─────────────────────────────

class TestValidSynchronizedCamera(unittest.TestCase):
    """Verify detection works when metadata supports cross-view pairing."""

    def test_h36m_two_camera_same_timestamp(self):
        """Exact H36M case: same subject/action/subaction, different cameras, same timestamp."""
        sources = [
            's_09_act_02_subact_01_ca_01_01234',
            's_09_act_02_subact_01_ca_02_01234',
            's_09_act_02_subact_01_ca_01_01235',
            's_09_act_02_subact_01_ca_02_01235',
        ]
        cameras = ['54138969', '55011271', '54138969', '55011271']
        result = inspect_prediction_metadata(sources, cameras)
        self.assertTrue(result['has_exact_timestamp'])
        self.assertTrue(result['has_multi_camera_sequences'])
        self.assertTrue(result['can_pair'])
        self.assertEqual(len(result['reasons']), 0)

    def test_valid_prediction_pairing(self):
        """Two cameras sharing same sequence+timestamp should allow pairing."""
        sources = [
            'seq1_1000', 'seq1_1000',  # cam A, cam B
            'seq1_1001', 'seq1_1001',  # cam A, cam B
            'seq2_2000', 'seq2_2000',  # cam A, cam B
        ]
        cameras = ['cam_A', 'cam_B', 'cam_A', 'cam_B', 'cam_A', 'cam_B']
        result = inspect_prediction_metadata(sources, cameras)
        self.assertTrue(result['has_exact_timestamp'])
        self.assertTrue(result['has_multi_camera_sequences'])
        self.assertTrue(result['can_pair'])
        self.assertEqual(len(result['reasons']), 0)

    def test_valid_gt_with_explicit_provenance(self):
        """GT with explicit provenance and plausible depth should enable floor."""
        gts = np.random.randn(50, 17, 3).astype(np.float32)  # z/xy ~ 1.0
        result = inspect_gt_metadata(gts, metric_3d_verified=True)
        self.assertTrue(result['depth_plausibility'])
        self.assertTrue(result['can_compute_floor'])
        self.assertEqual(len(result['reasons']), 0)

    def test_valid_gt_without_provenance_no_floor(self):
        """GT with plausible depth but no provenance should NOT enable floor."""
        gts = np.random.randn(50, 17, 3).astype(np.float32)
        result = inspect_gt_metadata(gts, metric_3d_verified=False)
        self.assertTrue(result['depth_plausibility'])
        self.assertFalse(result['can_compute_floor'])
        self.assertGreater(len(result['reasons']), 0)


# ─── Fixture: missing timestamp metadata ─────────────────────────────────────

class TestMissingTimestamp(unittest.TestCase):
    """Verify detection when source strings lack timestamps."""

    def test_no_timestamp_detected(self):
        """Sources without numeric last part should fail timestamp check."""
        sources = ['seq1_camA', 'seq1_camA', 'seq1_camB', 'seq1_camB']
        cameras = ['camA', 'camA', 'camB', 'camB']
        result = inspect_prediction_metadata(sources, cameras)
        self.assertFalse(result['has_exact_timestamp'])
        self.assertFalse(result['can_pair'])
        self.assertGreater(len(result['reasons']), 0)


# ─── Fixture: invalid GT metadata ────────────────────────────────────────────

class TestInvalidGT(unittest.TestCase):
    """Verify detection when GT is image-plane, not metric 3D."""

    def test_image_plane_gt_detected(self):
        """GT with small z relative to x,y should fail depth plausibility."""
        gts = np.zeros((50, 17, 3), dtype=np.float32)
        gts[:, :, :2] = np.random.randn(50, 17, 2) * 1000  # x,y in pixel range
        gts[:, :, 2] = np.random.randn(50, 17) * 10        # z much smaller
        result = inspect_gt_metadata(gts)
        self.assertFalse(result['depth_plausibility'])
        self.assertFalse(result['can_compute_floor'])
        self.assertGreater(len(result['reasons']), 0)


# ─── Fixture: single camera (no cross-view possible) ─────────────────────────

class TestSingleCamera(unittest.TestCase):
    """Verify detection when only one camera is present."""

    def test_single_camera_no_pairing(self):
        """All frames from same camera should prevent pairing."""
        sources = ['seq1_1000', 'seq1_1001', 'seq1_1002']
        cameras = ['cam_A', 'cam_A', 'cam_A']
        result = inspect_prediction_metadata(sources, cameras)
        self.assertTrue(result['has_exact_timestamp'])
        self.assertFalse(result['has_multi_camera_sequences'])
        self.assertFalse(result['can_pair'])


# ─── Fixture: H36M-like source format ────────────────────────────────────────

class TestH36MSourceFormat(unittest.TestCase):
    """Verify parser handles H36M source format correctly."""

    def test_h36m_format_parsed(self):
        """H36M source strings should be parsed into explicit fields."""
        from canonical.metadata_capability import _parse_source
        parsed = _parse_source('s_09_act_02_subact_01_ca_01')
        self.assertEqual(parsed['subject'], 's_09')
        self.assertEqual(parsed['action'], 'act_02')
        self.assertEqual(parsed['subaction'], 'subact_01')
        self.assertEqual(parsed['camera_id'], 'ca_01')
        self.assertIsNone(parsed['timestamp'])

    def test_h36m_format_with_timestamp(self):
        """Source strings with numeric timestamp should be detected."""
        from canonical.metadata_capability import _parse_source
        parsed = _parse_source('s_09_act_02_subact_01_ca_01_12345')
        self.assertEqual(parsed['subject'], 's_09')
        self.assertEqual(parsed['camera_id'], 'ca_01')
        self.assertEqual(parsed['timestamp'], '12345')

    def test_train_format_parsed(self):
        """Train-split format should also be parsed correctly."""
        from canonical.metadata_capability import _parse_source
        parsed = _parse_source('s_01_act_02_cam_01')
        self.assertEqual(parsed['subject'], 's_01')
        self.assertEqual(parsed['action'], 'act_02')
        self.assertEqual(parsed['camera_id'], 'cam_01')
        self.assertIsNone(parsed['subaction'])


# ─── Real pkl metadata tests ─────────────────────────────────────────────────

class TestRealPklPredictionCapability(unittest.TestCase):
    """Verify that current pkl metadata is insufficient for prediction pairing."""

    @classmethod
    def setUpClass(cls):
        cls.report = inspect_cross_view_metadata()

    def test_timestamps_missing(self):
        self.assertFalse(self.report['has_exact_timestamp'])

    def test_has_multi_camera_sequences(self):
        # The pkl has multi-camera sequences (valid), but can't pair without timestamps
        self.assertTrue(self.report['has_multi_camera_sequences'])

    def test_cannot_compute_prediction_cross_view(self):
        self.assertFalse(self.report['can_compute_prediction_cross_view'])

    def test_reasons_populated(self):
        self.assertGreater(len(self.report['reasons']), 0)


class TestRealPklGTCapability(unittest.TestCase):
    """Verify that current pkl GT does not enable cross-view floor."""

    @classmethod
    def setUpClass(cls):
        cls.report = inspect_cross_view_metadata()

    def test_depth_plausibility_is_false(self):
        self.assertFalse(self.report['depth_plausibility'])

    def test_cannot_compute_gt_floor(self):
        self.assertFalse(self.report['can_compute_gt_cross_view_floor'])

    def test_cannot_compute_full_evaluation(self):
        self.assertFalse(self.report['can_compute_full_cross_view_evaluation'])


class TestReportStructure(unittest.TestCase):
    """Verify report has all required keys."""

    def test_report_keys(self):
        report = inspect_cross_view_metadata()
        required = [
            'has_exact_timestamp', 'has_multi_camera_sequences',
            'can_compute_prediction_cross_view',
            'depth_plausibility', 'can_compute_gt_cross_view_floor',
            'can_compute_full_cross_view_evaluation',
            'reasons', 'details',
        ]
        for key in required:
            self.assertIn(key, report, f"Missing key: {key}")


if __name__ == "__main__":
    unittest.main()
