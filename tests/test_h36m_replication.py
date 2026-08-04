"""
Tests for the Human3.6M replication (6 tests). Synthetic arrays only: no
weights, no dataset, runs anywhere.

The aggregation tests matter most. `split_clips` resamples a video's tail when
its length does not divide evenly by 27, so a few frames land in two clips.
Getting that averaging wrong would silently corrupt the per-video sequences the
whole analysis is built on, and it would not show up as an error.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.bone_consistency import bone_ratios, deviation
from evaluation.h36m_replication import (
    COCO_FROM_H36M, N_FRAMES, aggregate_by_video, h36m_conf_to_coco, parse_video,
)
from evaluation.reliability import COCO_KEY_JOINTS


def _make_clips(n_clips, source, frame_idx):
    """Minimal stand-ins for the meta/pred archives aggregate_by_video reads."""
    rng = np.random.default_rng(0)
    meta = {
        "frame_idx": np.asarray(frame_idx),
        "clip_source": np.array(source),
        "gt": rng.normal(size=(n_clips, N_FRAMES, 17, 3)).astype(np.float32),
        "n_test_frames": int(np.asarray(frame_idx).max()) + 1,
    }
    pred = {
        "preds": rng.normal(size=(n_clips, N_FRAMES, 17, 3)).astype(np.float32),
        "conf": rng.uniform(size=(n_clips, N_FRAMES, 17)).astype(np.float32),
    }
    return meta, pred


class TestConfidenceMapping(unittest.TestCase):

    def test_key_joints_are_remapped_not_passed_through(self):
        """Every COCO slot the reliability score reads must be filled from H36M."""
        scores = np.arange(17, dtype=np.float64) / 100.0
        coco = h36m_conf_to_coco(scores)
        for coco_i in COCO_KEY_JOINTS:
            self.assertIn(coco_i, COCO_FROM_H36M,
                          "COCO slot %d is read by the reliability score but "
                          "has no H36M source" % coco_i)
            self.assertEqual(coco[coco_i], scores[COCO_FROM_H36M[coco_i]])

    def test_unread_slots_do_not_lower_the_minimum(self):
        """The score takes a min over key joints; filler must never win it."""
        scores = np.full(17, 0.5)
        coco = h36m_conf_to_coco(scores)
        self.assertAlmostEqual(float(np.min(coco[COCO_KEY_JOINTS])), 0.5)
        unread = [i for i in range(17) if i not in COCO_KEY_JOINTS]
        self.assertTrue(np.all(coco[unread] == 1.0))


class TestVideoParsing(unittest.TestCase):

    def test_source_string_splits_into_subject_action_camera(self):
        self.assertEqual(parse_video("s_09_act_02_subact_01_ca_01"),
                         ("S9", "act_02", "ca_01"))
        self.assertEqual(parse_video("s_11_act_16_subact_02_ca_04"),
                         ("S11", "act_16", "ca_04"))


class TestAggregation(unittest.TestCase):

    def test_disjoint_clips_concatenate_in_frame_order(self):
        """Two back-to-back clips of one video reconstruct the sequence exactly."""
        frame_idx = np.stack([np.arange(N_FRAMES), np.arange(N_FRAMES, 2 * N_FRAMES)])
        meta, pred = _make_clips(2, ["s_09_act_02_subact_01_ca_01"] * 2, frame_idx)

        out = aggregate_by_video(meta, pred, 2)
        self.assertEqual(list(out), ["s_09_act_02_subact_01_ca_01"])
        v = out["s_09_act_02_subact_01_ca_01"]
        self.assertEqual(v["pred"].shape, (2 * N_FRAMES, 17, 3))
        np.testing.assert_allclose(v["pred"], pred["preds"].reshape(-1, 17, 3),
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(v["frames"], np.arange(2 * N_FRAMES))

    def test_overlapping_tail_frames_are_averaged(self):
        """A frame appearing in two clips must come back as the mean of both."""
        # Second clip is the resampled tail: it repeats the last frame of clip 0.
        frame_idx = np.stack([np.arange(N_FRAMES),
                              np.r_[N_FRAMES - 1, np.arange(N_FRAMES, 2 * N_FRAMES - 1)]])
        meta, pred = _make_clips(2, ["s_09_act_02_subact_01_ca_01"] * 2, frame_idx)

        v = aggregate_by_video(meta, pred, 2)["s_09_act_02_subact_01_ca_01"]
        expected = (pred["preds"][0, N_FRAMES - 1] + pred["preds"][1, 0]) / 2.0
        np.testing.assert_allclose(v["pred"][N_FRAMES - 1], expected, rtol=1e-6, atol=1e-6)
        # Every other frame occurs once and must be untouched by the averaging.
        np.testing.assert_allclose(v["pred"][0], pred["preds"][0, 0], rtol=1e-6, atol=1e-6)

    def test_blocked_videos_are_dropped(self):
        """The three corrupted videos the official eval skips must not appear."""
        frame_idx = np.stack([np.arange(N_FRAMES), np.arange(N_FRAMES, 2 * N_FRAMES)])
        meta, pred = _make_clips(
            2, ["s_09_act_05_subact_02_ca_01", "s_11_act_02_subact_01_ca_01"], frame_idx)

        out = aggregate_by_video(meta, pred, 2)
        self.assertEqual(list(out), ["s_11_act_02_subact_01_ca_01"])


class TestSignalProperties(unittest.TestCase):

    def test_deviation_is_zero_for_a_rigid_skeleton_and_grows_with_distortion(self):
        """
        The signal must respond to bone-length change and nothing else. A rigidly
        rotated, uniformly rescaled skeleton has to score exactly zero, or the
        measurement is just reading global scale back out.
        """
        rng = np.random.default_rng(7)
        base = rng.normal(size=(17, 3))

        seq = []
        for t in range(20):
            th = 0.3 * t
            R = np.array([[np.cos(th), -np.sin(th), 0],
                          [np.sin(th), np.cos(th), 0],
                          [0, 0, 1]])
            seq.append((base @ R.T) * (1.0 + 0.1 * t))  # rotate and rescale
        seq = np.array(seq)

        ratios = bone_ratios(seq)
        dev = deviation(ratios, np.median(ratios, axis=0))
        # Floor is set by the 1e-9 epsilon in `deviation`, not by the geometry;
        # it sits five orders below the distorted-frame threshold checked next.
        np.testing.assert_allclose(dev, 0.0, atol=1e-6)

        # Now lengthen one limb on one frame: that frame, and only it, must move.
        distorted = seq.copy()
        distorted[10, 3] = distorted[10, 2] + (distorted[10, 3] - distorted[10, 2]) * 1.5
        ratios_d = bone_ratios(distorted)
        dev_d = deviation(ratios_d, np.median(ratios_d, axis=0))
        self.assertGreater(dev_d[10], 1e-3)
        self.assertLess(float(np.max(np.delete(dev_d, 10))), dev_d[10])


if __name__ == "__main__":
    unittest.main(verbosity=2)
