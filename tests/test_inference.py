"""
Regression tests for inference pipeline behavior.

Validates that:
1. Flip augmentation calls the model twice (original + flipped input)
2. Raw mode (apply_display_postprocess=False) skips camera_to_world/z-shift/scale
3. Default mode (apply_display_postprocess=True) applies official display processing
4. Canonical branches request raw root-relative output
5. Base-MPI builder uses 81 frames, not 27
6. estimate_poses returns raw_root_relative key

Source-routing tests read source text directly to avoid importing
YOLO/Gradio/MotionAGFormer at test time.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class TestFlipAugmentation(unittest.TestCase):
    """Test that flip augmentation runs the model on both original and flipped inputs."""

    @classmethod
    def setUpClass(cls):
        from demo_live.lifter import build_xs_model
        cls.model = build_xs_model(device="cpu")

    def test_lift_sequence_flip_calls_model_twice(self):
        """When use_flip=True, the model must be called on both original and flipped inputs."""
        np.random.seed(42)
        T = 5
        h36m = np.random.randn(T, 17, 3).astype(np.float32) * 0.5

        call_count = [0]
        original_forward = self.model.forward

        def counting_forward(x, *args, **kwargs):
            call_count[0] += 1
            return original_forward(x, *args, **kwargs)

        self.model.forward = counting_forward

        try:
            from demo_live.lifter import lift_sequence
            result = lift_sequence(
                self.model, h36m, img_size=(480, 640),
                use_flip=True, apply_display_postprocess=False
            )
            self.assertEqual(call_count[0], T * 2,
                             f"Expected {T*2} model calls (2 per frame), got {call_count[0]}")
            self.assertEqual(result.shape, (T, 17, 3))
        finally:
            self.model.forward = original_forward

    def test_lift_sequence_no_flip_calls_model_once(self):
        """When use_flip=False, the model is called once per frame."""
        np.random.seed(42)
        T = 5
        h36m = np.random.randn(T, 17, 3).astype(np.float32) * 0.5

        call_count = [0]
        original_forward = self.model.forward

        def counting_forward(x, *args, **kwargs):
            call_count[0] += 1
            return original_forward(x, *args, **kwargs)

        self.model.forward = counting_forward

        try:
            from demo_live.lifter import lift_sequence
            result = lift_sequence(
                self.model, h36m, img_size=(480, 640),
                use_flip=False, apply_display_postprocess=False
            )
            self.assertEqual(call_count[0], T,
                             f"Expected {T} model calls (1 per frame), got {call_count[0]}")
        finally:
            self.model.forward = original_forward

    def test_flip_data_swaps_left_right_and_negates_x(self):
        """Verify _flip_data correctly negates x and swaps left/right joints."""
        from demo_live.lifter import _flip_data
        LEFT_JOINTS = [1, 2, 3, 14, 15, 16]
        RIGHT_JOINTS = [4, 5, 6, 11, 12, 13]

        pose = np.random.randn(17, 3).astype(np.float32)
        flipped = _flip_data(pose)

        # Center joints: x negated, y/z unchanged
        for i in [0, 7, 8, 9, 10]:
            self.assertAlmostEqual(flipped[i, 0], -pose[i, 0], places=5)
            self.assertAlmostEqual(flipped[i, 1], pose[i, 1], places=5)
            self.assertAlmostEqual(flipped[i, 2], pose[i, 2], places=5)

        # Left joints: x is negated-right, y/z is right (swap only)
        for l, r in zip(LEFT_JOINTS, RIGHT_JOINTS):
            self.assertAlmostEqual(flipped[l, 0], -pose[r, 0], places=5)
            self.assertAlmostEqual(flipped[l, 1], pose[r, 1], places=5)


class TestRawMode(unittest.TestCase):
    """Test that apply_display_postprocess=False skips display transforms."""

    @classmethod
    def setUpClass(cls):
        from demo_live.lifter import build_xs_model
        cls.model = build_xs_model(device="cpu")

    def test_raw_mode_no_camera_to_world(self):
        """Raw output should be root-relative, not display-transformed."""
        np.random.seed(42)
        T = 3
        h36m = np.random.randn(T, 17, 3).astype(np.float32) * 0.5

        from demo_live.lifter import lift_sequence
        result = lift_sequence(
            self.model, h36m, img_size=(480, 640),
            mode="root", use_flip=False, apply_display_postprocess=False
        )

        self.assertTrue(np.any(result < 0),
                        "Raw output should contain negative values (not normalized to [0,1])")

    def test_default_mode_applies_display_processing(self):
        """Default display mode should apply camera_to_world and max-normalize."""
        np.random.seed(42)
        T = 3
        h36m = np.random.randn(T, 17, 3).astype(np.float32) * 0.5

        from demo_live.lifter import lift_sequence
        result = lift_sequence(
            self.model, h36m, img_size=(480, 640),
            mode="world", use_flip=False, apply_display_postprocess=True
        )

        self.assertAlmostEqual(result.max(), 1.0, places=4,
                               msg="Display output max should be 1.0 after normalization")

    def test_raw_vs_default_produce_different_outputs(self):
        """Raw and display modes must produce different results."""
        np.random.seed(42)
        T = 3
        h36m = np.random.randn(T, 17, 3).astype(np.float32) * 0.5

        from demo_live.lifter import lift_sequence
        raw = lift_sequence(
            self.model, h36m, img_size=(480, 640),
            mode="root", use_flip=False, apply_display_postprocess=False
        )
        display = lift_sequence(
            self.model, h36m, img_size=(480, 640),
            mode="world", use_flip=False, apply_display_postprocess=True
        )

        self.assertFalse(np.allclose(raw, display, atol=1e-4),
                         "Raw and display outputs should differ")


class TestSourceCodeChecks(unittest.TestCase):
    """Source-routing tests that read .py files directly — no YOLO/Gradio imports."""

    def _read_source(self, rel_path):
        full = os.path.join(REPO_ROOT, rel_path)
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

    def test_predict_image_canonical_raw(self):
        """predict_image must pass apply_display_postprocess=False for canonical."""
        src = self._read_source("backend/inference.py")
        self.assertIn("apply_display_postprocess", src)
        self.assertIn('mode != "canonical"', src)

    def test_build_base_model_uses_81_frames(self):
        """Base-MPI builder must use 81 frames, not 27."""
        src = self._read_source("demo_live/lifter.py")
        self.assertIn("MPI_BASE_NFRAMES = 81", src)
        # The XS model uses n_frames=N_FRAMES (27) — that's fine.
        # Check that build_base_model uses MPI_BASE_NFRAMES, not N_FRAMES.
        import re
        # Find the build_base_model function body
        match = re.search(r'def build_base_model.*?(?=\ndef |\Z)', src, re.DOTALL)
        self.assertIsNotNone(match, "build_base_model function not found")
        base_src = match.group(0)
        self.assertIn("n_frames=MPI_BASE_NFRAMES", base_src)
        self.assertNotIn("n_frames=N_FRAMES", base_src)

    def test_model_loader_uses_xs(self):
        """Default model loader should use build_xs_model."""
        src = self._read_source("backend/model_loader.py")
        self.assertIn("build_xs_model", src)

    def test_estimate_poses_has_raw_key(self):
        """estimate_poses must return raw_root_relative key."""
        src = self._read_source("backend/inference.py")
        self.assertIn("raw_root_relative", src)

    def test_estimate_poses_no_camera_pose_key(self):
        """The old 'camera_pose' key should no longer exist."""
        src = self._read_source("backend/inference.py")
        self.assertNotIn('"camera_pose"', src,
                         "camera_pose key should be renamed")

    def test_mpi_evaluator_uses_flip_and_raw(self):
        """MPI evaluator must use use_flip=True and apply_display_postprocess=False."""
        src = self._read_source("scripts/mpi_cross_view.py")
        self.assertIn("use_flip=True", src)
        self.assertIn("apply_display_postprocess=False", src)


if __name__ == "__main__":
    unittest.main()
