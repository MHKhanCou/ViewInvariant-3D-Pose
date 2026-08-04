"""
Tests for BVH export (7 tests). Synthetic skeletons only, no weights or data.

The first test is the one that matters. An earlier implementation took the
median of the bone VECTORS as the rest skeleton, which is wrong in a way that
produces a valid-looking file: rotating vectors average to something shorter
than themselves, so the rest skeleton came out undersized and no set of
rotations could reproduce the input. It cost 10.9 cm of round-trip error on a
rigidly rotating body, where the correct answer is exactly zero.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.reliability import BONE_CONNECTIONS
from presentation.bvh_export import (CHANNELS, CHILDREN, PARENT, _demo_sequence,
                                     _global_rotations, poses_to_bvh, rest_offsets)


def forward_kinematics(pose, off):
    """Rebuild joint positions from the rotations the exporter writes."""
    R = _global_rotations(pose, off)
    pos, stack = {0: pose[0].copy()}, [0]
    while stack:
        j = stack.pop()
        for c in CHILDREN[j]:
            pos[c] = pos[j] + R[j] @ off[c]
            stack.append(c)
    return np.stack([pos[i] for i in range(17)])


def rigid_sequence(n=40):
    """One skeleton rotated about the vertical. Bone lengths are constant."""
    base = _demo_sequence(1)[0] * 100.0
    out = []
    for t in np.linspace(0, 2 * np.pi, n):
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        out.append(base @ R.T)
    return np.array(out)


class TestRestSkeleton(unittest.TestCase):

    def test_rest_bone_lengths_match_the_input(self):
        """
        The rest skeleton must have the same bone lengths as the poses it
        describes. Averaging bone vectors would shrink them; averaging lengths
        does not.
        """
        seq = rigid_sequence()
        off = rest_offsets(seq)
        for c, p in PARENT.items():
            observed = np.median(np.linalg.norm(seq[:, c] - seq[:, p], axis=-1))
            self.assertAlmostEqual(float(np.linalg.norm(off[c])), float(observed),
                                   places=6,
                                   msg="rest offset for joint %d is the wrong length" % c)

    def test_exported_skeleton_has_constant_bone_lengths(self):
        """BVH cannot express a stretching bone; the rest skeleton fixes them."""
        off = rest_offsets(_demo_sequence() * 100.0)
        seq = _demo_sequence() * 100.0
        lengths = []
        for t in range(len(seq)):
            rec = forward_kinematics(seq[t], off)
            lengths.append([np.linalg.norm(rec[i] - rec[j]) for i, j in BONE_CONNECTIONS])
        lengths = np.array(lengths)
        np.testing.assert_allclose(lengths.std(axis=0), 0.0, atol=1e-8)


class TestRoundTrip(unittest.TestCase):

    def test_rigid_motion_reconstructs_exactly(self):
        """
        A rigidly rotating body is fully representable in BVH, so forward
        kinematics from the exported rotations must return the input exactly.
        This is the regression test for the shrunken-rest-skeleton bug.
        """
        seq = rigid_sequence()
        off = rest_offsets(seq)
        for t in range(len(seq)):
            rec = forward_kinematics(seq[t], off)
            np.testing.assert_allclose(rec, seq[t], atol=1e-6)

    def test_residual_on_articulated_input_is_bounded_by_bone_variation(self):
        """
        Where the input's own bone lengths vary, a fixed-length skeleton cannot
        match it and a residual is expected. It must stay small and must not be
        mistaken for a conversion error.
        """
        seq = _demo_sequence() * 100.0
        off = rest_offsets(seq)
        err = np.mean([np.linalg.norm(forward_kinematics(seq[t], off) - seq[t], axis=1).mean()
                       for t in range(len(seq))])
        bone = float(np.mean([np.linalg.norm(off[c]) for c in off]))
        self.assertLess(err / bone, 0.10)


class TestFileFormat(unittest.TestCase):

    def setUp(self):
        self.text = poses_to_bvh(_demo_sequence(12) * 100.0, fps=25.0)
        self.lines = self.text.splitlines()

    def test_declares_every_joint_once_and_closes_every_chain(self):
        self.assertEqual(self.text.count("ROOT") + self.text.count("JOINT"), 17)
        leaves = sum(1 for j in CHILDREN if not CHILDREN[j])
        self.assertEqual(self.text.count("End Site"), leaves)
        self.assertEqual(self.text.count("{"), 17 + leaves)
        self.assertEqual(self.text.count("}"), 17 + leaves)

    def test_channel_count_matches_the_header(self):
        """Six channels for the root, three for each of the other sixteen."""
        mi = self.lines.index("MOTION")
        self.assertEqual(int(self.lines[mi + 1].split(":")[1]), 12)
        for row in self.lines[mi + 3:]:
            self.assertEqual(len(row.split()), 6 + 16 * 3)

    def test_root_motion_can_be_pinned(self):
        """Canonical poses carry no meaningful translation, so it is optional."""
        text = poses_to_bvh(_demo_sequence(5) * 100.0, root_motion=False)
        mi = text.splitlines().index("MOTION")
        for row in text.splitlines()[mi + 3:]:
            x, y, z = (float(v) for v in row.split()[:3])
            self.assertEqual((x, y, z), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
