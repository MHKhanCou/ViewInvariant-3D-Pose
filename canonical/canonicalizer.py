"""
Stateful canonicalizer with temporal sign consistency across calls.
"""

import numpy as np
from canonical.body_frame import canonicalize_single, canonicalize_batch


class Canonicalizer:
    """
    Stateful wrapper that carries the last z-axis across calls for
    temporal sign consistency within a single sequence.

    Usage for metrics (no temporal leakage):
        can = Canonicalizer()
        for each independent frame pair:
            can.reset()
            ca, _ = can(pose_a)
            can.reset()
            cb, _ = can(pose_b)

    Usage for video visualization (temporal continuity):
        can = Canonicalizer()
        for each frame in sequence:
            ca, _ = can(frame)
        # then can.reset() before starting a new sequence
    """

    def __init__(self):
        self._prev_z = None

    def reset(self):
        """Reset temporal state. Call between independent sequences."""
        self._prev_z = None

    def __call__(self, poses, return_metadata=False):
        """
        Canonicalize poses with temporal sign consistency.

        Args:
            poses: (17, 3) or (T, 17, 3) float32 array.
            return_metadata: if True, also return metadata.

        Returns:
            canonical: (17, 3) or (T, 17, 3) canonicalized poses.
            R: (3, 3) or (T, 3, 3) rotation matrices.
            (optional) metadata: single dict or list of dicts.
        """
        poses = np.asarray(poses, dtype=np.float32)

        if poses.ndim == 2:
            # Single frame
            can, R, meta = canonicalize_single(poses, prev_z=self._prev_z)
            if meta["valid"]:
                self._prev_z = meta["z_axis"]
            if return_metadata:
                return can, R, meta
            return can, R

        elif poses.ndim == 3:
            # Sequence
            can_poses, R_metas, meta_list = canonicalize_batch(
                poses, prev_z=self._prev_z
            )
            # Update state to last valid z
            if meta_list and meta_list[-1]["valid"]:
                self._prev_z = meta_list[-1]["z_axis"]
            if return_metadata:
                return can_poses, R_metas, meta_list
            return can_poses, R_metas

        else:
            raise ValueError(f"Expected (17,3) or (T,17,3), got shape {poses.shape}")
