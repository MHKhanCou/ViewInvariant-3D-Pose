"""
MPI-INF-3DHP cross-view evaluation protocol.

Loads synchronized camera pairs, creates 27-frame windows from actual
camera streams (not repeated single images), and runs the full pipeline
(raw root-relative prediction -> canonicalization -> comparison).

Produces an auditable pairing table: subject, sequence, camera A/B,
exact synchronized frame IDs.
"""

import os
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MPI_ROOT = "E:/Thesis/mpi_inf_3dhp"

# Available camera pairs with extracted RGB frames
# Format: (subject, sequence, camera_a, camera_b, n_frames_a, n_frames_b)
AVAILABLE_PAIRS = [
    ("S1", "Seq1", 0, 1, 50, 50),  # cam0 and cam1, 50 frames each
]


def get_frame_dir(subject, sequence, camera_id):
    """Get the directory containing extracted JPEG frames."""
    return os.path.join(MPI_ROOT, subject, sequence, f"frames_cam{camera_id}")


def get_frame_paths(subject, sequence, camera_id):
    """Get sorted list of frame paths for a camera."""
    frame_dir = get_frame_dir(subject, sequence, camera_id)
    if not os.path.exists(frame_dir):
        return []
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    return [os.path.join(frame_dir, f) for f in frames]


def build_pairing_table():
    """
    Build the auditable pairing table for all available camera pairs.

    Returns:
        list of dicts, each containing:
            subject, sequence, cam_a, cam_b, frame_pairs
            (frame_pairs is a list of (frame_idx, path_a, path_b))
    """
    table = []
    for subject, sequence, cam_a, cam_b, n_a, n_b in AVAILABLE_PAIRS:
        paths_a = get_frame_paths(subject, sequence, cam_a)
        paths_b = get_frame_paths(subject, sequence, cam_b)

        if not paths_a or not paths_b:
            print(f"WARNING: No frames for {subject}/{sequence} cam{cam_a} or cam{cam_b}")
            continue

        # Match by frame index (MPI-INF-3DHP frame i in cam0 = frame i in cam1)
        n_matched = min(len(paths_a), len(paths_b))
        frame_pairs = [
            (i, paths_a[i], paths_b[i])
            for i in range(n_matched)
        ]

        table.append({
            "subject": subject,
            "sequence": sequence,
            "cam_a": cam_a,
            "cam_b": cam_b,
            "n_frames": n_matched,
            "frame_pairs": frame_pairs,
        })

        print(f"Pairing table: {subject}/{sequence} cam{cam_a} <-> cam{cam_b}: {n_matched} frame pairs")

    return table


def load_frames_as_sequence(frame_paths):
    """
    Load a list of frame paths as a numpy array of images.

    Returns:
        frames: (N, H, W, 3) uint8 RGB array
    """
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(frames) if frames else np.array([])


def create_27_frame_windows(n_frames, padding=13):
    """
    Create centered 27-frame window indices for a sequence.
    Frames near edges are replicate-padded.

    Returns:
        list of (window_start, window_end) tuples for each output frame.
    """
    windows = []
    for i in range(n_frames):
        lo = max(0, i - padding)
        hi = min(n_frames, i + padding + 1)
        windows.append((lo, hi))
    return windows
