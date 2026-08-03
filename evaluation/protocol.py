"""
MPI-INF-3DHP cross-view evaluation protocol.

Discovers every camera with extracted frames, produces per-camera frame lists
and an auditable pairing table over all camera combinations per sequence:
subject, sequence, camera A/B, exact synchronized frame IDs.

The corrected protocol evaluates only frames with a FULL 27-frame temporal
window (centers 13..N-14), so no replicate padding enters any reported number.

DEV_PAIR marks the development pair (all historical tuning happened there);
every other pair is held-out. S2 is the held-out subject.
"""

import os
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MPI_ROOT = "E:/Thesis/mpi_inf_3dhp"

# Sequences eligible for evaluation (have annot.mat + calibration).
SEQUENCES = [("S1", "Seq1"), ("S2", "Seq1"), ("S2", "Seq2")]

# Frame-directory prefixes = experimental conditions.
#   "frames_"     static window (frames 0..79), low body rotation
#   "frames_dyn_" dynamic window chosen by GT motion/rotation scan
CONDITIONS = {"frames_": "static", "frames_dyn_": "dynamic"}

# The pair every prior number/tuning decision was made on. Never held-out.
DEV_PAIR = ("S1", "Seq1", 0, 1)

WINDOW = 27
HALF = WINDOW // 2  # 13


def get_frame_dir(subject, sequence, camera_id, prefix="frames_"):
    """Directory containing extracted JPEG frames for one camera/condition."""
    return os.path.join(MPI_ROOT, subject, sequence, f"{prefix}cam{camera_id}")


def get_frame_paths(subject, sequence, camera_id, prefix="frames_"):
    """Sorted list of extracted frame paths for a camera."""
    frame_dir = get_frame_dir(subject, sequence, camera_id, prefix)
    if not os.path.exists(frame_dir):
        return []
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    return [os.path.join(frame_dir, f) for f in frames]


def frame_id_from_path(path):
    """
    True source-video frame index parsed from 'frame_%06d.jpg'.

    Extractions need not start at frame 0 (dynamic windows are extracted from
    the middle of a sequence), so positional indices are NOT valid annot.mat
    indices — every GT lookup must go through this.
    """
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def get_frame_ids(subject, sequence, camera_id):
    """True frame indices for a camera's extracted frames, sorted."""
    return [frame_id_from_path(p)
            for p in get_frame_paths(subject, sequence, camera_id)]


def discover_cameras():
    """
    Find every camera with extracted frames.

    Returns:
        list of dicts: {subject, sequence, camera, frame_paths, n_frames}
    """
    cameras = []
    for subject, sequence in SEQUENCES:
        seq_dir = os.path.join(MPI_ROOT, subject, sequence)
        if not os.path.exists(seq_dir):
            continue
        for name in sorted(os.listdir(seq_dir)):
            prefix = next((p for p in CONDITIONS
                           if name.startswith(p + "cam")), None)
            if prefix is None:
                continue
            cam = int(name[len(prefix) + 3:])
            paths = get_frame_paths(subject, sequence, cam, prefix)
            if paths:
                cameras.append({
                    "subject": subject,
                    "sequence": sequence,
                    "camera": cam,
                    "condition": CONDITIONS[prefix],
                    "prefix": prefix,
                    "frame_paths": paths,
                    "frame_ids": [frame_id_from_path(p) for p in paths],
                    "n_frames": len(paths),
                })
    return cameras


def evaluated_centers(n_frames):
    """Frame centers that have a full 27-frame window (no padding)."""
    return list(range(HALF, n_frames - HALF))


def build_pairing_table():
    """
    Auditable pairing table: all camera pairs within each sequence, matched by
    frame index (MPI-INF-3DHP cameras are hardware-synchronized: frame i in
    cam A corresponds to frame i in cam B).

    Returns:
        list of dicts: {subject, sequence, cam_a, cam_b, n_frames, frame_pairs,
                        is_dev_pair}
        frame_pairs lists only fully-windowed centers: (frame_idx, path_a, path_b)
    """
    cameras = discover_cameras()
    by_seq = {}
    for cam in cameras:
        # Pair only within the same sequence AND condition.
        by_seq.setdefault(
            (cam["subject"], cam["sequence"], cam["condition"]), []).append(cam)

    table = []
    for (subject, sequence, condition), cams in by_seq.items():
        cams = sorted(cams, key=lambda c: c["camera"])
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                a, b = cams[i], cams[j]
                n_matched = min(a["n_frames"], b["n_frames"])
                centers = evaluated_centers(n_matched)
                frame_pairs = [
                    (a["frame_ids"][k], a["frame_paths"][k], b["frame_paths"][k])
                    for k in centers
                ]
                is_dev = (condition == "static" and
                          (subject, sequence, a["camera"], b["camera"]) == DEV_PAIR)
                table.append({
                    "subject": subject,
                    "sequence": sequence,
                    "condition": condition,
                    "cam_a": a["camera"],
                    "cam_b": b["camera"],
                    "n_frames": len(frame_pairs),
                    "frame_pairs": frame_pairs,
                    "is_dev_pair": is_dev,
                })
                print(f"Pairing table: {subject}/{sequence} [{condition}] "
                      f"cam{a['camera']} <-> cam{b['camera']}: "
                      f"{len(frame_pairs)} fully-windowed frame pairs"
                      f"{'  [DEV PAIR]' if is_dev else ''}")
    return table


def load_frames_as_sequence(frame_paths):
    """Load frame paths as (N, H, W, 3) uint8 RGB array."""
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(frames) if frames else np.array([])


def create_27_frame_windows(n_frames, padding=HALF):
    """
    Centered 27-frame window indices; edge frames replicate-padded.
    LEGACY helper — the corrected protocol uses only full windows
    (see evaluated_centers) so reported numbers never contain padding.
    """
    windows = []
    for i in range(n_frames):
        lo = max(0, i - padding)
        hi = min(n_frames, i + padding + 1)
        windows.append((lo, hi))
    return windows
