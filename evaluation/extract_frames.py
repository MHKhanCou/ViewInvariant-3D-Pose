"""
Extract frames from local MPI-INF-3DHP AVIs for multi-camera cross-view evaluation.

Reads sequentially from frame 0 (no seeking — AVI seeks are unreliable), writes
frames_cam{c}/frame_%06d.jpg matching the existing extraction convention so
annot.mat frame indices line up 1:1 with jpg numbering.

Run:  ./venv/Scripts/python.exe -m evaluation.extract_frames
"""

import os
import cv2

MPI_ROOT = "E:/thesis/mpi_inf_3dhp"

# (subject_seq_dir, camera ids, number of frames)
JOBS = [
    ("S1/Seq1", [0, 1, 2, 4, 5, 6, 7, 8], 80),   # dev + held-out pairs
    ("S2/Seq1", [0, 1], 80),                      # held-out subject
]


def extract(seq_dir, cam, n_frames):
    video_path = os.path.join(MPI_ROOT, seq_dir, "imageSequence", f"video_{cam}.avi")
    out_dir = os.path.join(MPI_ROOT, seq_dir, f"frames_cam{cam}")
    if not os.path.exists(video_path):
        print(f"  MISSING {video_path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    # Skip work already done (existing 50-frame extractions stay byte-identical
    # only if we don't rewrite them; re-encoding jpg would change pixels).
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if existing >= n_frames:
        print(f"  {seq_dir} cam{cam}: {existing} frames already present, skip")
        return existing

    cap = cv2.VideoCapture(video_path)
    written = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  {seq_dir} cam{cam}: stream ended at frame {i}")
            break
        out_path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        if not os.path.exists(out_path):  # never overwrite prior extraction
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
    cap.release()
    print(f"  {seq_dir} cam{cam}: {written} frames ready in {out_dir}")
    return written


def main():
    for seq_dir, cams, n in JOBS:
        print(f"{seq_dir}:")
        for cam in cams:
            extract(seq_dir, cam, n)


if __name__ == "__main__":
    main()
