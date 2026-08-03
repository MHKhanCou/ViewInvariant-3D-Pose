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

ALL_CAMS = [0, 1, 2, 4, 5, 6, 7, 8]

# (subject_seq_dir, camera ids, n_frames, start_frame, dirname_suffix)
# Static windows start at frame 0. Dynamic windows are chosen by GT motion +
# body-rotation scan (see thesis_artifacts/fusion/): they contain large net
# body rotation, so per-frame view quality genuinely varies — the condition
# under which adaptive view selection can be tested.
JOBS = [
    ("S1/Seq1", ALL_CAMS, 80, 0, ""),        # dev + held-out pairs
    ("S2/Seq1", [0, 1], 80, 0, ""),          # held-out subject
    # Dynamic: 138 deg net body rotation, 2.4x the motion of the static window.
    ("S1/Seq1", ALL_CAMS, 146, 576, "_dyn"),
    # Dynamic held-out subject: 142 deg net rotation.
    ("S2/Seq2", ALL_CAMS, 146, 1908, "_dyn"),
]


def extract(seq_dir, cam, n_frames, start_frame=0, suffix=""):
    video_path = os.path.join(MPI_ROOT, seq_dir, "imageSequence", f"video_{cam}.avi")
    out_dir = os.path.join(MPI_ROOT, seq_dir, f"frames{suffix}_cam{cam}")
    if not os.path.exists(video_path):
        print(f"  MISSING {video_path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    # Files are named by TRUE source-frame index so annot.mat lookups stay valid
    # for windows that do not start at frame 0.
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if existing >= n_frames:
        print(f"  {seq_dir} cam{cam}{suffix}: {existing} frames present, skip")
        return existing

    cap = cv2.VideoCapture(video_path)
    written = 0
    for i in range(start_frame + n_frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  {seq_dir} cam{cam}{suffix}: stream ended at frame {i}")
            break
        if i < start_frame:
            continue  # decode-and-discard: AVI seeking is unreliable
        out_path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        if not os.path.exists(out_path):  # never overwrite prior extraction
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
    cap.release()
    print(f"  {seq_dir} cam{cam}{suffix}: {written} frames "
          f"[{start_frame}..{start_frame + written - 1}] -> {out_dir}")
    return written


def main():
    for seq_dir, cams, n, start, suffix in JOBS:
        print(f"{seq_dir}{suffix or ' (static)'}:")
        for cam in cams:
            extract(seq_dir, cam, n, start, suffix)


if __name__ == "__main__":
    main()
