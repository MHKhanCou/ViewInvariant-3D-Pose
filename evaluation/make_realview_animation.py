"""
The claim, as motion: eight cameras rotating into one shared frame.

The static figure (make_realview_figure) shows raw and canonical side by side.
This renders the operation between them. Nothing is re-inferred and nothing is
stylised: the endpoints are exactly the cached raw and canonical poses every
number in Chapter 5 derives from.

The middle is honest too. Canonicalization is a rotation, canonical = raw @ R,
so R is recovered per camera and interpolated along the geodesic from identity.
Every intermediate frame is therefore a real rotation of the real prediction,
not a linear blend between two endpoints.

Run:  python -m evaluation.make_realview_animation            # mp4 + gif
      python -m evaluation.make_realview_animation --check    # 3 stills only
"""

import argparse
import os
import shutil
import subprocess
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.make_explainer_figures import (_row_radius, draw_skeleton,
                                               mean_pairwise)
from evaluation.make_figures import INK, SURFACE
from evaluation.make_realview_figure import collect, PALETTE
from evaluation.run_eval import load_cache

OUT = os.path.join(REPO_ROOT, "thesis_artifacts", "figures")
REPORT_IMG = os.path.join(REPO_ROOT, "thesis_report", "images")
FFMPEG = shutil.which("ffmpeg") or glob.glob(os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
    "yt-dlp.FFmpeg*", "*", "bin", "ffmpeg.exe"))


def ffmpeg_path():
    if isinstance(FFMPEG, str):
        return FFMPEG
    return FFMPEG[0] if FFMPEG else None


def recover_rotation(src, dst):
    """The R with dst = src @ R, via SVD. Exact here: the map is a rotation."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def slerp_to(R, t):
    """Identity -> R along the geodesic, at fraction t."""
    c = (np.trace(R) - 1.0) / 2.0
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    if ang < 1e-8:
        return np.eye(3)
    K = (R - R.T) / (2.0 * np.sin(ang))          # skew generator
    a = ang * t
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def ease(x):
    """Smoothstep, so the rotation starts and stops rather than snapping."""
    return x * x * (3.0 - 2.0 * x)


def build(n_cams=8, subject="S1", sequence="Seq1"):
    cams, frame_i = collect(load_cache(), subject, sequence, n_cams, False)
    raw = [c["raw"] for c in cams]
    can = [c["canonical"] for c in cams]
    rots = [recover_rotation(r, k) for r, k in zip(raw, can)]
    # residual after recovering R; if the map were not a rotation this is large
    err = max(float(np.abs(r @ R - k).max()) for r, R, k in zip(raw, rots, can))
    return cams, raw, can, rots, frame_i, err


def draw(ax, poses, radius, title, sub, colour):
    ax.clear()
    ax.set_facecolor(SURFACE)
    for j, p in enumerate(poses):
        c = PALETTE[j % len(PALETTE)]
        draw_skeleton(ax, p, radius=radius, y_is_down=False,
                      colour_left=c, colour_right=c, alpha=0.85)
    ax.set_title(title, fontsize=15, color=colour, fontweight="bold", pad=6)
    ax.text2D(0.5, -0.04, sub, transform=ax.transAxes, ha="center",
              fontsize=11.5, color=INK)


def main():
    ap = argparse.ArgumentParser(description="The operation, animated")
    ap.add_argument("--cams", type=int, default=8)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--check", action="store_true", help="3 stills, no video")
    args = ap.parse_args()

    cams, raw, can, rots, frame_i, err = build(args.cams)
    d_raw, d_can = mean_pairwise(raw), mean_pairwise(can)
    drop = 100.0 * (1.0 - d_can / d_raw)
    print("cameras %d  frame %d  raw %.4f -> canonical %.4f  (-%.1f%%)"
          % (len(cams), frame_i, d_raw, d_can, drop))
    print("max |raw@R - canonical| = %.2e   (rotation recovered exactly)" % err)

    radius = max(_row_radius(raw, y_is_down=False),
                 _row_radius(can, y_is_down=False))

    HOLD, TURN, SETTLE = args.fps, int(args.fps * 2.6), int(args.fps * 1.8)
    total = HOLD + TURN + SETTLE

    frames_dir = os.path.join(OUT, "_anim")
    os.makedirs(frames_dir, exist_ok=True)
    for f in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(f)

    fig = plt.figure(figsize=(7.2, 6.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")

    picks = [0, HOLD + TURN // 2, total - 1] if args.check else range(total)
    for i in picks:
        if i < HOLD:
            t, azim = 0.0, -60.0
        elif i < HOLD + TURN:
            t, azim = ease((i - HOLD) / float(TURN - 1)), -60.0
        else:
            t = 1.0
            azim = -60.0 + 26.0 * ((i - HOLD - TURN) / float(SETTLE - 1))

        poses = [r @ slerp_to(R, t) for r, R in zip(raw, rots)]
        if t < 0.5:
            title, colour = "Raw predictions", "#eb6834"
            sub = "eight cameras, eight coordinate frames"
        else:
            title, colour = "After canonicalization", "#2a78d6"
            sub = "each camera canonicalized independently, no calibration"
        if 0.0 < t < 1.0:
            title, colour = "Rotating into the body frame", "#6a6a6a"
            sub = "the same rotation the method applies"

        draw(ax, poses, radius, title, sub, colour)
        ax.view_init(elev=12, azim=azim)
        d_now = mean_pairwise(poses)
        ax.text2D(0.5, 1.06, "mean pairwise distance  %.3f" % d_now,
                  transform=ax.transAxes, ha="center", fontsize=12.5,
                  color=INK, fontweight="bold")
        # No bbox_inches="tight" here: it crops to content, so the frame size
        # would drift as the skeletons move and h264 needs constant dimensions.
        fig.savefig(os.path.join(frames_dir, "f%04d.png" % i),
                    dpi=110, facecolor="white")
    plt.close(fig)

    if args.check:
        print("wrote 3 stills to %s" % frames_dir)
        return

    exe = ffmpeg_path()
    if not exe:
        print("ffmpeg not found - frames are in %s" % frames_dir)
        return

    mp4 = os.path.join(REPORT_IMG, "anim_realview.mp4")
    # pad to even dimensions; h264 requires it
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(frames_dir, "f%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                    "-movflags", "+faststart", mp4], check=True)
    gif = os.path.join(REPORT_IMG, "anim_realview.gif")
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(frames_dir, "f%04d.png"),
                    "-vf", "fps=12,scale=760:-1:flags=lanczos", gif], check=True)

    for p in (mp4, gif):
        print("wrote %s  (%.1f MB)" % (p, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
