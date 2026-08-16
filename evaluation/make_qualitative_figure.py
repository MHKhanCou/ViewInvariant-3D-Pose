"""
Qualitative results on real photographs, which the report otherwise lacks.

Every other figure in this report is a chart or a synthetic stick figure. For a
computer-vision thesis that is a conspicuous gap: a reader should be able to see
the system run on an actual person. This produces that figure, one row per
image, showing the input with detected 2D keypoints, the raw 3D prediction in
the camera frame, and the same prediction canonicalized.

Runs the real pipeline (`backend.estimate_poses`), not a reconstruction, so what
appears here is what the demonstration application produces.

Run:  ./venv/Scripts/python.exe -m evaluation.make_qualitative_figure
"""

import os
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.make_explainer_figures import draw_skeleton
from evaluation.make_figures import INK, INK2, SURFACE

OUT = os.path.join(REPO_ROOT, "thesis_artifacts", "figures")
REPORT_IMG = os.path.join(REPO_ROOT, "thesis_report", "images")
EXAMPLES = os.path.join(REPO_ROOT, "examples")

# Frames from the demonstration footage rather than stock photographs. The
# stock images in examples/ carry visible agency watermarks, which have no place
# in a submitted document, and video frames additionally show the system on the
# kind of input the application is actually given.
#
# The last row is deliberately a hard case. Seated and kneeling poses are where
# Section 5.9 measures the backbone to be weakest, and a qualitative figure that
# showed only successes would misrepresent it.
CANDIDATES = [
    {"video": "13605188_1080_1920_30fps (online-video-cutter.com).mp4", "frame": 45,
     "label": "Running, subject well separated from the background"},
    {"video": "5586539-uhd_3840_2160_25fps.mp4", "frame": 190,
     "label": "Standing at distance, subject small in frame"},
    {"video": "5586539-uhd_3840_2160_25fps.mp4", "frame": 60,
     "label": "Bent forward, a hard case:\nthe reconstruction is visibly poorer"},
]


def load_frame(spec):
    """Read either a still image or one frame of a video."""
    if "video" in spec:
        path = os.path.join(EXAMPLES, spec["video"])
        if not os.path.exists(path):
            return None, spec["video"]
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, spec["frame"])
        ok, bgr = cap.read()
        cap.release()
        return (bgr if ok else None), "%s @ frame %d" % (spec["video"][:24], spec["frame"])
    path = os.path.join(EXAMPLES, spec["image"])
    return cv2.imread(path), spec["image"]


def run_one(bgr, name):
    """Run the real pipeline on one frame; returns overlay and both poses."""
    from backend.inference import MAX_HEIGHT, estimate_poses
    from demo_live.visualize import draw_2d

    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        res = estimate_poses(rgb)
    except Exception as exc:                      # detection can legitimately fail
        print("   %s: %s" % (name[:40], exc))
        return None
    if res is None or float(np.mean(res["scores"])) < 0.15:
        return None

    # The returned keypoints live in the DOWNSCALED frame the detector saw
    # (backend/inference.py:96-102), not in the original image. Drawing them on
    # the full-resolution image puts the skeleton in the corner at the wrong
    # scale, so reproduce the same resize here before overlaying.
    h, w = bgr.shape[:2]
    if h > MAX_HEIGHT:
        bgr = cv2.resize(bgr, (int(round(w * MAX_HEIGHT / h)), MAX_HEIGHT))
    overlay = draw_2d(res["kpts_2d"], bgr.copy(), scores=res["scores"])
    return {
        "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        "raw": np.asarray(res["raw_root_relative"], dtype=np.float64),
        "canonical": np.asarray(res["view_invariant"], dtype=np.float64),
        "reliability": float(res.get("reliability", float("nan"))),
        "conf": float(np.mean(res["scores"])),
        "name": name,
    }


def main():
    rows = []
    print("Running the pipeline on demonstration footage...")
    for spec in CANDIDATES:
        bgr, name = load_frame(spec)
        if bgr is None:
            print("   unreadable:", name[:50])
            continue
        r = run_one(bgr, name)
        if r:
            r["label"] = spec.get("label", "")
            rows.append(r)
            print("   ok: %-40s conf %.2f" % (name[:40], r["conf"]))
    if not rows:
        raise SystemExit("no frame produced a usable detection")

    n = len(rows)
    # 6.3 in is too narrow for three column headers side by side: they collide,
    # and the wrapped row labels clip. 7.8 in still prints at 0.8 scale.
    fig = plt.figure(figsize=(7.8, 2.62 * n))
    fig.patch.set_facecolor(SURFACE)

    for i, r in enumerate(rows):
        ax = fig.add_subplot(n, 3, i * 3 + 1)
        ax.imshow(r["overlay"])
        ax.set_facecolor(SURFACE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(r["label"], fontsize=10, color=INK2, labelpad=6,
                      wrap=True)

        ax = fig.add_subplot(n, 3, i * 3 + 2, projection="3d")
        ax.set_facecolor(SURFACE)
        # Camera-frame poses carry image-vertical on y with down positive.
        draw_skeleton(ax, r["raw"], y_is_down=True)

        ax = fig.add_subplot(n, 3, i * 3 + 3, projection="3d")
        ax.set_facecolor(SURFACE)
        # Canonical poses carry body-up on +y.
        draw_skeleton(ax, r["canonical"], y_is_down=False)

    for x, t in ((0.19, "Input with detected 2D keypoints"),
                 (0.53, "3D prediction, camera frame"),
                 (0.85, "Same prediction, canonical frame")):
        fig.text(x, 0.965, t, ha="center", fontsize=11, color=INK)

    fig.text(0.5, 0.055,
             "Frames from the demonstration footage, neither evaluation dataset.\n"
             "The canonical column is what a downstream task consumes:\n"
             "it does not depend on where the camera stood.",
             ha="center", va="top", fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.115,
                        wspace=0.02, hspace=0.12)

    os.makedirs(OUT, exist_ok=True)
    fig.text(0.01, -0.022, "Source: examples/ via backend.estimate_poses",
             fontsize=8.5, color=INK2)
    p = os.path.join(OUT, "fig_qualitative.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    os.makedirs(REPORT_IMG, exist_ok=True)
    shutil.copy(p, os.path.join(REPORT_IMG, "fig_qualitative.png"))
    print("\nWrote %s  (%d rows)" % (p, n))


if __name__ == "__main__":
    main()
