"""
The claim on real photographs, from real cameras.

Every other cross-view figure in this report is a stick figure. This one puts
the actual camera images on top: one instant of MPI-INF-3DHP seen by several
synchronized cameras, the photographs themselves, then the raw predictions
superimposed, then the same predictions canonicalized.

Nothing is re-inferred. The poses come from the same cached predictions every
metric in Chapter 5 derives from, so the numbers printed here are the measured
ones.

Run:  python -m evaluation.make_realview_figure --cams 4
"""

import argparse
import json
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

from evaluation.make_explainer_figures import (_row_radius, draw_skeleton,
                                               mean_pairwise)
from evaluation.make_figures import INK, SURFACE
from evaluation.protocol import get_frame_dir
from evaluation.run_eval import load_cache

OUT = os.path.join(REPO_ROOT, "thesis_artifacts", "figures")
REPORT_IMG = os.path.join(REPO_ROOT, "thesis_report", "images")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#9467bd", "#17becf", "#8c564b", "#bcbd22"]


def frame_path(subject, sequence, cam_id, center, prefix="frames_"):
    """The photograph a cached prediction was centred on."""
    d = get_frame_dir(subject, sequence, cam_id, prefix)
    if not os.path.isdir(d):
        return None
    names = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    return os.path.join(d, names[center]) if center < len(names) else None


def collect(cache, subject="S1", sequence="Seq1", n_cams=4, dynamic=False):
    """Cameras that share a valid frame, with their images and poses."""
    # Static and dynamic windows share the cache; the dynamic ones carry a
    # '_dynamic' suffix and their frames live under a different prefix.
    stem = "%s_%s_cam" % (subject, sequence)
    keys = sorted(k for k in cache if k.startswith(stem)
                  and k.endswith("_dynamic") == dynamic)
    prefix = "frames_dyn_" if dynamic else "frames_"
    if not keys:
        raise SystemExit("no cached %s cameras for %s %s"
                         % ("dynamic" if dynamic else "static", subject, sequence))

    valid = np.ones(min(len(cache[k]["valid"]) for k in keys), dtype=bool)
    for k in keys:
        valid &= cache[k]["valid"][:len(valid)].astype(bool)
    if not valid.any():
        raise SystemExit("no frame is valid in every camera")

    # The median-disagreement frame, not the most flattering one: the same
    # choice rule the two-view figure uses.
    idx = [i for i in range(len(valid)) if valid[i]]
    spread = [mean_pairwise([cache[k]["raw"][i] for k in keys]) for i in idx]
    center_i = idx[int(np.argsort(spread)[len(spread) // 2])]

    out = []
    for k in keys:
        cam_id = int(k.rsplit("cam", 1)[1].split("_")[0])
        centre = int(cache[k]["centers"][center_i])
        img = frame_path(subject, sequence, cam_id, centre, prefix)
        if img is None:
            continue
        out.append({"cam": cam_id, "image": img,
                    "raw": cache[k]["raw"][center_i],
                    "canonical": cache[k]["canonical"][center_i]})
        if len(out) == n_cams:
            break
    if len(out) < 2:
        raise SystemExit("need at least two cameras with extracted frames")
    return out, center_i


def main():
    ap = argparse.ArgumentParser(description="Real camera views + the operation")
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--sequence", default="Seq1")
    ap.add_argument("--cams", type=int, default=4, help="how many cameras (2-8)")
    ap.add_argument("--dynamic", action="store_true",
                    help="the walking/gesturing window instead of the static one")
    ap.add_argument("--out", default="fig_realview.png")
    args = ap.parse_args()

    cams, frame_i = collect(load_cache(), args.subject, args.sequence,
                            args.cams, args.dynamic)
    n = len(cams)
    raw = [c["raw"] for c in cams]
    can = [c["canonical"] for c in cams]
    d_raw, d_can = mean_pairwise(raw), mean_pairwise(can)

    fig = plt.figure(figsize=(3.1 * n, 7.8))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2 * n, height_ratios=[1.0, 1.35],
                          hspace=0.18, wspace=0.05)

    for j, c in enumerate(cams):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.imshow(cv2.cvtColor(cv2.imread(c["image"]), cv2.COLOR_BGR2RGB))
        ax.set_title("Camera %d" % c["cam"], fontsize=11.5,
                     color=PALETTE[j % len(PALETTE)], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(PALETTE[j % len(PALETTE)])
            s.set_linewidth(2.5)

    r_raw = _row_radius(raw, y_is_down=True)
    r_can = _row_radius(can, y_is_down=False)
    for col, (poses, radius, down) in enumerate(((raw, r_raw, True),
                                                 (can, r_can, False))):
        ax = fig.add_subplot(gs[1, col * n:(col + 1) * n], projection="3d")
        ax.set_facecolor(SURFACE)
        for j, p in enumerate(poses):
            colour = PALETTE[j % len(PALETTE)]
            draw_skeleton(ax, p, radius=radius, y_is_down=down,
                          colour_left=colour, colour_right=colour, alpha=0.85)

    fig.text(0.5, 0.965, "%s %s, frame %d: one instant, %d real cameras"
             % (args.subject, args.sequence, frame_i, n),
             ha="center", fontsize=14, color=INK)
    # MPI-INF-3DHP predictions are cached in normalised units, not millimetres,
    # which is why Chapter 5 reports this dataset as a percentage. Labelling
    # these as mm would invent a scale the cache does not carry.
    fig.text(0.28, 0.505, "Raw predictions", ha="center", fontsize=12.5,
             color="#eb6834", weight="bold")
    fig.text(0.28, 0.475, "mean pairwise distance %.3f" % d_raw,
             ha="center", fontsize=10.5, color="#eb6834")
    fig.text(0.73, 0.505, "After canonicalization", ha="center", fontsize=12.5,
             color="#2a78d6", weight="bold")
    fig.text(0.73, 0.475, "mean pairwise distance %.3f" % d_can,
             ha="center", fontsize=10.5, color="#2a78d6")
    fig.text(0.5, 0.045,
             "Top: the actual photographs. Bottom: the predictions made from "
             "them, %.0f%% closer together after canonicalization "
             "(normalised units, as Chapter 5 reports this dataset)."
             % (100 * (1 - d_can / d_raw)),
             ha="center", fontsize=11, color=INK)
    fig.text(0.5, 0.018,
             "MPI-INF-3DHP. Each camera canonicalized independently, with no "
             "knowledge of the others and no calibration.",
             ha="center", fontsize=9.5, color="#555555")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, args.out)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.isdir(REPORT_IMG):
        shutil.copy2(path, os.path.join(REPORT_IMG, args.out))

    # The three numbers printed on the figure are quoted in the report, so they
    # have to be auditable like every other number in it. Written beside the
    # figure rather than inferred from the caption.
    with open(os.path.join(OUT, "realview.json"), "w", encoding="utf-8") as fh:
        json.dump({"subject": args.subject, "sequence": args.sequence,
                   "frame": int(frame_i), "n_cameras": int(n),
                   "cameras": [int(c["cam"]) for c in cams],
                   "mean_pairwise_raw": float(d_raw),
                   "mean_pairwise_canonical": float(d_can),
                   "improvement_pct": float(100 * (1 - d_can / d_raw)),
                   "units": "normalised, as the MPI-INF-3DHP cache stores them"},
                  fh, indent=1)
    print("wrote %s   raw %.4f -> canonical %.4f (normalised units, -%.1f%%)"
          % (path, d_raw, d_can, 100 * (1 - d_can / d_raw)))


if __name__ == "__main__":
    main()
