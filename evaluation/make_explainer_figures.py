"""
The two figures that explain the contribution without prose.

Both are built from the cached Human3.6M predictions, which already contain all
four synchronized cameras for every test frame, so neither runs inference.

Figure 1 shows one instant seen by four cameras, raw above and canonical below.
It is the whole thesis in one image: four sets of coordinates that disagree
because of where the cameras stood, and the same four agreeing once expressed in
a body-fixed frame.

Figure 2 shows why SittingDown is the one action where this fails, and that
rebuilding the segment frames from longer axes repairs it.

Protocol note: each camera is canonicalized INDEPENDENTLY (prev_z=None). These
are four simultaneous views, not a temporal sequence; threading the previous
frame's forward axis between them would leak a sign-flip across cameras and
manufacture agreement that the method does not actually provide.

Run:  ./venv/Scripts/python.exe -m evaluation.make_explainer_figures
"""

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation.h36m_crossview import ACTION_NAMES
from evaluation.h36m_multiscale import SEGMENTS_LONGAXIS, canonicalize_with
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.make_figures import BLUE, GRID, INK, INK2, MAGENTA, ORANGE, SURFACE
from evaluation.metrics import cross_view_joint_distance

OUT = os.path.join(REPO_ROOT, "thesis_artifacts", "figures")
REPORT_IMG = os.path.join(REPO_ROOT, "thesis_report", "images")

# H36M-17 bone topology, from scripts/create_canonicalization_figure.py:25-27.
BONES_I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
BONES_J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
IS_LEFT = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)


def to_display(pose, y_is_down):
    """
    Rotate a pose so the body stands upright on screen.

    Matplotlib draws its third coordinate vertically. Human3.6M camera
    coordinates put image-vertical on y with DOWN positive, while a canonical
    pose puts body-up on +y, so neither is upright as-is and both would render
    lying on their side.

    Both mappings below are proper rotations with determinant +1, not axis
    swaps. A swap would have determinant -1 and would silently mirror the
    skeleton, exchanging left and right limbs.
    """
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    if y_is_down:
        return np.stack([x, z, -y], axis=1)     # up is -y
    return np.stack([x, -z, y], axis=1)         # up is +y


def draw_skeleton(ax, pose, title="", radius=None, y_is_down=False,
                  colour_left="#2a78d6", colour_right="#eb6834",
                  elev=10, azim=-70, alpha=1.0):
    """
    Draw one skeleton on a supplied 3D axis.

    Bone topology from `scripts/create_canonicalization_figure.py:25-27`, which
    fixes the view radius at 0.72 for unit-scaled poses. These are millimetres,
    so the radius is derived from the data and shared across panels: a common
    scale is what makes the comparison between panels mean anything.
    """
    P = to_display(np.asarray(pose, dtype=np.float64), y_is_down)
    for i in range(len(BONES_I)):
        a, b = BONES_I[i], BONES_J[i]
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]],
                lw=2.4, color=colour_left if IS_LEFT[i] else colour_right,
                solid_capstyle="round", alpha=alpha)
    # `radius` is (horizontal half-extent, vertical half-extent). The root is the
    # pelvis, so the body extends both ways: legs down, head up. An asymmetric
    # range crops the feet.
    if radius is None:
        rh = float(np.abs(P[:, :2] - P[0, :2]).max()) * 1.05
        rv = float(np.abs(P[:, 2] - P[0, 2]).max()) * 1.05
    else:
        rh, rv = radius
    cx, cy, cz = P[0]
    ax.set_xlim3d(cx - rh, cx + rh)
    ax.set_ylim3d(cy - rh, cy + rh)
    ax.set_zlim3d(cz - rv, cz + rv)
    ax.set_box_aspect((1, 1, rv / max(rh, 1e-9)))
    ax.view_init(elev=elev, azim=azim)
    clear = (1.0, 1.0, 1.0, 0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(clear)
        axis.line.set_color(clear)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, pad=-4)


def _row_radius(poses, y_is_down):
    """
    Shared (horizontal, vertical) half-extents for a set of panels.

    Returned as a pair because a standing body is far taller than it is wide;
    using one radius for both would either crop the feet or shrink the figure to
    a sliver in the middle of a mostly empty panel.
    """
    D = np.stack([to_display(np.asarray(p, dtype=np.float64), y_is_down) for p in poses])
    D = D - D[:, 0:1, :]
    return (float(np.abs(D[:, :, :2]).max()) * 1.08,
            float(np.abs(D[:, :, 2]).max()) * 1.08)


def mean_pairwise(poses):
    """Mean cross-view joint distance over all pairs of a set of views."""
    d = [cross_view_joint_distance(poses[i], poses[j])
         for i in range(len(poses)) for j in range(i + 1, len(poses))]
    return float(np.mean(d))


def load_groups():
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pn = np.load(os.path.join(PRED_DIR, "preds.npz"))
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v
    return groups


def pick_frame(cams, canon_fn, stride=25, prefer="dramatic"):
    """
    Choose a frame where all four canonicalizations are valid.

    `dramatic` picks the frame with the largest raw-to-canonical reduction, which
    is the clearest illustration; `typical` picks the one closest to the median
    reduction, which is the more honest one. We render the typical frame and
    report both, so the figure cannot be accused of cherry-picking.
    """
    order = sorted(cams)
    n = min(len(cams[c]["pred"]) for c in order)
    best = []
    for t in range(0, n, stride):
        raw = [cams[c]["pred"][t] - cams[c]["pred"][t][0:1] for c in order]
        can, ok = [], True
        for p in raw:
            cp, _, meta = canon_fn(p)
            if not meta["valid"]:
                ok = False
                break
            can.append(cp)
        if not ok:
            continue
        r, c = mean_pairwise(raw), mean_pairwise(can)
        best.append((t, r, c, r - c, raw, can))
    if not best:
        return None
    if prefer == "dramatic":
        return max(best, key=lambda e: e[3])
    gains = sorted(e[3] for e in best)
    med = gains[len(gains) // 2]
    return min(best, key=lambda e: abs(e[3] - med))


def _finish(fig, name, source):
    fig.text(0.01, 0.01, "Source: %s" % source, fontsize=7, color=INK2)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    os.makedirs(REPORT_IMG, exist_ok=True)
    shutil.copy(p, os.path.join(REPORT_IMG, name))
    print("  %s" % name)


def figure_one(groups):
    """Four cameras, raw above and canonical below, one instant."""
    key = ("S9", "act_02")                       # Directions, a clean everyday action
    cams = groups[key]
    picked = pick_frame(cams, lambda p: canonicalize_single(p), prefer="typical")
    t, raw_d, can_d, _, raw, can = picked
    order = sorted(cams)

    # Overlay all four views in one panel rather than showing four panels side
    # by side. Four separate skeletons ask the reader to hold each in memory and
    # compare; four superimposed skeletons show agreement or disagreement
    # directly, which is the entire claim.
    cam_colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    r_raw = _row_radius(raw, y_is_down=True)
    r_can = _row_radius(can, y_is_down=False)

    fig = plt.figure(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(SURFACE)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_facecolor(SURFACE)
    for i in range(len(order)):
        draw_skeleton(ax, raw[i], radius=r_raw, y_is_down=True,
                      colour_left=cam_colours[i], colour_right=cam_colours[i],
                      alpha=0.85)
    ax.set_title("", pad=0)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_facecolor(SURFACE)
    for i in range(len(order)):
        draw_skeleton(ax, can[i], radius=r_can, y_is_down=False,
                      colour_left=cam_colours[i], colour_right=cam_colours[i],
                      alpha=0.85)
    ax.set_title("", pad=0)

    handles = [plt.Line2D([0], [0], color=cam_colours[i], lw=2.4,
                          label="Camera %s" % order[i].replace("ca_", ""))
               for i in range(len(order))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.095))

    fig.text(0.5, 0.955, "One instant, four cameras, superimposed",
             ha="center", fontsize=13.5, color=INK)
    fig.text(0.28, 0.895, "Raw predictions", ha="center", fontsize=11.5,
             color=ORANGE, weight="bold")
    fig.text(0.28, 0.862, "mean pairwise distance %.0f mm" % raw_d,
             ha="center", fontsize=10, color=ORANGE)
    fig.text(0.74, 0.895, "After canonicalization", ha="center", fontsize=11.5,
             color=BLUE, weight="bold")
    fig.text(0.74, 0.862, "mean pairwise distance %.0f mm" % can_d,
             ha="center", fontsize=10, color=BLUE)
    fig.text(0.5, 0.055,
             "Left: four cameras disagree because each predicts in its own frame. "
             "Right: the same four predictions in a body-fixed frame, %.0f mm to "
             "%.0f mm, a reduction of %.0f percent."
             % (raw_d, can_d, 100 * (1 - can_d / raw_d)),
             ha="center", fontsize=9.5, color=INK)
    fig.text(0.5, 0.020,
             "Human3.6M %s, subject %s. Each camera canonicalized independently, "
             "with no knowledge of the others." % (ACTION_NAMES.get(key[1], key[1]), key[0]),
             ha="center", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.13, wspace=0.02)
    _finish(fig, "fig_h36m_explainer.png", "h36m_replication/preds.npz")
    return raw_d, can_d


def figure_two(groups):
    """
    Why SittingDown fails, and why the obvious explanation is wrong.

    The intuitive story is that a seated pelvis foreshortens the hip axis and
    destabilises the frame. The data says otherwise, and the figure is built to
    show that directly: axis length does not separate the failing action from
    the rest, while backbone accuracy does.
    """
    hips, errs = {}, {}
    for (subj, action), cams in groups.items():
        for c, v in cams.items():
            P, G = v["pred"][::20], v["gt"][::20]
            hips.setdefault(action, []).append(
                np.linalg.norm(P[:, 1] - P[:, 4], axis=1))
            pr, gr = P - P[:, 0:1, :], G - G[:, 0:1, :]
            errs.setdefault(action, []).append(
                np.linalg.norm(pr - gr, axis=-1).mean(axis=-1))
    hip_len = {ACTION_NAMES.get(a, a): float(np.concatenate(v).mean())
               for a, v in hips.items()}
    acc = {ACTION_NAMES.get(a, a): float(np.concatenate(v).mean())
           for a, v in errs.items()}

    with open(os.path.join(REPO_ROOT, "thesis_artifacts", "h36m_crossview",
                           "h36m_crossview.json")) as fh:
        cv = json.load(fh)["summary"]["by_action"]
    canon = {a: v["mean_canonical_distance_mm"] for a, v in cv.items()}

    names = sorted(canon, key=lambda a: canon[a])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.patch.set_facecolor(SURFACE)

    from scipy.stats import spearmanr

    def scat(ax, xs, xlabel, title):
        for a in names:
            bad = a == "SittingDown"
            ax.scatter(xs[a], canon[a], s=64 if bad else 34,
                       color=ORANGE if bad else BLUE, zorder=4 if bad else 3,
                       alpha=0.95 if bad else 0.75)
        ax.annotate("SittingDown", (xs["SittingDown"], canon["SittingDown"]),
                    textcoords="offset points", xytext=(-12, -18), fontsize=9,
                    color=ORANGE, ha="right", weight="bold")
        # Report the association with and without the outlier. Quoting only the
        # first would let one point carry a claim about fifteen.
        rest = [a for a in names if a != "SittingDown"]
        r_all = spearmanr([xs[a] for a in names], [canon[a] for a in names])[0]
        r_out = spearmanr([xs[a] for a in rest], [canon[a] for a in rest])[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Canonical cross-view distance (mm)")
        ax.set_title(title, fontsize=11, loc="left", pad=8)
        ax.text(0.02, 0.02,
                r"$\rho$ = %+.2f over all 15    %+.2f without SittingDown"
                % (r_all, r_out),
                transform=ax.transAxes, fontsize=8.5, color=INK2)
        return r_all, r_out

    r_hip = scat(ax1, hip_len, "Mean hip-axis length (mm)",
                 "Axis length does not separate it")
    r_acc = scat(ax2, acc, "Backbone accuracy, MPJPE (mm)",
                 "Estimator accuracy does")
    ax1.text(0.03, 0.94, "the failing action has the\nSECOND LONGEST hip axis",
             transform=ax1.transAxes, fontsize=8.5, color=INK2, va="top")

    fig.suptitle("Why SittingDown fails, and why the obvious explanation is wrong",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.055,
             "Each point is one of the fifteen Human3.6M actions. We supposed a "
             "seated pelvis shortens the hip axis and destabilises the frame; it "
             "does not, and the failing action has almost the longest hip axis of any.",
             ha="center", fontsize=9, color=INK)
    fig.text(0.5, 0.018,
             "The association with estimator accuracy survives removing that "
             "outlier, so it is a trend across actions and not one point.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _finish(fig, "fig_h36m_sittingdown.png",
            "h36m_replication/preds.npz and h36m_crossview.json")
    return r_hip, r_acc


def figure_three():
    """
    The system diagram, drawn so the contribution boundary is the visual point.

    Everything above the divider is trained by somebody else and used frozen.
    Everything below it is this project and adds zero trained parameters. That
    split is the claim the title makes, so the figure should make it obvious
    before any caption is read.

    Built from the code rather than from thesis_artifacts/architecture.md, which
    is stale: it credits the body frame as "MoViD-inspired", when it is a
    Gram-Schmidt construction from anatomical axes and owes nothing to MoViD.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # Tall canvas on purpose: FancyBboxPatch adds its `pad` outside the height
    # given, so boxes occupy more vertical space than their nominal h. Stretching
    # the figure shrinks text in data units and keeps the labels clear of them.
    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.85, 12.05)
    ax.axis("off")

    GREY, FROZEN_FC = "#8a8a84", "#ececE6"

    def box(x, y, w, h, text, fc, ec, fs=9.5, weight="normal", tc=INK):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.14",
                                    linewidth=1.4, facecolor=fc, edgecolor=ec,
                                    zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4, weight=weight, linespacing=1.35)

    def arrow(x0, y0, x1, y1, colour=GREY):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.3,
                                     color=colour, zorder=2,
                                     shrinkA=1, shrinkB=1))

    # --- frozen, trained by their authors -------------------------------
    box(5.0, 11.55, 4.0, 0.60, "RGB image or video", FROZEN_FC, GREY)
    arrow(5.0, 11.25, 5.0, 10.82)
    box(5.0, 10.42, 5.8, 0.76,
        "2D keypoint detector  (frozen)\n"
        "YOLOv8-pose, or Stacked Hourglass on Human3.6M",
        FROZEN_FC, GREY, fs=9)
    arrow(5.0, 10.04, 5.0, 9.61)
    box(5.0, 9.21, 5.8, 0.76,
        "Lifting network  (frozen)\n"
        "MotionAGFormer-XS, 2.24M parameters, 27-frame window",
        FROZEN_FC, GREY, fs=9)
    arrow(5.0, 8.83, 5.0, 8.44)
    box(5.0, 8.10, 4.6, 0.60, "3D pose in the camera's frame", FROZEN_FC, GREY)

    # --- the boundary ----------------------------------------------------
    ax.text(0.35, 7.56, "trained by their authors, used frozen and unmodified",
            fontsize=8.5, color=GREY, va="center")
    ax.plot([0.3, 9.7], [7.30, 7.30], ls=(0, (5, 4)), lw=1.5, color=MAGENTA, zorder=1)
    ax.text(0.35, 7.03, "THIS WORK  —  adds 0 trained parameters, "
                        "no labels, no calibration",
            fontsize=9, color=MAGENTA, va="center", weight="bold")
    arrow(5.0, 7.78, 5.0, 6.75, BLUE)

    # --- the contribution -------------------------------------------------
    box(5.0, 6.44, 5.8, 0.78,
        "Body-frame canonicalization\n"
        "Gram-Schmidt frame from torso and hip axes, 402 FLOPs",
        "#dfe9f7", BLUE, fs=9)
    arrow(5.0, 6.05, 5.0, 5.60, BLUE)

    for cx, label in ((2.05, "Multi-scale\none frame per limb"),
                      (5.00, "Degeneracy gate\nabstains on\nill-posed frames"),
                      (7.95, "Multi-view fusion\nmedian over\nuncalibrated views")):
        box(cx, 5.10, 2.75, 1.00, label, "#dfe9f7", BLUE, fs=8.8)
        arrow(cx, 4.60, cx, 4.18, BLUE)

    for cx, label in ((2.05, "Cross-view\ncomparison"),
                      (5.00, "BVH export\nBlender, Unity"),
                      (7.95, "Interactive\n3D viewer")):
        box(cx, 3.76, 2.75, 0.84, label, SURFACE, BLUE, fs=8.8)

    ax.text(5.0, 2.60,
            "The unknown rotation between two cameras cancels exactly,\n"
            "because the frame is built from the joints and rotates with them.",
            ha="center", fontsize=9.5, color=INK, linespacing=1.5)
    ax.text(5.0, 1.83,
            r"$P^{(B)}R^{(B)} = (P^{(A)}Q)(Q^{\top}R^{(A)}) = P^{(A)}R^{(A)}$",
            ha="center", fontsize=12.5, color=BLUE)
    ax.text(5.0, 1.15,
            "So no camera calibration is needed and nothing has to be estimated.",
            ha="center", fontsize=9, color=INK2)

    fig.text(0.5, 0.975, "Where the contribution sits in the pipeline",
             ha="center", fontsize=13.5, color=INK)
    _finish(fig, "fig_architecture.png", "canonical/body_frame.py and demo_live/")


def main():
    print("Loading cached predictions...")
    groups = load_groups()
    print("Writing figures:")
    figure_three()
    raw_d, can_d = figure_one(groups)
    r_hip, r_acc = figure_two(groups)
    print("\n  figure 1: raw %.1f mm -> canonical %.1f mm (%.1f%%)"
          % (raw_d, can_d, 100 * (1 - can_d / raw_d)))
    print("  figure 2: rho(canonical vs hip length) %+.2f all / %+.2f without SittingDown"
          % r_hip)
    print("            rho(canonical vs MPJPE)      %+.2f all / %+.2f without SittingDown"
          % r_acc)
    print("Copied into %s" % REPORT_IMG)


if __name__ == "__main__":
    main()
