"""
Figure generation and two-view rendering for the canonicalization results.

One module serves both purposes deliberately. Every figure in the report is
produced here from a stored artifact, so a figure cannot drift from the number
it illustrates, and the same primitives render the synchronized two-view
comparison used in the defense. The report is the primary consumer; the viewer
is the same code called with different arguments.

What it will NOT do, and why:
  - No RGB panel. The prediction caches hold predicted joints and detector
    confidences, not frames. Showing the poses alone is also the more honest
    picture, since the claim is about 3D coordinates.
  - No camera glyphs. The method uses no extrinsics, and drawing cameras would
    illustrate information the pipeline never receives.

Run:  ./venv/Scripts/python.exe -m presentation.render            (all figures)
      ./venv/Scripts/python.exe -m presentation.render --twoview  (+ two-view)
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ART = os.path.join(REPO_ROOT, "thesis_artifacts")
IMG = os.path.join(REPO_ROOT, "thesis_report", "images")

# H36M-17 skeleton, drawn as five chains so limbs read as limbs.
CHAINS = [
    ([0, 7, 8, 9, 10], "#444444"),      # spine to head
    ([8, 11, 12, 13], "#1f77b4"),       # left arm
    ([8, 14, 15, 16], "#d62728"),       # right arm
    ([0, 4, 5, 6], "#1f77b4"),          # left leg
    ([0, 1, 2, 3], "#d62728"),          # right leg
]
INK = "#111111"
MUTED = "#8a8a8a"


def _load(rel):
    with open(os.path.join(ART, rel), encoding="utf-8") as f:
        return json.load(f)


def _style(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#666666")
    ax.tick_params(labelsize=8, colors="#333333", length=3)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_pose(ax, pose, color=None, alpha=1.0, lw=2.0, label=None, flip_y=False,
              plane="front"):
    """
    Project a pose for display. No analysis uses this projection.

    Two planes are offered because they show different things. The frontal
    plane (x, y) shows the pose. The transverse plane (x, z), seen from above,
    shows the azimuth, and azimuth is where most of the disagreement between two
    cameras of the same instant actually lives - a frontal view of two raw poses
    can look nearly identical while they are hundreds of millimetres apart.

    flip_y is for raw camera-frame poses, where Human3.6M's y axis points down
    and the subject would otherwise be drawn standing on their head. The
    canonical frame puts y along the torso, so canonical poses need no flip.
    """
    P = np.asarray(pose, dtype=float)
    if plane == "front":
        xy = np.column_stack([P[:, 0], -P[:, 1] if flip_y else P[:, 1]])
    else:
        xy = np.column_stack([P[:, 0], P[:, 2]])
    for chain, c in CHAINS:
        ax.plot(xy[chain, 0], xy[chain, 1], "-o", ms=2.6, lw=lw,
                color=color or c, alpha=alpha,
                label=label if (label and chain is CHAINS[0][0]) else None)
    return xy


def draw_frame_axes(ax, pose, scale=320.0, flip_y=False):
    """The two vectors TRIAD is built from, drawn on the raw pose."""
    P = np.asarray(pose, dtype=float)
    s = -1.0 if flip_y else 1.0
    root = np.array([P[0, 0], s * P[0, 1]])
    for j0, j1, colour in ((8, 0, "#2ca02c"), (1, 4, "#ff7f0e")):
        v = P[j0] - P[j1]
        v = np.array([v[0], s * v[1]])
        v = v / (np.linalg.norm(v) + 1e-9)
        ax.annotate("", xy=(root[0] + v[0] * scale, root[1] + v[1] * scale),
                    xytext=(root[0], root[1]),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.4))


# ---------------------------------------------------------------------------
# Figure 1: the three-level boundary. The primary contribution had no figure.
# ---------------------------------------------------------------------------

def fig_three_levels(out="fig_three_levels.png"):
    xs = _load("axis_law/axis_law.json")["by_subset"]["limb_levels"]
    mb = _load("axis_law/axis_law_motionbert.json")["by_subset"]["limb_levels"]

    fig = plt.figure(figsize=(12.5, 4.3))
    gs = GridSpec(1, 3, wspace=0.32, figure=fig)

    # Level 1 - between frame constructions. Holds.
    ax = fig.add_subplot(gs[0])
    for pts, lbl, c, m in ((xs, "MotionAGFormer-XS", "#1f77b4", "o"),
                           (mb, "MotionBERT", "#d62728", "s")):
        art = _load("axis_law/axis_law%s.json"
                    % ("" if lbl.startswith("MotionAG") else "_motionbert"))
        limbs = [p for p in art["points"]
                 if any(k in p["label"] for k in ("arm", "leg"))]
        ax.scatter([p["r_over_L"] for p in limbs], [p["d"] for p in limbs],
                   s=42, c=c, marker=m, label=lbl, zorder=3, alpha=0.85)
    ax.set_xlabel(r"lever arm over axis length  $\bar{r}/L$", fontsize=9)
    ax.set_ylabel("cross-view distance (mm)", fontsize=9)
    ax.set_title("Level 1  BETWEEN constructions\nHOLDS   " +
                 r"$\rho=+0.90,\ +0.88$", fontsize=9.5, color="#1a7f37")
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    # Level 2 - between frames of one construction. Fails.
    # The reliability curve is drawn only to set an honest vertical scale: on a
    # 2 mm axis the conditioning curve looks dramatic, which is the opposite of
    # the finding. Against a criterion that does move the number, it does not.
    ax = fig.add_subplot(gs[1])
    cdj = _load("conditioning/conditioning.json")
    cd = cdj["coverage_curves"]
    f = [100 * x for x in cd["fractions_dropped"]]
    ax.plot(f, cd["by_reliability"], ":", lw=1.5, color="#7fb08a",
            label="reliability score (for scale)")
    ax.plot(f, cd["by_conditioning"], "-o", ms=4, lw=2.0, color="#d62728",
            label="ordered by conditioning", zorder=3)
    ax.plot(f, cd["random"], "--", lw=1.5, color=MUTED,
            label="random ordering (null)")
    ax.set_xlabel("frames discarded (%)", fontsize=9)
    ax.set_ylabel("mean canonical distance (mm)", fontsize=9)
    ax.text(0.97, 0.74, "conditioning gains\n%+.2f mm over random;\nCI spans zero"
            % cdj["gain_at_10pct_dropped_mm"], transform=ax.transAxes,
            fontsize=7.8, color="#b3261e", ha="right", va="top")
    ax.set_title("Level 2  BETWEEN frames\nFAILS   tracks the random null",
                 fontsize=9.5, color="#b3261e")
    ax.legend(fontsize=7.2, frameon=False, loc="lower left")
    _style(ax)

    # Level 3 - between joints of one frame. Fails, and in the wrong direction.
    ax = fig.add_subplot(gs[2])
    rd = _load("radial/radial.json")
    ph = rd["post_hoc"]
    rr = np.array([490, 600])
    ax.plot(rr, rr * (ph["canonical_articulated_mm"]
                      / ph["mean_radius_articulated_mm"]), ":", color=MUTED,
            lw=1.3, label="proportional to radius")
    ax.scatter([ph["mean_radius_articulated_mm"]], [ph["canonical_articulated_mm"]],
               s=140, c="#b3261e", marker="s", zorder=3, label="beyond a hinge")
    ax.scatter([ph["mean_radius_torso_rigid_mm"]], [ph["canonical_torso_rigid_mm"]],
               s=140, c="#1a7f37", marker="o", zorder=3, label="rigid with torso")
    ax.annotate("larger radius,\nyet 2.7x smaller error", fontsize=8.2,
                color="#1a7f37", ha="center",
                xy=(ph["mean_radius_torso_rigid_mm"],
                    ph["canonical_torso_rigid_mm"]),
                xytext=(ph["mean_radius_torso_rigid_mm"] - 4, 128),
                arrowprops=dict(arrowstyle="-|>", color="#1a7f37", lw=1.2))
    ax.set_xlim(490, 600)
    ax.set_ylim(40, 235)
    ax.set_xlabel("mean radius from root (mm)", fontsize=9)
    ax.set_ylabel("cross-view distance (mm)", fontsize=9)
    ax.set_title("Level 3  BETWEEN joints\nFAILS   articulation dominates",
                 fontsize=9.5, color="#b3261e")
    ax.legend(fontsize=7.2, frameon=False, loc="lower left")
    _style(ax)

    fig.suptitle("Where the axis-length principle applies, and where it stops",
                 fontsize=11.5, y=1.02)
    fig.savefig(os.path.join(IMG, out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: matched radius. The cleanest evidence in the thesis, was a table.
# ---------------------------------------------------------------------------

def fig_matched_radius(out="fig_matched_radius.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, tag, name in ((axes[0], "", "MotionAGFormer-XS"),
                          (axes[1], "_motionbert", "MotionBERT")):
        ph = _load("radial/radial%s.json" % tag)["post_hoc"]
        pairs = ph["matched_radius_pairs"]
        labels, rig, art, gaps, ratios = [], [], [], [], []
        for m in pairs:
            labels.append("%s\nvs %s" % (m["rigid"].replace("_", " "),
                                         m["articulated"].replace("_", " ")))
            rig.append(m["canonical_rigid_mm"])
            art.append(m["canonical_articulated_mm"])
            gaps.append(m["radius_gap_pct"])
            ratios.append(m["ratio"])
        i = np.arange(len(labels))
        w = 0.36
        ax.bar(i - w / 2, rig, w, color="#1a7f37", label="rigid with torso")
        ax.bar(i + w / 2, art, w, color="#b3261e", label="beyond a hinge")
        for k in i:
            ax.text(k, max(rig[k], art[k]) * 1.06,
                    "radius differs %.1f%%\nerror differs %.2fx"
                    % (gaps[k], ratios[k]), ha="center", fontsize=7.6,
                    color="#333333")
        ax.set_xticks(i)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("cross-view distance (mm)", fontsize=9)
        ax.set_ylim(0, max(art) * 1.42)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        _style(ax)
    fig.suptitle("Matched radius, different chain position: radius is not the "
                 "governing variable", fontsize=11)
    fig.savefig(os.path.join(IMG, out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: the triage coverage curve, both backbones.
# ---------------------------------------------------------------------------

def fig_triage(out="fig_triage.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, tag, name in ((axes[0], "", "MotionAGFormer-XS"),
                          (axes[1], "_motionbert", "MotionBERT")):
        d = _load("conditioning/conditioning%s.json" % tag)
        c = d["coverage_curves"]
        f = [100 * x for x in c["fractions_dropped"]]
        ax.plot(f, c["by_reliability"], "-o", ms=4.5, lw=2.0, color="#1a7f37",
                label="ordered by reliability score", zorder=3)
        ax.plot(f, c["by_conditioning"], "-s", ms=4, lw=1.6, color="#d62728",
                label="ordered by conditioning (pre-registered)")
        ax.plot(f, c["random"], "--", lw=1.4, color=MUTED,
                label="random ordering (null)")
        rel = d["reliability_as_triage"]
        ax.text(0.03, 0.06,
                "gain at 10%% dropped  %+.2f mm\n95%% CI [%+.2f, %+.2f]"
                % (rel["gain_at_10pct_dropped_mm"], *rel["gain_ci95_mm"]),
                transform=ax.transAxes, fontsize=8, color="#1a7f37",
                va="bottom")
        ax.set_xlabel("frames abstained on (%)", fontsize=9)
        ax.set_ylabel("mean canonical cross-view distance (mm)", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7.2, frameon=False)
        _style(ax)
    fig.suptitle("Triage of canonicalization quality. The falsified score works "
                 "against this target; conditioning does not.", fontsize=10.5)
    fig.savefig(os.path.join(IMG, out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: the conceptual pipeline.
# ---------------------------------------------------------------------------

def fig_pipeline(out="fig_pipeline.png"):
    fig, ax = plt.subplots(figsize=(12.0, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    stages = [
        ("Camera A\nCamera B", "#e8e8e8", "two views,\nsame instant"),
        ("Frozen 2D\ndetector", "#dbe9f4", "no training"),
        ("Frozen 3D\nlifter", "#dbe9f4", "no training"),
        ("TRIAD body\nframe", "#d8efdc", "0 parameters\n402 FLOPs"),
        ("Canonical\nposes", "#d8efdc", "comparable\nacross views"),
    ]
    w, gap = 15.0, 4.2
    for i, (label, fc, note) in enumerate(stages):
        x = 2 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 11), w, 10, facecolor=fc,
                                   edgecolor="#555555", lw=1.0))
        ax.text(x + w / 2, 16, label, ha="center", va="center", fontsize=9.5)
        ax.text(x + w / 2, 8.4, note, ha="center", va="top", fontsize=7.8,
                color="#555555")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.6, 16), xytext=(x + w + 0.6, 16),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.annotate("", xy=(97, 24.5), xytext=(2, 24.5),
                arrowprops=dict(arrowstyle="-", color="#b3261e", lw=1.0,
                                linestyle=":"))
    ax.text(49.5, 25.4, "no camera calibration, no labels, no gradient step "
                        "anywhere on this path",
            ha="center", fontsize=8.4, color="#b3261e")
    ax.text(49.5, 3.2, "the frame is built from the prediction itself, which is "
                       "why it needs nothing the deployment cannot supply",
            ha="center", fontsize=8.2, color="#333333", style="italic")
    fig.savefig(os.path.join(IMG, out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Two-view comparison. Same primitives; this is the defense visual.
# ---------------------------------------------------------------------------

def two_view(subject=None, action=None, frame=None, cams=None,
             out="fig_twoview.png", preds=None):
    """One instant, two cameras, raw against canonical, with real numbers."""
    from evaluation.h36m_crossview import canonicalize_stream
    from evaluation.h36m_replication import OUT_DIR as PRED_DIR
    from evaluation.h36m_replication import aggregate_by_video, parse_video
    from evaluation.metrics import cross_view_joint_distance
    from evaluation.oracle import procrustes_cross_view_distance
    from evaluation.reliability import compute_reliability_score
    from evaluation.h36m_replication import h36m_conf_to_coco

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pn = np.load(preds or os.path.join(PRED_DIR, "preds.npz"))
    vids = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    groups = {}
    for vid, v in vids.items():
        s, a, c = parse_video(vid)
        groups.setdefault((s, a), {})[c] = v
    key = (subject, action) if subject else sorted(groups)[0]
    cams_avail = sorted(groups[key])

    def measure(c0, c1, idx):
        a = groups[key][c0]["pred"][idx:idx + 1]
        b = groups[key][c1]["pred"][idx:idx + 1]
        a = a - a[:, 0:1, :]
        b = b - b[:, 0:1, :]
        qa, oa = canonicalize_stream(a)
        qb, ob = canonicalize_stream(b)
        return (a, b, qa, qb, oa, ob,
                cross_view_joint_distance(a[0], b[0]),
                cross_view_joint_distance(qa[0], qb[0]))

    if cams and frame is not None:
        ca, cb, i = cams[0], cams[1], frame
    else:
        # Pick the frame whose improvement is closest to this group's median,
        # over every camera pair. Choosing the most flattering frame would make
        # the figure an advertisement rather than an illustration, and the
        # median is defensible in a way "frame 300 of the first pair" is not.
        cands = []
        n = min(len(groups[key][c]["pred"]) for c in cams_avail)
        for c0, c1 in [(cams_avail[p], cams_avail[q])
                       for p in range(len(cams_avail))
                       for q in range(p + 1, len(cams_avail))]:
            for idx in range(50, n - 50, max(1, (n - 100) // 12)):
                *_, dr, dc = measure(c0, c1, idx)
                if dr > 1e-6:
                    cands.append((100.0 * (dr - dc) / dr, c0, c1, idx))
        med = float(np.median([c[0] for c in cands]))
        _, ca, cb, i = min(cands, key=lambda t: abs(t[0] - med))

    A, B, cA, cB, okA, okB, d_raw, d_can = measure(ca, cb, i)
    d_orc = procrustes_cross_view_distance(cA[0], cB[0])
    relA = compute_reliability_score(
        A[0], h36m_conf_to_coco(groups[key][ca]["conf"][i]), None)[0]
    relB = compute_reliability_score(
        B[0], h36m_conf_to_coco(groups[key][cb]["conf"][i]), None)[0]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    for col, (pa, pb, title, dist, flip) in enumerate((
            (A[0], B[0], "RAW  camera-frame coordinates", d_raw, True),
            (cA[0], cB[0], "CANONICAL  body-frame coordinates", d_can, False))):
        for row, plane in enumerate(("front", "top")):
            ax = axes[row][col]
            draw_pose(ax, pa, color="#1f77b4", lw=2.2, flip_y=flip, plane=plane)
            draw_pose(ax, pb, color="#d62728", lw=2.2, alpha=0.85, flip_y=flip,
                      plane=plane)
            if col == 0 and plane == "front":
                draw_frame_axes(ax, pa, flip_y=flip)
            if row == 0:
                ax.set_title("%s\nmean joint distance  %.1f mm" % (title, dist),
                             fontsize=10.5, pad=8)
            ax.set_ylabel("seen from the front" if plane == "front"
                          else "seen from above", fontsize=8.5, color="#555555")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(0.8)
                s.set_color("#666666")
    # Both columns of a row must share a scale, or the canonical panel would
    # look tighter merely by being drawn smaller.
    for row in range(2):
        xs = [v for ax in axes[row] for v in ax.get_xlim()]
        ys = [v for ax in axes[row] for v in ax.get_ylim()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.05
        for ax in axes[row]:
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

    for c, lbl in (("#1f77b4", "camera %s" % ca), ("#d62728", "camera %s" % cb),
                   ("#2ca02c", "torso axis (primary)"),
                   ("#ff7f0e", "hip axis (secondary)")):
        axes[0][0].plot([], [], color=c, lw=2.2, label=lbl)
    axes[0][0].legend(fontsize=8.5, frameon=False, ncol=4,
                      loc="lower center", bbox_to_anchor=(1.06, 1.22))
    axes[1][1].text(0.5, -0.05,
                 "Procrustes oracle floor %.1f mm (needs both views)     "
                 "reliability %.2f / %.2f     gate: %s"
                 % (d_orc, relA, relB,
                    "accept" if (okA[0] and okB[0]) else "ABSTAIN"),
                 transform=axes[1][1].transAxes, fontsize=8.2, color="#333333",
                 ha="center", va="top")
    fig.suptitle("%s %s, frame %d: same instant, two viewpoints\n"
                 "%.1f mm apart raw, %.1f mm apart canonical  "
                 "(%.0f%% reduction, the median for this sequence)"
                 % (key[0], key[1], i, d_raw, d_can,
                    100.0 * (d_raw - d_can) / d_raw),
                 fontsize=11.5, y=1.06)
    fig.savefig(os.path.join(IMG, out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out, {"raw_mm": d_raw, "canonical_mm": d_can, "oracle_mm": d_orc,
                 "cameras": [ca, cb], "frame": int(i),
                 "reduction_pct": 100.0 * (d_raw - d_can) / d_raw,
                 "poses": (A[0], B[0], cA[0], cB[0])}


def fig_teaser(out="fig_teaser.png", preds=None, dpi=300):
    """
    The whole thesis in one row: disagreement, the frame, agreement.

    Reuses two_view's frame selection, so the instant shown is the one whose
    improvement is the median for its sequence rather than the most flattering,
    and the numbers printed are the real ones for that instant.

    The projection is from above throughout. Two cameras of a single instant
    differ mainly in azimuth, which a frontal view conceals almost entirely; the
    teaser would otherwise show two skeletons that look identical above a caption
    claiming they are 346 mm apart.
    """
    d = two_view(out="_teaser_scratch.png", preds=preds)[1]
    A, B, cA, cB = d["poses"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color("#cccccc")
            s.set_linewidth(0.9)

    draw_pose(axes[0], A, color="#1f77b4", lw=2.6, plane="top")
    draw_pose(axes[0], B, color="#d62728", lw=2.6, plane="top", alpha=0.9)
    axes[0].set_title("RAW\ntwo cameras, one instant", fontsize=11.5, pad=10)

    # Centre panel: the construction itself, drawn on one pose from the front.
    # The x limits are set explicitly because equal aspect on a tall, narrow
    # skeleton leaves a window too narrow for the horizontal hip arrow, which is
    # then silently clipped - the axis that is half the point of the panel.
    draw_pose(axes[1], A, color="#b8b8b8", lw=2.0, flip_y=True, plane="front")
    draw_frame_axes(axes[1], A, scale=330.0, flip_y=True)
    axes[1].set_xlim(A[0, 0] - 430, A[0, 0] + 430)
    axes[1].text(A[0, 0] + 40, -A[0, 1] + 330, "torso\n(primary)", fontsize=9.5,
                 color="#2ca02c", ha="left", va="center")
    axes[1].text(A[0, 0] + 350, -A[0, 1] + 25, "hip\n(secondary)", fontsize=9.5,
                 color="#ff7f0e", ha="center", va="bottom")
    axes[1].set_title("TRIAD BODY FRAME\nbuilt from the prediction itself",
                      fontsize=11.5, pad=10)

    draw_pose(axes[2], cA, color="#1f77b4", lw=2.6, plane="top")
    draw_pose(axes[2], cB, color="#d62728", lw=2.6, plane="top", alpha=0.9)
    axes[2].set_title("CANONICAL\nsame two cameras", fontsize=11.5, pad=10)

    for ax in (axes[0], axes[2]):
        xs, ys = ax.get_xlim(), ax.get_ylim()
        cx, cy = sum(xs) / 2, sum(ys) / 2
        half = max(xs[1] - xs[0], ys[1] - ys[0]) / 2 * 1.08
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    for ax, mm, colour in ((axes[0], d["raw_mm"], "#b3261e"),
                           (axes[2], d["canonical_mm"], "#1a7f37")):
        ax.text(0.5, -0.09, "%.0f mm apart" % mm, transform=ax.transAxes,
                ha="center", va="top", fontsize=20, color=colour)

    for x in (0.352, 0.655):
        fig.text(x, 0.52, r"$\Longrightarrow$", fontsize=26, color="#555555",
                 ha="center", va="center")
    fig.text(0.5, -0.02,
             "no training  $\\cdot$  no labels  $\\cdot$  no camera calibration"
             "  $\\cdot$  0 trained parameters  $\\cdot$  402 FLOPs per frame",
             ha="center", fontsize=12, color="#333333")
    fig.savefig(os.path.join(IMG, out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        os.remove(os.path.join(IMG, "_teaser_scratch.png"))
    except OSError:
        pass
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twoview", action="store_true")
    ap.add_argument("--teaser", action="store_true")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for fn in (fig_three_levels, fig_matched_radius, fig_triage, fig_pipeline):
        print("wrote", fn())
    if args.twoview:
        out, d = two_view()
        print("wrote %s   raw %.1f -> canonical %.1f mm (oracle %.1f)"
              % (out, d["raw_mm"], d["canonical_mm"], d["oracle_mm"]))
    if args.teaser:
        out, d = fig_teaser()
        print("wrote %s   raw %.0f -> canonical %.0f mm  (-%.0f%%)"
              % (out, d["raw_mm"], d["canonical_mm"], d["reduction_pct"]))


if __name__ == "__main__":
    main()
