"""
Improved 3D pose rendering for canonical poses.

Uses equal axis scaling, auto-fit bounds, joint markers, and thicker
bone lines.  This is the ONLY canonical rendering location — do not
add rendering code to demo_live/visualize.py.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# H36M-17 skeleton connections.
BONES_I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
BONES_J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)


def render_canonical_3d(pose3d, figsize=(8, 8), dpi=100):
    """
    Render a canonical 3D pose with equal axis scaling.

    Args:
        pose3d: (17, 3) float32 canonical pose.
        figsize: (width, height) in inches.
        dpi: matplotlib DPI.

    Returns:
        img: (H, W, 3) uint8 BGR image.
    """
    pose3d = np.asarray(pose3d, dtype=np.float32)

    fig = plt.figure(figsize=figsize, frameon=False, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # --- Equal axis scaling from pose extents ---
    margin = 0.15
    mins = pose3d.min(axis=0)
    maxs = pose3d.max(axis=0)
    center = (mins + maxs) / 2
    extent = maxs - mins
    max_extent = max(extent.max(), 1e-6)
    half_range = max_extent / 2 * (1 + margin)

    for axis, c in zip(["x", "y", "z"], center):
        getattr(ax, f"set_{axis}lim3d")([c - half_range, c + half_range])

    # --- Draw bones ---
    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)
    for i in range(len(BONES_I)):
        x = [pose3d[BONES_I[i], 0], pose3d[BONES_J[i], 0]]
        y = [pose3d[BONES_I[i], 1], pose3d[BONES_J[i], 1]]
        z = [pose3d[BONES_I[i], 2], pose3d[BONES_J[i], 2]]
        ax.plot(x, y, z, lw=3, color=lcolor if not LR[i] else rcolor)

    # --- Draw joint markers ---
    ax.scatter(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2],
               c="green", s=20, depthshade=True, zorder=5)

    # --- Clean background ---
    ax.view_init(elev=15, azim=70)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
