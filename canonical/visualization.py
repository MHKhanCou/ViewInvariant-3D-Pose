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


def render_canonical_3d(pose3d, figsize=(9.6, 5.4), dpi=100):
    """
    Render a canonical 3D pose using the same visual style as render_3d().

    Uses auto-fit axis limits based on the actual pose extent, so the
    skeleton fills the viewport properly. The visual style (colors, line
    thickness, viewport angle) matches render_3d() exactly.

    Args:
        pose3d: (17, 3) float32 canonical pose.
        figsize: (width, height) in inches.
        dpi: matplotlib DPI.

    Returns:
        img: (H, W, 3) uint8 BGR image.
    """
    pose3d = np.asarray(pose3d, dtype=np.float32)

    # Display-only axis remap. The canonical frame carries the body vertical on
    # +y, because it is built as thorax minus pelvis, but matplotlib draws its
    # THIRD argument as the screen vertical. Plotting (x, y, z) directly
    # therefore lays the skeleton on its side, which is what the video path did.
    #
    # (x, y, z) -> (x, -z, y) is a rotation of +90 degrees about x, with
    # determinant +1. A bare swap to (x, z, y) would have determinant -1 and
    # would mirror the body, silently exchanging left and right limbs.
    #
    # app.py applies the same remap on its image path before handing poses to
    # the interactive viewer; this is the corresponding fix for every caller
    # that renders through matplotlib, and it leaves scored coordinates alone.
    pose3d = np.column_stack([pose3d[:, 0], -pose3d[:, 2], pose3d[:, 1]])

    fig = plt.figure(figsize=figsize, frameon=False, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=15, azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
    J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]

    for i in range(len(I)):
        x = [pose3d[I[i], 0], pose3d[J[i], 0]]
        y = [pose3d[I[i], 1], pose3d[J[i], 1]]
        z = [pose3d[I[i], 2], pose3d[J[i], 2]]
        ax.plot(x, y, z, lw=2, color=lcolor if not LR[i] else rcolor)

    # Auto-fit axis limits based on pose extent (fills 60-80% of viewport).
    # This ensures the skeleton is large and natural, not tiny.
    pose_min = pose3d.min(axis=0)
    pose_max = pose3d.max(axis=0)
    pose_center = (pose_min + pose_max) / 2
    pose_extent = np.max(pose_max - pose_min)

    # Use 1.3x the pose extent as axis range (skeleton fills ~75% of viewport)
    half_range = pose_extent * 0.65

    ax.set_xlim3d([pose_center[0] - half_range, pose_center[0] + half_range])
    ax.set_ylim3d([pose_center[1] - half_range, pose_center[1] + half_range])
    ax.set_zlim3d([pose_center[2] - half_range, pose_center[2] + half_range])
    ax.set_aspect("auto")

    # Transparent panes and hidden ticks (matches official demo).
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
