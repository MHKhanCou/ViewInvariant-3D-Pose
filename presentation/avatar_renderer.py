"""Stylized avatar-like rendering for qualitative image demonstrations.

This renderer consumes predicted 3D joints only. It does not fit SMPL, infer
surface geometry, or change the MotionAGFormer prediction. It is deliberately
kept outside the model and canonical research modules.
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# H36M-17 bone connections with proper proportions.
_BONES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
_LEFT_BONES = {0, 1, 2, 13, 14, 15}

# Bone thickness hierarchy (thicker for torso, thinner for extremities)
_BONE_WIDTHS = {
    (0, 7): 14, (7, 8): 12, (8, 9): 10, (9, 10): 8,  # Spine/head
    (0, 1): 10, (1, 2): 8, (2, 3): 6,  # Left leg
    (0, 4): 10, (4, 5): 8, (5, 6): 6,  # Right leg
    (8, 11): 10, (11, 12): 7, (12, 13): 5,  # Right arm
    (8, 14): 10, (14, 15): 7, (15, 16): 5,  # Left arm
}

# Joint sizes (larger for torso, smaller for extremities)
_JOINT_SIZES = {
    0: 80, 7: 60, 8: 70, 9: 50, 10: 40,  # Torso/head
    1: 40, 2: 35, 3: 30,  # Left leg
    4: 40, 5: 35, 6: 30,  # Right leg
    11: 40, 12: 35, 13: 30,  # Right arm
    14: 40, 15: 35, 16: 30,  # Left arm
}


def render_stylized_avatar_3d(pose3d, figsize=(8.0, 8.0), dpi=100):
    """Render a clean avatar-like figure from a predicted 17-joint pose.

    This is an image-presentation layer, not Blender output and not a pose
    accuracy improvement. The input is an official-style display pose, not a
    fabricated trajectory.
    """
    pose = np.asarray(pose3d, dtype=np.float32)
    if pose.shape != (17, 3):
        raise ValueError(f"Expected pose shape (17, 3), got {pose.shape}")

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8f9fa")
    ax = fig.add_subplot(111, projection="3d", facecolor="#f8f9fa")
    ax.view_init(elev=15, azim=70)

    mins, maxs = pose.min(axis=0), pose.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = max(float((maxs - mins).max()), 1e-4)
    half = 0.65 * extent
    floor_z = mins[2] - 0.10 * extent
    gx = np.linspace(center[0] - half, center[0] + half, 8)
    gy = np.linspace(center[1] - half, center[1] + half, 8)
    xx, yy = np.meshgrid(gx, gy)
    ax.plot_surface(xx, yy, np.full_like(xx, floor_z), color="#e9ecef",
                    alpha=0.35, linewidth=0)

    # Draw bones with proper width hierarchy
    for index, (a, b) in enumerate(_BONES):
        xyz = pose[[a, b]]
        bone_key = (a, b)
        width = _BONE_WIDTHS.get(bone_key, 7)
        color = "#3498db" if index in _LEFT_BONES else "#e74c3c"
        # Shadow/outline
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#2c3e50", lw=width + 4,
                solid_capstyle="round", zorder=2)
        # Main bone
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=width,
                solid_capstyle="round", zorder=3)

    # Draw joints with proper sizing
    for idx in range(17):
        size = _JOINT_SIZES.get(idx, 40)
        if idx == 10:  # Head - special color
            ax.scatter([pose[idx, 0]], [pose[idx, 1]], [pose[idx, 2]], s=size,
                       c="#f39c12", edgecolors="#2c3e50", linewidths=1.6, zorder=5)
        else:
            ax.scatter([pose[idx, 0]], [pose[idx, 1]], [pose[idx, 2]], s=size,
                       c="#ecf0f1", edgecolors="#2c3e50", linewidths=1.4, depthshade=True, zorder=4)

    for axis, value in zip(("x", "y", "z"), center):
        getattr(ax, f"set_{axis}lim3d")([value - half, value + half])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
