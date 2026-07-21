"""
Visualization for the MotionAGFormer live demo.

Produces, per frame:
  * a 2D skeleton overlay drawn on the original RGB image (COCO-17 joints),
  * a 3D skeleton render (H36M-17 joints) via matplotlib,
and stacks them side-by-side into the final demo output.
"""

import os
import numpy as np
import cv2
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# COCO-17 skeleton (for the 2D overlay on detected joints).
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
]

# H36M-17 skeleton (for the lifted 3D joints).
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
]
# Exact left/right colour mask used by demo/vis.py:show3Dpose().
H36M_LR = np.array(
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    dtype=bool,
)


def draw_2d(kpts_2d, img, scores=None, thresh=0.3):
    """Draw COCO-17 skeleton overlay on a BGR image. Returns a new BGR image."""
    out = img.copy()
    H, W = out.shape[:2]
    if kpts_2d is None:
        return out

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    gcolor = (0, 255, 0)
    thickness = 3

    pts = kpts_2d.astype(int)
    conf = scores if scores is not None else np.ones(len(pts))

    for a, b in COCO_SKELETON:
        if a >= len(pts) or b >= len(pts):
            continue
        if conf[a] < thresh or conf[b] < thresh:
            continue
        pa = (int(np.clip(pts[a, 0], 0, W - 1)), int(np.clip(pts[a, 1], 0, H - 1)))
        pb = (int(np.clip(pts[b, 0], 0, W - 1)), int(np.clip(pts[b, 1], 0, H - 1)))
        cv2.line(out, pa, pb, lcolor, thickness)
        cv2.circle(out, pa, radius=4, color=gcolor, thickness=-1)
        cv2.circle(out, pb, radius=4, color=gcolor, thickness=-1)
    return out


def render_3d(pose3d, figsize=(9.6, 5.4), azimuth=70, elev=15):
    """Render an H36M-17 3D pose to a numpy BGR image (matches official demo style)."""
    fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azimuth)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
    J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]

    for i in range(len(I)):
        x = [pose3d[I[i], 0], pose3d[J[i], 0]]
        y = [pose3d[I[i], 1], pose3d[J[i], 1]]
        z = [pose3d[I[i], 2], pose3d[J[i], 2]]
        ax.plot(x, y, z, lw=2, color=lcolor if not H36M_LR[i] else rcolor)

    # Auto-fit axis limits based on pose extent (fills ~75% of viewport).
    pose_min = pose3d.min(axis=0)
    pose_max = pose3d.max(axis=0)
    pose_center = (pose_min + pose_max) / 2
    pose_extent = np.max(pose_max - pose_min)

    # Use 1.3x the pose extent as axis range
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


def _resize_to_width(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    return cv2.resize(img, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def make_frame(img_2d, img_3d):
    """Stack the 2D and 3D panels side-by-side, equal height."""
    # Match heights for horizontal stacking.
    h2 = img_2d.shape[0]
    h3 = img_3d.shape[0]
    target_h = max(h2, h3)
    # Resize both to exact target height to avoid 1px rounding errors
    if h2 != target_h:
        scale = target_h / float(h2)
        img_2d = cv2.resize(img_2d, (int(round(img_2d.shape[1] * scale)), target_h))
    if h3 != target_h:
        scale = target_h / float(h3)
        img_3d = cv2.resize(img_3d, (int(round(img_3d.shape[1] * scale)), target_h))
    return np.hstack([img_2d, img_3d])


def save_image_demo(img_2d, img_3d, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    combined = make_frame(img_2d, img_3d)
    cv2.imwrite(out_path, combined)


def save_video_demo(frames, out_path, fps=30):
    """frames: list of BGR combined images."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
