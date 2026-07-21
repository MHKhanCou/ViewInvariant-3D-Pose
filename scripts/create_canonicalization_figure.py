"""
Create a publication-quality figure showing canonicalization across cameras.

Layout:
  Row 1: Camera 0 RGB | Camera 1 RGB | Summary text
  Row 2: Camera 0 raw | Camera 1 raw | Camera 0 canonical | Camera 1 canonical
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.pose_detector import PoseDetector
from demo_live.lifter import build_xs_model, coco_to_h36m, lift_sequence, N_FRAMES
from canonical.canonicalizer import Canonicalizer

BONES_I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
BONES_J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)


def draw_pose_on_ax(ax, pose, title=""):
    lcolor, rcolor = (0, 0, 1), (1, 0, 0)
    for i in range(len(BONES_I)):
        x = [pose[BONES_I[i], 0], pose[BONES_J[i], 0]]
        y = [pose[BONES_I[i], 1], pose[BONES_J[i], 1]]
        z = [pose[BONES_I[i], 2], pose[BONES_J[i], 2]]
        ax.plot(x, y, z, lw=2, color=lcolor if not LR[i] else rcolor)
    R, RZ = 0.72, 0.7
    xr, yr, zr = pose[0]
    ax.set_xlim3d([-R+xr, R+xr]); ax.set_ylim3d([-R+yr, R+yr]); ax.set_zlim3d([-RZ+zr, RZ+zr])
    ax.set_aspect("auto"); ax.view_init(elev=15, azim=70)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white); ax.yaxis.set_pane_color(white); ax.zaxis.set_pane_color(white)
    ax.tick_params("x", labelbottom=False); ax.tick_params("y", labelleft=False); ax.tick_params("z", labelleft=False)
    ax.set_title(title, fontsize=10)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=os.path.join(REPO_ROOT, "..", "mpi_inf_3dhp"))
    parser.add_argument("--output", type=str, default="thesis_artifacts/figures/canonicalization_comparison.png")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading model and detector...")
    detector = PoseDetector(weights="yolov8n-pose.pt", device="cpu", conf=0.4)
    model = build_xs_model(device="cpu")

    cam0_path = os.path.join(args.data_root, "S1", "Seq1", "frames_cam0")
    cam1_path = os.path.join(args.data_root, "S1", "Seq1", "frames_cam1")
    frame_idx = 10

    frame0 = cv2.imread(os.path.join(cam0_path, f"frame_{frame_idx:06d}.jpg"))
    frame1 = cv2.imread(os.path.join(cam1_path, f"frame_{frame_idx:06d}.jpg"))
    if frame0 is None or frame1 is None:
        print("ERROR: Could not load frames"); return

    H, W = frame0.shape[:2]

    # Inference on both cameras
    for label, frame in [("Camera 0", frame0), ("Camera 1", frame1)]:
        kpts, scores = detector.detect(frame)
        if kpts is None: kpts = np.zeros((17, 2), dtype=np.float32); scores = np.zeros((17,), dtype=np.float32)
        kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
        scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)
        h36m = coco_to_h36m(kpts_seq, scores_seq)
        pose_raw = lift_sequence(model, h36m, img_size=(H, W), mode="root", use_flip=False)[-1]
        if label == "Camera 0": pose_a = pose_raw - pose_raw[0:1]
        else: pose_b = pose_raw - pose_raw[0:1]

    # Canonicalize
    can = Canonicalizer()
    can_a, _ = can(pose_a)
    can_b, _ = can(pose_b)

    raw_dist = float(np.mean(np.linalg.norm(pose_a - pose_b, axis=-1)))
    can_dist = float(np.mean(np.linalg.norm(can_a - can_b, axis=-1)))
    improvement = (1 - can_dist / raw_dist) * 100

    # Create figure: 2 rows × 3 columns
    fig = plt.figure(figsize=(16, 9))

    # Row 1: Camera images
    ax0 = fig.add_subplot(2, 3, 1)
    ax0.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    ax0.set_title("Camera 0 (viewpoint A)", fontsize=11, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(2, 3, 2)
    ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Camera 1 (viewpoint B)", fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax_summary = fig.add_subplot(2, 3, 3)
    ax_summary.axis('off')
    ax_summary.text(0.5, 0.85, "Cross-View Consistency", fontsize=13, fontweight='bold',
                   ha='center', transform=ax_summary.transAxes)
    ax_summary.text(0.5, 0.65, f"Raw distance:  {raw_dist:.4f}", fontsize=11,
                   ha='center', transform=ax_summary.transAxes)
    ax_summary.text(0.5, 0.45, f"Canonical distance:  {can_dist:.4f}", fontsize=11,
                   ha='center', transform=ax_summary.transAxes, color='green')
    ax_summary.text(0.5, 0.25, f"Improvement:  {improvement:.0f}%", fontsize=13,
                   fontweight='bold', ha='center', transform=ax_summary.transAxes, color='green')
    ax_summary.text(0.5, 0.05, "Canonicalization removes\norientation differences",
                   fontsize=10, ha='center', transform=ax_summary.transAxes, style='italic')

    # Row 2: Four pose subplots
    ax_r0 = fig.add_subplot(2, 3, 4, projection='3d')
    draw_pose_on_ax(ax_r0, pose_a, "Camera 0 (raw)")

    ax_r1 = fig.add_subplot(2, 3, 5, projection='3d')
    draw_pose_on_ax(ax_r1, pose_b, "Camera 1 (raw)")

    ax_c0 = fig.add_subplot(2, 3, 6, projection='3d')
    draw_pose_on_ax(ax_c0, can_a, "After Canonicalization")

    # Add annotation showing alignment
    fig.text(0.5, 0.02, "Canonical body-frame normalization aligns body orientation across viewpoints",
            ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {args.output}")


if __name__ == "__main__":
    main()
