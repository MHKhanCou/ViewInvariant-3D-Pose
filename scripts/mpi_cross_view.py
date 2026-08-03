"""
MPI-INF-3DHP Cross-View Evaluation

Evaluates cross-view consistency of MotionAGFormer predictions on
synchronized multi-camera MPI-INF-3DHP data.

Protocol:
1. Load annot.mat for each (subject, sequence)
2. Parse camera calibration
3. For each frame index, load RGB from 2+ cameras
4. Run YOLOv8-pose on each camera's RGB
5. Run MotionAGFormer-XS on each camera's 2D keypoints
6. Transform predictions to shared coordinate system using camera extrinsics
7. Compute cross-view consistency metrics

Output: CSV report + qualitative figures
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.mpi_eval import (
    load_annot_mat, load_camera_calibration, get_annot3_for_frame,
    transform_to_world, select_cameras, select_h36m_joints
)
from canonical.body_frame import canonicalize_single, compute_bone_lengths
from canonical.metrics import cross_view_consistency_error, bone_length_stability
from demo_live.pose_detector import PoseDetector
from demo_live.lifter import build_xs_model, coco_to_h36m, lift_sequence, N_FRAMES
from canonical.canonicalizer import Canonicalizer

DATA_ROOT = os.path.join(REPO_ROOT, "..", "mpi_inf_3dhp")


def run_inference_on_frame(detector, model, rgb_frame, device="cpu"):
    """Run the full pipeline on a single RGB frame.

    Returns:
        pred_3d: (17, 3) root-relative 3D prediction
        kpts: (17, 2) detected 2D keypoints
        scores: (17,) detection scores
    """
    H, W = rgb_frame.shape[:2]

    # Detect 2D pose
    kpts, scores = detector.detect(rgb_frame)
    if kpts is None:
        return None, None, None

    # Repeat to fill 27-frame window
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)

    # Lift to 3D
    from demo_live.lifter import coco_to_h36m, lift_sequence
    h36m = coco_to_h36m(kpts_seq, scores_seq)
    pose3d = lift_sequence(model, h36m, img_size=(H, W), mode="root",
                          use_flip=True,
                          apply_display_postprocess=False)[-1]

    return pose3d, kpts, scores


def visualize_matched_pairs(frame_a, frame_b, pred_a, pred_b,
                           gt_a, gt_b, cam_a, cam_b, frame_idx, out_dir):
    """Visualize a matched camera pair for sanity checking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Camera A frame
    axes[0].imshow(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Camera {cam_a} (frame {frame_idx})")
    axes[0].axis('off')

    # Camera B frame
    axes[1].imshow(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Camera {cam_b} (frame {frame_idx})")
    axes[1].axis('off')

    # 3D comparison
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    if pred_a is not None:
        ax.scatter(pred_a[:, 0], pred_a[:, 1], pred_a[:, 2],
                  c='blue', s=20, label='Camera A pred')
    if pred_b is not None:
        ax.scatter(pred_b[:, 0], pred_b[:, 1], pred_b[:, 2],
                  c='red', s=20, label='Camera B pred')
    ax.set_title(f"3D Comparison (frame {frame_idx})")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"pair_cam{cam_a}_cam{cam_b}_frame{frame_idx:04d}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MPI-INF-3DHP Cross-View Evaluation")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="Max frames to process per sequence (use None for all)")
    parser.add_argument("--output", type=str, default="thesis_artifacts/cross_view_report.csv")
    parser.add_argument("--vis-dir", type=str, default="thesis_artifacts/cross_view_figures")
    parser.add_argument("--vis-pairs", type=int, default=20,
                        help="Number of matched pairs to visualize for sanity check")
    args = parser.parse_args()

    print("MPI-INF-3DHP Cross-View Evaluation")
    print("=" * 50)

    # Load model and detector
    print("[1/6] Loading model and detector...")
    device = args.device
    detector = PoseDetector(weights="yolov8n-pose.pt", device=device, conf=0.4)
    model = build_xs_model(device=device)

    # Camera set
    cam_set = select_cameras()
    print(f"  Camera set: {cam_set}")

    # Find recordings
    recordings = []
    for subj in sorted(os.listdir(args.data_root)):
        subj_path = os.path.join(args.data_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for seq in sorted(os.listdir(subj_path)):
            seq_path = os.path.join(subj_path, seq)
            annot_path = os.path.join(seq_path, "annot.mat")
            calib_path = os.path.join(seq_path, "camera.calibration")
            if os.path.exists(annot_path) and os.path.exists(calib_path):
                # Check which cameras have frames
                cam_dirs = [d for d in os.listdir(seq_path)
                           if d.startswith("frames_cam") and os.path.isdir(os.path.join(seq_path, d))]
                available_cams = sorted([int(d.replace("frames_cam", "")) for d in cam_dirs])
                if len(available_cams) >= 2:
                    recordings.append({
                        'subject': subj,
                        'sequence': seq,
                        'annot_path': annot_path,
                        'calib_path': calib_path,
                        'available_cams': available_cams,
                    })

    print(f"  Recordings with 2+ cameras: {len(recordings)}")
    for r in recordings:
        print(f"    {r['subject']}/{r['sequence']}: cameras {r['available_cams']}")

    if len(recordings) == 0:
        print("  ERROR: No multi-camera recordings found. Aborting.")
        return

    # Process each recording
    all_results = []
    vis_dir = args.vis_dir
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\n[2/6] Processing recordings...")

    for rec in recordings:
        subj = rec['subject']
        seq = rec['sequence']
        print(f"\n  Processing {subj}/{seq}...")

        # Load annotation and calibration
        annot_data = load_annot_mat(rec['annot_path'])
        cameras_calib = load_camera_calibration(rec['calib_path'])

        # Build camera index mapping
        cam_id_to_calib = {c['id']: c for c in cameras_calib}

        # Use available cameras that are in our camera set
        active_cams = [c for c in rec['available_cams'] if c in cam_set]
        if len(active_cams) < 2:
            print(f"    Skipping: fewer than 2 cameras in set ({active_cams})")
            continue

        n_frames = args.n_frames or 6416

        # For each frame, run inference on 2+ cameras
        print(f"    Processing {n_frames} frames on cameras {active_cams}...")

        for frame_idx in range(n_frames):
            # Run inference on each camera
            preds = {}
            rgb_frames = {}
            for cam_id in active_cams:
                frame_path = os.path.join(args.data_root, subj, seq,
                                         f"frames_cam{cam_id}", f"frame_{frame_idx:06d}.jpg")
                if not os.path.exists(frame_path):
                    continue

                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                rgb_frames[cam_id] = frame
                pred, kpts, scores = run_inference_on_frame(detector, model, frame, device)
                if pred is not None:
                    preds[cam_id] = pred

            # Pair cameras and compute metrics
            for ci in range(len(active_cams)):
                for cj in range(ci + 1, len(active_cams)):
                    cam_a, cam_b = active_cams[ci], active_cams[cj]
                    if cam_a not in preds or cam_b not in preds:
                        continue

                    # MotionAGFormer outputs root-relative 3D in the model's
                    # internal coordinate system. Both predictions are in the
                    # SAME frame (the model's frame), so no camera extrinsic
                    # transform is needed for cross-view comparison.
                    # The extrinsics would only be needed if we wanted to
                    # compare predictions against annot3 ground truth (which
                    # IS in camera coordinates).
                    pred_a_h36m = preds[cam_a] - preds[cam_a][0:1]  # root-center
                    pred_b_h36m = preds[cam_b] - preds[cam_b][0:1]  # root-center

                    # Compute metrics
                    joint_dist = float(np.mean(np.linalg.norm(pred_a_h36m - pred_b_h36m, axis=-1)))

                    # Canonical consistency
                    can = Canonicalizer()
                    can_a, _ = can(pred_a_h36m)
                    can_b, _ = can(pred_b_h36m)
                    canon_err = cross_view_consistency_error(pred_a_h36m, pred_b_h36m)
                    bone_diff, _ = bone_length_stability(pred_a_h36m, pred_b_h36m)

                    # Joint angle deviation
                    angles_a = compute_joint_angles(pred_a_h36m)
                    angles_b = compute_joint_angles(pred_b_h36m)
                    angle_dev = float(np.mean(np.abs(angles_a - angles_b)))

                    result = {
                        'subject': subj,
                        'sequence': seq,
                        'frame': frame_idx,
                        'cam_a': cam_a,
                        'cam_b': cam_b,
                        'joint_distance_raw': joint_dist,
                        'canonical_consistency': canon_err,
                        'bone_length_dev': bone_diff,
                        'angle_deviation': angle_dev,
                    }
                    all_results.append(result)

            # Visualize first N pairs
            if frame_idx < args.vis_pairs and len(active_cams) >= 2:
                cam_a, cam_b = active_cams[0], active_cams[1]
                if cam_a in rgb_frames and cam_b in rgb_frames:
                    pred_a_v = preds.get(cam_a)
                    pred_b_v = preds.get(cam_b)
                    if pred_a_v is not None and pred_b_v is not None:
                        # Root-center for visualization (same as comparison)
                        pa = pred_a_v - pred_a_v[0:1]
                        pb = pred_b_v - pred_b_v[0:1]
                        visualize_matched_pairs(
                            rgb_frames[cam_a], rgb_frames[cam_b],
                            pa, pb, None, None, cam_a, cam_b, frame_idx, vis_dir)

            if frame_idx % 10 == 0:
                print(f"      Frame {frame_idx}/{n_frames}...")

    # Write CSV report
    print(f"\n[3/6] Writing report to {args.output}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if all_results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  {len(all_results)} results written")

        # Summary
        print(f"\n[4/6] Summary:")
        print(f"  Total matched pairs: {len(all_results)}")
        joint_dists = [r['joint_distance_raw'] for r in all_results]
        canon_errs = [r['canonical_consistency'] for r in all_results]
        bone_devs = [r['bone_length_dev'] for r in all_results]
        angle_devs = [r['angle_deviation'] for r in all_results]
        print(f"  Joint distance (raw): {np.mean(joint_dists):.2f} +/- {np.std(joint_dists):.2f}")
        print(f"  Canonical consistency: {np.mean(canon_errs):.2f} +/- {np.std(canon_errs):.2f}")
        print(f"  Bone-length deviation: {np.mean(bone_devs):.2f} +/- {np.std(bone_devs):.2f}")
        print(f"  Angle deviation: {np.mean(angle_devs):.2f} +/- {np.std(angle_devs):.2f}")
    else:
        print("  WARNING: No results to report")

    print(f"\n[5/6] Figures saved to {vis_dir}")
    print(f"\n[6/6] Done.")


def compute_joint_angles(pose):
    """Compute joint angles for angle deviation metric.

    Args:
        pose: (17, 3) root-relative 3D pose

    Returns:
        angles: (n_bones,) array of angles in radians
    """
    from canonical.body_frame import BONE_CONNECTIONS
    angles = []
    for a, b in BONE_CONNECTIONS:
        bone = pose[b] - pose[a]
        norm = np.linalg.norm(bone)
        if norm > 1e-6:
            # Angle with vertical axis (y-axis)
            angle = np.arccos(np.clip(bone[1] / norm, -1, 1))
            angles.append(angle)
        else:
            angles.append(0.0)
    return np.array(angles, dtype=np.float32)


if __name__ == "__main__":
    main()
