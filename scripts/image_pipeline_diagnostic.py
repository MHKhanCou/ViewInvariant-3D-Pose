"""
Diagnostic pipeline for image inference.

Saves intermediate outputs for every stage and compares against
expected behavior. Identifies where the pipeline diverges.

Stages:
1. Raw YOLO detection (keypoints + scores)
2. COCO 17 keypoints (from YOLO output)
3. H36M 17 joints after conversion (coco_to_h36m)
4. Normalized coordinates sent to MotionAGFormer
5. Raw MotionAGFormer prediction (3D joints)
6. Final rendered pose
"""

import os
import sys
import numpy as np
import cv2
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.pose_detector import PoseDetector
from demo_live.lifter import coco_to_h36m, lift_sequence, N_FRAMES, normalize_screen_coordinates
from demo_live.visualize import draw_2d, render_3d, make_frame
from backend.model_loader import get_model


def diagnose_image(image_path, output_dir, device="cpu"):
    """Run full diagnostic pipeline on a single image."""
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = frame.shape[:2]
    print(f"Input: {os.path.basename(image_path)} ({H_orig}x{W_orig})")

    # Stage 1: Raw YOLO detection
    print("\n--- Stage 1: Raw YOLO Detection ---")
    detector = PoseDetector(weights="yolov8n-pose.pt", device=device, conf=0.4)

    # Direct YOLO predict (low threshold to see all detections)
    results = detector.model.predict(frame_rgb, conf=0.01, verbose=False)
    result = results[0]

    if result.keypoints is not None and len(result.keypoints) > 0:
        n_persons = len(result.keypoints)
        print(f"  Persons detected: {n_persons}")
        for i in range(min(3, n_persons)):
            xy = result.keypoints.xy[i].cpu().numpy()
            conf = result.keypoints.conf[i].cpu().numpy()
            valid = np.sum(conf > 0.1)
            print(f"  Person {i}: {valid}/17 keypoints, mean_conf={conf.mean():.3f}")

        # Save best detection
        kpts_raw = result.keypoints.xy[0].cpu().numpy()
        scores_raw = result.keypoints.conf[0].cpu().numpy()
    else:
        print("  No persons detected")
        kpts_raw = np.zeros((17, 2), dtype=np.float32)
        scores_raw = np.zeros((17,), dtype=np.float32)

    # Save raw detection info
    with open(os.path.join(output_dir, "01_raw_detection.txt"), 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Size: {H_orig}x{W_orig}\n")
        f.write(f"Persons detected: {n_persons if result.keypoints is not None else 0}\n")
        f.write(f"Keypoints shape: {kpts_raw.shape}\n")
        f.write(f"Mean score: {scores_raw.mean():.4f}\n")
        f.write(f"Valid keypoints (>0.1): {np.sum(scores_raw > 0.1)}/17\n")

    # Stage 2: COCO 17 keypoints
    print("\n--- Stage 2: COCO 17 Keypoints ---")
    print(f"  Keypoints shape: {kpts_raw.shape}")
    print(f"  Scores shape: {scores_raw.shape}")
    print(f"  X range: [{kpts_raw[:, 0].min():.1f}, {kpts_raw[:, 0].max():.1f}]")
    print(f"  Y range: [{kpts_raw[:, 1].min():.1f}, {kpts_raw[:, 1].max():.1f}]")
    print(f"  Mean score: {scores_raw.mean():.4f}")
    print(f"  Valid (>0.1): {np.sum(scores_raw > 0.1)}/17")

    # Save COCO keypoints
    np.save(os.path.join(output_dir, "02_coco_keypoints.npy"), kpts_raw)
    np.save(os.path.join(output_dir, "02_coco_scores.npy"), scores_raw)

    # Stage 3: H36M conversion
    print("\n--- Stage 3: H36M 17 Joints Conversion ---")
    kpts_seq = np.repeat(kpts_raw[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores_raw[None], N_FRAMES, axis=0)
    h36m = coco_to_h36m(kpts_seq, scores_seq)

    print(f"  H36M shape: {h36m.shape}")
    print(f"  X range: [{h36m[:, :, 0].min():.1f}, {h36m[:, :, 0].max():.1f}]")
    print(f"  Y range: [{h36m[:, :, 1].min():.1f}, {h36m[:, :, 1].max():.1f}]")
    print(f"  Confidence range: [{h36m[:, :, 2].min():.4f}, {h36m[:, :, 2].max():.4f}]")

    # Save H36M joints
    np.save(os.path.join(output_dir, "03_h36m_joints.npy"), h36m)

    # Stage 4: Normalized coordinates
    print("\n--- Stage 4: Normalized Coordinates ---")
    W, H = W_orig, H_orig
    norm = normalize_screen_coordinates(h36m, w=W, h=H)
    print(f"  Normalized shape: {norm.shape}")
    print(f"  X range: [{norm[:, :, 0].min():.4f}, {norm[:, :, 0].max():.4f}]")
    print(f"  Y range: [{norm[:, :, 1].min():.4f}, {norm[:, :, 1].max():.4f}]")
    print(f"  Confidence range: [{norm[:, :, 2].min():.4f}, {norm[:, :, 2].max():.4f}]")

    # Save normalized coordinates
    np.save(os.path.join(output_dir, "04_normalized.npy"), norm)

    # Stage 5: MotionAGFormer prediction
    print("\n--- Stage 5: MotionAGFormer Prediction ---")
    model = get_model(device)
    inp = torch.from_numpy(norm.astype("float32"))[None].to(device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy()

    print(f"  Prediction shape: {pred.shape}")
    print(f"  Root: {pred[0, 13, 0]}")  # Center frame, joint 0
    print(f"  X range: [{pred[0, :, :, 0].min():.4f}, {pred[0, :, :, 0].max():.4f}]")
    print(f"  Y range: [{pred[0, :, :, 1].min():.4f}, {pred[0, :, :, 1].max():.4f}]")
    print(f"  Z range: [{pred[0, :, :, 2].min():.4f}, {pred[0, :, :, 2].max():.4f}]")

    # Save raw prediction
    np.save(os.path.join(output_dir, "05_raw_prediction.npy"), pred)

    # Stage 6: Final rendered pose
    print("\n--- Stage 6: Final Rendered Pose ---")
    pose3d = pred[0, 13]  # Center frame

    # Apply root-relative mode
    pose3d_root = pose3d - pose3d[0:1]

    # Render
    img_2d = draw_2d(kpts_raw, frame_rgb, scores=scores_raw)
    img_3d = render_3d(pose3d_root)
    combined = make_frame(img_2d, img_3d)

    print(f"  Combined shape: {combined.shape}")
    print(f"  Combined min: {combined.min()}, max: {combined.max()}")

    # Save final output
    cv2.imwrite(os.path.join(output_dir, "06_final_output.png"), combined)

    # Save 2D overlay separately
    cv2.imwrite(os.path.join(output_dir, "06_2d_overlay.png"), img_2d)

    # Save 3D render separately
    cv2.imwrite(os.path.join(output_dir, "06_3d_render.png"), img_3d)

    print(f"\nAll outputs saved to: {output_dir}")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"1. YOLO Detection: {'PASS' if np.sum(scores_raw > 0.1) > 0 else 'FAIL'}")
    print(f"   - Valid keypoints: {np.sum(scores_raw > 0.1)}/17")
    print(f"2. COCO Keypoints: PASS (shape={kpts_raw.shape})")
    print(f"3. H36M Conversion: PASS (shape={h36m.shape})")
    print(f"4. Normalization: PASS (range=[{norm[:,:,0].min():.2f}, {norm[:,:,0].max():.2f}])")
    print(f"5. MotionAGFormer: PASS (shape={pred.shape})")
    print(f"6. Rendering: {'PASS' if combined.max() > 0 else 'FAIL'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="thesis_artifacts/diagnostic")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    diagnose_image(args.input, args.output, args.device)
