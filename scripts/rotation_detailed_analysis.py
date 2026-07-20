"""
Detailed rotation analysis: saves all intermediate pipeline outputs
for each rotation and tests determinism across multiple runs.
"""

import os
import sys
import numpy as np
import cv2
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.pose_detector import PoseDetector
from demo_live.lifter import build_xs_model, coco_to_h36m, lift_sequence, N_FRAMES
from demo_live.visualize import draw_2d, render_3d, make_frame


def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def analyze_rotation(detector, model, frame_rgb, angle, out_dir, run_id=0, device="cpu"):
    """Run full pipeline and save every intermediate output."""
    H, W = frame_rgb.shape[:2]
    prefix = f"run{run_id}_rot{angle:03d}"

    # Stage 1: Input image
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_01_input.png"),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Stage 2: YOLOv8 detection — get full details
    results = detector.model.predict(
        frame_rgb, device=device, conf=0.01, verbose=False  # low conf to see all
    )
    result = results[0]

    detection_info = {
        'n_persons': len(result.keypoints) if result.keypoints is not None else 0,
        'has_keypoints': result.keypoints is not None and len(result.keypoints) > 0,
    }

    if result.keypoints is not None and len(result.keypoints) > 0:
        kpts_xy = result.keypoints.xy[0].cpu().numpy()
        kpts_conf = result.keypoints.conf[0].cpu().numpy()
        detection_info['n_keypoints'] = int(np.sum(kpts_conf > 0.1))
        detection_info['mean_confidence'] = float(kpts_conf.mean())
        detection_info['max_confidence'] = float(kpts_conf.max())
        detection_info['min_confidence'] = float(kpts_conf.min())

        # Bounding box
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes.xyxy[0].cpu().numpy()
            detection_info['bbox'] = box.tolist()
            detection_info['bbox_area'] = float((box[2]-box[0]) * (box[3]-box[1]))
    else:
        detection_info['n_keypoints'] = 0
        detection_info['mean_confidence'] = 0.0
        detection_info['max_confidence'] = 0.0

    # Save detection info
    with open(os.path.join(out_dir, f"{prefix}_02_detection.txt"), 'w') as f:
        for k, v in detection_info.items():
            f.write(f"{k}: {v}\n")

    # Stage 3: Standard detection path
    kpts, scores = detector.detect(frame_rgb)
    detection_ok = kpts is not None
    if not detection_ok:
        # Save zero-filled and return early
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_03_2d_overlay.png"),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(out_dir, f"{prefix}_04_kpts.txt"), 'w') as f:
            f.write("detection_failed: True\n")
        return {
            'angle': angle, 'run': run_id,
            'detection_ok': False,
            'n_persons': detection_info['n_persons'],
            'mean_conf': detection_info['mean_confidence'],
        }

    # Save 2D overlay
    img_2d = draw_2d(kpts, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), scores=scores)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_03_2d_overlay.png"), img_2d)

    # Stage 4: Keypoints info
    with open(os.path.join(out_dir, f"{prefix}_04_kpts.txt"), 'w') as f:
        f.write(f"n_detected: {int(np.sum(scores > 0.1))}\n")
        f.write(f"mean_score: {scores.mean():.4f}\n")
        f.write(f"kpts_range_x: [{kpts[:, 0].min():.1f}, {kpts[:, 0].max():.1f}]\n")
        f.write(f"kpts_range_y: [{kpts[:, 1].min():.1f}, {kpts[:, 1].max():.1f}]\n")

    # Stage 5: Normalized keypoints
    from demo_live.lifter import coco_to_h36m
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)
    h36m = coco_to_h36m(kpts_seq, scores_seq)

    with open(os.path.join(out_dir, f"{prefix}_05_normalized.txt"), 'w') as f:
        f.write(f"h36m_shape: {h36m.shape}\n")
        f.write(f"xy_range: [{h36m[:,:,:2].min():.4f}, {h36m[:,:,:2].max():.4f}]\n")
        f.write(f"conf_mean: {h36m[:,:,2].mean():.4f}\n")

    # Stage 6: MotionAGFormer inference
    pose3d = lift_sequence(model, h36m, img_size=(H, W), mode="root",
                          use_flip=False)[-1]

    with open(os.path.join(out_dir, f"{prefix}_06_raw_3d.txt"), 'w') as f:
        f.write(f"shape: {pose3d.shape}\n")
        f.write(f"root: {pose3d[0].tolist()}\n")
        f.write(f"extent: {float(np.max(pose3d) - np.min(pose3d)):.4f}\n")
        f.write(f"nan_count: {int(np.isnan(pose3d).sum())}\n")

    # Stage 7: Save 3D visualization
    img_3d = render_3d(pose3d)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_07_3d_raw.png"), img_3d)

    # Save final combined
    img_2d = draw_2d(kpts, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), scores=scores)
    combined = make_frame(img_2d, img_3d)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_08_final.png"), combined)

    return {
        'angle': angle, 'run': run_id,
        'detection_ok': True,
        'n_persons': detection_info['n_persons'],
        'mean_conf': detection_info['mean_confidence'],
        'n_kpts': detection_info['n_keypoints'],
        'extent': float(np.max(pose3d) - np.min(pose3d)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="thesis_artifacts/rotation_analysis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per rotation for determinism check")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model and detector...")
    detector = PoseDetector(weights="yolov8n-pose.pt", device=args.device, conf=0.4)
    model = build_xs_model(device=args.device)

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"ERROR: Cannot read {args.input}")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rotations = [0, 90, 180, 270]
    all_results = []

    for angle in rotations:
        print(f"\n{'='*50}")
        print(f"Testing {angle}° rotation ({args.runs} runs)")
        print(f"{'='*50}")

        rotated = rotate_image(frame_rgb, angle)

        for run in range(args.runs):
            result = analyze_rotation(detector, model, rotated, angle, args.output, run, args.device)
            all_results.append(result)
            status = "OK" if result['detection_ok'] else "DETECTION_FAILED"
            print(f"  Run {run}: {status}, conf={result.get('mean_conf', 0):.3f}, "
                  f"extent={result.get('extent', 0):.4f}")

    # Summary
    print(f"\n{'='*50}")
    print("DETERMINISM CHECK")
    print(f"{'='*50}")
    for angle in rotations:
        angle_results = [r for r in all_results if r['angle'] == angle]
        extents = [r['extent'] for r in angle_results if r['detection_ok']]
        if extents:
            print(f"  {angle}°: {len(extents)}/{args.runs} detections, "
                  f"extent range: [{min(extents):.4f}, {max(extents):.4f}], "
                  f"std: {np.std(extents):.6f}")
        else:
            print(f"  {angle}°: 0/{args.runs} detections")


if __name__ == "__main__":
    main()
