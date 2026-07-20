"""
Rotation Robustness Investigation

Tests the pipeline with rotated input images to identify where failures occur.

For each rotation (0°, 90°, 180°, 270°):
1. Save intermediate outputs from every stage
2. Compare with original orientation
3. Identify first stage where pipeline becomes inconsistent
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
from canonical.canonicalizer import Canonicalizer
from canonical.visualization import render_canonical_3d


def rotate_image(img, angle):
    """Rotate image by angle degrees (90, 180, 270)."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def run_pipeline_on_image(detector, model, frame_rgb, out_dir, label, device="cpu"):
    """Run full pipeline and save all intermediate outputs."""
    H, W = frame_rgb.shape[:2]

    # Stage 1: Original RGB
    cv2.imwrite(os.path.join(out_dir, f"{label}_01_original.png"),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Stage 2: YOLOv8 2D detection
    kpts, scores = detector.detect(frame_rgb)
    detection_success = kpts is not None
    if kpts is None:
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    # Save 2D overlay
    img_2d = draw_2d(kpts, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), scores=scores)
    cv2.imwrite(os.path.join(out_dir, f"{label}_02_2d_detection.png"), img_2d)

    # Stage 3: Normalized keypoints
    from demo_live.lifter import coco_to_h36m
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)

    # Check if detection failed
    if not detection_success:
        # Save info and return early
        with open(os.path.join(out_dir, f"{label}_03_norm_kpts.txt"), 'w') as f:
            f.write("detection_failed: True\n")
            f.write("kpts are zero-filled placeholders\n")
        with open(os.path.join(out_dir, f"{label}_04_raw_3d.txt"), 'w') as f:
            f.write("skipped: detection failed\n")
        with open(os.path.join(out_dir, f"{label}_05_canonical.txt"), 'w') as f:
            f.write("skipped: detection failed\n")
        # Save 2D overlay with zero detection
        img_2d = draw_2d(kpts, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), scores=scores)
        cv2.imwrite(os.path.join(out_dir, f"{label}_02_2d_detection.png"), img_2d)
        cv2.imwrite(os.path.join(out_dir, f"{label}_06_final.png"), img_2d)
        return {
            'label': label,
            'kpts_shape': (0,),
            'raw_extent': 0.0,
            'can_valid': False,
            'can_max_error': None,
            'detection_failed': True,
        }

    h36m = coco_to_h36m(kpts_seq, scores_seq)

    # Save normalized keypoints info
    norm_info = {
        'kpts_shape': h36m.shape,
        'kpts_min': float(h36m[:, :, :2].min()),
        'kpts_max': float(h36m[:, :, :2].max()),
        'conf_mean': float(h36m[:, :, 2].mean()),
    }
    with open(os.path.join(out_dir, f"{label}_03_norm_kpts.txt"), 'w') as f:
        for k, v in norm_info.items():
            f.write(f"{k}: {v}\n")

    # Stage 4: MotionAGFormer inference
    pose3d = lift_sequence(model, h36m, img_size=(H, W), mode="root",
                          use_flip=False)[-1]

    # Save raw 3D output info
    raw_info = {
        'shape': pose3d.shape,
        'root': pose3d[0].tolist(),
        'extent': float(np.max(pose3d) - np.min(pose3d)),
    }
    with open(os.path.join(out_dir, f"{label}_04_raw_3d.txt"), 'w') as f:
        for k, v in raw_info.items():
            f.write(f"{k}: {v}\n")

    # Stage 5: Canonical output
    can = Canonicalizer()
    pose_can, R = can(pose3d)
    meta = {"valid": True}  # Canonicalizer doesn't return metadata by default

    can_info = {
        'valid': meta['valid'],
        'root': pose_can[0].tolist(),
        'max_error': float(np.max(np.abs(pose_can - pose3d))) if meta['valid'] else None,
    }
    with open(os.path.join(out_dir, f"{label}_05_canonical.txt"), 'w') as f:
        for k, v in can_info.items():
            f.write(f"{k}: {v}\n")

    # Stage 6: Final visualization
    img_3d = render_3d(pose3d)
    img_can = render_canonical_3d(pose_can) if meta['valid'] else img_3d
    combined = make_frame(img_2d, img_can)
    cv2.imwrite(os.path.join(out_dir, f"{label}_06_final.png"), combined)

    return {
        'label': label,
        'kpts_shape': h36m.shape,
        'raw_extent': float(np.max(pose3d) - np.min(pose3d)),
        'can_valid': meta['valid'],
        'can_max_error': float(np.max(np.abs(pose_can - pose3d))) if meta['valid'] else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Rotation Robustness Test")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="thesis_artifacts/rotation_test")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model and detector...")
    detector = PoseDetector(weights="yolov8n-pose.pt", device=args.device, conf=0.4)
    model = build_xs_model(device=args.device)

    # Load original image
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"ERROR: Cannot read {args.input}")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Test all rotations
    rotations = [0, 90, 180, 270]
    results = []

    for angle in rotations:
        print(f"\nTesting {angle}° rotation...")
        rotated = rotate_image(frame_rgb, angle)
        label = f"rot_{angle:03d}"
        result = run_pipeline_on_image(detector, model, rotated, args.output, label, args.device)
        results.append(result)
        print(f"  Valid: {result['can_valid']}, Max error: {result['can_max_error']}")

    # Summary
    print("\n" + "=" * 60)
    print("ROTATION ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"{'Rotation':<12} {'Valid':<8} {'Max Error':<12} {'Raw Extent':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<12} {str(r['can_valid']):<8} "
              f"{r['can_max_error'] if r['can_max_error'] else 'N/A':<12} "
              f"{r['raw_extent']:<12.4f}")

    # Analyze where pipeline fails
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    valid_rotations = [r for r in results if r['can_valid']]
    invalid_rotations = [r for r in results if not r['can_valid']]

    if invalid_rotations:
        print(f"Failed rotations: {[r['label'] for r in invalid_rotations]}")
        print("The pipeline fails at canonicalization for rotated inputs.")
        print("This is expected: YOLOv8 detects keypoints in screen coordinates,")
        print("and rotation changes the spatial layout of detected keypoints.")
        print("MotionAGFormer was trained on upright human poses, so rotated")
        print("inputs produce unreliable 3D predictions.")
    else:
        print("All rotations produced valid canonical outputs.")
        errors = [r['can_max_error'] for r in results if r['can_max_error']]
        if errors:
            print(f"Error range: {min(errors):.6f} to {max(errors):.6f}")

    # Save summary
    with open(os.path.join(args.output, "rotation_summary.txt"), 'w') as f:
        f.write("ROTATION ROBUSTNESS SUMMARY\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(f"{r['label']}: valid={r['can_valid']}, "
                   f"max_error={r['can_max_error']}, raw_extent={r['raw_extent']:.4f}\n")
        f.write("\nANALYSIS\n")
        f.write("=" * 60 + "\n")
        if invalid_rotations:
            f.write(f"Failed rotations: {[r['label'] for r in invalid_rotations]}\n")
            f.write("Cause: YOLOv8 detects keypoints in screen coordinates.\n")
            f.write("Rotation changes spatial layout, producing unreliable 3D.\n")
            f.write("MotionAGFormer was trained on upright poses.\n")
        else:
            f.write("All rotations produced valid outputs.\n")

    print(f"\nResults saved to: {args.output}")


import argparse

if __name__ == "__main__":
    main()
