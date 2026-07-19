"""
Image demo: picture -> 2D detection -> MotionAGFormer-XS lift -> 3D render.

Usage:
    python demo_live/infer_image.py --input person.jpg
    python demo_live/infer_image.py --input person.jpg --output out.png --weights yolov8n-pose.pt

Outputs a side-by-side PNG: 2D skeleton overlay | 3D pose reconstruction.
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_detector import PoseDetector
from lifter import build_xs_model, coco_to_h36m, lift_sequence, N_FRAMES
from visualize import draw_2d, render_3d, save_image_demo


def main():
    parser = argparse.ArgumentParser(description="Image 3D pose demo (MotionAGFormer-XS)")
    parser.add_argument("--input", type=str, required=True, help="path to input image")
    parser.add_argument("--output", type=str, default=None, help="output image path")
    parser.add_argument("--det-weights", type=str, default="yolov8n-pose.pt",
                        help="Ultralytics pose model (.pt), COCO-17 output")
    parser.add_argument("--conf", type=float, default=0.4, help="detection confidence")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda id")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("demo_live", "output", base + "_demo.png")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    device = "cpu" if args.device == "cpu" or not _cuda_ok() else args.device

    print("[1/4] Loading detector ...")
    detector = PoseDetector(weights=args.det_weights, device=device, conf=args.conf)

    print("[2/4] Loading MotionAGFormer-XS lifter ...")
    model = build_xs_model(device=device)

    print("[3/4] Detecting 2D pose ...")
    frame = cv2.imread(args.input)
    if frame is None:
        raise IOError(f"Cannot read image: {args.input}")
    H, W = frame.shape[:2]

    kpts, scores = detector.detect(frame)
    if kpts is None:
        print("WARNING: no person detected in the image.")
        kpts = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros((17,), dtype=np.float32)

    # Repeat the single frame to fill the 27-frame window (static pose).
    kpts_seq = np.repeat(kpts[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores[None], N_FRAMES, axis=0)
    h36m = coco_to_h36m(kpts_seq, scores_seq)

    print("[4/4] Lifting to 3D and rendering ...")
    pose3d = lift_sequence(model, h36m, img_size=(H, W), device=device)[-1]

    img_2d = draw_2d(kpts, frame, scores=scores)
    img_3d = render_3d(pose3d)
    save_image_demo(img_2d, img_3d, args.output)

    print(f"Done. Output saved to: {args.output}")


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
