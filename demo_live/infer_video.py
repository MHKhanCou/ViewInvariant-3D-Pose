"""
Video demo: video -> per-frame 2D detection -> MotionAGFormer-XS 27-frame
sliding-window lift -> 2D overlay + 3D render -> saved .mp4.

Usage:
    python demo_live/infer_video.py --input person.mp4
    python demo_live/infer_video.py --input person.mp4 --output out.mp4

The 3D lifting uses a dense 27-frame sliding window, so every output frame
has a corresponding 3D pose. The result is an mp4 with the 2D skeleton overlay
on the left and the 3D reconstruction on the right.
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_detector import PoseDetector, detect_video
from lifter import build_xs_model, coco_to_h36m, lift_sequence
from visualize import draw_2d, render_3d, save_video_demo


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Video 3D pose demo (MotionAGFormer-XS)")
    parser.add_argument("--input", type=str, required=True, help="path to input video")
    parser.add_argument("--output", type=str, default=None, help="output mp4 path")
    parser.add_argument("--det-weights", type=str, default="yolov8n-pose.pt",
                        help="Ultralytics pose model (.pt), COCO-17 output")
    parser.add_argument("--conf", type=float, default=0.4, help="detection confidence")
    parser.add_argument("--det-width", type=int, default=640,
                        help="downscale frames to this width for detection (4K is slow); "
                             "set 0 to disable")
    parser.add_argument("--out-height", type=int, default=480,
                        help="display height of the combined output (keeps memory low on 4K)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda id")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("demo_live", "output", base + "_demo.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    device = "cpu" if args.device == "cpu" or not _cuda_ok() else args.device

    print("[1/5] Loading detector ...")
    detector = PoseDetector(weights=args.det_weights, device=device, conf=args.conf)

    print("[2/5] Loading MotionAGFormer-XS lifter ...")
    model = build_xs_model(device=device)

    print("[3/5] Detecting 2D poses per frame ...")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    keypoints, scores = detect_video(detector, args.input,
                                      det_width=None if args.det_width == 0 else args.det_width)
    T = keypoints.shape[0]
    print(f"      detected {T} frames (video reported {total}).")

    print("[4/5] Lifting to 3D (27-frame sliding window) ...")
    h36m = coco_to_h36m(keypoints, scores)
    poses3d = lift_sequence(model, h36m, img_size=(H, W), device=device)
    # Safety: align lengths if padding changed T.
    if len(poses3d) != T:
        poses3d = poses3d[-T:]

    print("[5/5] Rendering and saving mp4 ...")
    cap = cv2.VideoCapture(args.input)
    frames_out = []
    for i in range(T):
        ret, frame = cap.read()
        if not ret:
            break
        img_2d = draw_2d(keypoints[i], frame, scores=scores[i])
        if args.out_height and img_2d.shape[0] > args.out_height:
            s = args.out_height / float(img_2d.shape[0])
            img_2d = cv2.resize(img_2d, (int(round(img_2d.shape[1] * s)), args.out_height))

        # Skip 3D rendering when no person is detected (collapsed pose from zeros).
        mean_conf = float(scores[i].mean()) if scores[i] is not None else 0.0
        if mean_conf < 0.1:
            # Blank white 3D panel matching the 2D panel height.
            h = img_2d.shape[0]
            w = int(round(h * 6.4 / 5.4))  # match render_3d figsize ratio
            img_3d = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
            img_3d = render_3d(poses3d[i])
        frames_out.append(_stack(img_2d, img_3d))
    cap.release()

    save_video_demo(frames_out, args.output, fps=fps)
    print(f"Done. Output saved to: {args.output}")


def _stack(img_2d, img_3d):
    from visualize import make_frame
    return make_frame(img_2d, img_3d)


if __name__ == "__main__":
    main()
