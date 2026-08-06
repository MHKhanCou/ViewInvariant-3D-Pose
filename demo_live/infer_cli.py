"""
Same output as the Gradio app, from the command line.

    python demo_live/infer_cli.py --input photo.jpg --mode canonical
    python demo_live/infer_cli.py --input clip.mp4 --mode camera

Calls the same backend the app calls, so the rendering is identical.
Image modes: motionagformer | canonical | avatar.  Video modes: motionagformer | canonical.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import predict_image, predict_video

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    p = argparse.ArgumentParser(description="App-equivalent CLI (image or video)")
    p.add_argument("--input", required=True, help="image or video path")
    p.add_argument("--mode", default="canonical",
                   choices=["motionagformer", "canonical", "avatar", "camera"],
                   help="coordinate space / rendering mode (default: canonical)")
    p.add_argument("--output", default=None, help="output path (default: demo_live/output/)")
    p.add_argument("--bvh", action="store_true", help="video only: also export BVH")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    is_video = os.path.splitext(args.input)[1].lower() in VIDEO_EXT

    if is_video:
        out_path, secs = predict_video(args.input, mode=args.mode, export_bvh=args.bvh)
        if args.output:
            os.replace(out_path, args.output)
            out_path = args.output
    else:
        image_bgr = cv2.imread(args.input)
        if image_bgr is None:
            sys.exit(f"Could not read image: {args.input}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _, _, combined_rgb, secs = predict_image(image_rgb, mode=args.mode)
        out_path = args.output or os.path.join(OUT_DIR, f"{stem}_{args.mode}.png")
        cv2.imwrite(out_path, cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))

    print(f"{out_path}  ({secs:.2f}s, mode={args.mode})")


if __name__ == "__main__":
    main()
