"""
MotionAGFormer Live Demo - Gradio Web Application

Run:  python app.py
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.inference import predict_image, predict_video

# Map display labels to internal mode values.
MODE_MAP = {
    "MotionAGFormer View (default)": "motionagformer",
    "Canonical Body-Frame Pose (Research View)": "canonical",
    "Blender/Avatar View (Image-only, qualitative)": "avatar",
}


def on_image_run(image, mode_label, rotation_deg):
    if image is None:
        return None, None, None, "**Status:** Ready\n**Inference time:** -"
    try:
        mode = MODE_MAP.get(mode_label, "motionagformer")
        # Apply rotation if specified
        if rotation_deg is not None and rotation_deg != 0:
            import cv2
            import numpy as np
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        original, overlay, combined, t = predict_image(image, mode=mode)
        status = f"**Status:** Completed\n**Inference time:** {t:.2f}s"
        return original, overlay, combined, status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"**Status:** Error - {e}"


def on_video_run(video, mode_label):
    if video is None:
        return None, "**Status:** Ready\n**Inference time:** -"
    try:
        mode = MODE_MAP.get(mode_label, "motionagformer")
        if mode == "avatar":
            return None, ("**Status:** Avatar View is currently available for images only. "
                          "Use an image upload, or select MotionAGFormer View or "
                          "Canonical Body-Frame Pose for video.")
        out_path, t = predict_video(video, mode=mode)
        status = f"**Status:** Completed\n**Inference time:** {t:.2f}s"
        return out_path, status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"**Status:** Error - {e}"


with gr.Blocks(title="MotionAGFormer Demo") as demo:
    gr.Markdown("# MotionAGFormer Live Demo")
    gr.Markdown("Monocular 3D Human Pose Estimation from RGB Images")

    # ── Visualization Mode Selector ──
    with gr.Row():
        mode_select = gr.Radio(
            choices=[
                "MotionAGFormer View (default)",
                "Canonical Body-Frame Pose (Research View)",
                "Blender/Avatar View (Image-only, qualitative)",
            ],
            value="MotionAGFormer View (default)",
            label="Visualization Mode",
            info="Each option is a separate output layer. The default matches the "
                 "official MotionAGFormer display style; canonical is research-only; "
                 "avatar is presentation-only and image-only.",
        )

    with gr.Tabs():
        # ── Image Tab ──
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Image", type="numpy", sources=["upload"])
                    rotation_slider = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=1,
                        label="Rotation (degrees)",
                        info="Rotate input image before detection. Useful for tilted images."
                    )
                    img_run_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        img_original = gr.Image(label="Original", type="numpy", interactive=False)
                        img_overlay = gr.Image(label="2D Pose", type="numpy", interactive=False)
                    img_combined = gr.Image(label="2D + 3D Pose", type="numpy", interactive=False)
            img_status = gr.Markdown("**Status:** Ready\n**Inference time:** -")

        # ── Video Tab ──
        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video", sources=["upload"])
                    vid_run_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column(scale=2):
                    vid_output = gr.Video(label="Output", interactive=False)
            vid_status = gr.Markdown("**Status:** Ready\n**Inference time:** -")

    # ── Info Panel ──
    gr.Markdown(
        """
        ---
        ### Model Information
        | Property | Value |
        |----------|-------|
        | Model | MotionAGFormer-XS |
        | Dataset | Human3.6M |
        | Input Window | 27 Frames |
        | 2D Detector | YOLOv8 Pose |
        | Parameters | 2.2M |
        | MPJPE | 45.1 mm |
        | P-MPJPE | 36.9 mm |

        ### Visualization Modes
        - **MotionAGFormer View (default):** Official-demo-style display of the
          predicted root-relative pose: fixed visualization rotation, root-zeroing,
          normalization, camera angle, bounds, and colour convention. It is not
          world-space trajectory recovery.
        - **Canonical Body-Frame Pose (Research View):** The thesis contribution.
          It re-expresses the predicted pose in a body-fixed frame for qualitative
          viewpoint-normalized comparison. It is separate from the default view.
        - **Blender/Avatar View (Image-only):** A lightweight stylized renderer
          driven by predicted joints. It is not Blender, SMPL fitting, or an
          accuracy improvement; it is presentation-only.

        *No visualization mode changes the checkpoint, architecture, training,
        official evaluation, or benchmark results.*
        """
    )

    img_run_btn.click(
        fn=on_image_run,
        inputs=[img_input, mode_select, rotation_slider],
        outputs=[img_original, img_overlay, img_combined, img_status],
    )

    vid_run_btn.click(
        fn=on_video_run,
        inputs=[vid_input, mode_select],
        outputs=[vid_output, vid_status],
    )


if __name__ == "__main__":
    print("Loading models (first run may take a moment)...")
    from backend.model_loader import get_detector, get_model
    get_detector()
    get_model()
    print("Models loaded. Starting Gradio server...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
