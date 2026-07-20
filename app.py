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
    "Camera-relative root pose (qualitative)": "camera",
    "Canonical body-frame pose (qualitative)": "canonical",
}


def on_image_run(image, mode_label):
    if image is None:
        return None, None, None, "**Status:** Ready\n**Inference time:** -"
    try:
        mode = MODE_MAP.get(mode_label, "camera")
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
        mode = MODE_MAP.get(mode_label, "camera")
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
                "Camera-relative root pose (qualitative)",
                "Canonical body-frame pose (qualitative)",
            ],
            value="Camera-relative root pose (qualitative)",
            label="Visualization Mode",
            info="Camera-relative = root-zeroed pose rendered with fixed viewing angle. "
                 "Canonical = body-frame normalized pose with equal axis scaling.",
        )

    with gr.Tabs():
        # ── Image Tab ──
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Image", type="numpy", sources=["upload"])
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
        - **Camera-relative root pose**: Zeroes only the root joint. Non-root
          joints retain absolute positions. Rendered with a fixed viewing angle.
          Matches the official MotionAGFormer demo.
        - **Canonical body-frame pose**: Constructs a body-fixed coordinate
          system (vertical + hip axes) and expresses all joints in that frame.
          Reduces camera-orientation variation under approximately rigid and
          reliable 3D predictions. Uses equal axis scaling.

        *Both modes are qualitative visualizations, not benchmark evidence.*
        """
    )

    img_run_btn.click(
        fn=on_image_run,
        inputs=[img_input, mode_select],
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
