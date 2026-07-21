"""
MotionAGFormer Live Demo — Interactive 3D Viewer

Architecture:
    RGB Image / Video
            │
            ▼
    YOLOv8 Pose Detector
            │
            ▼
    MotionAGFormer-XS
            │
            ├───────────────┐
            │               │
            ▼               ▼
    Camera Pose      View-Invariant Pose
            │               │
            └──────┬────────┘
                   ▼
          Interactive Plotly Viewer

Run:  python app.py
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.inference import estimate_poses, predict_video
from demo_live.plotly_renderer import render_pose_plotly


def on_image_run(image, coord_space, rotation_deg):
    """Run inference and return the Plotly viewer + 2D overlay + status."""
    if image is None:
        return None, None, "**Status:** Ready"

    import cv2
    import numpy as np

    try:
        # Apply rotation if specified.
        if rotation_deg is not None and rotation_deg != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

        result = estimate_poses(image)
        if result is None:
            return None, None, "**Status:** No person detected"

        # Select pose based on coordinate space.
        if coord_space == "View-Invariant Coordinate System":
            pose = result["view_invariant"]
            space_label = "View-Invariant"
        else:
            pose = result["camera_pose"]
            space_label = "Camera"

        # Render interactive 3D viewer.
        fig = render_pose_plotly(pose, title=f"3D Pose — {space_label}")

        # Render 2D overlay for comparison.
        from demo_live.visualize import draw_2d
        overlay_bgr = draw_2d(result["kpts_2d"],
                               cv2.cvtColor(result["frame_rgb"], cv2.COLOR_RGB2BGR),
                               scores=result["scores"])
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        t = result["inference_time"]
        kp_valid = int(np.sum(result["scores"] > 0.3))
        status = (f"**Status:** Completed\n"
                  f"**Inference time:** {t:.2f}s\n"
                  f"**Keypoints:** {kp_valid}/17 (conf={result['scores'].mean():.3f})\n"
                  f"**Coordinate space:** {space_label}")

        return fig, overlay_rgb, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"**Status:** Error — {e}"


def on_coord_change(coord_space, _state_image, _state_result):
    """Switch coordinate space without re-running inference.
    NOTE: Gradio doesn't support stateful callbacks easily, so for now
    we re-run the full pipeline. A stateful version can be added later."""
    # For the initial implementation, coordinate switching triggers re-inference.
    # This is acceptable because inference takes ~3s on CPU.
    return gr.update()


with gr.Blocks(title="MotionAGFormer — View-Invariant 3D Pose", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MotionAGFormer — Interactive 3D Pose Viewer")
    gr.Markdown(
        "Upload an RGB image or video to reconstruct 3D human pose. "
        "Switch between **Camera** and **View-Invariant** coordinate systems "
        "to see how the thesis contribution transforms the pose representation."
    )

    with gr.Tabs():
        # ── Image Tab ──
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        sources=["upload"],
                    )
                    coord_space = gr.Radio(
                        choices=[
                            "Camera Coordinate System",
                            "View-Invariant Coordinate System",
                        ],
                        value="Camera Coordinate System",
                        label="Representation",
                        info="Switching changes the pose coordinates. Same renderer, same skeleton.",
                    )
                    rotation_slider = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=1,
                        label="Rotation (degrees)",
                        info="Rotate input image before detection. Useful for tilted images.",
                    )
                    img_run_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column(scale=2):
                    viewer_3d = gr.Plot(
                        label="Interactive 3D Viewer",
                        show_label=True,
                    )
                    img_overlay = gr.Image(
                        label="2D Pose Detection",
                        type="numpy",
                        interactive=False,
                    )

            img_status = gr.Markdown("**Status:** Ready")

        # ── Video Tab ──
        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video", sources=["upload"])
                    vid_coord = gr.Radio(
                        choices=[
                            "Camera Coordinate System",
                            "View-Invariant Coordinate System",
                        ],
                        value="Camera Coordinate System",
                        label="Representation",
                    )
                    vid_run_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column(scale=2):
                    vid_output = gr.Video(label="Output", interactive=False)

            vid_status = gr.Markdown("**Status:** Ready")

    # ── Architecture Info ──
    gr.Markdown(
        """
        ---
        ### Architecture

        ```
        RGB Image / Video
                │
                ▼
        YOLOv8 Pose Detector
                │
                ▼
        MotionAGFormer-XS (2.2M params, H36M)
                │
                ├───────────────┐
                │               │
                ▼               ▼
        Camera Pose      View-Invariant Pose
                │               │
                └──────┬────────┘
                       ▼
              Interactive 3D Viewer
        ```

        | Property | Value |
        |----------|-------|
        | Model | MotionAGFormer-XS |
        | Dataset | Human3.6M |
        | Input Window | 27 Frames |
        | 2D Detector | YOLOv8 Pose |
        | Parameters | 2.2M |
        | MPJPE | 45.1 mm |
        | P-MPJPE | 36.9 mm |
        | Viewer | Plotly WebGL (orbit, pan, zoom) |

        ### Coordinate Spaces
        - **Camera Coordinate System**: Standard MotionAGFormer output. The pose is reconstructed as observed from the input camera.
        - **View-Invariant Coordinate System**: The thesis contribution. Body orientation is normalized, viewpoint dependency is removed, motion is preserved.
        """
    )

    # ── Event Handlers ──
    img_run_btn.click(
        fn=on_image_run,
        inputs=[img_input, coord_space, rotation_slider],
        outputs=[viewer_3d, img_overlay, img_status],
    )

    # Re-run when coordinate space changes (after inference).
    coord_space.change(
        fn=on_image_run,
        inputs=[img_input, coord_space, rotation_slider],
        outputs=[viewer_3d, img_overlay, img_status],
    )

    vid_run_btn.click(
        fn=lambda video, cs: predict_video(
            video,
            mode="canonical" if "View-Invariant" in cs else "motionagformer",
        ),
        inputs=[vid_input, vid_coord],
        outputs=[vid_output, vid_status],
    )


if __name__ == "__main__":
    print("Loading models (first run may take a moment)...")
    from backend.model_loader import get_detector, get_model
    get_detector()
    get_model()
    print("Models loaded. Starting Gradio server...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
