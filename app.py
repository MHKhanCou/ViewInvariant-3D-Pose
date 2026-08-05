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
import warnings
warnings.filterwarnings("ignore", category=EncodingWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.inference import estimate_poses, predict_video
from demo_live.plotly_renderer import render_pose_plotly, get_bone_length_summary


def on_image_run(image, coord_space, rotation_deg, show_avatar=False,
                 equalize_limbs=True):
    """Run inference and return the Plotly viewer + 2D overlay + status."""
    if image is None:
        return None, None, "**Status:** Ready", "", gr.update(visible=False)

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
            return None, None, "**Status:** No person detected", "", gr.update(visible=False)

        # Select pose based on coordinate space.
        if coord_space == "View-Invariant Coordinate System":
            pose = np.asarray(result["view_invariant"], dtype=np.float64)
            # Display-only axis remap: the canonical body frame carries the
            # body vertical on +y, but the viewer draws +z as up, which made
            # the skeleton appear to lie on its side. Rotate +90° about x —
            # (x, y, z) -> (x, -z, y) — a proper rotation (no mirroring), so
            # the body stands upright. Evaluated coordinates are untouched.
            pose = np.column_stack([pose[:, 0], -pose[:, 2], pose[:, 1]])
            space_label = "View-Invariant"
        else:
            pose = result["motionagformer_display_pose"]
            space_label = "Camera"

        # Display-only limb-length equalization (never touches scored poses).
        if equalize_limbs:
            from backend.inference import equalize_limb_lengths
            pose = equalize_limb_lengths(pose)
            space_label += ", limbs equalized (display)"

        # Render interactive 3D viewer.
        fig = render_pose_plotly(pose, title=f"3D Pose — {space_label}")

        # Render 2D overlay for comparison.
        from demo_live.visualize import draw_2d
        overlay_bgr = draw_2d(result["kpts_2d"],
                               cv2.cvtColor(result["frame_rgb"], cv2.COLOR_RGB2BGR),
                               scores=result["scores"])
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Bone length summary.
        bone_summary = get_bone_length_summary(pose)

        t = result["inference_time"]
        kp_valid = int(np.sum(result["scores"] > 0.3))

        # --- Geometric plausibility gate ---
        # This score is NOT an accuracy estimate and must not be presented as
        # one. The report falsifies it as an error predictor along seven
        # independent axes: it is invariant to the depth error that dominates
        # monocular estimation, because a pose can be symmetric, correctly
        # proportioned and well conditioned while being wrong in depth. What it
        # does detect is corruption and degeneracy (rho = -0.813 under the
        # controlled degradation sweep). The wording below is deliberately
        # limited to that, so the demonstration cannot contradict the thesis.
        rel = result["reliability"]
        if result["hard_failure"]:
            verdict = (f"🔴 **Degenerate geometry** — {result['failure_reason']}. "
                       f"The body frame cannot be built reliably here.")
        elif result["abstain"]:
            verdict = (f"🟠 **Implausible geometry** — plausibility {rel:.3f}, "
                       f"below the 0.5 gate. Likely a corrupted or degenerate skeleton.")
        else:
            verdict = (f"🟢 **Geometry plausible** — plausibility {rel:.3f}. "
                       f"This says the skeleton is well formed. It does *not* "
                       f"estimate depth accuracy.")
        comps = result["reliability_components"]
        comp_line = " · ".join(
            f"{name.replace('_', ' ')} {val:.2f}" for name, val in comps.items())

        status = (f"**Status:** Completed\n\n"
                  f"**Geometric plausibility:** {verdict}\n"
                  f"<sub>{comp_line}</sub>\n\n"
                  # No section number here on purpose: it was written as 5.7,
                  # the report has since renumbered it to 5.8, and it will move
                  # again. A reference that cannot go stale is worth more than a
                  # precise one that does.
                  f"<sub>⚠️ Plausibility is a degeneracy gate, not a confidence "
                  f"score. The report falsifies it as a predictor of accuracy "
                  f"along five independent axes; a pose that is smoothly wrong "
                  f"in depth still scores highly.</sub>\n\n"
                  f"**Inference time:** {t:.2f}s\n"
                  f"**Keypoints:** {kp_valid}/17 (conf={result['scores'].mean():.3f})\n"
                  f"**Coordinate space:** {space_label}")

        # Optional qualitative avatar render (presentation layer only).
        if show_avatar:
            from presentation.avatar_renderer import render_stylized_avatar_3d
            avatar_bgr = render_stylized_avatar_3d(
                result["motionagformer_display_pose"])
            avatar_rgb = cv2.cvtColor(avatar_bgr, cv2.COLOR_BGR2RGB)
            avatar_update = gr.update(value=avatar_rgb, visible=True)
        else:
            avatar_update = gr.update(visible=False)

        return fig, overlay_rgb, status, bone_summary, avatar_update

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"**Status:** Error — {e}", "", gr.update(visible=False)


def on_coord_change(coord_space, _state_image, _state_result):
    """Switch coordinate space without re-running inference.
    NOTE: Gradio doesn't support stateful callbacks easily, so for now
    we re-run the full pipeline. A stateful version can be added later."""
    # For the initial implementation, coordinate switching triggers re-inference.
    # This is acceptable because inference takes ~3s on CPU.
    return gr.update()


with gr.Blocks(title="View-Invariant 3D Pose — Training-Free Canonicalization") as demo:
    gr.Markdown("# View-Invariant 3D Pose — Interactive Demonstration")
    gr.Markdown(
        "A frozen monocular estimator predicts a skeleton in the coordinate "
        "frame of whichever camera happened to record it, so the same motion "
        "seen from two viewpoints yields two different sets of numbers. This "
        "demonstration applies a body-fixed frame **after** prediction, adding "
        "no trained parameters, no labels and no calibration.\n\n"
        "Switch **Representation** below to see the transform. In the camera "
        "frame the coordinates depend on where the camera stood; in the "
        "view-invariant frame they do not."
    )
    with gr.Accordion("What the measured results are, and what this demo can "
                      "and cannot show", open=False):
        gr.Markdown(
            "**Measured on held-out data.** On 180 held-out camera pairs of "
            "Human3.6M, a dataset that played no part in developing the method, "
            "the body frame reduces cross-view joint distance from 320.4 mm to "
            "75.3 mm, an improvement of 74.1 percent with a 95 percent interval "
            "of [+69.8, +77.2]. It improves 179 of the 180 pairs and closes "
            "90.5 percent of the gap to an oracle that aligns each pair "
            "optimally using knowledge of both views.\n\n"
            "**What this single-image demo cannot show.** The claim is about "
            "*agreement between simultaneous views*, which needs two cameras. "
            "With one image you can see that the representation changes and "
            "that the skeleton is well formed, but not that two views agree. "
            "Treat this as an illustration of the transform, not as evidence "
            "for it; the evidence is in Chapter 5.\n\n"
            "**What the plausibility score is not.** It gates degenerate and "
            "corrupted skeletons. It does not estimate accuracy, and the "
            "report falsifies it as an error predictor along seven independent "
            "axes. A prediction that is smoothly wrong in depth scores highly, "
            "which is precisely the failure mode it cannot see."
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
                        info="Same skeleton, same renderer, different coordinates. "
                             "The camera frame depends on where the camera stood; "
                             "the view-invariant frame does not.",
                    )
                    rotation_slider = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=1,
                        label="Rotation (degrees)",
                        info="Rotate input image before detection. Useful for tilted images.",
                    )
                    equalize_limbs = gr.Checkbox(
                        value=True,
                        label="Equalize limb lengths (display only)",
                        info="The estimator predicts left and right limbs at "
                             "different lengths, by up to 23 percent on the "
                             "forearm. This averages them for display only. "
                             "Raw predictions and all scoring are unaffected, "
                             "and no reported number uses this.",
                    )
                    show_avatar = gr.Checkbox(
                        value=False,
                        label="Show stylized avatar",
                        info="Qualitative presentation render of the predicted pose.",
                    )
                    img_run_btn = gr.Button("Run Inference", variant="primary")
                    img_overlay = gr.Image(
                        label="2D Pose Detection",
                        type="numpy",
                        interactive=False,
                    )

                with gr.Column(scale=2):
                    viewer_3d = gr.Plot(
                        label="Interactive 3D Viewer",
                        show_label=True,
                    )
                    img_status = gr.Markdown("**Status:** Ready")
                    bone_metrics = gr.Markdown("")
                    avatar_view = gr.Image(
                        label="Stylized Avatar (qualitative)",
                        type="numpy",
                        interactive=False,
                        visible=False,
                    )

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
                    vid_bvh = gr.Checkbox(
                        value=False,
                        label="Also export BVH (motion capture)",
                        info="Writes a .bvh file for Blender, Unity or "
                             "MotionBuilder. BVH stores joint rotations relative "
                             "to each parent and contains no camera, which is "
                             "what the view-invariant frame produces.",
                    )
                    vid_run_btn = gr.Button("Run Inference", variant="primary")
                with gr.Column(scale=2):
                    vid_output = gr.Video(label="Output", interactive=False)
                    vid_bvh_file = gr.File(label="Motion capture (.bvh)",
                                           visible=False, interactive=False)

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
        inputs=[img_input, coord_space, rotation_slider, show_avatar, equalize_limbs],
        outputs=[viewer_3d, img_overlay, img_status, bone_metrics, avatar_view],
    )

    # Re-run when coordinate space or display options change (after inference).
    coord_space.change(
        fn=on_image_run,
        inputs=[img_input, coord_space, rotation_slider, show_avatar, equalize_limbs],
        outputs=[viewer_3d, img_overlay, img_status, bone_metrics, avatar_view],
    )
    equalize_limbs.change(
        fn=on_image_run,
        inputs=[img_input, coord_space, rotation_slider, show_avatar, equalize_limbs],
        outputs=[viewer_3d, img_overlay, img_status, bone_metrics, avatar_view],
    )

    def on_video_run(video, cs, want_bvh):
        if video is None:
            return None, "**Status:** Ready", gr.update(visible=False)
        try:
            mode = "canonical" if "View-Invariant" in cs else "motionagformer"
            if want_bvh:
                out_path, t, bvh_path = predict_video(video, mode=mode, export_bvh=True)
                note = ("\n\n**BVH exported.** Import into Blender with "
                        "*File → Import → Motion Capture (.bvh)*. Bone lengths are "
                        "fixed to the sequence median, and rotation about a bone's "
                        "own axis is not recoverable from joint positions, so "
                        "twist is zero.")
                if mode != "canonical":
                    note += (" Exported from the camera frame, so the body will "
                             "import arbitrarily oriented; the view-invariant "
                             "representation is the one BVH expects.")
                bvh_update = gr.update(value=bvh_path, visible=bvh_path is not None)
            else:
                out_path, t = predict_video(video, mode=mode)
                note, bvh_update = "", gr.update(visible=False)
            return (out_path,
                    f"**Status:** Completed\n**Inference time:** {t:.2f}s" + note,
                    bvh_update)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"**Status:** Error — {e}", gr.update(visible=False)

    vid_run_btn.click(
        fn=on_video_run,
        inputs=[vid_input, vid_coord, vid_bvh],
        outputs=[vid_output, vid_status, vid_bvh_file],
    )


if __name__ == "__main__":
    print("Loading models (first run may take a moment)...")
    from backend.model_loader import get_detector, get_model
    get_detector()
    get_model()
    print("Models loaded. Starting Gradio server...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft())
