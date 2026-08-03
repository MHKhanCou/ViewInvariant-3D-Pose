"""
Run full diagnostic pipeline on all example images.
Saves: input, 2D overlay, 3D render, bone lengths, reliability score.
Generates a summary report.
"""

import os
import sys
import time
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.model_loader import get_model, get_detector
from backend.inference import estimate_poses
from evaluation.reliability import compute_reliability_score, should_abstain, has_hard_geometric_failure
from demo_live.visualize import draw_2d
from demo_live.plotly_renderer import render_pose_plotly, get_bone_length_summary


def run_example(image_path, output_dir):
    """Run full pipeline on a single example image."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # Truncate basename for filesystem safety
    basename = basename[:80]

    frame = cv2.imread(image_path)
    if frame is None:
        print("  SKIP: Cannot read %s" % os.path.basename(image_path))
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    result = estimate_poses(frame_rgb)
    t1 = time.time()

    if result is None:
        print("  SKIP: No detection for %s" % os.path.basename(image_path))
        return None

    # Get reliability score on RAW root-relative pose (not display-transformed)
    raw_pose = result["raw_root_relative"]
    reliability, components, metadata = compute_reliability_score(
        raw_pose, result["scores"]
    )
    hard_failure, failure_reason = has_hard_geometric_failure(raw_pose)
    abstain = should_abstain(reliability, hard_failure)

    # Save images
    cv2.imwrite(os.path.join(output_dir, "input.jpg"),
                cv2.cvtColor(result["frame_rgb"], cv2.COLOR_RGB2BGR))

    overlay_bgr = draw_2d(result["kpts_2d"],
                           cv2.cvtColor(result["frame_rgb"], cv2.COLOR_RGB2BGR),
                           scores=result["scores"])
    cv2.imwrite(os.path.join(output_dir, "2d_overlay.jpg"),
                cv2.cvtColor(overlay_bgr, cv2.COLOR_RGB2BGR))

    # Save Plotly viewer as static image (skip if kaleido not available)
    try:
        display_pose = result["motionagformer_display_pose"]
        fig = render_pose_plotly(display_pose, title="3D Pose")
        fig.write_image(os.path.join(output_dir, "3d_viewer.png"), width=960, height=540)
    except (ImportError, ValueError, Exception):
        pass  # kaleido not installed, skip static export

    # Save bone length summary
    bone_summary = get_bone_length_summary(display_pose)
    with open(os.path.join(output_dir, "bone_lengths.md"), "w", encoding="utf-8") as f:
        f.write(bone_summary)

    # Save reliability score
    with open(os.path.join(output_dir, "reliability.txt"), "w", encoding="utf-8") as f:
        f.write("Reliability Score: %.4f\n" % reliability)
        f.write("Hard Failure: %s\n" % hard_failure)
        if failure_reason:
            f.write("Failure Reason: %s\n" % failure_reason)
        f.write("Abstain: %s\n" % abstain)
        f.write("Components:\n")
        for name, val in components.items():
            f.write("  %s: %.4f\n" % (name, val))
        f.write("Inference time: %.2fs\n" % (t1 - t0))
        f.write("Keypoints: %d/17 (conf=%.3f)\n" % (
            int(np.sum(result["scores"] > 0.3)),
            result["scores"].mean()
        ))

    return {
        "name": basename,
        "reliability": reliability,
        "hard_failure": hard_failure,
        "failure_reason": failure_reason,
        "abstain": abstain,
        "kp_valid": int(np.sum(result["scores"] > 0.3)),
        "conf": result["scores"].mean(),
        "time": t1 - t0,
        "components": components,
    }


def main():
    import glob
    example_dir = os.path.join(REPO_ROOT, "examples")
    output_dir = os.path.join(REPO_ROOT, "thesis_artifacts", "example_results")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join(example_dir, ext)))
    image_paths = sorted(set(image_paths))

    print("Loading models...")
    get_model()
    get_detector()
    print("Processing %d images..." % len(image_paths))

    results = []
    for i, img_path in enumerate(image_paths):
        print("\n[%d/%d] %s" % (i + 1, len(image_paths), os.path.basename(img_path)[:60]))
        basename = os.path.splitext(os.path.basename(img_path))[0][:80]
        result_dir = os.path.join(output_dir, basename)
        result = run_example(img_path, result_dir)
        if result:
            results.append(result)

    # Generate summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (%d images)" % len(results))
    print("=" * 70)
    print("%-50s %8s %7s %5s %7s %7s %6s" % (
        "Image", "Reliab", "Abstain", "Reason", "KP", "Conf", "Time"
    ))
    print("-" * 95)

    n_abstain = 0
    for r in results:
        abstain_mark = "YES" if r["abstain"] else ""
        reason_mark = r["failure_reason"] if r["hard_failure"] else ""
        if r["abstain"]:
            n_abstain += 1
        print("%-50s %.4f %7s %5s %5d/17 %.3f  %.2fs" % (
            r["name"][:50],
            r["reliability"],
            abstain_mark,
            reason_mark,
            r["kp_valid"],
            r["conf"],
            r["time"],
        ))

    print("-" * 95)
    reliabilities = [r["reliability"] for r in results]
    print("Mean reliability: %.4f" % np.mean(reliabilities))
    print("Min reliability:  %.4f" % np.min(reliabilities))
    print("Max reliability:  %.4f" % np.max(reliabilities))
    print("Abstain count: %d/%d (%.0f%%)" % (
        n_abstain, len(results), 100 * n_abstain / max(len(results), 1)
    ))

    # Save summary CSV
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    with open(csv_path, "w") as f:
        f.write("image,reliability,hard_failure,failure_reason,abstain,kp_valid,confidence,time\n")
        for r in results:
            f.write("%s,%.4f,%s,%s,%s,%d,%.4f,%.2f\n" % (
                r["name"], r["reliability"], r["hard_failure"],
                r["failure_reason"], r["abstain"],
                r["kp_valid"], r["conf"], r["time"]
            ))
    print("\nResults saved to: %s" % output_dir)


if __name__ == "__main__":
    main()
