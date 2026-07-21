"""
Batch diagnostic pipeline for Milestone 1 benchmarking.

For every image, saves all intermediate outputs:
  1. Input image (copy)
  2. Bounding box (from YOLO detection)
  3. 2D keypoints overlay (COCO-17)
  4. Converted H36M joints (npy)
  5. Normalized coordinates (npy)
  6. MotionAGFormer raw output (npy)
  7. Camera-coordinate 3D visualization (png)
  8. View-invariant 3D visualization (png)

Produces a summary CSV with per-image metrics.
"""

import os
import sys
import csv
import glob
import numpy as np
import cv2
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo_live.pose_detector import PoseDetector
from demo_live.lifter import coco_to_h36m, lift_sequence, N_FRAMES, normalize_screen_coordinates
from demo_live.visualize import draw_2d, render_3d, make_frame
from canonical.canonicalizer import Canonicalizer
from canonical.visualization import render_canonical_3d
from backend.model_loader import get_model
from demo_live.visualize import H36M_SKELETON


def compute_bone_lengths(pose):
    """Compute 16 bone lengths for the H36M-17 skeleton."""
    bones = []
    for i, j in H36M_SKELETON:
        bl = np.linalg.norm(pose[i] - pose[j])
        bones.append(bl)
    return np.array(bones)


def diagnose_image(image_path, output_dir, detector, model, canonicalizer, device="cpu"):
    """Run full diagnostic on a single image. Returns a results dict."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    frame = cv2.imread(image_path)
    if frame is None:
        return {"image": basename, "status": "ERROR: cannot read"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = frame.shape[:2]

    # --- Stage 1: YOLO Detection ---
    results = detector.model.predict(frame_rgb, conf=0.01, verbose=False)
    result = results[0]
    n_persons = 0
    if result.keypoints is not None and len(result.keypoints) > 0:
        n_persons = len(result.keypoints)
        xy = result.keypoints.xy
        conf = result.keypoints.conf
        # Pick best person by mean confidence
        mean_confs = conf.mean(dim=1)
        best = int(mean_confs.argmax())
        kpts_raw = xy[best].cpu().numpy().astype(np.float32)
        scores_raw = conf[best].cpu().numpy().astype(np.float32)
    else:
        kpts_raw = np.zeros((17, 2), dtype=np.float32)
        scores_raw = np.zeros((17,), dtype=np.float32)

    valid_kpts = int(np.sum(scores_raw > 0.3))
    mean_conf = float(scores_raw.mean())

    # Save bounding box from detection
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        best_box = boxes[0].astype(int)
        cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "02_bbox.jpg"), frame)

    # Save 2D overlay
    img_2d = draw_2d(kpts_raw, frame_rgb, scores=scores_raw)
    cv2.imwrite(os.path.join(output_dir, "03_2d_keypoints.jpg"),
                cv2.cvtColor(img_2d, cv2.COLOR_RGB2BGR))

    # --- Stage 2-3: H36M Conversion ---
    kpts_seq = np.repeat(kpts_raw[None], N_FRAMES, axis=0)
    scores_seq = np.repeat(scores_raw[None], N_FRAMES, axis=0)
    h36m = coco_to_h36m(kpts_seq, scores_seq)
    np.save(os.path.join(output_dir, "04_h36m_joints.npy"), h36m)

    # --- Stage 4: Normalization ---
    # Use processing resolution (downscale to MAX_HEIGHT if needed)
    MAX_HEIGHT = 480
    if H_orig > MAX_HEIGHT:
        proc_scale = MAX_HEIGHT / H_orig
        proc_H = MAX_HEIGHT
        proc_W = int(round(W_orig * proc_scale))
    else:
        proc_scale = 1.0
        proc_H = H_orig
        proc_W = W_orig

    # Re-run detection at processing resolution for consistent normalization
    if proc_scale < 1.0:
        small = cv2.resize(frame_rgb, (proc_W, proc_H))
        small_results = detector.model.predict(small, conf=0.01, verbose=False)
        small_result = small_results[0]
        if small_result.keypoints is not None and len(small_result.keypoints) > 0:
            xy = small_result.keypoints.xy
            cf = small_result.keypoints.conf
            best = int(cf.mean(dim=1).argmax())
            kpts_proc = xy[best].cpu().numpy().astype(np.float32)
            scores_proc = cf[best].cpu().numpy().astype(np.float32)
        else:
            kpts_proc = kpts_raw * proc_scale
            scores_proc = scores_raw

        kpts_seq = np.repeat(kpts_proc[None], N_FRAMES, axis=0)
        scores_seq = np.repeat(scores_proc[None], N_FRAMES, axis=0)
        h36m = coco_to_h36m(kpts_seq, scores_seq)

    norm = normalize_screen_coordinates(h36m, w=proc_W, h=proc_H)
    np.save(os.path.join(output_dir, "05_normalized.npy"), norm)

    # --- Stage 5: MotionAGFormer Prediction ---
    inp = torch.from_numpy(norm.astype("float32"))[None].to(device)
    with torch.no_grad():
        pred = model(inp).cpu().numpy()
    np.save(os.path.join(output_dir, "06_raw_prediction.npy"), pred)

    pose3d = pred[0, 13]  # Center frame

    # --- Stage 6a: Camera-coordinate visualization ---
    pose_camera = pose3d.copy()
    pose_camera[0] = 0  # Root zero
    from lib.utils import camera_to_world
    VIS_ROT = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    pose_camera = camera_to_world(pose_camera.astype(np.float64), R=VIS_ROT, t=0)
    pose_camera[:, 2] -= np.min(pose_camera[:, 2])
    max_val = np.max(pose_camera)
    if max_val > 0:
        pose_camera /= max_val

    img_3d_camera = render_3d(pose_camera)
    cv2.imwrite(os.path.join(output_dir, "07_camera_view.jpg"), img_3d_camera)
    np.save(os.path.join(output_dir, "07_camera_pose.npy"), pose_camera)

    # --- Stage 6b: View-invariant visualization ---
    pose_root = pose3d - pose3d[0:1]  # Root-relative
    try:
        pose_canonical, _ = canonicalizer(pose_root)
        img_3d_canonical = render_canonical_3d(pose_canonical)
        cv2.imwrite(os.path.join(output_dir, "08_view_invariant.jpg"), img_3d_canonical)
        np.save(os.path.join(output_dir, "08_view_invariant_pose.npy"), pose_canonical)
        can_status = "OK"
    except Exception as e:
        img_3d_canonical = np.full((540, 960, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(output_dir, "08_view_invariant.jpg"), img_3d_canonical)
        np.save(os.path.join(output_dir, "08_view_invariant_pose.npy"), np.zeros((17, 3)))
        can_status = f"FAIL: {e}"

    # --- Save combined output ---
    combined = make_frame(img_2d, img_3d_camera)
    cv2.imwrite(os.path.join(output_dir, "09_combined_output.jpg"), combined)

    # --- Save input copy ---
    cv2.imwrite(os.path.join(output_dir, "01_input.jpg"),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # --- Compute bone lengths ---
    bone_lengths_camera = compute_bone_lengths(pose_camera)
    bone_lengths_vi = compute_bone_lengths(pose_canonical) if can_status == "OK" else np.zeros(16)

    return {
        "image": basename,
        "size": f"{W_orig}x{H_orig}",
        "persons_detected": n_persons,
        "valid_keypoints": valid_kpts,
        "mean_confidence": f"{mean_conf:.4f}",
        "h36m_shape": str(h36m.shape),
        "pred_x_range": f"[{pose3d[:,0].min():.4f}, {pose3d[:,0].max():.4f}]",
        "pred_y_range": f"[{pose3d[:,1].min():.4f}, {pose3d[:,1].max():.4f}]",
        "pred_z_range": f"[{pose3d[:,2].min():.4f}, {pose3d[:,2].max():.4f}]",
        "bone_length_mean": f"{bone_lengths_camera.mean():.4f}",
        "bone_length_std": f"{bone_lengths_camera.std():.4f}",
        "canonical_status": can_status,
        "status": "PASS" if valid_kpts > 0 else "FAIL: no detection",
    }


def run_benchmark(benchmark_dir, output_root, device="cpu"):
    """Run diagnostic on all images in benchmark_dir."""
    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for pat in image_patterns:
        image_paths.extend(glob.glob(os.path.join(benchmark_dir, pat)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {benchmark_dir}")
        return

    print(f"Found {len(image_paths)} benchmark images")

    # Load models once
    print("Loading models...")
    detector = PoseDetector(weights="yolov8n-pose.pt", device=device, conf=0.4)
    model = get_model(device)
    canonicalizer = Canonicalizer()

    results = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img_output = os.path.join(output_root, basename)
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(img_path)}")
        print(f"{'='*60}")

        try:
            result = diagnose_image(img_path, img_output, detector, model, canonicalizer, device)
            results.append(result)
            print(f"  Status: {result['status']}")
            print(f"  Valid keypoints: {result['valid_keypoints']}/17")
            print(f"  Canonical: {result['canonical_status']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"image": basename, "status": f"ERROR: {e}"})

    # Write summary CSV
    csv_path = os.path.join(output_root, "benchmark_summary.csv")
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)
    print(f"Detection: {n_pass}/{n_total} images detected")
    for r in results:
        status_marker = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{status_marker}] {r['image']}: {r.get('valid_keypoints', '?')}/17 kp, "
              f"conf={r.get('mean_confidence', '?')}, canonical={r.get('canonical_status', '?')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default="thesis_artifacts/benchmark")
    parser.add_argument("--output", type=str, default="thesis_artifacts/benchmark/results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    benchmark_dir = os.path.join(REPO_ROOT, args.benchmark_dir)
    output_dir = os.path.join(REPO_ROOT, args.output)
    run_benchmark(benchmark_dir, output_dir, args.device)
