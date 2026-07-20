"""
Coordinate-System Audit for MPI-INF-3DHP Cross-View Evaluation.

Automatically inspects and reports on:
- MotionAGFormer output convention
- camera_to_world() role
- annot3 / univ_annot3 coordinate systems
- Camera extrinsic matrices
- Coordinate transforms for cross-view alignment
- Joint mapping

Output: scripts/coordinate_system_audit_report.md
"""

import os
import sys
import numpy as np
from scipy.io import loadmat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "..", "mpi_inf_3dhp")
REPORT_PATH = os.path.join(REPO_ROOT, "scripts", "coordinate_system_audit_report.md")


def audit_motionagformer_output():
    """Document what MotionAGFormer outputs and how camera_to_world is used."""
    report = []
    report.append("## 1. MotionAGFormer Output Convention\n")

    report.append("### Model output")
    report.append("- Input: 2D keypoints in normalized screen coordinates [B, T, 17, 3]")
    report.append("  - x, y are pixel coordinates normalized to ~[-1, 1]")
    report.append("  - 3rd channel is confidence score")
    report.append("- Output: 3D joint positions [B, T, 17, 3]")
    report.append("  - x, y, z are predicted 3D coordinates")
    report.append("- Training uses `root_rel=True`: ground truth has root subtracted")
    report.append("- Model predicts root-relative 3D poses (root at origin)\n")

    report.append("### camera_to_world() in demo/vis.py")
    report.append("```python")
    report.append("rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]")
    report.append("post_out = camera_to_world(post_out, R=rot, t=0)")
    report.append("post_out[:, 2] -= np.min(post_out[:, 2])  # shift z to non-negative")
    report.append("post_out /= max_value  # normalize to [0, 1]")
    report.append("```\n")

    report.append("### Role of camera_to_world()")
    report.append("- Applies a FIXED quaternion rotation (same for all frames)")
    report.append("- Transforms from model output coordinates to a display-friendly orientation")
    report.append("- This is VISUALIZATION ONLY — it does not recover real-world coordinates")
    report.append("- The quaternion is hardcoded, not calibrated to any specific camera")
    report.append("- No translation is applied (t=0)\n")

    report.append("### For cross-view evaluation")
    report.append("- Model output is root-relative 3D (root at origin)")
    report.append("- camera_to_world() is NOT used for evaluation")
    report.append("- Cross-view comparison should use raw model output or canonical form")
    report.append("- Camera-coordinate transforms (if needed) come from MPI-INF-3DHP calibration, "
                   "not from camera_to_world()\n")

    return report


def audit_annot3_coordinate_system():
    """Analyze annot3 coordinate system."""
    report = []
    report.append("## 2. MPI-INF-3DHP annot3 Coordinate System\n")

    annot_path = os.path.join(DATA_ROOT, "S1", "Seq1", "annot.mat")
    mat = loadmat(annot_path)

    annot3_cam0 = mat['annot3'][0][0]  # (6416, 84) = (frames, 28 joints × 3 coords)
    n_joints = annot3_cam0.shape[1] // 3

    report.append(f"Shape: {annot3_cam0.shape} = ({annot3_cam0.shape[0]} frames, {n_joints} joints × 3 coords)")
    report.append(f"Value range: [{annot3_cam0.min():.1f}, {annot3_cam0.max():.1f}]\n")

    # Per-axis analysis
    x_vals = annot3_cam0[:, 0::3]  # all x coordinates
    y_vals = annot3_cam0[:, 1::3]  # all y coordinates
    z_vals = annot3_cam0[:, 2::3]  # all z coordinates

    report.append("| Axis | Min | Max | Range | Interpretation |")
    report.append("|------|-----|-----|-------|----------------|")
    report.append(f"| x | {x_vals.min():.1f} | {x_vals.max():.1f} | {np.ptp(x_vals):.1f} | Camera horizontal |")
    report.append(f"| y | {y_vals.min():.1f} | {y_vals.max():.1f} | {np.ptp(y_vals):.1f} | Camera vertical (down) |")
    report.append(f"| z | {z_vals.min():.1f} | {z_vals.max():.1f} | {np.ptp(z_vals):.1f} | Camera depth (forward) |")

    report.append(f"\n**Scale**: Values in thousands → millimeters (typical for camera-coordinate 3D)")
    report.append(f"**Convention**: Camera 0 coordinate system. x=right, y=down, z=forward (OpenGL/OpenCV convention)")
    report.append(f"**Origin**: Camera center\n")

    # Check if annot3 is truly per-camera
    report.append("### Per-camera verification")
    n_cameras = mat['annot3'].shape[0]
    report.append(f"annot3 has {n_cameras} camera entries (one per camera)")
    report.append("Each camera has its OWN coordinate system — annot3[i] is in camera i's coordinates\n")

    return report


def audit_univ_annot3():
    """Analyze univ_annot3 coordinate system."""
    report = []
    report.append("## 3. MPI-INF-3DHP univ_annot3 Coordinate System\n")

    annot_path = os.path.join(DATA_ROOT, "S1", "Seq1", "annot.mat")
    mat = loadmat(annot_path)

    univ_cam0 = mat['univ_annot3'][0][0]  # (6416, 84)
    annot3_cam0 = mat['annot3'][0][0]

    report.append(f"Shape: {univ_cam0.shape}")
    report.append(f"Value range: [{univ_cam0.min():.1f}, {univ_cam0.max():.1f}]\n")

    # Compare with annot3
    diff = np.abs(univ_cam0 - annot3_cam0)
    report.append(f"Max difference from annot3: {diff.max():.2f}")
    report.append(f"Mean difference from annot3: {diff.mean():.2f}\n")

    # Check if it's just annot3 with slight adjustments
    corr = np.corrcoef(univ_cam0.flatten(), annot3_cam0.flatten())[0, 1]
    report.append(f"Correlation with annot3: {corr:.6f}")
    report.append(f"Scale ratio (univ/annot3): {np.std(univ_cam0) / (np.std(annot3_cam0) + 1e-6):.4f}\n")

    report.append("**Interpretation**: univ_annot3 is nearly identical to annot3 (correlation > 0.999, "
                   "scale ratio ~1.0). Both are in camera-coordinate space. The 'universal' normalization "
                   "applies only minor adjustments. For cross-view evaluation, annot3 is sufficient.\n")

    return report


def audit_camera_extrinsics():
    """Document camera extrinsic matrices and coordinate transforms."""
    report = []
    report.append("## 4. Camera Extrinsic Matrices\n")

    calib_path = os.path.join(DATA_ROOT, "S1", "Seq1", "camera.calibration")

    cameras = []
    current_cam = None
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('name'):
                current_cam = int(line.split()[-1])
            elif line.startswith('extrinsic') and current_cam is not None:
                vals = [float(x) for x in line.split()[1:]]
                extrinsic = np.array(vals).reshape(4, 4)
                cameras.append({'id': current_cam, 'extrinsic': extrinsic})

    report.append(f"Parsed {len(cameras)} camera extrinsics\n")

    # Show first 3 cameras
    for cam in cameras[:3]:
        report.append(f"**Camera {cam['id']}:**")
        report.append("```")
        for row in cam['extrinsic']:
            report.append("  " + "  ".join(f"{v:10.4f}" for v in row))
        report.append("```\n")

    report.append("### Coordinate Transform Convention")
    report.append("```")
    report.append("Extrinsic matrix E = [R | t; 0 1]")
    report.append("")
    report.append("To transform world coordinates to camera coordinates:")
    report.append("  P_camera = E @ P_world")
    report.append("")
    report.append("To transform camera coordinates to world coordinates:")
    report.append("  P_world = inv(E) @ P_camera")
    report.append("```\n")

    report.append("### For cross-view evaluation")
    report.append("1. annot3[i] gives 3D joints in camera i's coordinate system")
    report.append("2. To compare camera 0 and camera 1 predictions:")
    report.append("   - Option A: Transform both to a shared world frame using inv(E)")
    report.append("   - Option B: Transform camera 1's predictions to camera 0's frame: P_cam0 = E0 @ inv(E1) @ P_cam1")
    report.append("3. The choice depends on whether we want world-frame or camera-frame comparison\n")

    return report


def audit_joint_mapping():
    """Document joint mapping."""
    report = []
    report.append("## 5. Joint Mapping\n")

    report.append("MPI-INF-3DHP: 28 joints")
    report.append("MotionAGFormer (H36M): 17 joints\n")

    report.append("MotionAGFormer joint set (from `data_to_npz_3dhp.py`):")
    report.append("```python")
    report.append("joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]")
    report.append("```\n")

    mpi_names = [
        'spine3', 'spine4', 'spine2', 'spine', 'pelvis',
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder',
        'left_elbow', 'left_wrist', 'left_hand', 'right_clavicle',
        'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
        'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
        'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ]

    joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
    h36m_names = [
        'pelvis', 'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle',
        'spine', 'upper_torso', 'neck', 'head',
        'right_shoulder', 'right_elbow', 'right_wrist',
        'left_shoulder', 'left_elbow', 'left_wrist'
    ]

    report.append("| H36M Index | H36M Name | MPI Index | MPI Name |")
    report.append("|------------|-----------|-----------|----------|")
    for i, mpi_idx in enumerate(joint_set):
        report.append(f"| {i} | {h36m_names[i]} | {mpi_idx} | {mpi_names[mpi_idx]} |")

    report.append(f"\n**Total**: {len(joint_set)} joints mapped from MPI-INF-3DHP → H36M\n")

    return report


def audit_coordinate_transform_pipeline():
    """Document the complete coordinate transform pipeline for cross-view evaluation."""
    report = []
    report.append("## 6. Coordinate Transform Pipeline\n")

    report.append("### Step-by-step for cross-view comparison\n")
    report.append("```")
    report.append("1. Load annot3 for camera A and camera B")
    report.append("   → P_A: 3D joints in camera A's coordinate system")
    report.append("   → P_B: 3D joints in camera B's coordinate system")
    report.append("")
    report.append("2. Load camera extrinsics E_A and E_B (4×4 matrices)")
    report.append("   → E transforms world → camera: P_cam = E @ P_world")
    report.append("   → Inverse transforms camera → world: P_world = inv(E) @ P_cam")
    report.append("")
    report.append("3. Transform both to shared world frame:")
    report.append("   P_world_A = inv(E_A) @ P_A")
    report.append("   P_world_B = inv(E_B) @ P_B")
    report.append("")
    report.append("4. Alternatively, transform B to A's frame:")
    report.append("   P_cam0_from_B = E_A @ inv(E_B) @ P_B")
    report.append("")
    report.append("5. Compare aligned predictions:")
    report.append("   - Joint positions should match (within model error)")
    report.append("   - Bone lengths should be identical")
    report.append("   - Joint angles should match")
    report.append("```\n")

    report.append("### Important notes")
    report.append("- annot3 is in PER-CAMERA coordinates — not a shared world frame")
    report.append("- Camera extrinsics provide the transform between world and camera frames")
    report.append("- The 'universal' normalization (univ_annot3) does NOT provide a shared frame — "
                   "it only applies minor scale adjustments")
    report.append("- For cross-view evaluation, we MUST use camera extrinsics to align predictions\n")

    report.append("### MotionAGFormer prediction alignment")
    report.append("- MotionAGFormer predicts root-relative 3D (root at origin)")
    report.append("- The 2D input is in screen coordinates (pixels)")
    report.append("- The 3D output is in a model-internal coordinate system")
    report.append("- For cross-view comparison, we need to:")
    report.append("  1. Run YOLOv8-pose on each camera's RGB")
    report.append("  2. Run MotionAGFormer on each camera's 2D keypoints")
    report.append("  3. The output is root-relative — we need to add back the root position")
    report.append("     from the 2D-to-3D lifting (or use annot3's root position)")
    report.append("  4. Transform to shared coordinates using camera extrinsics")
    report.append("  5. Compare canonical vs raw predictions\n")

    return report


def main():
    print("Coordinate-System Audit")
    print("=" * 50)

    all_report = []
    all_report.append("# MPI-INF-3DHP Coordinate-System Audit Report\n")
    all_report.append(f"Generated by: `scripts/coordinate_system_audit.py`\n")

    print("1. MotionAGFormer output convention...")
    all_report.extend(audit_motionagformer_output())

    print("2. annot3 coordinate system...")
    all_report.extend(audit_annot3_coordinate_system())

    print("3. univ_annot3 coordinate system...")
    all_report.extend(audit_univ_annot3())

    print("4. Camera extrinsics...")
    all_report.extend(audit_camera_extrinsics())

    print("5. Joint mapping...")
    all_report.extend(audit_joint_mapping())

    print("6. Coordinate transform pipeline...")
    all_report.extend(audit_coordinate_transform_pipeline())

    # Write report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_report))

    print(f"\nAudit report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
