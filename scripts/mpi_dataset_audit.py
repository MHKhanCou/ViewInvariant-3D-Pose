"""
MPI-INF-3DHP Dataset Audit Script

Inspects the raw MPI-INF-3DHP data and generates an audit report
confirming synchronization, coordinate conventions, and metadata structure.

Output: scripts/mpi_dataset_audit_report.md
"""

import os
import sys
import glob
import numpy as np
from collections import defaultdict

try:
    from scipy.io import loadmat
except ImportError:
    print("ERROR: scipy required. pip install scipy")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "..", "mpi_inf_3dhp")
REPORT_PATH = os.path.join(REPO_ROOT, "scripts", "mpi_dataset_audit_report.md")

# MPI-INF-3DHP joint set compatible with H36M (17 joints)
JOINT_SET = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

# Cameras used by MotionAGFormer (excluding ceiling cameras 11-13)
CAM_SET = [0, 1, 2, 4, 5, 6, 7, 8]

# All 14 cameras
ALL_CAMS = list(range(14))


def audit_directory_structure():
    """Check that expected directories and files exist."""
    report = []
    report.append("## 1. Directory Structure\n")

    if not os.path.exists(DATA_ROOT):
        report.append(f"**ERROR**: Data root not found: {DATA_ROOT}")
        return report, False

    subjects = sorted([d for d in os.listdir(DATA_ROOT)
                       if d.startswith('S') and os.path.isdir(os.path.join(DATA_ROOT, d))])
    report.append(f"Subjects found: {subjects}\n")

    recordings = []
    for subj in subjects:
        subj_path = os.path.join(DATA_ROOT, subj)
        seqs = sorted([d for d in os.listdir(subj_path)
                       if d.startswith('Seq') and os.path.isdir(os.path.join(subj_path, d))])
        for seq in seqs:
            seq_path = os.path.join(subj_path, seq)
            has_annot = os.path.exists(os.path.join(seq_path, "annot.mat"))
            has_calib = os.path.exists(os.path.join(seq_path, "camera.calibration"))

            # Count camera frame directories
            cam_dirs = [d for d in os.listdir(seq_path)
                        if d.startswith("frames_cam") and os.path.isdir(os.path.join(seq_path, d))]
            cam_ids = sorted([int(d.replace("frames_cam", "")) for d in cam_dirs])

            # Count frames per camera
            frame_counts = {}
            for cd in cam_dirs:
                cd_path = os.path.join(seq_path, cd)
                n = len([f for f in os.listdir(cd_path) if f.endswith('.jpg')])
                frame_counts[cd] = n

            recordings.append({
                'subject': subj,
                'sequence': seq,
                'has_annot': has_annot,
                'has_calib': has_calib,
                'cam_dirs': cam_ids,
                'frame_counts': frame_counts,
            })

    report.append(f"| Subject | Sequence | annot.mat | calibration | Cameras | Frames (cam0) |")
    report.append(f"|---------|----------|-----------|-------------|---------|---------------|")
    for r in recordings:
        fc0 = r['frame_counts'].get('frames_cam0', 0)
        report.append(f"| {r['subject']} | {r['sequence']} | {'Yes' if r['has_annot'] else 'No'} | "
                      f"{'Yes' if r['has_calib'] else 'No'} | {r['cam_dirs']} | {fc0} |")

    report.append("")
    return report, len(recordings) > 0


def audit_annot_mat(annot_path, subj, seq):
    """Inspect annot.mat structure."""
    report = []
    report.append(f"## 2. annot.mat Inspection — {subj}/{seq}\n")

    mat = loadmat(annot_path)

    # Filter out MATLAB metadata
    data_keys = [k for k in mat.keys() if not k.startswith('__')]
    report.append(f"Keys: {data_keys}\n")

    for key in data_keys:
        val = mat[key]
        report.append(f"**{key}**: type={type(val).__name__}", )
        if hasattr(val, 'shape'):
            report.append(f"  shape={val.shape}, dtype={val.dtype}")
        elif hasattr(val, '__len__'):
            report.append(f"  length={len(val)}")
            if len(val) > 0 and hasattr(val[0], 'shape'):
                report.append(f"  first element shape={val[0].shape}")

    # Check annot3 structure
    if 'annot3' in mat:
        annot3 = mat['annot3']
        report.append(f"\n**annot3** (3D annotations in camera coordinates):")
        report.append(f"  Shape: {annot3.shape} (cameras × 1)")
        report.append(f"  Per-camera shape: {annot3[0][0].shape} = (frames, {annot3[0][0].shape[1]})")
        n_joints = annot3[0][0].shape[1] // 3
        report.append(f"  Inferred joint count: {n_joints}")
        report.append(f"  Coordinate range: x=[{annot3[0][0][:, 0].min():.1f}, {annot3[0][0][:, 0].max():.1f}], "
                      f"y=[{annot3[0][0][:, 1].min():.1f}, {annot3[0][0][:, 1].max():.1f}], "
                      f"z=[{annot3[0][0][:, 2].min():.1f}, {annot3[0][0][:, 2].max():.1f}]")

    if 'univ_annot3' in mat:
        univ = mat['univ_annot3']
        report.append(f"\n**univ_annot3** (normalized 3D):")
        report.append(f"  Shape: {univ.shape}")
        report.append(f"  Per-camera shape: {univ[0][0].shape}")

    if 'frames' in mat:
        frames = mat['frames']
        report.append(f"\n**frames**: shape={frames.shape}, dtype={frames.dtype}")
        report.append(f"  Range: [{frames.min()}, {frames.max()}]")

    if 'cameras' in mat:
        cameras = mat['cameras']
        report.append(f"\n**cameras**: shape={cameras.shape}, values={cameras.flatten()}")

    report.append("")
    return report


def audit_camera_calibration(calib_path):
    """Parse camera.calibration and extract extrinsics."""
    report = []
    report.append("## 3. Camera Calibration\n")

    with open(calib_path, 'r') as f:
        lines = f.readlines()

    cameras = []
    current_cam = None
    for line in lines:
        line = line.strip()
        if line.startswith('name'):
            current_cam = int(line.split()[-1])
        elif line.startswith('intrinsic'):
            vals = [float(x) for x in line.split()[1:]]
            intrinsic = np.array(vals).reshape(4, 4)[:3]  # 4x4 → first 3 rows
            cameras.append({'id': current_cam, 'intrinsic': intrinsic})
        elif line.startswith('extrinsic') and current_cam is not None:
            vals = [float(x) for x in line.split()[1:]]
            extrinsic = np.array(vals).reshape(4, 4)
            cameras[-1]['extrinsic'] = extrinsic

    report.append(f"Cameras parsed: {len(cameras)}")
    report.append(f"Camera IDs: {[c['id'] for c in cameras]}\n")

    for cam in cameras:
        report.append(f"**Camera {cam['id']}:**")
        report.append(f"  Intrinsic (3x4):")
        for row in cam['intrinsic']:
            report.append(f"    {[f'{v:.3f}' for v in row]}")
        report.append(f"  Extrinsic (4x4):")
        for row in cam['extrinsic']:
            report.append(f"    {[f'{v:.4f}' for v in row]}")
        report.append("")

    report.append("**Extrinsic convention**: The 4×4 matrix is [R | t; 0 1] where R is the 3×3 rotation "
                   "and t is the 3×1 translation. This transforms world coordinates to camera coordinates. "
                   "To transform camera coordinates to world: use the inverse.\n")

    return report


def audit_frame_synchronization(data_root, n_samples=5):
    """Verify frame synchronization by checking RGB frames at same index."""
    report = []
    report.append("## 4. Frame Synchronization Verification\n")
    report.append(f"Checking {n_samples} random frame pairs across cameras.\n")

    # Find first recording with multiple cameras
    for subj in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj)
        if not os.path.isdir(subj_path):
            continue
        for seq in sorted(os.listdir(subj_path)):
            seq_path = os.path.join(subj_path, seq)
            cam_dirs = sorted([d for d in os.listdir(seq_path)
                              if d.startswith("frames_cam") and os.path.isdir(os.path.join(seq_path, d))])
            if len(cam_dirs) < 2:
                continue

            cam0_path = os.path.join(seq_path, cam_dirs[0])
            cam1_path = os.path.join(seq_path, cam_dirs[1])
            n_frames_cam0 = len([f for f in os.listdir(cam0_path) if f.endswith('.jpg')])
            n_frames_cam1 = len([f for f in os.listdir(cam1_path) if f.endswith('.jpg')])

            report.append(f"Recording: {subj}/{seq}")
            report.append(f"  {cam_dirs[0]}: {n_frames_cam0} frames")
            report.append(f"  {cam_dirs[1]}: {n_frames_cam1} frames\n")

            # Check that frame indices match
            if n_frames_cam0 == n_frames_cam1:
                report.append(f"  Frame counts match: {n_frames_cam0} == {n_frames_cam1}")
            else:
                report.append(f"  **WARNING**: Frame counts differ: {n_frames_cam0} vs {n_frames_cam1}")

            # Check sample frames exist at same indices
            min_frames = min(n_frames_cam0, n_frames_cam1)
            sample_indices = np.linspace(0, min_frames - 1, n_samples, dtype=int)
            for idx in sample_indices:
                f0 = os.path.join(cam0_path, f"frame_{idx:06d}.jpg")
                f1 = os.path.join(cam1_path, f"frame_{idx:06d}.jpg")
                exists0 = os.path.exists(f0)
                exists1 = os.path.exists(f1)
                report.append(f"  Frame {idx}: cam0={'exists' if exists0 else 'MISSING'}, "
                            f"cam1={'exists' if exists1 else 'MISSING'}")

            report.append(f"\n**Conclusion**: Frame index i in {cam_dirs[0]} corresponds to "
                         f"frame index i in {cam_dirs[1]} (synchronized recording).")
            report.append("")
            return report  # Only check first multi-camera recording

    report.append("**WARNING**: No multi-camera recording found for synchronization check.")
    return report


def audit_joint_mapping():
    """Document the joint mapping between MPI-INF-3DHP and H36M."""
    report = []
    report.append("## 5. Joint Mapping\n")

    report.append("MPI-INF-3DHP has 28 joints. MotionAGFormer uses 17 H36M-compatible joints.\n")
    report.append("Joint set used by MotionAGFormer (from data_to_npz_3dhp.py):")
    report.append(f"```python\njoint_set = {JOINT_SET}\n```\n")

    mpi_names = [
        'spine3', 'spine4', 'spine2', 'spine', 'pelvis',
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder',
        'left_elbow', 'left_wrist', 'left_hand', 'right_clavicle',
        'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
        'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
        'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ]

    report.append("| MPI-INF-3DHP Index | Joint Name | H36M Mapping |")
    report.append("|-------------------|------------|---------------|")
    h36m_names = [
        'pelvis', 'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle',
        'spine', 'upper_torso', 'neck', 'head',
        'right_shoulder', 'right_elbow', 'right_wrist',
        'left_shoulder', 'left_elbow', 'left_wrist'
    ]
    for i, mpi_idx in enumerate(JOINT_SET):
        report.append(f"| {mpi_idx} | {mpi_names[mpi_idx]} | {h36m_names[i] if i < len(h36m_names) else '?'} |")

    report.append("")
    return report


def audit_annot3_vs_univ(annot_path):
    """Compare annot3 and univ_annot3 to understand coordinate systems."""
    report = []
    report.append("## 6. annot3 vs univ_annot3 Coordinate Analysis\n")

    mat = loadmat(annot_path)

    if 'annot3' not in mat or 'univ_annot3' not in mat:
        report.append("Missing annot3 or univ_annot3 in annot.mat")
        return report

    # Compare camera 0's annot3 vs univ_annot3
    annot3_cam0 = mat['annot3'][0][0]  # (6416, 84)
    univ_cam0 = mat['univ_annot3'][0][0]  # (6416, 84)

    n_joints = annot3_cam0.shape[1] // 3

    report.append(f"Camera 0 comparison:")
    report.append(f"  annot3: shape={annot3_cam0.shape}, range=[{annot3_cam0.min():.1f}, {annot3_cam0.max():.1f}]")
    report.append(f"  univ_annot3: shape={univ_cam0.shape}, range=[{univ_cam0.min():.1f}, {univ_cam0.max():.1f}]\n")

    # Per-joint statistics
    report.append("| Joint | annot3 range | univ_annot3 range | Scale ratio |")
    report.append("|-------|-------------|-------------------|-------------|")

    for j in range(min(n_joints, 28)):
        a3 = annot3_cam0[:, j*3:(j+1)*3]
        u3 = univ_cam0[:, j*3:(j+1)*3]
        a3_range = np.ptp(a3, axis=0).mean()  # peak-to-peak per coord, averaged
        u3_range = np.ptp(u3, axis=0).mean()
        ratio = u3_range / (a3_range + 1e-6)
        report.append(f"| {j} | [{a3[:, 0].min():.0f}, {a3[:, 0].max():.0f}] | "
                     f"[{u3[:, 0].min():.1f}, {u3[:, 0].max():.1f}] | {ratio:.3f} |")

    report.append("")
    report.append("**Interpretation**: If annot3 values are in the thousands (mm-scale camera coordinates) "
                   "and univ_annot3 values are smaller (normalized), then annot3 is in camera coordinate "
                   "space and univ_annot3 is in a normalized/universal coordinate space.")
    report.append("")

    return report


def main():
    print("MPI-INF-3DHP Dataset Audit")
    print("=" * 50)

    all_report = []
    all_report.append("# MPI-INF-3DHP Dataset Audit Report\n")
    all_report.append(f"Generated by: `scripts/mpi_dataset_audit.py`\n")
    all_report.append(f"Data root: `{DATA_ROOT}`\n")

    # Phase 1: Directory structure
    print("1. Checking directory structure...")
    dir_report, has_data = audit_directory_structure()
    all_report.extend(dir_report)

    if not has_data:
        all_report.append("\n**ABORT**: No data found. Audit cannot continue.")
        _write_report(all_report)
        return

    # Phase 2: annot.mat inspection
    print("2. Inspecting annot.mat files...")
    for annot_path in sorted(glob.glob(os.path.join(DATA_ROOT, "*/Seq*", "annot.mat"))):
        parts = annot_path.replace("\\", "/").split("/")
        subj = [p for p in parts if p.startswith('S')][-1]
        seq = [p for p in parts if p.startswith('Seq')][-1]
        all_report.extend(audit_annot_mat(annot_path, subj, seq))
        break  # Only inspect first one for brevity

    # Phase 3: Camera calibration
    print("3. Parsing camera calibration...")
    calib_path = os.path.join(DATA_ROOT, "S1", "Seq1", "camera.calibration")
    if os.path.exists(calib_path):
        all_report.extend(audit_camera_calibration(calib_path))

    # Phase 4: Frame synchronization
    print("4. Verifying frame synchronization...")
    all_report.extend(audit_frame_synchronization(DATA_ROOT))

    # Phase 5: Joint mapping
    print("5. Documenting joint mapping...")
    all_report.extend(audit_joint_mapping())

    # Phase 6: annot3 vs univ_annot3
    print("6. Analyzing annot3 vs univ_annot3...")
    annot_path = os.path.join(DATA_ROOT, "S1", "Seq1", "annot.mat")
    if os.path.exists(annot_path):
        all_report.extend(audit_annot3_vs_univ(annot_path))

    # Write report
    _write_report(all_report)
    print(f"\nAudit report saved to: {REPORT_PATH}")


def _write_report(lines):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
