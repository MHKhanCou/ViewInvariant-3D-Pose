"""
MPI-INF-3DHP cross-view evaluation functions.

Provides utilities for loading MPI-INF-3DHP data, running inference
across cameras, and computing cross-view consistency metrics.
"""

import os
import numpy as np
from scipy.io import loadmat


def load_annot_mat(annot_path):
    """Load MPI-INF-3DHP annot.mat file.

    Returns:
        dict with keys: annot2, annot3, univ_annot3, cameras, frames
    """
    mat = loadmat(annot_path)
    return {
        'annot2': mat['annot2'],
        'annot3': mat['annot3'],
        'univ_annot3': mat['univ_annot3'],
        'cameras': mat['cameras'],
        'frames': mat['frames'],
    }


def load_camera_calibration(calib_path):
    """Parse camera.calibration file.

    Returns:
        list of dicts with 'id', 'intrinsic' (3x4), 'extrinsic' (4x4)
    """
    cameras = []
    current_cam = None
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('name'):
                current_cam = int(line.split()[-1])
            elif line.startswith('intrinsic'):
                vals = [float(x) for x in line.split()[1:]]
                intrinsic = np.array(vals).reshape(4, 4)[:3]
                cameras.append({'id': current_cam, 'intrinsic': intrinsic})
            elif line.startswith('extrinsic') and current_cam is not None:
                vals = [float(x) for x in line.split()[1:]]
                extrinsic = np.array(vals).reshape(4, 4)
                cameras[-1]['extrinsic'] = extrinsic
    return cameras


def get_annot3_for_frame(annot_data, cam_idx, frame_idx):
    """Get 3D joints for a specific camera and frame.

    Args:
        annot_data: loaded annot.mat dict
        cam_idx: camera index (0-13)
        frame_idx: frame index

    Returns:
        (28, 3) array of 3D joint positions in camera coordinates
    """
    annot3 = annot_data['annot3'][cam_idx][0]  # (6416, 84)
    joints = annot3[frame_idx].reshape(28, 3)
    return joints


def get_annot3_sequence(annot_data, cam_idx, n_frames=None):
    """Get 3D joints for all frames of a specific camera.

    Args:
        annot_data: loaded annot.mat dict
        cam_idx: camera index (0-13)
        n_frames: number of frames (None = all)

    Returns:
        (T, 28, 3) array of 3D joint positions
    """
    annot3 = annot_data['annot3'][cam_idx][0]  # (6416, 84)
    if n_frames is not None:
        annot3 = annot3[:n_frames]
    return annot3.reshape(-1, 28, 3)


def transform_to_world(joints_3d, extrinsic):
    """Transform camera-coordinate 3D joints to world coordinates.

    Args:
        joints_3d: (N, 28, 3) or (28, 3) joints in camera coordinates
        extrinsic: (4, 4) camera-to-world matrix [R|t; 0,1]

    Returns:
        joints_world: joints in world coordinates
    """
    joints_3d = np.asarray(joints_3d, dtype=np.float64)
    original_shape = joints_3d.shape
    flat = joints_3d.reshape(-1, 3)

    # Invert the extrinsic to go camera -> world
    # extrinsic is [R|t; 0,1] which transforms world -> camera
    # So inv(extrinsic) transforms camera -> world
    E_inv = np.linalg.inv(extrinsic)

    # Apply rotation (ignore translation for pose comparison)
    R_inv = E_inv[:3, :3]
    joints_world = (R_inv @ flat.T).T

    return joints_world.reshape(original_shape).astype(np.float32)


def select_cameras(cam_set=None):
    """Select cameras for cross-view evaluation.

    Uses the 'relevant' camera set (chest-high, non-ceiling cameras)
    as defined in the MPI-INF-3DHP documentation.
    """
    if cam_set is None:
        # Relevant cameras (excluding ceiling cameras 11-13)
        cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
    return cam_set


# Joint mapping: MPI-INF-3DHP 28 joints → H36M 17 joints
MPI_TO_H36M_JOINT_SET = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

H36M_JOINT_NAMES = [
    'pelvis', 'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
    'spine', 'upper_torso', 'neck', 'head',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist'
]

MPI_JOINT_NAMES = [
    'spine3', 'spine4', 'spine2', 'spine', 'pelvis',
    'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder',
    'left_elbow', 'left_wrist', 'left_hand', 'right_clavicle',
    'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe'
]


def select_h36m_joints(joints_28):
    """Select 17 H36M-compatible joints from 28 MPI-INF-3DHP joints.

    Args:
        joints_28: (T, 28, 3) or (28, 3)

    Returns:
        joints_17: (T, 17, 3) or (17, 3)
    """
    return joints_28[..., MPI_TO_H36M_JOINT_SET, :]
