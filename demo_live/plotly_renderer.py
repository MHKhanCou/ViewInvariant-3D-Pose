"""
Interactive 3D skeleton renderer using Plotly.

Produces a WebGL-based 3D viewer with orbit, pan, zoom, and reset controls.
Single renderer for all coordinate spaces — only the pose data changes.
"""

import numpy as np
import plotly.graph_objects as go

# H36M-17 skeleton connections (16 bones).
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3],       # Right leg
    [0, 4], [4, 5], [5, 6],       # Left leg
    [0, 7], [7, 8],                # Spine + thorax
    [8, 9], [9, 10],               # Neck + head
    [8, 11], [11, 12], [12, 13],  # Left arm
    [8, 14], [14, 15], [15, 16],  # Right arm
]

# Left/right color assignment per bone.
H36M_LR = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

# Anatomical bone width (relative thickness) — thicker for torso/upper limbs,
# thinner for extremities. Makes the skeleton look more human.
BONE_WIDTHS = [
    6, 6, 5,          # Right leg:  thigh, shin, ankle
    6, 6, 5,          # Left leg:   thigh, shin, ankle
    7, 8,             # Spine, thorax
    6, 4,             # Neck, head
    5, 4, 3,          # Right arm: shoulder, elbow, wrist
    5, 4, 3,          # Left arm:  shoulder, elbow, wrist
]

# Joint sizes — larger for torso/core, smaller for extremities.
BONE_NAMES = [
    "R_Thigh", "R_Shin", "R_Ankle",
    "L_Thigh", "L_Shin", "L_Ankle",
    "Spine", "Thorax",
    "Neck", "Head",
    "R_UpperArm", "R_Forearm", "R_Hand",
    "L_UpperArm", "L_Forearm", "L_Hand",
]

# Colors — match the official demo (matplotlib 'b'/'r').
LEFT_COLOR = "#0000FF"
RIGHT_COLOR = "#FF0000"
JOINT_COLOR = "#333333"
ROOT_COLOR = "#F59E0B"

# Official demo bone order + left/right flags (demo_live/visualize.py).
OFFICIAL_I = [0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9]
OFFICIAL_J = [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]
OFFICIAL_LR = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
OFFICIAL_BONE_NAMES = [
    "Pelvis-L_Hip", "Pelvis-R_Hip", "L_Thigh", "R_Thigh", "L_Shin", "R_Shin",
    "Spine", "Thorax", "L_Shoulder", "R_Shoulder", "L_UpperArm", "L_Forearm",
    "R_UpperArm", "R_Forearm", "Neck", "Head",
]


def compute_bone_lengths(pose3d):
    """Compute 16 bone lengths for the H36M-17 skeleton."""
    lengths = np.array([
        np.linalg.norm(pose3d[i] - pose3d[j]) for i, j in H36M_SKELETON
    ])
    return lengths


def compute_limb_symmetry(pose3d):
    """Compute left-right limb length ratios."""
    symmetries = {}
    pairs = [
        ("Thigh", 1, 4),    # R_Hip, L_Hip
        ("Shin", 2, 5),      # R_Knee, L_Knee
        ("Ankle", 3, 6),     # R_Ankle, L_Ankle
        ("UpperArm", 11, 14), # R_Shoulder, L_Shoulder
        ("Forearm", 12, 15),  # R_Elbow, L_Elbow
        ("Hand", 13, 16),     # R_Wrist, L_Wrist
    ]
    for name, r_idx, l_idx in pairs:
        # Find bones connected to these joints
        r_bones = [i for i, (a, b) in enumerate(H36M_SKELETON) if a == r_idx or b == r_idx]
        l_bones = [i for i, (a, b) in enumerate(H36M_SKELETON) if a == l_idx or b == l_idx]
        if r_bones and l_bones:
            r_len = np.linalg.norm(pose3d[H36M_SKELETON[r_bones[0]][0]] - pose3d[H36M_SKELETON[r_bones[0]][1]])
            l_len = np.linalg.norm(pose3d[H36M_SKELETON[l_bones[0]][0]] - pose3d[H36M_SKELETON[l_bones[0]][1]])
            ratio = r_len / max(l_len, 1e-6)
            symmetries[name] = ratio
    return symmetries


def render_pose_plotly(
    pose3d,
    title="3D Pose",
    show_grid=True,
    show_axes=True,
    height=700,
):
    """
    Render an H36M-17 3D pose as an interactive Plotly figure.

    Args:
        pose3d: (17, 3) numpy array of 3D joint positions.
        title: plot title string.
        show_grid: whether to show the grid floor.
        show_axes: whether to show axis labels.
        height: figure height in pixels.

    Returns:
        A plotly.graph_objects.Figure with orbit/pan/zoom controls.
    """
    fig = go.Figure()

    # --- Draw bones: official demo style (thin uniform blue/red lines) ---
    for idx in range(len(OFFICIAL_I)):
        i, j = OFFICIAL_I[idx], OFFICIAL_J[idx]
        color = RIGHT_COLOR if OFFICIAL_LR[idx] else LEFT_COLOR
        bone_len = np.linalg.norm(pose3d[i] - pose3d[j])
        fig.add_trace(go.Scatter3d(
            x=[float(pose3d[i, 0]), float(pose3d[j, 0])],
            y=[float(pose3d[i, 1]), float(pose3d[j, 1])],
            z=[float(pose3d[i, 2]), float(pose3d[j, 2])],
            mode="lines",
            line=dict(width=4, color=color),
            hovertemplate=f"{OFFICIAL_BONE_NAMES[idx]}<br>Length: {bone_len:.3f}<extra></extra>",
            showlegend=False,
        ))

    # --- Draw joints as markers ---
    joint_names = [
        "Pelvis", "R_Hip", "R_Knee", "R_Ankle",
        "L_Hip", "L_Knee", "L_Ankle",
        "Spine", "Thorax", "Neck", "Head",
        "R_Shoulder", "R_Elbow", "R_Wrist",
        "L_Shoulder", "L_Elbow", "L_Wrist",
    ]

    # Small uniform markers — official style has no joint balls; keep tiny
    # ones purely as hover targets.
    joint_sizes = [3] * 17

    joint_colors = []
    for name in joint_names:
        if "R_" in name:
            joint_colors.append(RIGHT_COLOR)
        elif "L_" in name:
            joint_colors.append(LEFT_COLOR)
        else:
            joint_colors.append(JOINT_COLOR)

    fig.add_trace(go.Scatter3d(
        x=[float(pose3d[i, 0]) for i in range(17)],
        y=[float(pose3d[i, 1]) for i in range(17)],
        z=[float(pose3d[i, 2]) for i in range(17)],
        mode="markers",
        marker=dict(size=joint_sizes, color=joint_colors),
        hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        text=joint_names,
        showlegend=False,
    ))

    # --- Root joint: small diamond, subtle ---
    fig.add_trace(go.Scatter3d(
        x=[float(pose3d[0, 0])],
        y=[float(pose3d[0, 1])],
        z=[float(pose3d[0, 2])],
        mode="markers",
        marker=dict(size=5, color=ROOT_COLOR, symbol="diamond"),
        hovertemplate="Pelvis (root)<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        showlegend=False,
    ))

    # --- Auto-fit axis limits ---
    pose_min = pose3d.min(axis=0)
    pose_max = pose3d.max(axis=0)
    pose_center = (pose_min + pose_max) / 2
    pose_extent = float(np.max(pose_max - pose_min))
    half_range = pose_extent * 0.65

    # Official-demo-like axes: light gray grid on softly shaded panes.
    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgba(245,245,245,0.9)",
        gridcolor="#d9d9d9",
        showgrid=show_grid,
        showticklabels=False,
        zeroline=False,
    )
    scene = dict(
        xaxis=dict(range=[pose_center[0] - half_range, pose_center[0] + half_range],
                   title="X" if show_axes else "", **axis_style),
        yaxis=dict(range=[pose_center[1] - half_range, pose_center[1] + half_range],
                   title="Y" if show_axes else "", **axis_style),
        zaxis=dict(range=[pose_center[2] - half_range, pose_center[2] + half_range],
                   title="Z" if show_axes else "", **axis_style),
        aspectmode="data",
        bgcolor="rgba(255,255,255,1)",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=scene,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        showlegend=False,
        # Match the official demo viewpoint (elev=15, azim=70).
        scene_camera=dict(
            eye=dict(x=0.66, y=1.82, z=0.52),
            up=dict(x=0, y=0, z=1),
        ),
    )

    return fig


def get_bone_length_summary(pose3d):
    """Return a markdown-formatted summary of bone lengths and limb symmetry."""
    lengths = compute_bone_lengths(pose3d)
    sym = compute_limb_symmetry(pose3d)

    total_height = float(np.linalg.norm(pose3d[10] - pose3d[0]))  # Head to pelvis
    total_span = float(pose3d[:, 0].max() - pose3d[:, 0].min())  # Shoulder to shoulder approx

    lines = [
        "**Bone Lengths:**",
        "| Segment | Length |",
        "|---------|--------|",
    ]
    for idx, name in enumerate(BONE_NAMES):
        lines.append(f"| {name} | {lengths[idx]:.4f} |")

    lines.append("")
    lines.append(f"**Total height** (pelvis→head): {total_height:.4f}")
    lines.append(f"**Body span** (L→R): {total_span:.4f}")
    lines.append("")
    lines.append("**Left-Right Symmetry:**")
    for name, ratio in sym.items():
        diff = abs(ratio - 1.0) * 100
        status = "OK" if diff < 15 else "ASYMMETRIC"
        lines.append(f"- {name}: R/L = {ratio:.2f} ({diff:.1f}% diff) [{status}]")

    return "\n".join(lines)
