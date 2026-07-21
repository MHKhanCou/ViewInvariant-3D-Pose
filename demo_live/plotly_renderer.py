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
    [8, 11], [11, 12], [12, 13],  # Right arm
    [8, 14], [14, 15], [15, 16],  # Left arm
]

# Left/right color assignment per bone (matches H36M_LR).
# 0 = blue (left side of body), 1 = red (right side of body)
H36M_LR = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

# Colors
LEFT_COLOR = "#2563EB"   # Blue
RIGHT_COLOR = "#DC2626"  # Red
JOINT_COLOR = "#059669"  # Green
ROOT_COLOR = "#F59E0B"   # Amber (pelvis/root joint)


def render_pose_plotly(
    pose3d,
    title="3D Pose",
    show_grid=True,
    show_axes=True,
    height=600,
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

    # --- Draw bones as lines ---
    for idx, (i, j) in enumerate(H36M_SKELETON):
        color = RIGHT_COLOR if H36M_LR[idx] else LEFT_COLOR
        fig.add_trace(go.Scatter3d(
            x=[float(pose3d[i, 0]), float(pose3d[j, 0])],
            y=[float(pose3d[i, 1]), float(pose3d[j, 1])],
            z=[float(pose3d[i, 2]), float(pose3d[j, 2])],
            mode="lines",
            line=dict(width=4, color=color),
            hoverinfo="none",
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

    # Determine colors per joint
    joint_colors = []
    for name in joint_names:
        if "R_" in name:
            joint_colors.append(RIGHT_COLOR)
        elif "L_" in name:
            joint_colors.append(LEFT_COLOR)
        elif name in ("Pelvis", "Spine", "Thorax", "Neck", "Head"):
            joint_colors.append(JOINT_COLOR)
        else:
            joint_colors.append(JOINT_COLOR)

    fig.add_trace(go.Scatter3d(
        x=[float(pose3d[i, 0]) for i in range(17)],
        y=[float(pose3d[i, 1]) for i in range(17)],
        z=[float(pose3d[i, 2]) for i in range(17)],
        mode="markers+text",
        marker=dict(size=5, color=joint_colors),
        text=joint_names,
        textposition="top center",
        textfont=dict(size=9, color="#666666"),
        hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        showlegend=False,
    ))

    # --- Highlight root joint ---
    fig.add_trace(go.Scatter3d(
        x=[float(pose3d[0, 0])],
        y=[float(pose3d[0, 1])],
        z=[float(pose3d[0, 2])],
        mode="markers",
        marker=dict(size=8, color=ROOT_COLOR, symbol="diamond"),
        hovertemplate="Pelvis (root)<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        showlegend=False,
    ))

    # --- Compute axis limits (auto-fit with padding) ---
    pose_min = pose3d.min(axis=0)
    pose_max = pose3d.max(axis=0)
    pose_center = (pose_min + pose_max) / 2
    pose_extent = float(np.max(pose_max - pose_min))
    half_range = pose_extent * 0.65

    scene = dict(
        xaxis=dict(
            range=[pose_center[0] - half_range, pose_center[0] + half_range],
            showbackground=False,
            showgrid=show_grid,
            showticklabels=False,
            title="X" if show_axes else "",
            zeroline=False,
        ),
        yaxis=dict(
            range=[pose_center[1] - half_range, pose_center[1] + half_range],
            showbackground=False,
            showgrid=show_grid,
            showticklabels=False,
            title="Y" if show_axes else "",
            zeroline=False,
        ),
        zaxis=dict(
            range=[pose_center[2] - half_range, pose_center[2] + half_range],
            showbackground=False,
            showgrid=show_grid,
            showticklabels=False,
            title="Z" if show_axes else "",
            zeroline=False,
        ),
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
        scene_camera=dict(
            eye=dict(x=1.5, y=0.8, z=0.8),
            up=dict(x=0, y=0, z=1),
        ),
    )

    return fig
