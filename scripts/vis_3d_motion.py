"""Interactive 3D visualization of retargeted robot motion + object.

Usage:
    python scripts/vis_3d_motion.py --output_dir output/siheng_dual_engineai_pm01 \
        --obj_mesh assets/box_real.usd [--frame 0]

Generates an interactive HTML file with:
  - Two robots as stick figures (colored links)
  - Object mesh as wireframe box
  - Frame slider
"""

import argparse
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ---------- Robot kinematic chain (parent → child) ----------
# body indices from meta.json body_names
LINKS = [
    # Left leg: BASE → HIP_PITCH_L → ... → ANKLE_ROLL_L
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    # Right leg
    (0, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    # Torso
    (0, 13),
    # Left arm
    (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # Right arm
    (13, 19), (19, 20), (20, 21), (21, 22), (22, 23),
    # Head
    (13, 24),
]


def load_box_vertices(mesh_path):
    """Load box mesh vertices in local frame."""
    ext = Path(mesh_path).suffix.lower()
    if ext in (".usd", ".usda", ".usdc"):
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(mesh_path))
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                usd_mesh = UsdGeom.Mesh(prim)
                pts = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
                xform = np.array(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    ),
                    dtype=np.float64,
                )
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                return (pts_h @ xform)[:, :3]
    else:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return np.asarray(mesh.vertices)


def quat_to_rot(q):
    """wxyz quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def compute_box_edges(verts_local):
    """Find box edges in local frame (before rotation)."""
    n = len(verts_local)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(verts_local[i] - verts_local[j])
            n_small = np.sum(diff < 0.01)
            if n_small >= 2:
                edges.append((i, j))
    return edges


def transform_box_vertices(verts_local, pos, quat):
    """Transform box vertices to world frame."""
    R = quat_to_rot(quat)
    return (R @ verts_local.T).T + pos


def make_robot_traces(pos, name, color, show_legend=True):
    """Create scatter + line traces for one robot at one frame."""
    traces = []

    # Joint positions
    traces.append(go.Scatter3d(
        x=pos[:25, 0], y=pos[:25, 1], z=pos[:25, 2],
        mode="markers",
        marker=dict(size=4, color=color),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    ))

    # Links
    link_x, link_y, link_z = [], [], []
    for p, c in LINKS:
        link_x.extend([pos[p, 0], pos[c, 0], None])
        link_y.extend([pos[p, 1], pos[c, 1], None])
        link_z.extend([pos[p, 2], pos[c, 2], None])

    traces.append(go.Scatter3d(
        x=link_x, y=link_y, z=link_z,
        mode="lines",
        line=dict(width=4, color=color),
        name=f"{name} links",
        legendgroup=name,
        showlegend=False,
    ))

    return traces


def make_box_trace(verts_world, edges, name="box", color="green", show_legend=True):
    """Create wireframe trace for box."""
    ex, ey, ez = [], [], []
    for i, j in edges:
        ex.extend([verts_world[i, 0], verts_world[j, 0], None])
        ey.extend([verts_world[i, 1], verts_world[j, 1], None])
        ez.extend([verts_world[i, 2], verts_world[j, 2], None])

    return go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=6, color=color),
        name=name,
        legendgroup="box",
        showlegend=show_legend,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--obj_mesh", default=None)
    parser.add_argument("--frame", type=int, default=None,
                        help="Single frame to show (default: slider over all frames)")
    parser.add_argument("--step", type=int, default=10,
                        help="Frame step for slider (default: 10)")
    parser.add_argument("--out_html", default=None,
                        help="Output HTML path (default: <output_dir>/vis_3d.html)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # Load data
    agents = []
    for agent_dir in sorted(out_dir.glob("agent_*")):
        data = np.load(agent_dir / "motion.npz", allow_pickle=True)
        with open(agent_dir / "meta.json") as f:
            meta = json.load(f)
        agents.append({"data": data, "meta": meta, "name": agent_dir.name})

    n_frames = agents[0]["data"]["body_pos_w"].shape[0]
    n_bodies = agents[0]["data"]["body_pos_w"].shape[1]
    has_object = n_bodies > 25  # box_real is body 25

    # Load box mesh vertices (local frame, for wireframe)
    box_verts_local = None
    box_edges = None
    if args.obj_mesh and has_object:
        box_verts_local = load_box_vertices(args.obj_mesh)
        box_edges = compute_box_edges(box_verts_local)
        extents = box_verts_local.max(axis=0) - box_verts_local.min(axis=0)
        print(f"Box mesh: {len(box_verts_local)} vertices, {len(box_edges)} edges, extents: {extents}")

    # Determine frames to visualize
    if args.frame is not None:
        frames_to_show = [args.frame]
    else:
        frames_to_show = list(range(0, n_frames, args.step))

    colors = ["royalblue", "orangered", "green", "purple"]

    # Build figure with slider
    fig = go.Figure()

    # Create frames for animation
    slider_steps = []
    all_frames = []

    for fi, frame_idx in enumerate(frames_to_show):
        frame_traces = []

        for ai, agent in enumerate(agents):
            pos = agent["data"]["body_pos_w"][frame_idx]
            robot_traces = make_robot_traces(
                pos, agent["name"], colors[ai % len(colors)],
                show_legend=(fi == 0),
            )
            frame_traces.extend(robot_traces)

        # Box wireframe (from agent_0, same for all)
        if box_verts_local is not None and box_edges is not None and has_object:
            box_pos = agents[0]["data"]["body_pos_w"][frame_idx, 25]
            box_quat = agents[0]["data"]["body_quat_w"][frame_idx, 25]
            verts_w = transform_box_vertices(box_verts_local, box_pos, box_quat)
            box_trace = make_box_trace(verts_w, box_edges, show_legend=(fi == 0))
            frame_traces.append(box_trace)

            # Also add box vertices as markers
            frame_traces.append(go.Scatter3d(
                x=verts_w[:, 0], y=verts_w[:, 1], z=verts_w[:, 2],
                mode="markers",
                marker=dict(size=3, color="green"),
                name="box vertices",
                legendgroup="box",
                showlegend=False,
            ))

        all_frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))
        slider_steps.append(dict(
            args=[[str(frame_idx)], dict(frame=dict(duration=0, redraw=True),
                                         mode="immediate")],
            label=str(frame_idx),
            method="animate",
        ))

    # Add initial frame traces to figure
    if all_frames:
        for trace in all_frames[0].data:
            fig.add_trace(trace)

    # Add ground plane
    ground_size = 3
    cx = np.mean([agents[0]["data"]["body_pos_w"][:, 0, 0].mean(),
                   agents[1]["data"]["body_pos_w"][:, 0, 0].mean()])
    cy = np.mean([agents[0]["data"]["body_pos_w"][:, 0, 1].mean(),
                   agents[1]["data"]["body_pos_w"][:, 0, 1].mean()])
    fig.add_trace(go.Mesh3d(
        x=[cx - ground_size, cx + ground_size, cx + ground_size, cx - ground_size],
        y=[cy - ground_size, cy - ground_size, cy + ground_size, cy + ground_size],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="lightgray", opacity=0.3,
        name="ground",
        showlegend=False,
    ))

    # Layout
    fig.frames = all_frames
    fig.update_layout(
        title=f"3D Motion Visualization — {out_dir.name}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0, y=-2.5, z=1.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        width=1200, height=800,
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Frame: "),
            steps=slider_steps,
        )] if len(frames_to_show) > 1 else [],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                        mode="immediate")]),
            ],
        )] if len(frames_to_show) > 1 else [],
    )

    out_html = args.out_html or str(out_dir / "vis_3d.html")
    fig.write_html(out_html)
    print(f"\nSaved interactive 3D visualization to: {out_html}")
    print(f"Open in browser to view.")


if __name__ == "__main__":
    main()
