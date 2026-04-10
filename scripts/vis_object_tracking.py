"""
Visualize object 6DoF tracking results by projecting the tracked mesh onto video frames.

Usage:
    cd /home/sihengzhao/research/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/vis_object_tracking.py \
        --output_dir outputs/demo_test/siheng_dual \
        --obj_mesh /path/to/box.usd
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path


def project_mesh_to_frame(mesh_verts, T_obj2world, K, c2w, scale):
    """Project scaled mesh vertices through object pose → world → camera → image.

    Args:
        mesh_verts: (V, 3) mesh vertices in mesh local frame
        T_obj2world: (4, 4) object-to-world transformation from ICP
        K: (3, 3) camera intrinsics
        c2w: (4, 4) camera-to-world
        scale: float, mesh scale factor

    Returns:
        pts_2d: (V, 2) projected pixel coordinates
        depths: (V,) depth values (for visibility check)
    """
    # Scale mesh vertices
    verts_scaled = mesh_verts * scale

    # Apply object pose (ICP transformation: model → world)
    verts_h = np.hstack([verts_scaled, np.ones((len(verts_scaled), 1))])
    verts_world = (T_obj2world @ verts_h.T).T[:, :3]  # (V, 3)

    # World → camera
    w2c = np.linalg.inv(c2w)
    verts_cam_h = np.hstack([verts_world, np.ones((len(verts_world), 1))])
    verts_cam = (w2c @ verts_cam_h.T).T[:, :3]  # (V, 3)

    # Camera → image
    depths = verts_cam[:, 2]
    pts_2d = (K @ verts_cam.T).T  # (V, 3)
    pts_2d = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)

    return pts_2d, depths


def draw_mesh_wireframe(frame, pts_2d, triangles, depths, color=(0, 255, 0), alpha=0.5):
    """Draw mesh wireframe on frame.

    Only draws edges where both vertices are in front of camera.
    """
    overlay = frame.copy()
    H, W = frame.shape[:2]

    # Collect unique edges from triangles
    edges = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edges.add(e)

    for i, j in edges:
        if depths[i] <= 0 or depths[j] <= 0:
            continue
        p1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
        p2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
        # Clip to image bounds (with margin)
        if (p1[0] < -W or p1[0] > 2 * W or p1[1] < -H or p1[1] > 2 * H):
            continue
        if (p2[0] < -W or p2[0] > 2 * W or p2[1] < -H or p2[1] > 2 * H):
            continue
        cv2.line(overlay, p1, p2, color, 2)

    # Draw vertices as small circles
    for k in range(len(pts_2d)):
        if depths[k] <= 0:
            continue
        p = (int(pts_2d[k, 0]), int(pts_2d[k, 1]))
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.circle(overlay, p, 3, (0, 0, 255), -1)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def main():
    parser = argparse.ArgumentParser(description="Visualize object 6DoF tracking")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="GVHMR output dir (contains scene_data.pt, obj_poses.pt)")
    parser.add_argument("--obj_mesh", type=str, required=True,
                        help="Path to object mesh (.obj, .ply, .stl, or .usd)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to render (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # Load obj_poses.pt
    pose_path = out_dir / "preprocess" / "obj_poses_scaled.pt"
    assert pose_path.exists(), f"obj_poses.pt not found: {pose_path}. Run track_object.py first."
    pose_data = torch.load(pose_path, map_location='cpu')

    obj_pos = np.array(pose_data["obj_pos"])       # (T, 3)
    obj_quat = np.array(pose_data["obj_quat"])     # (T, 4) xyzw
    obj_scale = float(pose_data["obj_scale"])
    fitness = np.array(pose_data["fitness"])         # (T,)
    fps = float(pose_data.get("fps", 30.0))

    T = len(obj_pos)
    print(f"[VisObj] {T} frames, scale={obj_scale:.4f}, fps={fps}")

    # Reconstruct 4x4 poses from pos + quat
    from scipy.spatial.transform import Rotation
    poses = np.zeros((T, 4, 4))
    for i in range(T):
        poses[i, :3, :3] = Rotation.from_quat(obj_quat[i]).as_matrix()
        poses[i, :3, 3] = obj_pos[i]
        poses[i, 3, 3] = 1.0

    # Load mesh
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from track_object import load_mesh
    mesh = load_mesh(args.obj_mesh)
    mesh_verts = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print(f"[VisObj] Mesh: {len(mesh_verts)} vertices, {len(triangles)} triangles")

    # Load scene_data for camera K and c2w
    scene_path = out_dir / "preprocess" / "scene_data.pt"
    assert scene_path.exists(), f"scene_data.pt not found"
    scene_data = torch.load(scene_path, map_location='cpu')
    frame_names = scene_data[0]
    im_K_dict = scene_data[3]
    im_poses_dict = scene_data[4]

    # Load video
    from hmr4d.utils.video_io_utils import read_video_np
    video_path = str(out_dir / "0_input_video.mp4")
    frames = read_video_np(video_path)
    F, H, W, _ = frames.shape
    print(f"[VisObj] Video: {F} frames, {W}x{H}")

    n_render = min(F, T, len(frame_names))
    if args.max_frames > 0:
        n_render = min(n_render, args.max_frames)

    # Output video
    vis_path = str(out_dir / "vis_object_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vis_path, fourcc, fps, (W, H))

    for fidx in range(n_render):
        frame_bgr = cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR)
        fn = frame_names[fidx]

        K = np.array(im_K_dict[fn], dtype=np.float64)
        c2w = np.array(im_poses_dict[fn], dtype=np.float64)

        if fitness[fidx] > 0:
            pts_2d, depths = project_mesh_to_frame(
                mesh_verts, poses[fidx], K, c2w, obj_scale)
            frame_bgr = draw_mesh_wireframe(frame_bgr, pts_2d, triangles, depths)

        # Draw info text
        fit_str = f"fit={fitness[fidx]:.2f}" if fitness[fidx] > 0 else "NO TRACK"
        cv2.putText(frame_bgr, f"Frame {fidx} {fit_str} scale={obj_scale:.4f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame_bgr)

        if fidx % 50 == 0:
            print(f"  Rendered {fidx}/{n_render}")

    writer.release()
    print(f"[VisObj] Saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
