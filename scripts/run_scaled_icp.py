"""
Re-run ICP object tracking on depth-scaled MoGe point cloud.

Loads existing object masks (from SAM2), scales scene_data pts3d by depth_scale,
then runs the same ICP pipeline as track_object.py. Saves results alongside original.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/run_scaled_icp.py \
        --output_dir outputs/demo_test/siheng_dual \
        --obj_mesh /home/sihengzhao/research/HDMI-Multi-datapipeline/assets/box_real.usd
"""

import argparse
import json
import numpy as np
import torch
import sys
from pathlib import Path

# Import ICP functions from track_object.py (reuse, don't copy)
sys.path.insert(0, str(Path(__file__).parent))
from track_object import (
    load_mesh,
    extract_object_pointclouds,
    estimate_scale_from_masks,
    prepare_model_pcd,
    run_icp_tracking,
    smooth_trajectory,
    CENTROID_CLIP_RADIUS_FACTOR,
)


def load_object_masks(mask_dir, total_frames):
    """Load pre-computed object masks from SAM2 npz files.

    Args:
        mask_dir: Path to object_mask_data/ directory
        total_frames: total number of frames

    Returns:
        dict {frame_idx: (H, W) bool array}
    """
    masks = {}
    for fidx in range(total_frames):
        mask_path = mask_dir / f"mask_{fidx:05d}.npz"
        if mask_path.exists():
            data = np.load(mask_path)
            # SAM2 masks are stored as 'mask' key
            mask = data[list(data.keys())[0]]
            masks[fidx] = mask.astype(bool)
    print(f"[ScaledICP] Loaded {len(masks)}/{total_frames} object masks")
    return masks


def main():
    parser = argparse.ArgumentParser(description="ICP tracking on depth-scaled MoGe")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--obj_mesh", type=str, required=True,
                        help="Path to object mesh (.obj, .ply, .usd)")
    parser.add_argument("--smooth_window", type=int, default=21)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load depth_scale ---
    scale_path = preprocess_dir / "depth_scale.json"
    assert scale_path.exists(), f"depth_scale.json not found. Run compute_depth_scale.py first."
    with open(scale_path) as f:
        scale_data = json.load(f)
    depth_scale = scale_data["depth_scale"]
    print(f"[ScaledICP] depth_scale = {depth_scale:.4f}")

    # --- Load scene_data ---
    scene_data = torch.load(preprocess_dir / "scene_data.pt", map_location="cpu")
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]
    total_frames = len(frame_names)
    print(f"[ScaledICP] {total_frames} frames")

    # --- Scale pts3d ---
    print(f"[ScaledICP] Scaling pts3d by {depth_scale:.4f} ...")
    for fn in frame_names:
        pts3d_dict[fn] = pts3d_dict[fn] * depth_scale
    # Update scene_data tuple (pts3d_dict is mutable, already modified)
    # normals don't change under uniform scaling (direction-only)

    # --- Load mesh ---
    mesh = load_mesh(args.obj_mesh)
    print(f"[ScaledICP] Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # --- Load object masks ---
    mask_dir = preprocess_dir / "masks" / "object_mask_data"
    assert mask_dir.exists(), f"Object masks not found: {mask_dir}"
    obj_masks = load_object_masks(mask_dir, total_frames)

    # --- Extract point clouds from scaled pts3d ---
    obj_pts, obj_norms = extract_object_pointclouds(scene_data, obj_masks)

    # --- Estimate obj_scale (now = GVHMR depth / real depth) ---
    mesh_verts = np.asarray(mesh.vertices)
    mesh_extents = mesh_verts.max(axis=0) - mesh_verts.min(axis=0)
    mesh_max_extent = float(mesh_extents.max())

    im_K_dict = scene_data[3]
    K = np.array(im_K_dict[frame_names[0]], dtype=np.float64)
    fx = K[0, 0]

    obj_scale = estimate_scale_from_masks(obj_masks, obj_pts, fx, mesh_max_extent)
    print(f"[ScaledICP] New obj_scale = {obj_scale:.4f} (GVHMR depth / real depth)")

    # --- Prepare model point cloud ---
    model_pcd = prepare_model_pcd(mesh, obj_scale, n_points=5000)

    # --- ICP tracking ---
    mesh_diagonal = float(np.linalg.norm(mesh_extents))
    clip_radius = CENTROID_CLIP_RADIUS_FACTOR * mesh_diagonal * obj_scale
    print(f"[ScaledICP] Clip radius: {clip_radius:.4f}m")

    import cv2
    video_path = str(out_dir / "0_input_video.mp4")
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    poses, fitness = run_icp_tracking(
        model_pcd, obj_pts, obj_norms, total_frames,
        obj_masks=obj_masks, mesh_verts=mesh_verts, K=K, scale=obj_scale,
        H=H, W=W, clip_radius=clip_radius)

    # --- Smooth ---
    pos, quat, lin_vel, ang_vel = smooth_trajectory(
        poses, fps, window=args.smooth_window)

    # --- Save ---
    save_path = preprocess_dir / "obj_poses_scaled.pt"
    result = {
        "obj_pos": pos,
        "obj_quat": quat,
        "obj_lin_vel": lin_vel,
        "obj_ang_vel": ang_vel,
        "obj_scale": obj_scale,
        "depth_scale": depth_scale,
        "obj_mesh_path": str(args.obj_mesh),
        "fitness": fitness,
        "fps": fps,
    }
    torch.save(result, save_path)
    print(f"\n[ScaledICP] Saved: {save_path}")
    print(f"  obj_scale: {obj_scale:.4f}")
    print(f"  depth_scale: {depth_scale:.4f}")
    print(f"  Mean fitness: {fitness[fitness > 0].mean():.3f}")
    print(f"  Frames tracked: {(fitness > 0).sum()}/{total_frames}")


if __name__ == "__main__":
    main()
