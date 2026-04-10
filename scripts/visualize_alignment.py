"""
Visualize alignment results: BEV trajectory, Z time series, 3D render.

Compares before (aligned_results + obj_poses) vs after (aligned_results_xz + obj_poses_scaled).

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/visualize_alignment.py \
        --output_dir outputs/demo_test/siheng_dual
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_data(out_dir):
    """Load before/after human + object trajectories."""
    preprocess_dir = out_dir / "preprocess"
    G_old = torch.load(preprocess_dir / "G_refined.pt", map_location="cpu").numpy()

    # Use G_refined_scaled if available (from scaled grounding), else original
    g_scaled_path = preprocess_dir / "G_refined_scaled.pt"
    G_new = torch.load(g_scaled_path, map_location="cpu").numpy() if g_scaled_path.exists() else G_old

    # "Before" = original aligned_results (unscaled grounding)
    ar_old = torch.load(out_dir / "aligned_results.pt", map_location="cpu")
    # "After" = XZ-corrected on top of scaled grounding
    ar_new = torch.load(out_dir / "aligned_results_xz.pt", map_location="cpu")

    obj_old = torch.load(preprocess_dir / "obj_poses.pt", map_location="cpu")
    obj_new = torch.load(preprocess_dir / "obj_poses_scaled.pt", map_location="cpu")

    # Object: camera frame → gravity frame (each with its own G)
    obj_old_grav = (G_old @ obj_old["obj_pos"].T).T
    obj_new_grav = (G_new @ obj_new["obj_pos"].T).T

    pids = sorted(ar_old.keys())
    humans_old = {pid: ar_old[pid]["smpl_params_global"]["transl"].numpy() for pid in pids}
    humans_new = {pid: ar_new[pid]["smpl_params_global"]["transl"].numpy() for pid in pids}

    return humans_old, humans_new, obj_old_grav, obj_new_grav, pids, ar_new, G_new


def plot_bev_and_z(humans_old, humans_new, obj_old, obj_new, pids, save_dir):
    """Generate BEV XZ plot and Z time series."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- BEV XZ: Before ---
    ax = axes[0, 0]
    for pid in pids:
        t = humans_old[pid]
        ax.plot(t[:, 0], t[:, 2], label=f"Human {pid}", linewidth=1.5)
    ax.plot(obj_old[:, 0], obj_old[:, 2], "k--", label="Box", linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("BEFORE: BEV XZ trajectory")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- BEV XZ: After ---
    ax = axes[0, 1]
    for pid in pids:
        t = humans_new[pid]
        ax.plot(t[:, 0], t[:, 2], label=f"Human {pid}", linewidth=1.5)
    ax.plot(obj_new[:, 0], obj_new[:, 2], "k--", label="Box", linewidth=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("AFTER: BEV XZ trajectory")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Z time series: Before ---
    ax = axes[1, 0]
    frames = np.arange(len(obj_old))
    for pid in pids:
        ax.plot(frames, humans_old[pid][:, 2], label=f"Human {pid}", linewidth=1.5)
    ax.plot(frames, obj_old[:, 2], "k--", label="Box", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z (m)")
    ax.set_title("BEFORE: Z (depth) over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Z time series: After ---
    ax = axes[1, 1]
    for pid in pids:
        ax.plot(frames, humans_new[pid][:, 2], label=f"Human {pid}", linewidth=1.5)
    ax.plot(frames, obj_new[:, 2], "k--", label="Box", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Z (m)")
    ax.set_title("AFTER: Z (depth) over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "alignment_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Saved: {path}")


def render_3d_video(out_dir, ar_new, obj_new_grav, obj_data, pids, G):
    """Render bird's-eye 3D view with human SMPL meshes + box wireframe."""
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.video_io_utils import get_video_lwh, get_writer
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from einops import einsum
    from scipy.spatial.transform import Rotation
    import cv2

    video_path = str(out_dir / "0_input_video.mp4")
    length, width, height = get_video_lwh(video_path)

    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cpu()

    COLORS = [[0.8, 0.5, 0.5], [0.5, 0.5, 0.8]]

    # --- Compute human verts in gravity frame ---
    all_verts = {}
    for pid in pids:
        params = ar_new[pid]["smpl_params_global"]
        params_no_transl = {k: v for k, v in params.items() if k != "transl"}
        smplx_out = smplx_model(**to_cuda(params_no_transl))
        pelvis = smplx_out.joints[:, [0]]
        verts_centered = smplx_out.vertices - pelvis
        verts = torch.stack([smplx2smpl @ v for v in verts_centered]).cpu()
        transl = params["transl"].cpu()
        verts = verts + transl.unsqueeze(1)
        all_verts[pid] = verts

    # --- Scene centering: offset + T rotation (same as 05_render.py) ---
    anchor = all_verts[pids[0]].clone()
    offset = einsum(J_regressor, anchor[0], "j v, v i -> j i")[0]
    all_min_y = min(all_verts[pid][:, :, 1].min().item() for pid in pids)
    offset[1] = all_min_y
    T_rot = compute_T_ayfz2ay(
        einsum(J_regressor, (anchor - offset)[[0]], "j v, l v i -> l j i"),
        inverse=True)
    T_rot_3x3 = T_rot[0, :3, :3]  # (3, 3) rotation only

    # Apply to all humans: center then rotate
    moved = {}
    for pid in pids:
        centered = all_verts[pid] - offset
        moved[pid] = apply_T_on_points(centered, T_rot)

    # --- Box mesh for PyTorch3D rendering ---
    obj_data_full = torch.load(out_dir / "preprocess" / "obj_poses_scaled.pt", map_location="cpu")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from track_object import load_mesh
    mesh_o3d = load_mesh(obj_data_full["obj_mesh_path"])
    mesh_verts_np = np.asarray(mesh_o3d.vertices)  # real meters from USD
    mesh_faces_np = np.asarray(mesh_o3d.triangles)

    obj_pos_cam = obj_data_full["obj_pos"]
    obj_quat = obj_data_full["obj_quat"]

    # --- Side-view camera: similar to original video angle ---
    combined = torch.cat([moved[pid] for pid in pids], dim=1)

    # Side view, elevated like original video camera (~22° down tilt)
    # vec_rot=90 looks from the side; cam_height_degree=25 for elevation
    global_R, global_T, lights = get_global_cameras_static(
        combined, beta=3.5, cam_height_degree=30, target_center_height=1.2, vec_rot=270.0)

    _, _, K_render = create_camera_sensor(width, height, 24)

    renderer_human = Renderer(width, height, device="cuda", faces=faces_smpl, K=K_render, bin_size=0)

    joints0 = einsum(J_regressor, moved[pids[0]], "j v, l v i -> l j i")
    scale_g, cx_g, cz_g = get_ground_params_from_points(joints0[:, 0], moved[pids[0]])
    renderer_human.set_ground(scale_g * 2.0, cx_g, cz_g)

    tracks_data = torch.load(out_dir / "preprocess" / "tracks.pt", map_location="cpu")
    fps = tracks_data["fps"]
    output_path = str(out_dir / "3_global_with_box.mp4")
    writer = get_writer(output_path, fps=fps, crf=23)

    offset_np = offset.numpy()

    # Pre-compute box faces tensor
    from pytorch3d.structures import Meshes, join_meshes_as_scene
    from pytorch3d.renderer import (
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesVertex, PerspectiveCameras,
    )

    box_faces_t = torch.from_numpy(mesh_faces_np).long()
    smpl_faces_t = torch.from_numpy(faces_smpl.astype(np.int64)).long()

    raster_settings = RasterizationSettings(
        image_size=(height, width), blur_radius=0.0,
        faces_per_pixel=1, bin_size=0)

    for i in tqdm(range(length), desc="Rendering 3D"):
        # First render ground only
        cameras = renderer_human.create_camera(global_R[i], global_T[i])

        # Box: SAME transform as humans (G_refined → center → T_rot), NO y_offset
        R_obj = Rotation.from_quat(obj_quat[i]).as_matrix()
        box_cam = (R_obj @ mesh_verts_np.T).T + obj_pos_cam[i]
        box_grav = (G @ box_cam.T).T
        box_centered = box_grav - offset_np
        box_t = torch.from_numpy(box_centered).float().unsqueeze(0)
        box_render_verts = apply_T_on_points(box_t, T_rot)[0]  # (V, 3)

        # Build combined scene: all humans + box in ONE mesh for proper depth
        mesh_list = []
        for j, pid in enumerate(pids):
            v = moved[pid][i].cuda()  # (V_smpl, 3)
            f = smpl_faces_t.cuda()
            c = torch.tensor(COLORS[j % len(COLORS)]).float().expand(v.shape[0], 3).cuda()
            mesh_list.append(Meshes(verts=[v], faces=[f],
                                    textures=TexturesVertex(verts_features=[c])))

        # Add box
        bv = box_render_verts.cuda().float()
        bf = box_faces_t.cuda()
        bc = torch.tensor([0.9, 0.8, 0.3]).float().expand(bv.shape[0], 3).cuda()
        mesh_list.append(Meshes(verts=[bv], faces=[bf],
                                textures=TexturesVertex(verts_features=[bc])))

        # Join all meshes into single scene (depth handled correctly)
        scene = join_meshes_as_scene(mesh_list)

        # Render ground first (humans only — for ground plane texture)
        verts_frame = torch.cat([moved[pid][[i]] for pid in pids], dim=0).cuda()
        h_colors = torch.tensor([COLORS[j % len(COLORS)] for j in range(len(pids))]).float().cuda()
        ground_img = renderer_human.render_with_ground(verts_frame, h_colors, cameras, lights)

        # Render combined scene (humans + box) with proper depth via PyTorch3D
        cam_pt3d = PerspectiveCameras(
            device="cuda",
            R=renderer_human.R.mT, T=renderer_human.T,
            K=renderer_human.K_full,
            image_size=renderer_human.image_sizes, in_ndc=False)
        if i == 0:
            scene_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cam_pt3d, raster_settings=raster_settings),
                shader=SoftPhongShader(device="cuda", cameras=cam_pt3d, lights=lights))
        else:
            scene_renderer.rasterizer.cameras = cam_pt3d
            scene_renderer.shader.cameras = cam_pt3d

        scene_img = scene_renderer(scene)  # (1, H, W, 4)
        rgba = scene_img[0].cpu().numpy()
        alpha = rgba[:, :, 3:4]
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

        # Composite depth-correct scene onto ground
        ground_np = np.array(ground_img) if not isinstance(ground_img, np.ndarray) else ground_img
        mask = alpha > 0.1
        ground_np[mask.squeeze(-1)] = rgb[mask.squeeze(-1)]
        img = ground_np

        writer.write_frame(img)

    writer.close()
    del smplx_model, smplx2smpl, renderer_human
    torch.cuda.empty_cache()
    print(f"[Viz] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_3d", action="store_true", help="Skip 3D render (slow)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    print("[Viz] Loading data...")
    humans_old, humans_new, obj_old, obj_new, pids, ar_new, G = load_all_data(out_dir)

    # --- Plots ---
    plot_bev_and_z(humans_old, humans_new, obj_old, obj_new, pids, preprocess_dir)

    # --- 3D render ---
    if not args.skip_3d:
        print("[Viz] Rendering 3D video...")
        obj_data = torch.load(preprocess_dir / "obj_poses_scaled.pt", map_location="cpu")
        render_3d_video(out_dir, ar_new, obj_new, obj_data, pids, G)
    else:
        print("[Viz] Skipping 3D render (--skip_3d)")

    print("[Viz] Done.")


if __name__ == "__main__":
    main()
