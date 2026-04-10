"""
Visualize computed contact offsets as red spheres on the 3D rendered scene,
with lines from each contact sphere to the corresponding wrist joint.

Based on 08_visualize.py's render_3d_video(), adds:
  - Red balls at contact_target_pos_offset positions on the object mesh
  - Colored lines from each contact point to its assigned wrist
    (crossed lines = L/R mapping is wrong)

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python ../scripts/vis_contact_offset.py \
        --output_dir outputs/demo_test/koi \
        --obj_mesh ../assets/box_real.usd
"""

import argparse
import json
import numpy as np
import torch
import trimesh as tm
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def project_3d_to_2d(pts_3d, cam_pt3d, image_size):
    """Project 3D points (render coords) to 2D pixel coords via PyTorch3D camera.

    Args:
        pts_3d: (N, 3) numpy array in render world coords
        cam_pt3d: PyTorch3D PerspectiveCameras object
        image_size: (H, W) tuple

    Returns:
        (N, 2) numpy array of pixel coords [u, v]
    """
    pts = torch.from_numpy(pts_3d).float().unsqueeze(0).to(cam_pt3d.device)  # (1, N, 3)
    screen = cam_pt3d.transform_points_screen(pts, image_size=(image_size,))  # (1, N, 3)
    return screen[0, :, :2].cpu().numpy()  # (N, 2) [u, v]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize contact offsets as red spheres + wrist lines on 3D render")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--obj_mesh", type=str, default=None,
                        help="Path to object mesh. Falls back to obj_poses metadata.")
    parser.add_argument("--sphere_radius", type=float, default=0.02,
                        help="Radius of contact marker spheres (meters)")
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated frame indices to render as PNG (e.g., '0,100,544'). "
                             "If not set, renders full video.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load contact offsets ---
    offsets_path = preprocess_dir / "contact_offsets.json"
    assert offsets_path.exists(), f"Run compute_contact_offset.py first: {offsets_path}"
    with open(offsets_path) as f:
        contact_offsets = json.load(f)
    print(f"[VisContact] Loaded contact offsets: {offsets_path}")
    for agent_key, data in contact_offsets.items():
        print(f"  {agent_key}: {data['contact_target_pos_offset']}")

    # --- Build per-agent offset data: (agent_idx, side, offset_pt) ---
    # side 0 = det[9]=0 (mapped to wrist index 0 = left_wrist joint 20)
    # side 1 = det[9]=1 (mapped to wrist index 1 = right_wrist joint 21)
    agent_offset_info = []  # list of (agent_idx, side, offset_pt_meshlocal)
    all_offset_pts = []
    for agent_key, data in sorted(contact_offsets.items()):
        agent_idx = int(agent_key.split("_")[1])
        for side, offset in enumerate(data["contact_target_pos_offset"]):
            pt = np.array(offset)
            if np.linalg.norm(pt) > 1e-6:
                agent_offset_info.append((agent_idx, side, pt))
                all_offset_pts.append(pt)

    if not all_offset_pts:
        print("[VisContact] No non-zero contact offsets found, nothing to visualize.")
        return

    # --- Create sphere meshes ---
    sphere_template = tm.creation.icosphere(subdivisions=2, radius=args.sphere_radius)
    sphere_verts_list = []
    sphere_faces_list = []
    vert_offset = 0
    for pt in all_offset_pts:
        sv = np.array(sphere_template.vertices) + pt
        sf = np.array(sphere_template.faces) + vert_offset
        sphere_verts_list.append(sv)
        sphere_faces_list.append(sf)
        vert_offset += len(sv)

    sphere_verts_local = np.concatenate(sphere_verts_list, axis=0)
    sphere_faces_all = np.concatenate(sphere_faces_list, axis=0)
    print(f"[VisContact] {len(all_offset_pts)} contact spheres, "
          f"{len(sphere_verts_local)} verts total")

    # --- Render setup (adapted from 08_visualize.py) ---
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.video_io_utils import get_video_lwh, get_writer
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from einops import einsum
    import cv2
    import sys, os

    _SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_SCRIPTS_DIR))
    from track_object import load_mesh

    G = torch.load(preprocess_dir / "G_refined_scaled.pt", map_location="cpu").numpy()
    ar = torch.load(out_dir / "aligned_results_xz.pt", map_location="cpu", weights_only=False)
    obj_data = torch.load(preprocess_dir / "obj_poses_scaled.pt",
                          map_location="cpu", weights_only=False)
    pids = sorted(ar.keys())

    obj_pos_cam = obj_data["obj_pos"]
    obj_quat_cam = obj_data["obj_quat"]

    video_path = str(out_dir / "0_input_video.mp4")
    length, width, height = get_video_lwh(video_path)

    # Resolve GVHMR root from the output_dir path (output_dir is under GVHMR/outputs/...)
    _gvhmr_root = out_dir
    while _gvhmr_root.name != "GVHMR" and _gvhmr_root != _gvhmr_root.parent:
        _gvhmr_root = _gvhmr_root.parent
    _body_model_dir = _gvhmr_root / "hmr4d" / "utils" / "body_model"

    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(_body_model_dir / "smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(_body_model_dir / "smpl_neutral_J_regressor.pt").cpu()

    COLORS = [[0.8, 0.5, 0.5], [0.5, 0.5, 0.8]]
    CONTACT_COLOR = [1.0, 0.2, 0.2]
    LINE_COLORS = [(0, 255, 0), (255, 255, 0)]  # BGR: green for side=0, yellow for side=1

    # Compute human verts + wrist joints in gravity frame
    all_verts = {}
    all_wrists = {}  # {pid: (T, 2, 3)} — [left_wrist, right_wrist] in gravity frame
    for pid in pids:
        params = ar[pid]["smpl_params_global"]
        params_no_transl = {k: v for k, v in params.items() if k != "transl"}
        with torch.no_grad():
            smplx_out = smplx_model(**to_cuda(params_no_transl))
        pelvis = smplx_out.joints[:, [0]]
        verts_centered = smplx_out.vertices - pelvis
        verts = torch.stack([smplx2smpl @ v for v in verts_centered]).cpu()
        transl = params["transl"].cpu()
        verts = verts + transl.unsqueeze(1)
        all_verts[pid] = verts

        # Wrist joints (gravity frame)
        wrist_lr = smplx_out.joints[:, [20, 21]] - pelvis  # (T, 2, 3)
        all_wrists[pid] = (wrist_lr.cpu() + transl.unsqueeze(1)).numpy()

    # Scene centering
    anchor = all_verts[pids[0]].clone()
    offset = einsum(J_regressor, anchor[0], "j v, v i -> j i")[0]
    all_min_y = min(all_verts[pid][:, :, 1].min().item() for pid in pids)
    offset[1] = all_min_y
    T_rot = compute_T_ayfz2ay(
        einsum(J_regressor, (anchor - offset)[[0]], "j v, l v i -> l j i"),
        inverse=True)
    offset_np = offset.numpy()

    moved = {}
    for pid in pids:
        centered = all_verts[pid] - offset
        moved[pid] = apply_T_on_points(centered, T_rot)

    # Transform wrists to render coords (same chain: gravity → center → T_rot)
    wrists_render = {}
    for pid in pids:
        w = all_wrists[pid]  # (T, 2, 3) gravity
        w_centered = w - offset_np  # (T, 2, 3)
        # Apply T_rot: need to reshape for apply_T_on_points which expects (L, V, 3)
        T_frames = w_centered.shape[0]
        w_flat = torch.from_numpy(w_centered).float()  # (T, 2, 3)
        w_render = apply_T_on_points(w_flat, T_rot).numpy()  # (T, 2, 3)
        wrists_render[pid] = w_render

    # Load object mesh
    mesh_path = args.obj_mesh or obj_data.get("obj_mesh_path")
    mesh_o3d = load_mesh(mesh_path)
    mesh_verts_np = np.asarray(mesh_o3d.vertices)
    mesh_faces_np = np.asarray(mesh_o3d.triangles)

    combined = torch.cat([moved[pid] for pid in pids], dim=1)
    global_R, global_T, lights = get_global_cameras_static(
        combined, beta=3.5, cam_height_degree=30, target_center_height=1.2, vec_rot=270.0)
    _, _, K_render = create_camera_sensor(width, height, 24)

    renderer_human = Renderer(width, height, device="cuda", faces=faces_smpl, K=K_render, bin_size=0)
    joints0 = einsum(J_regressor, moved[pids[0]], "j v, l v i -> l j i")
    scale_g, cx_g, cz_g = get_ground_params_from_points(joints0[:, 0], moved[pids[0]])
    renderer_human.set_ground(scale_g * 2.0, cx_g, cz_g)

    tracks_data = torch.load(out_dir / "preprocess" / "tracks.pt", map_location="cpu")
    fps = tracks_data["fps"]

    from pytorch3d.structures import Meshes, join_meshes_as_scene
    from pytorch3d.renderer import (
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesVertex, PerspectiveCameras,
    )

    box_faces_t = torch.from_numpy(mesh_faces_np).long()
    sphere_faces_t = torch.from_numpy(sphere_faces_all.astype(np.int64)).long()
    smpl_faces_t = torch.from_numpy(faces_smpl.astype(np.int64)).long()

    raster_settings = RasterizationSettings(
        image_size=(height, width), blur_radius=0.0,
        faces_per_pixel=1, bin_size=0)

    # Decide which frames to render
    if args.frames:
        frame_indices = [int(x.strip()) for x in args.frames.split(",")]
        render_video = False
    else:
        frame_indices = list(range(length))
        render_video = True

    if render_video:
        output_path = str(out_dir / "3_global_with_contact.mp4")
        writer = get_writer(output_path, fps=fps, crf=23)

    scene_renderer = None

    for i in tqdm(frame_indices, desc="Rendering contact viz"):
        if i >= length:
            continue

        cameras = renderer_human.create_camera(global_R[i], global_T[i])

        # Box verts: mesh-local → camera → gravity → centered → T_rot
        R_obj = Rotation.from_quat(obj_quat_cam[i]).as_matrix()
        box_cam = (R_obj @ mesh_verts_np.T).T + obj_pos_cam[i]
        box_grav = (G @ box_cam.T).T
        box_centered = box_grav - offset_np
        box_t = torch.from_numpy(box_centered).float().unsqueeze(0)
        box_render_verts = apply_T_on_points(box_t, T_rot)[0]

        # Sphere verts: same transform
        sphere_cam = (R_obj @ sphere_verts_local.T).T + obj_pos_cam[i]
        sphere_grav = (G @ sphere_cam.T).T
        sphere_centered = sphere_grav - offset_np
        sphere_t = torch.from_numpy(sphere_centered).float().unsqueeze(0)
        sphere_render_verts = apply_T_on_points(sphere_t, T_rot)[0]

        # Also transform individual contact offset centers (for line endpoints)
        contact_centers_render = []
        for pt in all_offset_pts:
            pt_cam = R_obj @ pt + obj_pos_cam[i]
            pt_grav = G @ pt_cam
            pt_centered = pt_grav - offset_np
            pt_r = apply_T_on_points(
                torch.from_numpy(pt_centered).float().reshape(1, 1, 3), T_rot
            )[0, 0].numpy()
            contact_centers_render.append(pt_r)

        # Build scene: humans + box + contact spheres
        mesh_list = []
        for j, pid in enumerate(pids):
            v = moved[pid][i].cuda()
            f = smpl_faces_t.cuda()
            c = torch.tensor(COLORS[j % len(COLORS)]).float().expand(v.shape[0], 3).cuda()
            mesh_list.append(Meshes(verts=[v], faces=[f],
                                    textures=TexturesVertex(verts_features=[c])))

        bv = box_render_verts.cuda().float()
        bf = box_faces_t.cuda()
        bc = torch.tensor([0.9, 0.8, 0.3]).float().expand(bv.shape[0], 3).cuda()
        mesh_list.append(Meshes(verts=[bv], faces=[bf],
                                textures=TexturesVertex(verts_features=[bc])))

        sv = sphere_render_verts.cuda().float()
        sf = sphere_faces_t.cuda()
        sc = torch.tensor(CONTACT_COLOR).float().expand(sv.shape[0], 3).cuda()
        mesh_list.append(Meshes(verts=[sv], faces=[sf],
                                textures=TexturesVertex(verts_features=[sc])))

        scene = join_meshes_as_scene(mesh_list)

        # Render ground
        verts_frame = torch.cat([moved[pid][[i]] for pid in pids], dim=0).cuda()
        h_colors = torch.tensor([COLORS[j % len(COLORS)] for j in range(len(pids))]).float().cuda()
        ground_img = renderer_human.render_with_ground(verts_frame, h_colors, cameras, lights)

        # Render scene
        cam_pt3d = PerspectiveCameras(
            device="cuda",
            R=renderer_human.R.mT, T=renderer_human.T,
            K=renderer_human.K_full,
            image_size=renderer_human.image_sizes, in_ndc=False)
        if scene_renderer is None:
            scene_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cam_pt3d, raster_settings=raster_settings),
                shader=SoftPhongShader(device="cuda", cameras=cam_pt3d, lights=lights))
        else:
            scene_renderer.rasterizer.cameras = cam_pt3d
            scene_renderer.shader.cameras = cam_pt3d

        scene_img = scene_renderer(scene)
        rgba = scene_img[0].cpu().numpy()
        alpha = rgba[:, :, 3:4]
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

        ground_np = np.array(ground_img) if not isinstance(ground_img, np.ndarray) else ground_img
        mask = alpha > 0.1
        ground_np[mask.squeeze(-1)] = rgb[mask.squeeze(-1)]
        img = ground_np  # RGB uint8

        # --- Draw lines from contact points to wrist joints (2D overlay) ---
        for k, (agent_idx, side, _) in enumerate(agent_offset_info):
            pid = pids[agent_idx]
            wrist_side = side  # 0=left_wrist, 1=right_wrist (to be verified)

            # Contact point in render coords
            cp_render = contact_centers_render[k]
            # Wrist in render coords
            wr_render = wrists_render[pid][i, wrist_side]

            # Project both to 2D via PyTorch3D camera
            both_pts = np.stack([cp_render, wr_render], axis=0)  # (2, 3)
            uv = project_3d_to_2d(both_pts, cam_pt3d, (height, width))

            cp_2d = tuple(int(round(x)) for x in uv[0])
            wr_2d = tuple(int(round(x)) for x in uv[1])

            # Draw line (RGB color on RGB image)
            line_color = LINE_COLORS[side % len(LINE_COLORS)]
            cv2.line(img, cp_2d, wr_2d, line_color, thickness=2, lineType=cv2.LINE_AA)
            # Draw small circle at wrist
            cv2.circle(img, wr_2d, 4, line_color, -1, lineType=cv2.LINE_AA)

        if render_video:
            writer.write_frame(img)
        else:
            save_path = str(preprocess_dir / f"contact_viz_f{i:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {save_path}")

    if render_video:
        writer.close()
        print(f"[VisContact] Saved: {output_path}")

    del smplx_model, smplx2smpl, renderer_human
    torch.cuda.empty_cache()
    print("[VisContact] Done.")


if __name__ == "__main__":
    main()
