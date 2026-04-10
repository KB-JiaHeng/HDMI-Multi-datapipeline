"""
Compute depth scale factor between GVHMR and MoGe-2 depth spaces.

Builds two independent pelvis trajectories in gravity frame:
1. GVHMR: incam transl[0] → G_refined → gravity init + raw global velocity integral
2. MoGe: vitpose_wholebody feet (6 kpts) + FK pelvis-foot offset → G_refined → gravity

Scale = median(GVHMR_traj / MoGe_traj) per frame.

Outputs:
  - preprocess/depth_scale.json: scale value + statistics
  - preprocess/debug_scale_*.png: verification images

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/compute_depth_scale.py \
        --output_dir outputs/demo_test/siheng_dual
"""

import argparse
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda


# ---------------------------------------------------------------------------
# MoGe sampling: 3D position at vitpose foot keypoints
# ---------------------------------------------------------------------------

def sample_moge_at_feet(pts3d, vp_wb_frame, hw=2, conf_thresh=0.5):
    """Sample MoGe 3D at vitpose_wholebody foot keypoints.

    Uses 6 foot keypoints: 17-19 (L_big_toe, L_small_toe, L_heel),
                           20-22 (R_big_toe, R_small_toe, R_heel).
    Returns per-foot average 3D position (camera frame).

    Args:
        pts3d: (H, W, 3) float array, camera-frame 3D points
        vp_wb_frame: (133, 3) tensor, [x, y, conf] per keypoint
        hw: half-window for median patch sampling
        conf_thresh: minimum confidence

    Returns:
        foot_l_cam: (3,) or None — left foot avg MoGe position
        foot_r_cam: (3,) or None — right foot avg MoGe position
    """
    H, W = pts3d.shape[:2]

    results = {}
    for foot_name, kp_range in [("left", [17, 18, 19]), ("right", [20, 21, 22])]:
        positions = []
        for kp_idx in kp_range:
            px, py, conf = vp_wb_frame[kp_idx]
            px, py, conf = px.item(), py.item(), conf.item()
            if conf < conf_thresh:
                continue
            px_i = int(min(max(round(px), 0), W - 1))
            py_i = int(min(max(round(py), 0), H - 1))

            # 5x5 patch median
            patch = pts3d[max(0, py_i - hw):min(H, py_i + hw + 1),
                          max(0, px_i - hw):min(W, px_i + hw + 1)]
            ok = np.isfinite(patch).all(axis=-1) & (patch[..., 2] > 0)
            if ok.sum() > 0:
                positions.append(np.median(patch[ok], axis=0))

        results[foot_name] = np.mean(positions, axis=0) if positions else None

    return results.get("left"), results.get("right")


# ---------------------------------------------------------------------------
# FK: pelvis-to-foot offset in camera frame
# ---------------------------------------------------------------------------

def compute_fk_foot_to_pelvis(smplx_model, incam_params_frame):
    """Compute pelvis-foot offset from GVHMR incam params (single frame).

    Returns offsets such that: pelvis_cam = foot_cam + offset

    Args:
        smplx_model: SMPLX body model (cuda)
        incam_params_frame: dict with global_orient, body_pose, betas (each [1, ...])

    Returns:
        offset_l: (3,) pelvis - L_foot (camera frame)
        offset_r: (3,) pelvis - R_foot (camera frame)
    """
    with torch.no_grad():
        out = smplx_model(**to_cuda(incam_params_frame))
    joints = out.joints[0].cpu().numpy()  # (J, 3)
    pelvis = joints[0]
    foot_l = joints[10]  # L_Foot
    foot_r = joints[11]  # R_Foot
    return pelvis - foot_l, pelvis - foot_r


# ---------------------------------------------------------------------------
# Build GVHMR trajectory in gravity frame
# ---------------------------------------------------------------------------

def build_gvhmr_trajectory(hmr_data, G_refined):
    """Return raw GVHMR global trajectory directly.

    Uses smpl_params_global['transl'] — velocity-integrated in GVHMR's
    gravity frame (GVHMR-ay ≈ G_refined). No incam transl, no bridge rescale.

    Args:
        hmr_data: hmr4d_results dict
        G_refined: (3, 3) ndarray — unused (kept for API consistency)

    Returns:
        (F, 3) ndarray — pelvis trajectory in GVHMR gravity frame
    """
    return hmr_data["smpl_params_global"]["transl"].numpy().copy()


# ---------------------------------------------------------------------------
# Build MoGe trajectory in gravity frame
# ---------------------------------------------------------------------------

def build_moge_trajectory(scene_data, hmr_data, vp_wb, G_refined, smplx_model,
                          sample_step=1):
    """Build MoGe-derived pelvis trajectory in G_refined gravity frame.

    Per frame: MoGe feet (6 wholebody kpts) + FK offset → pelvis (cam) → gravity.

    Args:
        scene_data: 7-tuple
        hmr_data: hmr4d_results dict
        vp_wb: (F, 133, 3) tensor
        G_refined: (3, 3) ndarray
        smplx_model: SMPLX model (cuda)
        sample_step: process every Nth frame

    Returns:
        frame_indices: list of int
        moge_traj: (len, 3) ndarray — pelvis in gravity frame
    """
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]
    nf = len(frame_names)

    frame_indices = []
    moge_traj = []

    for f in tqdm(range(0, nf, sample_step), desc="MoGe trajectory"):
        pts3d = pts3d_dict[frame_names[f]]
        if torch.is_tensor(pts3d):
            pts3d = pts3d.float().numpy()
        else:
            pts3d = np.array(pts3d, dtype=np.float32)

        # Sample MoGe at feet
        foot_l_cam, foot_r_cam = sample_moge_at_feet(pts3d, vp_wb[f])

        if foot_l_cam is None and foot_r_cam is None:
            continue

        # FK offset
        incam_params = {
            "global_orient": hmr_data["smpl_params_incam"]["global_orient"][[f]],
            "body_pose": hmr_data["smpl_params_incam"]["body_pose"][[f]],
            "betas": hmr_data["smpl_params_incam"]["betas"][[f]],
        }
        offset_l, offset_r = compute_fk_foot_to_pelvis(smplx_model, incam_params)

        # MoGe pelvis from each foot
        pelvis_estimates = []
        if foot_l_cam is not None:
            pelvis_estimates.append(foot_l_cam + offset_l)
        if foot_r_cam is not None:
            pelvis_estimates.append(foot_r_cam + offset_r)

        pelvis_moge_cam = np.mean(pelvis_estimates, axis=0)
        pelvis_moge_grav = G_refined @ pelvis_moge_cam

        frame_indices.append(f)
        moge_traj.append(pelvis_moge_grav)

    return frame_indices, np.array(moge_traj)


# ---------------------------------------------------------------------------
# Compute scale
# ---------------------------------------------------------------------------

def compute_scale(gvhmr_traj, moge_traj, frame_indices):
    """Compute scale using DISPLACEMENT from frame 0 (origins differ).

    Scale = LS fit of GVHMR_displacement vs MoGe_displacement.

    Returns:
        scale_global: float — uniform scale (LS fit on all axes)
        scale_per_axis: dict — {X, Y, Z} LS scales
        per_frame_ratios: dict — per-frame ratio arrays for plotting
    """
    gvhmr_sampled = gvhmr_traj[frame_indices]
    moge_sampled = moge_traj

    # Displacement from first sample frame
    gvhmr_disp = gvhmr_sampled - gvhmr_sampled[0]
    moge_disp = moge_sampled - moge_sampled[0]

    # Skip frame 0 (displacement = 0)
    gvhmr_disp = gvhmr_disp[1:]
    moge_disp = moge_disp[1:]

    # Per-axis least squares on displacements: s = (m_disp^T g_disp) / (m_disp^T m_disp)
    scale_per_axis = {}
    for ax, name in enumerate(["X", "Y", "Z"]):
        m = moge_disp[:, ax]
        g = gvhmr_disp[:, ax]
        denom = np.dot(m, m)
        if abs(denom) > 1e-8:
            s = np.dot(m, g) / denom
            residual = g - s * m
            scale_per_axis[name] = {
                "scale": float(s),
                "residual_std": float(residual.std()),
                "residual_max": float(np.abs(residual).max()),
            }
        else:
            scale_per_axis[name] = {"scale": float("nan"), "residual_std": 0, "residual_max": 0}

    # Global uniform scale on displacements
    m_flat = moge_disp.flatten()
    g_flat = gvhmr_disp.flatten()
    denom = np.dot(m_flat, m_flat)
    scale_global = float(np.dot(m_flat, g_flat) / denom) if abs(denom) > 1e-8 else 1.0

    # Per-frame displacement magnitude ratio for plotting
    gvhmr_disp_full = gvhmr_sampled - gvhmr_sampled[0]
    moge_disp_full = moge_sampled - moge_sampled[0]
    gvhmr_disp_norm = np.linalg.norm(gvhmr_disp_full, axis=1)
    moge_disp_norm = np.linalg.norm(moge_disp_full, axis=1)
    disp_ratio = np.where(moge_disp_norm > 0.01,
                          gvhmr_disp_norm / moge_disp_norm, np.nan)

    return scale_global, scale_per_axis, {
        "frame_indices": frame_indices,
        "disp_ratio": [float(x) if np.isfinite(x) else None for x in disp_ratio],
        "gvhmr_disp_norm": gvhmr_disp_norm.tolist(),
        "moge_disp_norm": moge_disp_norm.tolist(),
        "gvhmr_z": gvhmr_sampled[:, 2].tolist(),
        "moge_z": moge_sampled[:, 2].tolist(),
    }


# ---------------------------------------------------------------------------
# Verification visualization
# ---------------------------------------------------------------------------

def render_verification_images(video_path, hmr_data_dict, vp_wb_dict, output_dir,
                               scene_data=None, depth_scale=None,
                               sample_frames=[0, 100, 300, 500]):
    """Render SMPL mesh + MoGe scaled pelvis projection on video frames."""
    from hmr4d.utils.vis.renderer import Renderer

    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use GVHMR K from first person (same as 05_render.py line 55)
    first_pid = sorted(hmr_data_dict.keys())[0]
    K = hmr_data_dict[first_pid]["K_fullimg"][0]
    renderer = Renderer(W, H, device="cuda", faces=faces_smpl, K=K)

    COLORS = [[0.8, 0.5, 0.5], [0.5, 0.5, 0.8]]  # red-ish, blue-ish

    for f_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # cv2 BGR → RGB for renderer
        img = frame[..., ::-1].copy()

        for j, pid in enumerate(sorted(hmr_data_dict.keys())):
            hmr = hmr_data_dict[pid]
            # Exact 05_render.py:render_incam logic (lines 63-64)
            smplx_out = smplx_model(**to_cuda({
                k: v[[f_idx]] for k, v in hmr["smpl_params_incam"].items()
            }))
            verts = (smplx2smpl @ smplx_out.vertices[0]).unsqueeze(0)
            color = COLORS[j % len(COLORS)]
            img = renderer.render_mesh(verts[0].cuda(), img, color)

        # Overlay vitpose foot keypoints (green circles) on top of mesh
        for pid in sorted(hmr_data_dict.keys()):
            vp_wb = vp_wb_dict[pid]
            for kp_idx in range(17, 23):
                px, py, conf = vp_wb[f_idx, kp_idx]
                if conf > 0.5:
                    cv2.circle(img, (int(px.item()), int(py.item())), 5, (0, 255, 0), 2)

        # --- Project MoGe-derived pelvis (scaled) to 2D ---
        # This is the KEY verification: does scaled MoGe pelvis project
        # to the correct 2D position (matching ViTPose)?
        if scene_data is not None and depth_scale is not None:
            frame_names_sd = scene_data[0]
            pts3d_dict_sd = scene_data[1]
            K_moge = scene_data[3][frame_names_sd[0]]
            if torch.is_tensor(K_moge):
                K_moge_np = K_moge.numpy()
            else:
                K_moge_np = np.array(K_moge)
            fx_m, fy_m = K_moge_np[0, 0], K_moge_np[1, 1]
            cx_m, cy_m = K_moge_np[0, 2], K_moge_np[1, 2]

            for j, pid in enumerate(sorted(hmr_data_dict.keys())):
                hmr = hmr_data_dict[pid]
                vp_wb = vp_wb_dict[pid]

                pts3d = pts3d_dict_sd[frame_names_sd[f_idx]]
                if torch.is_tensor(pts3d):
                    pts3d_np = pts3d.float().numpy()
                else:
                    pts3d_np = np.array(pts3d, dtype=np.float32)

                # MoGe feet + FK → pelvis (camera frame)
                foot_l, foot_r = sample_moge_at_feet(pts3d_np, vp_wb[f_idx])

                incam_params = {
                    "global_orient": hmr["smpl_params_incam"]["global_orient"][[f_idx]],
                    "body_pose": hmr["smpl_params_incam"]["body_pose"][[f_idx]],
                    "betas": hmr["smpl_params_incam"]["betas"][[f_idx]],
                }
                offset_l, offset_r = compute_fk_foot_to_pelvis(smplx_model, incam_params)

                pelvis_estimates = []
                if foot_l is not None:
                    pelvis_estimates.append(foot_l + offset_l)
                if foot_r is not None:
                    pelvis_estimates.append(foot_r + offset_r)

                if pelvis_estimates:
                    pelvis_moge_cam = np.mean(pelvis_estimates, axis=0)
                    # Scale by depth_scale
                    pelvis_scaled = pelvis_moge_cam * depth_scale

                    # Project to 2D using MoGe K (scene K)
                    u_moge = int(fx_m * pelvis_scaled[0] / pelvis_scaled[2] + cx_m)
                    v_moge = int(fy_m * pelvis_scaled[1] / pelvis_scaled[2] + cy_m)

                    # Also project UNSCALED for comparison
                    u_raw = int(fx_m * pelvis_moge_cam[0] / pelvis_moge_cam[2] + cx_m)
                    v_raw = int(fy_m * pelvis_moge_cam[1] / pelvis_moge_cam[2] + cy_m)

                    # Cyan diamond = scaled MoGe pelvis
                    cv2.drawMarker(img, (u_moge, v_moge), (255, 255, 0),
                                   cv2.MARKER_DIAMOND, 18, 2)
                    # Magenta cross = unscaled MoGe pelvis
                    cv2.drawMarker(img, (u_raw, v_raw), (255, 0, 255),
                                   cv2.MARKER_CROSS, 15, 2)

        cv2.putText(img, f"Frame {f_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Mesh=GVHMR, Cyan diamond=Scaled MoGe pelvis, Magenta cross=Raw MoGe",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # RGB → BGR for cv2.imwrite
        out_path = output_dir / f"debug_scale_frame{f_idx}.png"
        cv2.imwrite(str(out_path), img[..., ::-1])
        print(f"  Saved: {out_path}")

    cap.release()
    del smplx_model, smplx2smpl, renderer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute GVHMR/MoGe depth scale")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_step", type=int, default=5,
                        help="Process every Nth frame for scale computation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load data ---
    print("[DepthScale] Loading data...")
    scene_data = torch.load(preprocess_dir / "scene_data.pt", map_location="cpu")
    G_refined = torch.load(preprocess_dir / "G_refined.pt", map_location="cpu").numpy()

    # Per-person data
    hmr_data_dict = {}
    vp_wb_dict = {}
    person_dirs = sorted(preprocess_dir.glob("person_*"))
    for pd in person_dirs:
        pid = int(pd.name.split("_")[1])
        hmr_data_dict[pid] = torch.load(pd / "hmr4d_results.pt", map_location="cpu")
        vp_wb_dict[pid] = torch.load(pd / "vitpose_wholebody.pt", map_location="cpu")

    print(f"[DepthScale] {len(hmr_data_dict)} person(s), "
          f"{len(scene_data[0])} frames")

    # --- FK model ---
    smplx_model = make_smplx("supermotion").cuda()

    # --- Build trajectories and compute scale per person ---
    all_scales = []
    all_ratios = {}

    for pid in sorted(hmr_data_dict.keys()):
        print(f"\n[DepthScale] Person {pid}")
        hmr = hmr_data_dict[pid]
        vp_wb = vp_wb_dict[pid]

        # GVHMR trajectory (gravity frame)
        gvhmr_traj = build_gvhmr_trajectory(hmr, G_refined)
        print(f"  GVHMR traj: shape={gvhmr_traj.shape}, "
              f"Z range=[{gvhmr_traj[:,2].min():.3f}, {gvhmr_traj[:,2].max():.3f}]")

        # MoGe trajectory (gravity frame)
        frame_indices, moge_traj = build_moge_trajectory(
            scene_data, hmr, vp_wb, G_refined, smplx_model,
            sample_step=args.sample_step)
        print(f"  MoGe traj: {len(frame_indices)} frames, "
              f"Z range=[{moge_traj[:,2].min():.3f}, {moge_traj[:,2].max():.3f}]")

        # Scale
        scale_global, scale_per_axis, ratios = compute_scale(
            gvhmr_traj, moge_traj, frame_indices)

        print(f"  Global LS scale: {scale_global:.4f}")
        for ax_name, info in scale_per_axis.items():
            print(f"  {ax_name}: scale={info['scale']:.4f}, "
                  f"residual_std={info['residual_std']:.4f}m")

        all_scales.append(scale_global)
        all_ratios[pid] = ratios

    del smplx_model
    torch.cuda.empty_cache()

    # --- Aggregate scale across persons ---
    depth_scale = float(np.median(all_scales))
    print(f"\n[DepthScale] Final depth_scale = {depth_scale:.4f} "
          f"(from {len(all_scales)} persons)")

    # --- Save ---
    result = {
        "depth_scale": depth_scale,
        "per_person_scales": {str(pid): float(s)
                              for pid, s in zip(sorted(hmr_data_dict.keys()), all_scales)},
        "per_person_ratios": {str(pid): r for pid, r in all_ratios.items()},
    }
    scale_path = preprocess_dir / "depth_scale.json"
    with open(scale_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DepthScale] Saved: {scale_path}")

    # --- Verification images ---
    print("\n[DepthScale] Generating verification images...")
    video_path = out_dir / "0_input_video.mp4"
    render_verification_images(video_path, hmr_data_dict, vp_wb_dict, preprocess_dir,
                               scene_data=scene_data, depth_scale=depth_scale)

    # --- Scale plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Displacement magnitude ratio per frame
        ax = axes[0]
        for pid, ratios in all_ratios.items():
            valid = [i for i, r in enumerate(ratios["disp_ratio"])
                     if r is not None]
            ax.plot([ratios["frame_indices"][i] for i in valid],
                    [ratios["disp_ratio"][i] for i in valid],
                    label=f"Person {pid}", alpha=0.7)
        ax.axhline(y=depth_scale, color="r", linestyle="--",
                    label=f"LS scale={depth_scale:.4f}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("|GVHMR_disp| / |MoGe_disp|")
        ax.set_title("Displacement magnitude ratio (from frame 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Displacement norm over time
        ax = axes[1]
        for pid, ratios in all_ratios.items():
            ax.plot(ratios["frame_indices"], ratios["gvhmr_disp_norm"],
                    label=f"GVHMR P{pid}", linewidth=2)
            ax.plot(ratios["frame_indices"], ratios["moge_disp_norm"],
                    label=f"MoGe P{pid}", linewidth=1, linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Displacement from frame 0 (m)")
        ax.set_title("GVHMR vs MoGe cumulative displacement")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = preprocess_dir / "debug_scale_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[DepthScale] Saved: {plot_path}")
    except ImportError:
        print("[DepthScale] matplotlib not available, skipping plot")

    print("\n[DepthScale] Done. Review verification images before proceeding.")


if __name__ == "__main__":
    main()
