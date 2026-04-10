"""
XZ-axis Kalman+RTS filter: fuse GVHMR global velocity with MoGe-derived pelvis.

For each person, runs independent Kalman filters on X and Z axes (gravity frame).
- Prediction: GVHMR raw global frame-to-frame displacement
- Measurement: MoGe feet (vitpose 6 kpts) + FK → pelvis → G_refined → gravity
- Gating: visibility-only (vitpose conf + finite MoGe)
- RTS backward smoother for offline refinement

Y-axis is NOT touched (already handled by scene_grounding RANSAC planes).

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/run_xz_kalman.py \
        --output_dir outputs/demo_test/siheng_dual
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda

# Reuse MoGe sampling + FK from compute_depth_scale
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compute_depth_scale import sample_moge_at_feet, compute_fk_foot_to_pelvis


# ---------------------------------------------------------------------------
# Kalman filter + RTS smoother (1D per axis)
# ---------------------------------------------------------------------------

def run_kalman_rts_1d(gvhmr_disp, measurements, valid_mask,
                      Q_pos=2.5e-5, Q_bias=1e-7, R_meas=5e-2,
                      gate_chi2=3.84, init_pos=None):
    """Forward Kalman + RTS backward smoother for one axis.

    State: [position, velocity_bias]
    Predict: pos += (gvhmr_displacement - bias)
    Measure: MoGe-derived pelvis position (when valid)

    Args:
        gvhmr_disp: (N,) per-frame displacement from GVHMR global
        measurements: (N,) MoGe-derived pelvis position per frame
        valid_mask: (N,) bool — True when measurement is available
        Q_pos, Q_bias, R_meas, gate_chi2: Kalman parameters
        init_pos: initial position (from first valid measurement, or GVHMR)

    Returns:
        smoothed: (N,) RTS-smoothed position
        forward: (N,) forward-only Kalman position (for comparison)
        bias: (N,) estimated velocity bias
        n_accepted: number of accepted measurement updates
    """
    N = len(gvhmr_disp)
    F_mat = np.array([[1.0, -1.0], [0.0, 1.0]])
    B_vec = np.array([1.0, 0.0])
    H_mat = np.array([[1.0, 0.0]])

    # Initialize from first valid measurement
    if init_pos is None:
        first_valid = np.where(valid_mask)[0]
        init_pos = measurements[first_valid[0]] if len(first_valid) > 0 else 0.0

    x = np.array([init_pos, 0.0])
    P = np.array([[0.1, 0.0], [0.0, 1e-5]])
    Q_mat = np.array([[Q_pos, 0.0], [0.0, Q_bias]])

    # Storage for forward pass (needed by RTS)
    x_fwd = np.zeros((N, 2))
    P_fwd = np.zeros((N, 2, 2))
    x_pred = np.zeros((N, 2))
    P_pred = np.zeros((N, 2, 2))

    n_accepted = 0
    n_gated = 0

    for f in range(N):
        # --- Predict ---
        x_p = F_mat @ x + B_vec * gvhmr_disp[f]
        P_p = F_mat @ P @ F_mat.T + Q_mat

        x_pred[f] = x_p.copy()
        P_pred[f] = P_p.copy()

        # --- Update (if valid measurement) ---
        if valid_mask[f]:
            innovation = measurements[f] - (H_mat @ x_p)[0]
            S = (H_mat @ P_p @ H_mat.T)[0, 0] + R_meas
            NIS = innovation ** 2 / S

            if NIS <= gate_chi2:
                K_gain = (P_p @ H_mat.T) / S
                x_p = x_p + K_gain[:, 0] * innovation
                I_KH = np.eye(2) - K_gain @ H_mat
                P_p = I_KH @ P_p @ I_KH.T + np.outer(K_gain[:, 0], K_gain[:, 0]) * R_meas
                n_accepted += 1
            else:
                n_gated += 1

        x = x_p
        P = P_p
        x_fwd[f] = x.copy()
        P_fwd[f] = P.copy()

    # --- RTS backward smoother ---
    x_smooth = np.zeros((N, 2))
    P_smooth = np.zeros((N, 2, 2))
    x_smooth[-1] = x_fwd[-1]
    P_smooth[-1] = P_fwd[-1]

    for f in range(N - 2, -1, -1):
        P_pred_next = P_pred[f + 1]
        # C = P_fwd[f] @ F^T @ inv(P_pred[f+1])
        C = P_fwd[f] @ F_mat.T @ np.linalg.inv(P_pred_next)
        x_smooth[f] = x_fwd[f] + C @ (x_smooth[f + 1] - x_pred[f + 1])
        P_smooth[f] = P_fwd[f] + C @ (P_smooth[f + 1] - P_pred_next) @ C.T

    return x_smooth[:, 0], x_fwd[:, 0], x_smooth[:, 1], n_accepted, n_gated


# ---------------------------------------------------------------------------
# Build MoGe measurement trajectory
# ---------------------------------------------------------------------------

def build_moge_measurements(scene_data, hmr_data, vp_wb, G_refined, smplx_model,
                            depth_scale):
    """Per-frame MoGe-derived pelvis in gravity frame.

    Returns:
        moge_pelvis_grav: (F, 3) — pelvis position (NaN where invalid)
        valid_mask: (F,) bool
    """
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]
    nf = len(frame_names)

    moge_pelvis_grav = np.full((nf, 3), np.nan)
    valid_mask = np.zeros(nf, dtype=bool)

    for f in tqdm(range(nf), desc="MoGe measurements"):
        pts3d = pts3d_dict[frame_names[f]]
        if torch.is_tensor(pts3d):
            pts3d_np = pts3d.float().numpy()
        else:
            pts3d_np = np.array(pts3d, dtype=np.float32)

        # Scale pts3d
        pts3d_scaled = pts3d_np * depth_scale

        # Sample feet
        foot_l, foot_r = sample_moge_at_feet(pts3d_scaled, vp_wb[f])
        if foot_l is None and foot_r is None:
            continue

        # FK offset
        incam_params = {
            "global_orient": hmr_data["smpl_params_incam"]["global_orient"][[f]],
            "body_pose": hmr_data["smpl_params_incam"]["body_pose"][[f]],
            "betas": hmr_data["smpl_params_incam"]["betas"][[f]],
        }
        offset_l, offset_r = compute_fk_foot_to_pelvis(smplx_model, incam_params)

        pelvis_estimates = []
        if foot_l is not None:
            pelvis_estimates.append(foot_l + offset_l)
        if foot_r is not None:
            pelvis_estimates.append(foot_r + offset_r)

        pelvis_cam = np.mean(pelvis_estimates, axis=0)
        pelvis_grav = G_refined @ pelvis_cam

        moge_pelvis_grav[f] = pelvis_grav
        valid_mask[f] = True

    n_valid = valid_mask.sum()
    print(f"  Valid measurements: {n_valid}/{nf} ({100*n_valid/nf:.1f}%)")
    return moge_pelvis_grav, valid_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XZ Kalman+RTS filter")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--Q_pos", type=float, default=2.5e-5)
    parser.add_argument("--Q_bias", type=float, default=1e-7)
    parser.add_argument("--R_meas", type=float, default=5e-2)
    parser.add_argument("--aligned_input", type=str, default=None,
                        help="Path to aligned_results .pt (default: <output_dir>/aligned_results.pt)")
    parser.add_argument("--g_refined_path", type=str, default=None,
                        help="Path to G_refined .pt (default: preprocess/G_refined.pt)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load data ---
    print("[XZ-Kalman] Loading data...")
    with open(preprocess_dir / "depth_scale.json") as f:
        depth_scale = json.load(f)["depth_scale"]
    print(f"  depth_scale = {depth_scale:.4f}")

    scene_data = torch.load(preprocess_dir / "scene_data.pt", map_location="cpu")
    g_path = args.g_refined_path or str(preprocess_dir / "G_refined.pt")
    G_refined = torch.load(g_path, map_location="cpu").numpy()
    ar_path = args.aligned_input or str(out_dir / "aligned_results.pt")
    aligned_results = torch.load(ar_path, map_location="cpu")
    print(f"  aligned_input = {ar_path}")
    print(f"  G_refined = {g_path}")

    hmr_data_dict = {}
    vp_wb_dict = {}
    for pd in sorted(preprocess_dir.glob("person_*")):
        pid = int(pd.name.split("_")[1])
        hmr_data_dict[pid] = torch.load(pd / "hmr4d_results.pt", map_location="cpu")
        vp_wb_dict[pid] = torch.load(pd / "vitpose_wholebody.pt", map_location="cpu")

    smplx_model = make_smplx("supermotion").cuda()

    # --- Process each person ---
    new_results = {}
    for pid in sorted(hmr_data_dict.keys()):
        print(f"\n[XZ-Kalman] Person {pid}")
        hmr = hmr_data_dict[pid]
        vp_wb = vp_wb_dict[pid]
        nf = hmr["smpl_params_global"]["transl"].shape[0]

        # GVHMR raw global trajectory (GVHMR-ay ≈ G_refined)
        raw_global = hmr["smpl_params_global"]["transl"].numpy()

        # GVHMR frame-to-frame displacement
        gvhmr_disp = np.zeros((nf, 3))
        gvhmr_disp[1:] = raw_global[1:] - raw_global[:-1]

        # MoGe measurements (gravity frame)
        moge_pelvis, valid_mask = build_moge_measurements(
            scene_data, hmr, vp_wb, G_refined, smplx_model, depth_scale)

        # Get Y-corrected trajectory from existing aligned_results
        t_aligned = aligned_results[pid]["smpl_params_global"]["transl"].numpy()

        # Run Kalman+RTS for X and Z independently
        corrected = t_aligned.copy()

        for axis, axis_name in [(0, "X"), (2, "Z")]:
            print(f"\n  {axis_name}-axis Kalman+RTS:")

            # Initial position from first valid MoGe measurement
            first_valid_idx = np.where(valid_mask)[0]
            init_pos = moge_pelvis[first_valid_idx[0], axis] if len(first_valid_idx) > 0 else t_aligned[0, axis]

            smoothed, forward, bias, n_acc, n_gated = run_kalman_rts_1d(
                gvhmr_disp[:, axis],
                moge_pelvis[:, axis],
                valid_mask,
                Q_pos=args.Q_pos,
                Q_bias=args.Q_bias,
                R_meas=args.R_meas,
                init_pos=init_pos,
            )

            # Stats
            valid_meas = moge_pelvis[valid_mask, axis]
            valid_smooth = smoothed[valid_mask]
            residual = valid_meas - valid_smooth
            drift_before = abs(t_aligned[-1, axis] - t_aligned[0, axis])
            drift_after = abs(smoothed[-1] - smoothed[0])
            print(f"    Accepted/Gated: {n_acc}/{n_gated}")
            print(f"    Residual: mean={residual.mean():.4f}m, std={residual.std():.4f}m")
            print(f"    Bias final: {bias[-1]*1000:.3f} mm/frame")
            print(f"    Range before: [{t_aligned[:,axis].min():.3f}, {t_aligned[:,axis].max():.3f}]")
            print(f"    Range after:  [{smoothed.min():.3f}, {smoothed.max():.3f}]")

            corrected[:, axis] = smoothed

        # Build output (preserve Y from aligned_results, replace X and Z)
        new_results[pid] = {
            "smpl_params_global": {
                "global_orient": aligned_results[pid]["smpl_params_global"]["global_orient"],
                "body_pose": aligned_results[pid]["smpl_params_global"]["body_pose"],
                "transl": torch.from_numpy(corrected).float(),
                "betas": aligned_results[pid]["smpl_params_global"]["betas"],
            },
            "K_fullimg": aligned_results[pid]["K_fullimg"],
            "optimized_scale": aligned_results[pid].get("optimized_scale", 1.0),
            "transl_camera_frame": torch.from_numpy(
                (G_refined.T @ corrected.T).T
            ).float(),
        }

    del smplx_model
    torch.cuda.empty_cache()

    # --- Save ---
    save_path = out_dir / "aligned_results_xz.pt"
    torch.save(new_results, save_path)
    print(f"\n[XZ-Kalman] Saved: {save_path}")

    # --- Quick comparison ---
    for pid in sorted(new_results.keys()):
        t_old = aligned_results[pid]["smpl_params_global"]["transl"].numpy()
        t_new = new_results[pid]["smpl_params_global"]["transl"].numpy()
        diff = t_new - t_old
        print(f"  Person {pid}: XZ shift mean=({diff[:,0].mean():.4f}, {diff[:,2].mean():.4f})m, "
              f"max=({np.abs(diff[:,0]).max():.4f}, {np.abs(diff[:,2]).max():.4f})m")


if __name__ == "__main__":
    main()
