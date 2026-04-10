"""
Re-run human Y-axis grounding using depth-scaled MoGe point cloud.

The original scene_grounding uses unscaled MoGe for RANSAC plane fitting.
This script scales the point cloud by depth_scale first, so ground planes
are in the same metric space as the scaled ICP object tracking.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/run_scaled_grounding.py \
        --output_dir outputs/demo_test/siheng_dual
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from hmr4d.utils.multihuman.scene_grounding import align_with_scene_grounding


def main():
    parser = argparse.ArgumentParser(description="Re-run grounding on scaled MoGe")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load depth_scale ---
    with open(preprocess_dir / "depth_scale.json") as f:
        depth_scale = json.load(f)["depth_scale"]
    print(f"[ScaledGround] depth_scale = {depth_scale:.4f}")

    # --- Load scene_data and scale pts3d ---
    print("[ScaledGround] Loading scene_data...")
    scene_data = torch.load(preprocess_dir / "scene_data.pt", map_location="cpu")
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]

    print(f"[ScaledGround] Scaling pts3d by {depth_scale:.4f}...")
    for fn in frame_names:
        pts3d_dict[fn] = pts3d_dict[fn] * depth_scale
    # normals unchanged (direction-only, scale-invariant)

    # --- Load per-person GVHMR results ---
    per_person_results = {}
    for pd in sorted(preprocess_dir.glob("person_*")):
        pid = int(pd.name.split("_")[1])
        per_person_results[pid] = torch.load(pd / "hmr4d_results.pt", map_location="cpu")
    print(f"[ScaledGround] {len(per_person_results)} person(s)")

    # --- Run scene grounding on scaled point cloud ---
    # This is the same function as 04_align.py uses, but with scaled scene_data
    print("[ScaledGround] Running scene grounding on scaled MoGe...")
    aligned_results, G_refined = align_with_scene_grounding(
        per_person_results, scene_data)

    # --- Save ---
    # Save aligned results (Y-corrected with scaled ground planes)
    save_path = out_dir / "aligned_results_scaled.pt"
    torch.save(aligned_results, save_path)
    print(f"[ScaledGround] Saved: {save_path}")

    # Save G_refined (might differ slightly from original due to scaled points)
    g_path = preprocess_dir / "G_refined_scaled.pt"
    torch.save(torch.from_numpy(G_refined).float(), g_path)
    print(f"[ScaledGround] Saved: {g_path}")

    # --- Compare with original ---
    ar_old = torch.load(out_dir / "aligned_results.pt", map_location="cpu")
    obj = torch.load(preprocess_dir / "obj_poses_scaled.pt", map_location="cpu")
    G_old = torch.load(preprocess_dir / "G_refined.pt", map_location="cpu").numpy()

    obj_grav_old = (G_old @ obj["obj_pos"].T).T
    obj_grav_new = (G_refined @ obj["obj_pos"].T).T

    print("\n[ScaledGround] Y-axis comparison (gravity frame):")
    print("Frame   H1_old_Y  H1_new_Y  H2_old_Y  H2_new_Y  Box_old_Y  Box_new_Y")
    for f in [0, 100, 200, 300, 400, 500, 640]:
        pids = sorted(aligned_results.keys())
        h1_old_y = ar_old[pids[0]]["smpl_params_global"]["transl"][f, 1].item()
        h1_new_y = aligned_results[pids[0]]["smpl_params_global"]["transl"][f, 1].item()
        h2_old_y = ar_old[pids[1]]["smpl_params_global"]["transl"][f, 1].item()
        h2_new_y = aligned_results[pids[1]]["smpl_params_global"]["transl"][f, 1].item()
        box_old_y = obj_grav_old[f, 1]
        box_new_y = obj_grav_new[f, 1]
        print(f"  {f:5d}  {h1_old_y:8.3f}  {h1_new_y:8.3f}  {h2_old_y:8.3f}  "
              f"{h2_new_y:8.3f}  {box_old_y:9.3f}  {box_new_y:9.3f}")

    # Check if box and human Y are now closer
    print("\n[ScaledGround] Y gap (human avg - box):")
    for label, ar, obj_grav in [("BEFORE (unscaled)", ar_old, obj_grav_old),
                                 ("AFTER (scaled)", aligned_results, obj_grav_new)]:
        gaps = []
        for f in range(0, 641, 10):
            h_avg_y = 0.5 * (ar[pids[0]]["smpl_params_global"]["transl"][f, 1].item() +
                             ar[pids[1]]["smpl_params_global"]["transl"][f, 1].item())
            gaps.append(h_avg_y - obj_grav[f, 1])
        gaps = np.array(gaps)
        print(f"  {label}: mean={gaps.mean():.3f}m, std={gaps.std():.3f}m, "
              f"range=[{gaps.min():.3f}, {gaps.max():.3f}]")


if __name__ == "__main__":
    main()
