"""
Compute contact_target_pos_offset automatically from pipeline outputs.

For each agent (person), for each EEF (left/right wrist):
  1. Run SMPLX FK to get wrist joint positions (gravity frame)
  2. Transform to object-local frame via inverse ICP pose
  3. Ray-cast from mesh centroid through wrist to find contact surface point
  4. Aggregate across contact frames via component-wise median

Outputs preprocess/contact_offsets.json with per-agent offsets in mesh-local frame,
directly usable as HDMI-Multi's contact_target_pos_offset.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python ../scripts/compute_contact_offset.py \
        --output_dir outputs/demo_test/koi \
        --obj_mesh ../assets/box_real.usd
"""

import argparse
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda


def compute_hand_positions(aligned_results, pids):
    """Run SMPLX FK to get palm center positions in gravity frame.

    Uses average of four finger base joints (index1, middle1, ring1, pinky1)
    as palm center — closer to actual contact point than the wrist joint.

    Returns dict {pid: (T, 2, 3)} where dim 1 is [left_palm, right_palm].
    """
    # Finger base joint indices (SMPLX standard)
    # Average of four finger base joints (closer to object surface than wrist)
    LEFT_PALM_JOINTS = [25, 28, 34, 31]   # left index1, middle1, ring1, pinky1
    RIGHT_PALM_JOINTS = [40, 43, 49, 46]  # right index1, middle1, ring1, pinky1

    smplx_model = make_smplx("supermotion").cuda()
    hands = {}
    for pid in pids:
        params = aligned_results[pid]["smpl_params_global"]
        params_no_transl = {k: v for k, v in params.items() if k != "transl"}
        with torch.no_grad():
            out = smplx_model(**to_cuda(params_no_transl))
        pelvis = out.joints[:, [0]]  # (T, 1, 3)
        left_palm = out.joints[:, LEFT_PALM_JOINTS].mean(dim=1)   # (T, 3)
        right_palm = out.joints[:, RIGHT_PALM_JOINTS].mean(dim=1) # (T, 3)
        palm_lr = torch.stack([left_palm, right_palm], dim=1) - pelvis  # (T, 2, 3)
        transl = params["transl"].cpu()
        hands[pid] = (palm_lr.cpu() + transl.unsqueeze(1)).numpy()
    del smplx_model
    torch.cuda.empty_cache()
    return hands


def parse_contact_frames(contact_labels, pids):
    """Extract contact frame indices per person.

    Uses contact_per_person (either hand has contact → True).
    Both hands are computed for every contact frame, since when carrying
    an object both hands are always on it — 100DOH may miss one hand
    due to occlusion but the other confirms the frame.

    Returns dict {pid: [frame_indices]}
    """
    contact_per_person = contact_labels["contact_per_person"]
    result = {}
    for pid in pids:
        result[pid] = [t for t in range(len(contact_per_person[pid]))
                       if contact_per_person[pid][t]]
    return result


def raycast_contact_point(mesh_tri, mesh_centroid, hand_obj):
    """Find contact face and project wrist onto it.

    1. Ray-cast from centroid through wrist to find WHICH face.
    2. Project wrist perpendicularly onto that face's plane.

    Returns:
        (face_id, projected_point) or (None, None) if failed.
    """
    ray_dir = hand_obj - mesh_centroid
    norm = np.linalg.norm(ray_dir)
    if norm < 1e-6:
        return None, None
    ray_dir = ray_dir / norm

    locations, _, face_ids = mesh_tri.ray.intersects_location(
        ray_origins=[mesh_centroid],
        ray_directions=[ray_dir]
    )
    if len(locations) == 0:
        return None, None

    # Pick the face whose ray intersection is closest to the wrist
    dists = np.linalg.norm(locations - hand_obj, axis=1)
    best_face = face_ids[np.argmin(dists)]

    # Project wrist perpendicularly onto that face's plane
    face_normal = mesh_tri.face_normals[best_face]
    face_vertex = mesh_tri.vertices[mesh_tri.faces[best_face][0]]
    d = np.dot(hand_obj - face_vertex, face_normal)
    projected = hand_obj - d * face_normal

    return int(best_face), projected


def main():
    parser = argparse.ArgumentParser(
        description="Compute contact_target_pos_offset from pipeline outputs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="GVHMR output dir (contains aligned_results_xz.pt)")
    parser.add_argument("--obj_mesh", type=str, required=True,
                        help="Path to object mesh (.obj, .ply, .usd)")
    parser.add_argument("--fitness_thresh", type=float, default=0.3,
                        help="Skip frames with ICP fitness below this")
    parser.add_argument("--min_contact_frames", type=int, default=3,
                        help="Minimum contact frames for reliable offset")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    preprocess_dir = out_dir / "preprocess"

    # --- Load data ---
    print("[ContactOffset] Loading data...")
    aligned = torch.load(out_dir / "aligned_results_xz.pt",
                         map_location="cpu", weights_only=False)
    obj_data = torch.load(preprocess_dir / "obj_poses_scaled.pt",
                          map_location="cpu", weights_only=False)
    G = torch.load(preprocess_dir / "G_refined_scaled.pt",
                   map_location="cpu", weights_only=False).numpy()
    contact_labels = torch.load(preprocess_dir / "contact_labels.pt",
                                map_location="cpu", weights_only=False)

    pids = sorted(aligned.keys())
    obj_pos_cam = np.array(obj_data["obj_pos"])     # (T, 3) camera frame
    obj_quat_cam = np.array(obj_data["obj_quat"])   # (T, 4) xyzw, camera frame
    obj_scale = float(obj_data["obj_scale"])
    fitness = np.array(obj_data["fitness"])          # (T,)
    T = obj_pos_cam.shape[0]

    print(f"  {len(pids)} persons, {T} frames, obj_scale={obj_scale:.4f}")

    # --- Load mesh as trimesh ---
    import sys, os
    _SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_SCRIPTS_DIR))
    from track_object import load_mesh
    mesh_o3d = load_mesh(args.obj_mesh)
    mesh_verts = np.asarray(mesh_o3d.vertices)
    mesh_faces = np.asarray(mesh_o3d.triangles)
    mesh_tri = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
    mesh_centroid = mesh_tri.centroid
    print(f"  Mesh: {len(mesh_verts)} verts, centroid={mesh_centroid}")

    # --- Object poses → gravity frame ---
    R_G = Rotation.from_matrix(G)
    obj_rot_cam = Rotation.from_quat(obj_quat_cam)   # (T,)
    obj_rot_grav = R_G * obj_rot_cam                   # (T,) gravity frame
    obj_pos_grav = (G @ obj_pos_cam.T).T               # (T, 3)

    # --- SMPLX FK → wrist positions (gravity frame) ---
    print("[ContactOffset] Running SMPLX FK...")
    wrists = compute_hand_positions(aligned, pids)

    # --- Contact frames (either hand → both hands computed) ---
    contact_frames_per_person = parse_contact_frames(contact_labels, pids)
    for pid in pids:
        print(f"  Person {pid}: {len(contact_frames_per_person[pid])} contact frames (either hand)")

    # --- Ray-cast for each contact frame, both hands ---
    print("[ContactOffset] Computing contact surface points...")
    per_agent_results = {}

    for agent_idx, pid in enumerate(pids):
        palm_pos = wrists[pid]  # (T, 2, 3) gravity frame, [left_palm, right_palm]
        contact_frames = contact_frames_per_person[pid]

        hand_data = {}  # wrist_idx → {face_id: [projected_pts]}
        hand_stats = {}  # wrist_idx → (n_skipped_fitness, n_skipped_raycast)

        for wrist_idx in [0, 1]:
            face_hits = {}
            n_skipped_fitness = 0
            n_skipped_raycast = 0

            for t in contact_frames:
                if t >= T:
                    continue
                if fitness[t] < args.fitness_thresh:
                    n_skipped_fitness += 1
                    continue

                hand_grav = palm_pos[t, wrist_idx]
                R_obj_g = obj_rot_grav[t].as_matrix()
                t_obj_g = obj_pos_grav[t]
                # No obj_scale division: ICP pose maps real mesh → camera frame
                # (obj_scale is for ICP matching only, not the output pose)
                hand_obj = R_obj_g.T @ (hand_grav - t_obj_g)

                face_id, projected = raycast_contact_point(mesh_tri, mesh_centroid, hand_obj)
                if face_id is None:
                    n_skipped_raycast += 1
                    continue

                face_hits.setdefault(face_id, []).append(projected)

            hand_data[wrist_idx] = face_hits
            hand_stats[wrist_idx] = (n_skipped_fitness, n_skipped_raycast)

        # Print per-face distribution and aggregate
        agent_offsets = []
        agent_counts = {}
        for wrist_idx in [0, 1]:
            wrist_name = "left" if wrist_idx == 0 else "right"
            face_hits = hand_data[wrist_idx]
            n_sf, n_sr = hand_stats[wrist_idx]
            total_hits = sum(len(v) for v in face_hits.values())
            print(f"  Agent {agent_idx} (pid={pid}), {wrist_name}: "
                  f"{total_hits} valid, {n_sf} low-fitness, {n_sr} raycast-fail")
            for fid, pts_list in sorted(face_hits.items(), key=lambda x: -len(x[1])):
                n = mesh_tri.face_normals[fid]
                print(f"    face {fid}: {len(pts_list)} hits ({100*len(pts_list)/max(total_hits,1):.0f}%), "
                      f"normal=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})")

            # Majority vote: median only from dominant face
            if face_hits:
                dominant_face = max(face_hits, key=lambda k: len(face_hits[k]))
                pts = np.array(face_hits[dominant_face])
            else:
                pts = np.zeros((0, 3))
                dominant_face = None

            if len(pts) >= args.min_contact_frames:
                median_pt = np.median(pts, axis=0)
                std_pt = np.std(pts, axis=0)
                snapped, dist, _ = trimesh.proximity.closest_point(mesh_tri, [median_pt])
                final_pt = snapped[0]
                print(f"    → face {dominant_face} ({len(pts)}/{total_hits}): "
                      f"median=({median_pt[0]:+.4f},{median_pt[1]:+.4f},{median_pt[2]:+.4f}) "
                      f"std=({std_pt[0]:.4f},{std_pt[1]:.4f},{std_pt[2]:.4f}) "
                      f"snap_dist={dist[0]:.4f}")
                agent_offsets.append(final_pt.tolist())
                agent_counts[wrist_name] = int(len(pts))
            else:
                print(f"    → WARNING only {len(pts)} frames, setting to [0,0,0]")
                agent_offsets.append([0.0, 0.0, 0.0])
                agent_counts[wrist_name] = int(len(pts))

        per_agent_results[f"agent_{agent_idx}"] = {
            "contact_target_pos_offset": agent_offsets,
            "num_contact_frames": agent_counts,
            "person_id": int(pid),
        }

    # --- Save ---
    save_path = preprocess_dir / "contact_offsets.json"
    with open(save_path, "w") as f:
        json.dump(per_agent_results, f, indent=2)
    print(f"\n[ContactOffset] Saved: {save_path}")
    for agent_key, data in per_agent_results.items():
        print(f"  {agent_key}: {data['contact_target_pos_offset']}")


if __name__ == "__main__":
    main()
