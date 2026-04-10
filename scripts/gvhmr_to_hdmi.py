"""
End-to-end: GVHMR aligned_results.pt → GMR retargeting → HDMI motion format.

Supports both single-person (hmr4d_results.pt) and multi-person (aligned_results.pt).
Each person is retargeted to the same robot and saved as agent_N/ in HDMI format.

Optionally adds object contact + 6DoF trajectory from auto-detection pipeline.

Usage:
    cd HDMI-Multi-datapipeline
    python scripts/gvhmr_to_hdmi.py \
        --gvhmr_pred_file GVHMR/outputs/demo_multi/video_name/aligned_results.pt \
        --robot unitree_g1 \
        --output_dir output/video_name_g1 \
        --auto_contact GVHMR/outputs/.../contact_labels.pt \
        --obj_pose_file GVHMR/outputs/.../preprocess/obj_poses.pt
"""

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Add submodule paths
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
GMR_ROOT = REPO_ROOT / "GMR"
GVHMR_ROOT = REPO_ROOT / "GVHMR"

sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, str(GMR_ROOT / "scripts"))  # for importing GMR_pkl_to_HDMI_npz

# ---------------------------------------------------------------------------
# GMR imports
# ---------------------------------------------------------------------------
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_gvhmr_pred_file,
    get_gvhmr_data_offline_fast,
)
from GMR_pkl_to_HDMI_npz import (
    HDMI_BODY_NAMES,
    ROBOT_XML_PATHS,
    compute_body_poses_from_qpos,
    filter_and_reorder_bodies,
    _finite_diff_first_order,
    _angular_velocity_from_quat,
)

# ---------------------------------------------------------------------------
# HDMI joint names (matches HDMI_BODY_NAMES ordering minus root for joints)
# ---------------------------------------------------------------------------
HDMI_JOINT_NAMES = {
    "g1": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ],
    "pm01": [
        "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
        "J12_WAIST_YAW",
        "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
        "J23_HEAD_YAW",
    ],
}

# GMR robot name → our robot key
ROBOT_KEY_MAP = {
    "unitree_g1": "g1",
    "engineai_pm01": "pm01",
}

# Semantic body role → body name per robot (for object contact computation)
ROBOT_BODY_ROLES = {
    "g1": {
        "left_ankle":  "left_ankle_roll_link",
        "right_ankle": "right_ankle_roll_link",
        "left_eef":    "left_wrist_yaw_link",
        "right_eef":   "right_wrist_yaw_link",
    },
    "pm01": {
        "left_ankle":  "LINK_ANKLE_ROLL_L",
        "right_ankle": "LINK_ANKLE_ROLL_R",
        "left_eef":    "LINK_ELBOW_YAW_L",
        "right_eef":   "LINK_ELBOW_YAW_R",
    },
}


def retarget_one_person(pred_file, smplx_folder, robot_name, visualize=False,
                        loop=False, record_video=False, rate_limit=False, video_path=None):
    """Retarget one person's GVHMR output to robot. Returns GMR pkl dict."""

    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        pred_file, smplx_folder
    )

    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot_name,
    )

    viewer = None
    if visualize:
        viewer = RobotMotionViewer(
            robot_type=robot_name,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=record_video,
            video_path=video_path or f"videos/{robot_name}.mp4",
        )

    qpos_list = []
    i = -1
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        if loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        smplx_data = smplx_data_frames[i]
        qpos = retarget.retarget(smplx_data)
        qpos_list.append(qpos)

        if viewer is not None:
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 2.0:
                print(f"  Rendering FPS: {fps_counter / (current_time - fps_start_time):.1f}")
                fps_counter = 0
                fps_start_time = current_time

            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=rate_limit,
            )

    if viewer is not None:
        viewer.close()

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz → xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])

    return {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def pkl_to_hdmi_npz(pkl_data, robot_key):
    """Convert GMR pkl dict to HDMI npz arrays."""

    root_pos = pkl_data["root_pos"]
    root_rot = pkl_data["root_rot"]  # xyzw
    dof_pos = pkl_data["dof_pos"]
    fps = pkl_data["fps"]
    dt = 1.0 / fps

    # Filter DOF if G1 with hands (43 → 29)
    if dof_pos.shape[1] == 43:
        dof_pos = np.concatenate([dof_pos[:, :22], dof_pos[:, 29:36]], axis=1)

    # FK → body poses
    body_pos_w, body_quat_w, gmr_body_names = compute_body_poses_from_qpos(
        root_pos, root_rot, dof_pos, ROBOT_XML_PATHS[robot_key]
    )

    # Filter to HDMI bodies
    body_pos_w, body_quat_w = filter_and_reorder_bodies(
        body_pos_w, body_quat_w, gmr_body_names, HDMI_BODY_NAMES[robot_key]
    )

    # Velocities
    body_lin_vel_w = _finite_diff_first_order(body_pos_w, dt)
    body_ang_vel_w = _angular_velocity_from_quat(body_quat_w, dt)
    joint_vel = _finite_diff_first_order(dof_pos, dt)

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_pos": dof_pos,
        "joint_vel": joint_vel,
    }


def save_agent(agent_dir, npz_data, body_names, joint_names, fps):
    """Save one agent's HDMI data."""
    os.makedirs(agent_dir, exist_ok=True)

    np.savez(os.path.join(agent_dir, "motion.npz"), **npz_data)

    with open(os.path.join(agent_dir, "meta.json"), "w") as f:
        json.dump({"body_names": body_names, "joint_names": joint_names, "fps": fps}, f, indent=2)



def append_object_to_npz(npz_data, obj_data):
    """Append one object body to npz_data arrays and add contact."""
    T = npz_data['body_pos_w'].shape[0]
    npz_data['body_pos_w'] = np.concatenate([npz_data['body_pos_w'], obj_data['obj_pos'][:T, None, :]], axis=1)
    npz_data['body_quat_w'] = np.concatenate([npz_data['body_quat_w'], obj_data['obj_quat'][:T, None, :]], axis=1)
    npz_data['body_lin_vel_w'] = np.concatenate([npz_data['body_lin_vel_w'], obj_data['obj_lin_vel'][:T, None, :]], axis=1)
    npz_data['body_ang_vel_w'] = np.concatenate([npz_data['body_ang_vel_w'], obj_data['obj_ang_vel'][:T, None, :]], axis=1)
    npz_data['object_contact'] = obj_data['obj_contact'][:T]
    return npz_data


def main():
    parser = argparse.ArgumentParser(description="GVHMR → GMR → HDMI end-to-end pipeline")
    parser.add_argument("--gvhmr_pred_file", type=str, required=True,
                        help="Path to aligned_results.pt or hmr4d_results.pt")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", "tienkung"])
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for HDMI format (agent_0/, agent_1/, ...)")
    parser.add_argument("--save_pkl", action="store_true",
                        help="Also save intermediate GMR pkl files")
    parser.add_argument("--auto_contact", type=str, default=None,
                        help="Path to auto-detected contact_labels.pt (from detect_contact.py)")
    parser.add_argument("--obj_pose_file", type=str, default=None,
                        help="Path to obj_poses.pt from track_object.py (auto 6DoF tracking)")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    args = parser.parse_args()

    robot_key = ROBOT_KEY_MAP.get(args.robot, args.robot)
    if robot_key not in HDMI_BODY_NAMES:
        print(f"Error: robot '{robot_key}' not in HDMI_BODY_NAMES. Available: {list(HDMI_BODY_NAMES.keys())}")
        sys.exit(1)

    smplx_folder = GMR_ROOT / "assets" / "body_models"

    # Load and detect format
    raw = torch.load(args.gvhmr_pred_file, weights_only=False)
    if "smpl_params_global" in raw:
        persons = {0: raw}
    else:
        persons = {pid: raw[pid] for pid in sorted(raw.keys())}
    print(f"[gvhmr_to_hdmi] {len(persons)} person(s), robot={args.robot}")

    # Write per-person temp files for load_gvhmr_pred_file
    import tempfile
    temp_files = {}
    for pid, pdata in persons.items():
        if isinstance(args.gvhmr_pred_file, str) and len(persons) == 1 and "smpl_params_global" in raw:
            temp_files[pid] = args.gvhmr_pred_file
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=f"_p{pid}.pt", delete=False)
            torch.save(pdata, tmp.name)
            temp_files[pid] = tmp.name

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1 & 2: Retarget and convert all agents (keep in memory)
    all_npz_data = []
    all_fps = []

    for agent_idx, (pid, pred_file) in enumerate(temp_files.items()):
        print(f"\n=== Person {pid} (agent_{agent_idx}) ===")

        pkl_data = retarget_one_person(
            pred_file, str(smplx_folder), args.robot,
            visualize=True,
            loop=args.loop,
            record_video=args.record_video,
            rate_limit=args.rate_limit,
            video_path=os.path.join(args.output_dir, f"agent_{agent_idx}_{args.robot}.mp4"),
        )
        print(f"  Retargeted: {pkl_data['root_pos'].shape[0]} frames, {pkl_data['dof_pos'].shape[1]} DOF")

        if args.save_pkl:
            pkl_path = os.path.join(args.output_dir, f"agent_{agent_idx}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(pkl_data, f)
            print(f"  Saved pkl: {pkl_path}")

        npz_data = pkl_to_hdmi_npz(pkl_data, robot_key)
        print(f"  FK done: body_pos_w={npz_data['body_pos_w'].shape}")

        all_npz_data.append(npz_data)
        all_fps.append(int(pkl_data["fps"]))

    # ===================================================================
    # Step 3a: Contact information (binary labels from 100DOH auto-detection)
    # ===================================================================
    body_names = list(HDMI_BODY_NAMES[robot_key])  # copy so we can append object names

    if args.auto_contact:
        auto_labels = torch.load(args.auto_contact, map_location='cpu', weights_only=False)
        contact_per_person = auto_labels["contact_per_person"]
        video_fps = auto_labels.get("fps", 30.0)
        motion_fps = all_fps[0]
        T_motion = all_npz_data[0]["body_pos_w"].shape[0]

        person_ids = sorted(persons.keys())
        for agent_idx, pid in enumerate(person_ids):
            if pid in contact_per_person:
                contact_arr = contact_per_person[pid].numpy().astype(bool)
            else:
                print(f"  Warning: No contact data for person {pid}, defaulting to False")
                contact_arr = np.zeros(T_motion, dtype=bool)

            # Resample if video fps != motion fps
            if abs(video_fps - motion_fps) > 0.5:
                indices = np.round(np.arange(T_motion) * video_fps / motion_fps).astype(int)
                indices = np.clip(indices, 0, len(contact_arr) - 1)
                contact_arr = contact_arr[indices]
            else:
                contact_arr = contact_arr[:T_motion]

            all_npz_data[agent_idx]['object_contact'] = contact_arr.reshape(-1, 1)
            n_contact = int(contact_arr.sum())
            print(f"  Agent {agent_idx} (pid {pid}): {n_contact}/{T_motion} contact frames ({n_contact/T_motion*100:.1f}%)")

    # ===================================================================
    # Step 3b: Object 6DoF trajectory (from track_object.py ICP tracking)
    #   Coordinate transform: camera/SLAM frame → gravity Y-up → Z-up → pelvis-scaled
    # ===================================================================
    if args.obj_pose_file:
        from scipy.spatial.transform import Rotation as Rot
        import json as _json

        obj_data_raw = torch.load(args.obj_pose_file, map_location='cpu', weights_only=False)
        T_motion = all_npz_data[0]["body_pos_w"].shape[0]
        obj_fps = obj_data_raw.get("fps", 30.0)
        motion_fps = all_fps[0]

        # Resample helper
        def _resample(arr, src_fps, dst_fps, T_out):
            arr = np.array(arr)
            if abs(src_fps - dst_fps) > 0.5:
                indices = np.round(np.arange(T_out) * src_fps / dst_fps).astype(int)
                indices = np.clip(indices, 0, len(arr) - 1)
                return arr[indices]
            arr_out = arr[:T_out]
            if len(arr_out) < T_out:
                arr_out = np.pad(arr_out, [(0, T_out - len(arr_out))] + [(0, 0)] * (arr_out.ndim - 1), mode='edge')
            return arr_out

        obj_pos = _resample(obj_data_raw["obj_pos"], obj_fps, motion_fps, T_motion)
        obj_quat_xyzw = _resample(obj_data_raw["obj_quat"], obj_fps, motion_fps, T_motion)
        obj_lin_vel = _resample(obj_data_raw["obj_lin_vel"], obj_fps, motion_fps, T_motion)
        obj_ang_vel = _resample(obj_data_raw["obj_ang_vel"], obj_fps, motion_fps, T_motion)

        # --- Load G_refined (gravity rotation: camera frame → Y-up) ---
        gvhmr_dir = pathlib.Path(args.gvhmr_pred_file).parent
        G_refined_path = gvhmr_dir / "preprocess" / "G_refined.pt"
        if not G_refined_path.exists():
            # Try one level up (if gvhmr_pred_file is in preprocess/person_X/)
            G_refined_path = gvhmr_dir.parent / "G_refined.pt"
        if not G_refined_path.exists():
            print(f"[gvhmr_to_hdmi] ERROR: G_refined.pt not found. Run 04_align.py first.")
            print(f"  Searched: {gvhmr_dir / 'preprocess' / 'G_refined.pt'}")
            sys.exit(1)
        G_refined = torch.load(G_refined_path, map_location='cpu').numpy().astype(np.float64)
        print(f"\n  Loaded G_refined from {G_refined_path}")

        # --- Step 1: G_refined rotation (camera/SLAM frame → gravity Y-up) ---
        obj_pos = (G_refined @ obj_pos.T).T
        obj_lin_vel = (G_refined @ obj_lin_vel.T).T
        obj_ang_vel = (G_refined @ obj_ang_vel.T).T
        R_G = Rot.from_matrix(G_refined)
        obj_rots = Rot.from_quat(obj_quat_xyzw)
        obj_quat_xyzw = (R_G * obj_rots).as_quat()

        # --- Step 2: Y-up → Z-up (same rotation as GMR's get_gvhmr_data_offline_fast) ---
        R_y2z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        obj_pos = obj_pos @ R_y2z.T
        obj_lin_vel = obj_lin_vel @ R_y2z.T
        obj_ang_vel = obj_ang_vel @ R_y2z.T
        R_y2z_rot = Rot.from_matrix(R_y2z)
        obj_quat_xyzw = (R_y2z_rot * Rot.from_quat(obj_quat_xyzw)).as_quat()

        # --- Convert quaternion xyzw → wxyz (HDMI convention) ---
        obj_quat_wxyz = np.zeros_like(obj_quat_xyzw)
        obj_quat_wxyz[:, 0] = obj_quat_xyzw[:, 3]  # w
        obj_quat_wxyz[:, 1:] = obj_quat_xyzw[:, :3]  # xyz

        # --- Step 3: pelvis_scale (same as GMR's scale_human_data on root) ---
        from general_motion_retargeting.params import IK_CONFIG_DICT
        ik_cfg_path = IK_CONFIG_DICT["smplx"][args.robot]
        with open(ik_cfg_path) as f:
            ik_cfg = _json.load(f)
        base_pelvis_scale = ik_cfg["human_scale_table"]["pelvis"]
        assumed_height = ik_cfg["human_height_assumption"]

        first_pid = sorted(persons.keys())[0]
        betas = persons[first_pid]["smpl_params_global"]["betas"]
        beta0 = float(betas[0, 0]) if betas.dim() == 2 else float(betas[0])
        actual_height = 1.66 + 0.1 * beta0
        pelvis_scale = base_pelvis_scale * (actual_height / assumed_height)
        print(f"  Pelvis scale: {base_pelvis_scale} × ({actual_height:.3f}/{assumed_height}) = {pelvis_scale:.4f}")

        obj_pos = pelvis_scale * obj_pos
        obj_lin_vel = pelvis_scale * obj_lin_vel
        # ang_vel is radians/sec, scale-invariant

        obj_name = pathlib.Path(obj_data_raw.get("obj_mesh_path", "object")).stem
        for agent_idx, npz_data in enumerate(all_npz_data):
            existing_contact = npz_data.get('object_contact', np.zeros((T_motion, 1)))
            obj_data = {
                "obj_pos": obj_pos,
                "obj_quat": obj_quat_wxyz,
                "obj_lin_vel": obj_lin_vel,
                "obj_ang_vel": obj_ang_vel,
                "obj_contact": existing_contact,
            }
            append_object_to_npz(npz_data, obj_data)

        body_names.append(obj_name)
        print(f"\n  Object 6DoF trajectory: '{obj_name}'")

    else:
        # Warn if contact exists but no trajectory
        has_contact = any('object_contact' in npz and npz['object_contact'].any() for npz in all_npz_data)
        if has_contact:
            print("\n  Warning: Contact labels exist but no --obj_pose_file provided.")
            print("    Object trajectory will NOT be in motion.npz.")
            print("    Run track_object.py to generate obj_poses.pt.")

    # Step 2.5: Ground offset — shift all bodies (including object) so global lowest Z = 0
    global_min_z = min(npz['body_pos_w'][:, :, 2].min() for npz in all_npz_data)
    if abs(global_min_z) > 1e-4:
        print(f"\n  Ground offset: shifting Z by {-global_min_z:+.4f}m (lowest body Z was {global_min_z:.4f}m)")
        for npz in all_npz_data:
            npz['body_pos_w'][:, :, 2] -= global_min_z

    # Step 4: Save all agents
    for agent_idx, npz_data in enumerate(all_npz_data):
        agent_dir = os.path.join(args.output_dir, f"agent_{agent_idx}")
        save_agent(
            agent_dir, npz_data,
            body_names=body_names,
            joint_names=HDMI_JOINT_NAMES[robot_key],
            fps=all_fps[agent_idx],
        )
        print(f"  Saved HDMI: {agent_dir}/")

    # Cleanup temp files
    for pid, path in temp_files.items():
        if path != args.gvhmr_pred_file:
            os.unlink(path)

    print(f"\n[gvhmr_to_hdmi] Done → {args.output_dir}")


if __name__ == "__main__":
    main()
