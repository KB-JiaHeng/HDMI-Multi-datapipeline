"""
v5: Fixes from audit agent:
  1. Centroid-distance clip (replaces Z-only clip that cuts off tilted surfaces)
  2. Adaptive max_corr_dist (widens during fast motion)
  3. Temporal coherence in reinit (biases toward previous orientation)
  4. Intermediate pitch angles in full_rotation_init (covers 45° transitions)

Usage:
    cd /home/sihengzhao/research/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/vis_icp_v5.py
"""

import numpy as np
import torch
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter

# ── Paths ──
OUT_DIR = Path("/home/sihengzhao/research/GVHMR/outputs/demo_test/siheng_dual")
MESH_PATH = Path("/home/sihengzhao/research/HDMI-Multi-datapipeline/assets/box_real.obj")
SCENE_DATA_PATH = OUT_DIR / "preprocess" / "scene_data.pt"
MASK_DIR = OUT_DIR / "preprocess" / "masks" / "object_mask_data"
VIDEO_PATH = OUT_DIR / "0_input_video.mp4"
OUTPUT_VIDEO = OUT_DIR / "vis_icp_v5.mp4"
OUTPUT_VIDEO_SMOOTH = OUT_DIR / "vis_icp_v5_smooth.mp4"

# Params
SCALE = 0.7152
CENTROID_CLIP_RADIUS = 0.40  # max distance from centroid (box diagonal ~0.55m * 0.7152 ≈ 0.39m)
VOXEL_SIZE = 0.005
TUKEY_K = 0.10
MAX_CORR_INIT = 0.08
MAX_CORR_TRACK_BASE = 0.05
MAX_CORR_TRACK_MAX = 0.15   # upper bound for adaptive expansion
REINIT_THRESHOLD = 0.3
IOU_REINIT_THRESHOLD = 0.3
N_MODEL_POINTS = 5000
TEMPORAL_BIAS_WEIGHT = 0.3   # weight for angular proximity in reinit scoring


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    return mesh


def sample_model_pcd(mesh, scale):
    verts = np.asarray(mesh.vertices) * scale
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.compute_vertex_normals()
    pcd = m.sample_points_uniformly(number_of_points=N_MODEL_POINTS)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def extract_frame_pts(pts3d_dict, normals_dict, frame_name, mask):
    pts = np.array(pts3d_dict[frame_name], dtype=np.float64)
    nrm = np.array(normals_dict[frame_name], dtype=np.float64)
    pts_m = pts[mask]
    nrm_m = nrm[mask]
    finite = np.isfinite(pts_m).all(axis=1) & np.isfinite(nrm_m).all(axis=1)
    return pts_m[finite], nrm_m[finite]


# FIX 1: Centroid-distance clip instead of Z-only clip
def centroid_clip(pts, norms, radius=CENTROID_CLIP_RADIUS):
    """Clip points by distance from centroid. View-independent, works for any tilt."""
    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    keep = dists <= radius
    return pts[keep], norms[keep]


def make_target(pts, norms):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(norms)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def angular_distance(T1, T2):
    """Angular distance between two poses (in radians)."""
    R_diff = T1[:3, :3] @ T2[:3, :3].T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


# FIX 4: Include intermediate pitch angles (30°, 45°, 60°)
def full_rotation_init(model_pcd, target_pcd, T_prev=None, n_yaws_per_face=8):
    """Full-rotation reinit with intermediate pitch angles.

    Searches 6 cardinal face orientations + 8 intermediate tilts = 14 orientations,
    each × n_yaws_per_face in-plane rotations = 112 candidates.

    If T_prev is provided (FIX 3), scores are biased toward the previous orientation
    for temporal coherence.
    """
    mc = model_pcd.get_center()
    tc = target_pcd.get_center()

    # Cardinal face rotations (6)
    face_rotations = [
        Rotation.from_euler('y', 90, degrees=True),
        Rotation.from_euler('y', -90, degrees=True),
        Rotation.from_euler('x', -90, degrees=True),
        Rotation.from_euler('x', 90, degrees=True),
        Rotation.identity(),
        Rotation.from_euler('y', 180, degrees=True),
    ]

    # Intermediate tilts: 45° around X and Y (covers the transition angles)
    intermediate_rotations = [
        Rotation.from_euler('x', 45, degrees=True),
        Rotation.from_euler('x', -45, degrees=True),
        Rotation.from_euler('y', 45, degrees=True),
        Rotation.from_euler('y', -45, degrees=True),
        Rotation.from_euler('x', 135, degrees=True),
        Rotation.from_euler('x', -135, degrees=True),
        Rotation.from_euler('y', 135, degrees=True),
        Rotation.from_euler('y', -135, degrees=True),
    ]

    all_rotations = face_rotations + intermediate_rotations

    best_score, best_fit, best_T = -1, -1, np.eye(4)

    for R_face in all_rotations:
        for j in range(n_yaws_per_face):
            yaw = 2.0 * np.pi * j / n_yaws_per_face
            R_yaw = Rotation.from_euler('z', yaw)
            R_total = (R_yaw * R_face).as_matrix()

            T_init = np.eye(4)
            T_init[:3, :3] = R_total
            T_init[:3, 3] = tc - R_total @ mc

            res = o3d.pipelines.registration.registration_icp(
                model_pcd, target_pcd, MAX_CORR_INIT, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=TUKEY_K)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

            # FIX 3: Temporal coherence — bias toward previous orientation
            if T_prev is not None and res.fitness > 0:
                ang_dist = angular_distance(res.transformation, T_prev)
                # Proximity score: 1.0 at 0°, 0.0 at 180°
                proximity = 1.0 - ang_dist / np.pi
                score = (1.0 - TEMPORAL_BIAS_WEIGHT) * res.fitness + TEMPORAL_BIAS_WEIGHT * proximity
            else:
                score = res.fitness

            if score > best_score:
                best_score = score
                best_fit = res.fitness
                best_T = res.transformation

    return best_T, best_fit


def track_frame(model_pcd, target_pcd, T_init, max_corr_dist=MAX_CORR_TRACK_BASE):
    res = o3d.pipelines.registration.registration_icp(
        model_pcd, target_pcd, max_corr_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=TUKEY_K)),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    return res.transformation, res.fitness


def predict_pose(T_prev, T_prev2):
    """Constant-velocity prediction."""
    delta_T = T_prev @ np.linalg.inv(T_prev2)
    T_predicted = delta_T @ T_prev
    return T_predicted


# FIX 2: Compute motion magnitude for adaptive correspondence distance
def compute_motion_magnitude(T_prev, T_prev2):
    """Compute translation + rotation magnitude between two poses."""
    delta_T = T_prev @ np.linalg.inv(T_prev2)
    trans_mag = np.linalg.norm(delta_T[:3, 3])
    ang_mag = angular_distance(T_prev, T_prev2)
    # Convert angular motion to approximate surface displacement
    # For an object of ~0.5m radius, 1 radian = 0.5m surface displacement
    return trans_mag + 0.5 * ang_mag


def adaptive_corr_dist(motion_mag):
    """KISS-ICP style: widen correspondence distance when motion is large."""
    # Base + proportional to motion, capped at MAX
    return min(MAX_CORR_TRACK_BASE + 2.0 * motion_mag, MAX_CORR_TRACK_MAX)


def compute_silhouette_iou(mesh_verts, T, K, scale, mask_gt, H, W):
    verts_s = mesh_verts * scale
    verts_h = np.hstack([verts_s, np.ones((len(verts_s), 1))])
    verts_w = (T @ verts_h.T).T[:, :3]
    depths = verts_w[:, 2]
    if (depths <= 0).all():
        return 0.0
    pts_2d = (K @ verts_w.T).T
    pts_2d = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)
    valid = depths > 0
    pts_2d_valid = pts_2d[valid].astype(np.float32)
    if len(pts_2d_valid) < 3:
        return 0.0
    pts_2d_valid[:, 0] = np.clip(pts_2d_valid[:, 0], -W, 2 * W)
    pts_2d_valid[:, 1] = np.clip(pts_2d_valid[:, 1], -H, 2 * H)
    try:
        hull = ConvexHull(pts_2d_valid)
        hull_pts = pts_2d_valid[hull.vertices].astype(np.int32)
    except Exception:
        return 0.0
    mask_rendered = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask_rendered, hull_pts.reshape(-1, 1, 2), 1)
    inter = np.logical_and(mask_rendered > 0, mask_gt > 0).sum()
    union = np.logical_or(mask_rendered > 0, mask_gt > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def draw_wireframe(frame_bgr, mesh_verts, mesh_tris, T, K, scale, color=(0, 255, 0)):
    verts_s = mesh_verts * scale
    verts_h = np.hstack([verts_s, np.ones((len(verts_s), 1))])
    verts_w = (T @ verts_h.T).T[:, :3]
    depths = verts_w[:, 2]
    pts_2d = (K @ verts_w.T).T
    pts_2d = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)
    edges = set()
    for tri in mesh_tris:
        for i in range(3):
            edges.add(tuple(sorted([tri[i], tri[(i + 1) % 3]])))
    for i, j in edges:
        if depths[i] > 0 and depths[j] > 0:
            p1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
            p2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
            cv2.line(frame_bgr, p1, p2, color, 2)


def main():
    print("Loading data ...")
    mesh = load_mesh(str(MESH_PATH))
    mesh_verts = np.asarray(mesh.vertices)
    mesh_tris = np.asarray(mesh.triangles)
    model_pcd = sample_model_pcd(mesh, SCALE)

    scene_data = torch.load(SCENE_DATA_PATH, weights_only=False)
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]
    normals_dict = scene_data[6]
    im_K_dict = scene_data[3]
    K = np.array(im_K_dict[frame_names[0]], dtype=np.float64)
    F = len(frame_names)

    from hmr4d.utils.video_io_utils import read_video_np
    frames_rgb = read_video_np(str(VIDEO_PATH))
    H, W = frames_rgb[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, 30, (W, H))

    print(f"Tracking {F} frames with v5 fixes ...")

    poses = [None] * F
    fitness_arr = np.zeros(F)
    iou_arr = np.zeros(F)
    n_init, n_track, n_skip, n_iou_reinit = 0, 0, 0, 0

    for fidx in range(F):
        frame_bgr = cv2.cvtColor(frames_rgb[fidx], cv2.COLOR_RGB2BGR)

        mask_path = MASK_DIR / f"mask_{fidx:05d}.npz"
        skip = False
        mask_gt = None
        if not mask_path.exists():
            skip = True
        else:
            mask_gt = np.load(mask_path)["mask"].astype(bool)
            if not mask_gt.any():
                skip = True

        if skip:
            if fidx > 0 and poses[fidx - 1] is not None:
                poses[fidx] = poses[fidx - 1]
                draw_wireframe(frame_bgr, mesh_verts, mesh_tris, poses[fidx], K, SCALE, (0, 200, 0))
            n_skip += 1
            writer.write(frame_bgr)
            continue

        pts, norms = extract_frame_pts(pts3d_dict, normals_dict, frame_names[fidx], mask_gt)
        if len(pts) < 50:
            if fidx > 0 and poses[fidx - 1] is not None:
                poses[fidx] = poses[fidx - 1]
                draw_wireframe(frame_bgr, mesh_verts, mesh_tris, poses[fidx], K, SCALE, (0, 200, 0))
            n_skip += 1
            writer.write(frame_bgr)
            continue

        # FIX 1: Centroid-distance clip instead of Z-only clip
        pts_c, norms_c = centroid_clip(pts, norms)
        if len(pts_c) < 50:
            pts_c, norms_c = pts, norms
        target = make_target(pts_c, norms_c)
        if len(target.points) < 50:
            if fidx > 0 and poses[fidx - 1] is not None:
                poses[fidx] = poses[fidx - 1]
                draw_wireframe(frame_bgr, mesh_verts, mesh_tris, poses[fidx], K, SCALE, (0, 200, 0))
            n_skip += 1
            writer.write(frame_bgr)
            continue

        # ── Decide: init or track ──
        need_init = (poses[fidx - 1] is None) if fidx > 0 else True
        if not need_init and fitness_arr[max(fidx - 1, 0)] < REINIT_THRESHOLD:
            need_init = True

        mode = "?"
        if need_init:
            T_prev_for_bias = poses[fidx - 1] if fidx > 0 else None
            T_result, fit = full_rotation_init(model_pcd, target, T_prev=T_prev_for_bias)
            mode = "INIT"
            n_init += 1
        else:
            T_prev = poses[fidx - 1]
            T_prev2 = poses[fidx - 2] if fidx >= 2 and poses[fidx - 2] is not None else None

            if T_prev2 is not None:
                T_predicted = predict_pose(T_prev, T_prev2)
                # FIX 2: Adaptive correspondence distance
                motion_mag = compute_motion_magnitude(T_prev, T_prev2)
                corr_dist = adaptive_corr_dist(motion_mag)
            else:
                T_predicted = T_prev
                corr_dist = MAX_CORR_TRACK_BASE

            T_result, fit = track_frame(model_pcd, target, T_predicted, max_corr_dist=corr_dist)
            mode = f"VEL d={corr_dist:.3f}" if T_prev2 is not None else "TRACK"
            n_track += 1

        # ── Silhouette IoU gating ──
        iou = compute_silhouette_iou(mesh_verts, T_result, K, SCALE, mask_gt, H, W)

        if iou < IOU_REINIT_THRESHOLD and mode != "INIT":
            # FIX 3: Temporal coherence — pass T_prev to bias reinit
            T_prev_for_bias = poses[fidx - 1] if fidx > 0 else None
            T_reinit, fit_reinit = full_rotation_init(model_pcd, target, T_prev=T_prev_for_bias)
            iou_reinit = compute_silhouette_iou(mesh_verts, T_reinit, K, SCALE, mask_gt, H, W)
            if iou_reinit > iou:
                T_result = T_reinit
                fit = fit_reinit
                iou = iou_reinit
                mode = "IoU_REINIT"
                n_iou_reinit += 1

        poses[fidx] = T_result
        fitness_arr[fidx] = fit
        iou_arr[fidx] = iou

        # Draw
        if iou >= IOU_REINIT_THRESHOLD:
            color = (0, 255, 0)
        elif fit >= REINIT_THRESHOLD:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        draw_wireframe(frame_bgr, mesh_verts, mesh_tris, T_result, K, SCALE, color)

        cv2.putText(frame_bgr, f"F{fidx} {mode} fit={fit:.2f} IoU={iou:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(frame_bgr)

        if fidx % 100 == 0:
            print(f"  Frame {fidx}/{F}: {mode} fit={fit:.4f} IoU={iou:.4f}")

    writer.release()
    valid = fitness_arr > 0
    print(f"\nRaw tracking done. Saved to {OUTPUT_VIDEO}")
    print(f"  Mean fitness:   {fitness_arr[valid].mean():.4f}")
    print(f"  Mean IoU:       {iou_arr[valid].mean():.4f}")
    print(f"  Init/Track/Skip/IoU_reinit: {n_init}/{n_track}/{n_skip}/{n_iou_reinit}")
    print(f"  Frames fit>0.3: {(fitness_arr > 0.3).sum()}/{F}")
    print(f"  Frames IoU>0.3: {(iou_arr > 0.3).sum()}/{F}")

    # ── Savitzky-Golay smoothing (post-hoc, window=21) ──
    print(f"\nApplying Savitzky-Golay smoothing (window=21) ...")
    SMOOTH_WINDOW = 21
    SMOOTH_POLY = 3

    # Build raw pose arrays from tracking results
    raw_poses_4x4 = np.zeros((F, 4, 4))
    for i in range(F):
        if poses[i] is not None:
            raw_poses_4x4[i] = poses[i]
        elif i > 0:
            raw_poses_4x4[i] = raw_poses_4x4[i - 1]
        else:
            raw_poses_4x4[i] = np.eye(4)

    raw_pos = raw_poses_4x4[:, :3, 3]  # (F, 3)
    raw_quat = Rotation.from_matrix(raw_poses_4x4[:, :3, :3]).as_quat()  # (F, 4) xyzw

    # Smooth position
    smooth_pos = savgol_filter(raw_pos, SMOOTH_WINDOW, SMOOTH_POLY, axis=0)

    # Smooth quaternion: sign-flip for continuity, savgol, renormalize
    smooth_quat = raw_quat.copy()
    for i in range(1, F):
        if np.dot(smooth_quat[i], smooth_quat[i - 1]) < 0:
            smooth_quat[i] = -smooth_quat[i]
    smooth_quat = savgol_filter(smooth_quat, SMOOTH_WINDOW, SMOOTH_POLY, axis=0)
    smooth_quat /= np.linalg.norm(smooth_quat, axis=1, keepdims=True)

    # Reconstruct smoothed 4x4 poses
    smooth_poses = np.zeros((F, 4, 4))
    smooth_R = Rotation.from_quat(smooth_quat).as_matrix()
    for i in range(F):
        smooth_poses[i, :3, :3] = smooth_R[i]
        smooth_poses[i, :3, 3] = smooth_pos[i]
        smooth_poses[i, 3, 3] = 1.0

    # ── Second render pass with smoothed poses ──
    print(f"Rendering smoothed video ...")
    writer2 = cv2.VideoWriter(str(OUTPUT_VIDEO_SMOOTH), fourcc, 30, (W, H))

    for fidx in range(F):
        frame_bgr = cv2.cvtColor(frames_rgb[fidx], cv2.COLOR_RGB2BGR)

        # Compute smoothed IoU
        mask_path = MASK_DIR / f"mask_{fidx:05d}.npz"
        if mask_path.exists():
            mask_gt = np.load(mask_path)["mask"].astype(bool)
            iou_s = compute_silhouette_iou(mesh_verts, smooth_poses[fidx], K, SCALE, mask_gt, H, W)
        else:
            iou_s = 0.0

        color = (0, 255, 0) if iou_s >= 0.3 else (0, 0, 255)
        draw_wireframe(frame_bgr, mesh_verts, mesh_tris, smooth_poses[fidx], K, SCALE, color)

        cv2.putText(frame_bgr, f"F{fidx} SMOOTH IoU={iou_s:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer2.write(frame_bgr)

    writer2.release()

    # Compute smoothed IoU stats
    smooth_ious = np.zeros(F)
    for fidx in range(F):
        mask_path = MASK_DIR / f"mask_{fidx:05d}.npz"
        if mask_path.exists():
            mask_gt = np.load(mask_path)["mask"].astype(bool)
            smooth_ious[fidx] = compute_silhouette_iou(mesh_verts, smooth_poses[fidx], K, SCALE, mask_gt, H, W)

    valid_s = smooth_ious > 0
    print(f"\nSmoothed results saved to {OUTPUT_VIDEO_SMOOTH}")
    print(f"  Raw   mean IoU: {iou_arr[valid].mean():.4f}")
    print(f"  Smooth mean IoU: {smooth_ious[valid_s].mean():.4f}")
    print(f"  Raw   frames IoU>0.5: {(iou_arr > 0.5).sum()}/{F}")
    print(f"  Smooth frames IoU>0.5: {(smooth_ious > 0.5).sum()}/{F}")


if __name__ == "__main__":
    main()
