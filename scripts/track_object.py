"""
Object 6DoF tracking from video + known mesh.

Pipeline:
  1. User draws bbox on a frame → SAM2 propagates object mask across all frames
  2. Object mask + scene_data.pt pts3d → per-frame object point clouds (world coords)
  3. Auto-estimate depth scale (MoGe-2 depth vs real depth from mask + focal length)
  4. Full-rotation ICP init (14 orientations × 8 yaw = 112 candidates)
  5. Frame-to-frame ICP tracking with velocity prediction + IoU gating
  6. Savitzky-Golay smoothing + finite-diff velocities
  7. Save obj_poses.pt

Expects mesh in real-world meters (e.g. .usd or .obj with metersPerUnit=1).

Supports both .obj and .usd mesh files.

Usage:
    cd /home/sihengzhao/research/GVHMR
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    conda activate gvhmr
    python /home/sihengzhao/research/HDMI-Multi-datapipeline/scripts/track_object.py \
        --output_dir outputs/demo_test/siheng_dual \
        --obj_mesh /path/to/box.usd
"""

import os
import sys
import argparse
import numpy as np
import torch
# open3d imported lazily inside functions that need it — importing at module
# level breaks cv2 GUI (selectROI) due to OpenGL context conflict.
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

# ICP parameters (validated in v5 diagnostic)
TUKEY_K = 0.10              # relaxed for MoGe-2 ~5cm depth noise
MAX_CORR_INIT = 0.08        # init search correspondence distance
MAX_CORR_TRACK_BASE = 0.05  # base tracking correspondence distance
MAX_CORR_TRACK_MAX = 0.15   # upper bound for adaptive expansion
CENTROID_CLIP_RADIUS_FACTOR = 0.75  # clip at 75% of scaled mesh diagonal
REINIT_THRESHOLD = 0.3      # fitness-based reinit
IOU_REINIT_THRESHOLD = 0.3  # silhouette IoU gating
TEMPORAL_BIAS_WEIGHT = 0.3   # weight for angular proximity in reinit scoring


# ---------------------------------------------------------------------------
# 0. Mesh loading (supports .obj, .ply, .stl, .usd)
# ---------------------------------------------------------------------------

def load_mesh(mesh_path):
    """Load a triangle mesh from various formats including USD.

    For USD files, extracts the first Mesh prim and applies its local xform ops.
    Returns an Open3D TriangleMesh.
    """
    import open3d as o3d
    ext = Path(mesh_path).suffix.lower()

    if ext == ".usd" or ext == ".usda" or ext == ".usdc" or ext == ".usdz":
        return _load_mesh_from_usd(mesh_path)
    else:
        # Open3D native formats: .obj, .ply, .stl, .off, .gltf
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh or unsupported format: {mesh_path}")
        mesh.compute_vertex_normals()
        return mesh


def _load_mesh_from_usd(usd_path):
    """Extract triangle mesh from a USD file using pxr.

    Finds the first Mesh prim, reads vertices/faces, applies local xform ops
    (translate, orient, scale) from the mesh prim and its ancestors.
    """
    import open3d as o3d
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Cannot open USD file: {usd_path}")

    # Find the first Mesh prim
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_prim = prim
            break

    if mesh_prim is None:
        raise ValueError(f"No Mesh prim found in USD: {usd_path}")

    usd_mesh = UsdGeom.Mesh(mesh_prim)
    points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
    face_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
    face_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())

    if len(points) == 0:
        raise ValueError(f"Empty mesh in USD: {usd_path}")

    # Compute the full world transform for this mesh prim
    xformable = UsdGeom.Xformable(mesh_prim)
    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    xform_matrix = np.array(world_xform, dtype=np.float64)  # 4x4, row-major in pxr
    # pxr returns row-major (transposed vs OpenGL convention), so:
    # point_world = point_local @ xform_matrix (row vector convention)
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_world = (points_h @ xform_matrix)[:, :3]

    # Triangulate faces (some faces may be quads)
    triangles = []
    idx = 0
    for count in face_counts:
        if count == 3:
            triangles.append([face_indices[idx], face_indices[idx + 1], face_indices[idx + 2]])
        elif count == 4:
            # Split quad into two triangles
            triangles.append([face_indices[idx], face_indices[idx + 1], face_indices[idx + 2]])
            triangles.append([face_indices[idx], face_indices[idx + 2], face_indices[idx + 3]])
        else:
            # Fan triangulation for n-gons
            for i in range(1, count - 1):
                triangles.append([face_indices[idx], face_indices[idx + i], face_indices[idx + i + 1]])
        idx += count

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points_world)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    o3d_mesh.compute_vertex_normals()

    print(f"[TrackObj] Loaded USD mesh: {len(points_world)} vertices, {len(triangles)} triangles")
    verts = np.asarray(o3d_mesh.vertices)
    print(f"  Bounds: {verts.min(0)} to {verts.max(0)}")
    print(f"  Extents: {verts.max(0) - verts.min(0)}")
    return o3d_mesh


# ---------------------------------------------------------------------------
# 1. Object segmentation (user-interactive bbox → SAM2 propagation)
# ---------------------------------------------------------------------------

def run_object_segmentation(output_dir, frames):
    """Segment the object via user-drawn bounding box + SAM2 propagation.

    Flow:
      1. If cached masks exist, load and return them.
      2. Show a video frame, user draws a bounding box with cv2.selectROI.
      3. SAM2 image predictor converts bbox → mask on that frame.
      4. SAM2 video predictor propagates the mask across all frames.
      5. Save per-frame masks as npz.

    Returns dict {frame_idx: (H,W) bool mask}.
    """  
    import cv2
    import shutil
    import tempfile

    mask_base = Path(output_dir) / "preprocess" / "masks"
    obj_mask_dir = mask_base / "object_mask_data"
    obj_done_flag = obj_mask_dir / "_done"

    # --- Check cache ---
    if obj_done_flag.exists():
        print(f"[TrackObj] Object masks already exist: {obj_mask_dir}")
        return _load_object_masks(obj_mask_dir, frames.shape[0], frames.shape[1:3])

    obj_mask_dir.mkdir(parents=True, exist_ok=True)

    F, H, W, _ = frames.shape

    # --- Step 1: User selects bounding box ---
    prompt_fidx = F // 2  # middle frame
    frame_bgr = cv2.cvtColor(frames[prompt_fidx], cv2.COLOR_RGB2BGR)

    print(f"\n[TrackObj] Draw a bounding box around the object in frame {prompt_fidx}.")
    print("  Press ENTER/SPACE to confirm, 'c' to cancel and redraw.")
    # Force window to be visible (some WMs hide new windows)
    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Select Object", 100, 100)
    cv2.imshow("Select Object", frame_bgr)
    cv2.waitKey(1)  # pump event loop to render window
    roi = cv2.selectROI("Select Object", frame_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[TrackObj] ERROR: No bounding box selected.")
        sys.exit(1)

    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)  # (1, 4) xyxy
    print(f"[TrackObj] User bbox: x={x}, y={y}, w={w}, h={h}")

    # --- Step 2: SAM2 segmentation from bbox ---
    print("[TrackObj] Running SAM2 propagation ...")
    masks_dict = _run_sam2_from_bbox(frames, prompt_fidx, bbox, obj_mask_dir)

    # Write done flag
    obj_done_flag.touch()
    return masks_dict


def _run_sam2_from_bbox(frames, prompt_fidx, bbox, obj_mask_dir):
    """Run SAM2 image predictor + video predictor from a user-provided bbox.

    Args:
        frames: (F, H, W, 3) uint8 RGB
        prompt_fidx: frame index where user drew the bbox
        bbox: (1, 4) xyxy float32
        obj_mask_dir: directory to save masks

    Returns dict {frame_idx: (H,W) bool mask}.
    """
    import torch
    import cv2
    import tempfile
    import shutil

    from hmr4d.utils.multihuman.sam2_wrapper import (
        SAM2_CHECKPOINT, SAM2_MODEL_CFG, _lazy_import_sam2,
    )

    deps = _lazy_import_sam2()
    build_sam2_video_predictor = deps["build_sam2_video_predictor"]
    build_sam2 = deps["build_sam2"]
    SAM2ImagePredictor = deps["SAM2ImagePredictor"]

    F, H, W, _ = frames.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Save frames as temp JPEGs (SAM2 video predictor needs image dir)
    tmp_dir = tempfile.mkdtemp(prefix="sam2_obj_")
    frame_names = []
    try:
        print(f"[TrackObj] Writing {F} frames to temp dir ...")
        for i in range(F):
            fname = f"{i:05d}.jpg"
            cv2.imwrite(os.path.join(tmp_dir, fname), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            frame_names.append(fname)

        # Build models
        video_predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT)
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)

        checkpoint = torch.load(SAM2_CHECKPOINT, map_location="cpu", weights_only=True)
        video_predictor.load_state_dict(checkpoint["model"])
        sam2_model.load_state_dict(checkpoint["model"])

        video_predictor = video_predictor.to(device).eval()
        sam2_model = sam2_model.to(device).eval()
        image_predictor = SAM2ImagePredictor(sam2_model)

        # --- Image predictor: bbox → mask on prompt frame ---
        image_predictor.set_image(frames[prompt_fidx])
        input_boxes = torch.tensor(bbox, device=device)
        masks_pred, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # masks_pred: (1, H, W) bool or (1, 1, H, W)
        if masks_pred.ndim == 4:
            masks_pred = masks_pred.squeeze(1)
        init_mask = torch.tensor(masks_pred[0], device=device, dtype=torch.bool)

        print(f"[TrackObj] Initial mask: {init_mask.sum().item()} pixels")

        # --- Video predictor: propagate ---
        inference_state = video_predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        video_predictor.reset_state(inference_state)

        # Add mask at prompt frame (object_id=1)
        video_predictor.add_new_mask(
            inference_state, prompt_fidx, obj_id=1, mask=init_mask,
        )

        # Propagate forward from prompt frame
        all_masks = {}
        for out_fidx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_fidx,
            max_frame_num_to_track=F,  # propagate to end
        ):
            mask_bool = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # (H, W)
            all_masks[out_fidx] = mask_bool

        # Propagate backward from prompt frame
        video_predictor.reset_state(inference_state)
        video_predictor.add_new_mask(
            inference_state, prompt_fidx, obj_id=1, mask=init_mask,
        )
        for out_fidx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_fidx,
            max_frame_num_to_track=F,
            reverse=True,
        ):
            mask_bool = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
            all_masks[out_fidx] = mask_bool

        print(f"[TrackObj] Propagated to {len(all_masks)}/{F} frames")

        # --- Save masks as npz ---
        masks_out = {}
        for fidx in range(F):
            if fidx in all_masks:
                mask = all_masks[fidx]
            else:
                mask = np.zeros((H, W), dtype=bool)
            mask_uint16 = mask.astype(np.uint16)  # 0=bg, 1=object
            npz_path = obj_mask_dir / f"mask_{fidx:05d}.npz"
            np.savez_compressed(str(npz_path), mask=mask_uint16)
            masks_out[fidx] = mask

        n_valid = sum(1 for m in masks_out.values() if m.any())
        print(f"[TrackObj] Object detected in {n_valid}/{F} frames")
        return masks_out

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def _load_object_masks(obj_mask_dir, num_frames, hw_shape):
    """Load cached object masks from npz files."""
    masks = {}
    for fidx in range(num_frames):
        mask_path = obj_mask_dir / f"mask_{fidx:05d}.npz"
        if mask_path.exists():
            m = np.load(mask_path)["mask"]
            masks[fidx] = m.astype(bool)
        else:
            masks[fidx] = np.zeros(hw_shape, dtype=bool)
    return masks


# ---------------------------------------------------------------------------
# 2. Extract per-frame object point clouds from scene_data
# ---------------------------------------------------------------------------

def extract_object_pointclouds(scene_data, obj_masks):
    """Extract per-frame object point clouds in world coordinates.

    Args:
        scene_data: 7-tuple from scene_data.pt
        obj_masks: dict {frame_idx: (H,W) bool}

    Returns:
        dict {frame_idx: (N,3) float64 array} — world-coordinate points
        dict {frame_idx: (N,3) float64 array} — world-coordinate normals
    """
    frame_names = scene_data[0]
    pts3d_dict = scene_data[1]
    normals_dict = scene_data[6]

    obj_pts = {}
    obj_norms = {}
    for fidx, fn in enumerate(frame_names):
        if fidx not in obj_masks:
            continue
        mask = obj_masks[fidx]
        pts = np.array(pts3d_dict[fn], dtype=np.float64)   # (H, W, 3)
        nrm = np.array(normals_dict[fn], dtype=np.float64)  # (H, W, 3)

        # Apply mask
        pts_masked = pts[mask]  # (N, 3)
        nrm_masked = nrm[mask]  # (N, 3)

        # Filter NaN (MoGe-2 produces NaN in invalid regions)
        finite = np.isfinite(pts_masked).all(axis=1) & np.isfinite(nrm_masked).all(axis=1)
        pts_masked = pts_masked[finite]
        nrm_masked = nrm_masked[finite]

        if len(pts_masked) > 0:
            obj_pts[fidx] = pts_masked
            obj_norms[fidx] = nrm_masked

    print(f"[TrackObj] Extracted point clouds for {len(obj_pts)}/{len(frame_names)} frames")
    return obj_pts, obj_norms


# ---------------------------------------------------------------------------
# 3. Scale estimation
# ---------------------------------------------------------------------------

def estimate_scale_from_masks(obj_masks, obj_pts_dict, fx, mesh_max_extent, n_frames=20):
    """Estimate scale from 2D mask width + known mesh extent + focal length.

    MoGe-2 metric depth has systematic scale error (~0.7x). This estimates
    the correction factor by comparing MoGe-2 depth with the expected depth
    from the object's known size and its pixel extent in the image.

    scale = median_depth / (fx * mesh_max_extent / pixel_width)

    Args:
        obj_masks: dict {frame_idx: (H,W) bool}
        obj_pts_dict: dict {frame_idx: (N,3) array} — world-coordinate points
        fx: focal length in pixels
        mesh_max_extent: largest AABB extent of the mesh (meters)
        n_frames: number of frames to use for estimation

    Returns:
        scale: float — multiply mesh vertices by this to match MoGe-2 depth space
    """
    scales = []
    valid_frames = sorted(obj_pts_dict.keys())[:n_frames]

    for fidx in valid_frames:
        if fidx not in obj_masks:
            continue
        mask = obj_masks[fidx]
        pts = obj_pts_dict[fidx]

        # Mask pixel width
        ys, xs = np.where(mask)
        if len(xs) < 10:
            continue
        pixel_width = xs.max() - xs.min()
        if pixel_width < 10:
            continue

        # MoGe-2 depth (median Z of object points)
        depth_med = float(np.median(pts[:, 2]))
        if depth_med <= 0:
            continue

        # Expected real depth from known dimensions + focal length
        real_depth = fx * mesh_max_extent / pixel_width
        scale = depth_med / real_depth
        scales.append(scale)

    if not scales:
        print("[TrackObj] WARNING: Could not estimate scale from masks, defaulting 1.0")
        return 1.0

    scale = float(np.median(scales))
    print(f"[TrackObj] Mask-based scale estimation:")
    print(f"  Samples: {len(scales)}, Median scale: {scale:.4f}")
    print(f"  (MoGe-2 depth is {scale:.0%} of real depth)")
    return scale


# ---------------------------------------------------------------------------
# 4. ICP tracking
# ---------------------------------------------------------------------------

def prepare_model_pcd(mesh, scale, n_points=5000):
    """Sample points from scaled mesh and compute normals."""
    import open3d as o3d
    # Scale mesh
    mesh_scaled = o3d.geometry.TriangleMesh(mesh)
    mesh_scaled.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) * scale
    )
    mesh_scaled.compute_vertex_normals()

    # Sample point cloud
    pcd = mesh_scaled.sample_points_uniformly(number_of_points=n_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.02, max_nn=30))
    return pcd


def angular_distance(T1, T2):
    """Angular distance (radians) between two 4x4 poses."""
    R_diff = T1[:3, :3] @ T2[:3, :3].T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def symmetry_aware_angular_distance(T1, T2, symmetries=None):
    """Min angular distance over symmetry-equivalent orientations.

    Args:
        symmetries: list of 3x3 rotation matrices (symmetry group).
            None = identity only (equivalent to angular_distance).
    """
    if symmetries is None:
        return angular_distance(T1, T2)
    R1, R2 = T1[:3, :3], T2[:3, :3]
    min_dist = float('inf')
    for S in symmetries:
        R_diff = R1 @ (R2 @ S).T
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        dist = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        if dist < min_dist:
            min_dist = dist
    return min_dist


def canonicalize_rotation(T, T_ref, symmetries):
    """Pick the D2 element S that makes T's rotation closest to T_ref in raw angular distance.

    Locks the rotation to the same symmetry branch as T_ref,
    preventing frame-to-frame oscillation between equivalent orientations.

    Args:
        T: (4, 4) pose, modified in-place
        T_ref: (4, 4) reference pose
        symmetries: list of (3, 3) rotation matrices
    Returns:
        T (modified in-place)
    """
    if symmetries is None or T_ref is None:
        return T
    R, R_ref = T[:3, :3], T_ref[:3, :3]
    best_S = min(symmetries, key=lambda S:
        np.linalg.norm(Rotation.from_matrix(R @ S @ R_ref.T).as_rotvec()))
    T[:3, :3] = R @ best_S
    return T


def centroid_clip(pts, norms, radius):
    """Clip points by distance from centroid. View-independent."""
    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    keep = dists <= radius
    return pts[keep], norms[keep]


def visibility_filter(model_pcd, T_pose, min_points=100):
    """Filter model PCD to keep only points visible from camera.

    Uses Open3D hidden_point_removal (Katz et al. 2007 spherical flipping).
    Handles self-occlusion, not just back-face culling.

    Args:
        model_pcd: Open3D PointCloud in model frame
        T_pose: 4x4 model-to-world/camera transform
        min_points: fallback to full model if fewer visible

    Returns:
        Filtered Open3D PointCloud (model frame, visible points only)
    """
    import open3d as o3d
    pts = np.asarray(model_pcd.points)
    R, t = T_pose[:3, :3], T_pose[:3, 3]
    # Transform model points to camera frame
    pts_cam = (R @ pts.T).T + t
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pts_cam)
    diameter = np.linalg.norm(pts_cam.max(0) - pts_cam.min(0))
    try:
        _, visible_idx = pcd_cam.hidden_point_removal([0, 0, 0], diameter * 100)
    except Exception:
        return model_pcd
    if len(visible_idx) < min_points:
        return model_pcd
    return model_pcd.select_by_index(visible_idx)


def predict_pose(T_prev, T_prev2):
    """Constant-velocity prediction (KISS-ICP style)."""
    delta_T = T_prev @ np.linalg.inv(T_prev2)
    return delta_T @ T_prev


def adaptive_corr_dist(T_prev, T_prev2):
    """Widen correspondence distance when motion is large."""
    delta_T = T_prev @ np.linalg.inv(T_prev2)
    trans_mag = np.linalg.norm(delta_T[:3, 3])
    ang_mag = angular_distance(T_prev, T_prev2)
    motion_mag = trans_mag + 0.5 * ang_mag
    return min(MAX_CORR_TRACK_BASE + 2.0 * motion_mag, MAX_CORR_TRACK_MAX)


def compute_silhouette_iou(mesh_verts, T, K, scale, mask_gt, H, W):
    """Project mesh at pose, compute convex hull IoU with SAM2 mask."""
    import cv2
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


def full_rotation_init(model_pcd, target_pcd, T_prev=None, n_yaws_per_face=8,
                       symmetries=None):
    """Full-rotation reinit: 14 orientations × n_yaws = 112 candidates.

    Searches 6 cardinal face orientations + 8 intermediate tilts (±45°, ±135°),
    each with in-plane yaw sweep. If T_prev provided, scores are biased toward
    the previous orientation for temporal coherence.
    """
    import open3d as o3d
    mc = model_pcd.get_center()
    tc = target_pcd.get_center()
    best_score, best_fit, best_T = -1, -1, np.eye(4)

    face_rotations = [
        Rotation.from_euler('y', 90, degrees=True),
        Rotation.from_euler('y', -90, degrees=True),
        Rotation.from_euler('x', -90, degrees=True),
        Rotation.from_euler('x', 90, degrees=True),
        Rotation.identity(),
        Rotation.from_euler('y', 180, degrees=True),
    ]
    intermediate_rotations = [
        Rotation.from_euler('x', a, degrees=True)
        for a in [45, -45, 135, -135]
    ] + [
        Rotation.from_euler('y', a, degrees=True)
        for a in [45, -45, 135, -135]
    ]
    all_rotations = face_rotations + intermediate_rotations

    for R_face in all_rotations:
        for j in range(n_yaws_per_face):
            yaw = 2.0 * np.pi * j / n_yaws_per_face
            R_yaw = Rotation.from_euler('z', yaw)
            R_total = (R_yaw * R_face).as_matrix()

            T_init = np.eye(4)
            T_init[:3, :3] = R_total
            T_init[:3, 3] = tc - R_total @ mc

            model_vis = visibility_filter(model_pcd, T_init)
            res = o3d.pipelines.registration.registration_icp(
                model_vis, target_pcd, MAX_CORR_INIT, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=TUKEY_K)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

            if T_prev is not None and res.fitness > 0:
                ang_dist = symmetry_aware_angular_distance(res.transformation, T_prev, symmetries)
                proximity = 1.0 - ang_dist / np.pi
                score = (1.0 - TEMPORAL_BIAS_WEIGHT) * res.fitness + TEMPORAL_BIAS_WEIGHT * proximity
            else:
                score = res.fitness

            if score > best_score:
                best_score = score
                best_fit = res.fitness
                best_T = res.transformation

    return best_T, best_fit


def track_icp_frame(model_pcd, target_pcd, T_init, max_corr_dist=MAX_CORR_TRACK_BASE):
    """Track object in one frame using ICP with previous/predicted pose as init."""
    import open3d as o3d
    model_vis = visibility_filter(model_pcd, T_init)
    result = o3d.pipelines.registration.registration_icp(
        model_vis, target_pcd,
        max_correspondence_distance=max_corr_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=TUKEY_K)),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    return result.transformation, result.fitness


def run_icp_tracking(model_pcd, obj_pts_dict, obj_norms_dict, total_frames,
                     obj_masks=None, mesh_verts=None, K=None, scale=1.0, H=0, W=0,
                     clip_radius=0.40, min_points=50, voxel_size=0.005,
                     max_angular_diff=None, reinit_threshold=None,
                     stretch_D=None, symmetries=None):
    """Run ICP tracking across all frames.

    Uses centroid-distance clip, velocity prediction, adaptive correspondence
    distance, full-rotation reinit, and silhouette IoU gating.

    Args:
        max_angular_diff: (optional) max rotation per frame in radians. If ICP
            rotation exceeds this, keep ICP translation but replace rotation with
            decayed constant-velocity prediction. Default None = disabled.
        reinit_threshold: (optional) fitness below which to reinit. Default None
            = uses REINIT_THRESHOLD constant (0.3).
        stretch_D: (optional) np.array (3,) stretch factors. If provided,
            un-stretch poses before IoU projection (SAM2 mask is unstretched).
        symmetries: (optional) list of 3x3 rotation matrices for the object's
            symmetry group. Used for symmetry-aware angular distance in clamp
            and IoU reinit gating. None = no symmetry (identity only).

    Returns:
        poses: (T, 4, 4) array of transformations (model → world)
        fitness: (T,) array of ICP fitness scores
    """
    import open3d as o3d
    from scipy.spatial.transform import Rotation as Rot

    _reinit_thr = reinit_threshold if reinit_threshold is not None else REINIT_THRESHOLD

    poses_list = [None] * total_frames
    fitness = np.zeros(total_frames)
    iou_gating_available = (obj_masks is not None and mesh_verts is not None
                            and K is not None and H > 0 and W > 0)
    n_init, n_track, n_iou_reinit, n_clamped = 0, 0, 0, 0
    clamp_count = 0
    T_anchor = None  # last pose where ICP converged naturally (clamp did not fire)

    def _unstretched_T(T):
        """Un-stretch a pose for IoU projection."""
        if stretch_D is None:
            return T
        D_inv = 1.0 / stretch_D
        T_u = T.copy()
        T_u[:3, :3] = np.diag(D_inv) @ T[:3, :3] @ np.diag(stretch_D)
        T_u[:3, 3] = D_inv * T[:3, 3]
        return T_u

    # --- Debug: model PCD extents ---
    _mpts = np.asarray(model_pcd.points)
    print(f"[DEBUG] Model PCD: {len(_mpts)} pts, "
          f"extents={np.array2string(_mpts.max(0)-_mpts.min(0), precision=4)}, "
          f"center={np.array2string(_mpts.mean(0), precision=4)}")

    for fidx in range(total_frames):
        if fidx not in obj_pts_dict or len(obj_pts_dict[fidx]) < min_points:
            if fidx > 0 and poses_list[fidx - 1] is not None:
                poses_list[fidx] = poses_list[fidx - 1]
            continue

        # Centroid-distance clip + voxel downsample + statistical outlier removal
        pts, norms = obj_pts_dict[fidx], obj_norms_dict[fidx]
        n_raw = len(pts)
        pts_c, norms_c = centroid_clip(pts, norms, radius=clip_radius)
        n_clip = len(pts_c)
        if len(pts_c) < min_points:
            pts_c, norms_c = pts, norms
            n_clip = n_raw

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(pts_c)
        target.normals = o3d.utility.Vector3dVector(norms_c)
        target = target.voxel_down_sample(voxel_size)
        target, _ = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        n_final = len(target.points)

        if n_final < min_points:
            if fidx > 0 and poses_list[fidx - 1] is not None:
                poses_list[fidx] = poses_list[fidx - 1]
            continue

        # Debug: target PCD info (first frame + every 50th)
        if fidx == 0 or fidx % 50 == 0:
            _tpts = np.asarray(target.points)
            print(f"[DEBUG] F{fidx:04d} target: {n_raw}→{n_clip}→{n_final} pts, "
                  f"extents={np.array2string(_tpts.max(0)-_tpts.min(0), precision=4)}, "
                  f"center={np.array2string(_tpts.mean(0), precision=4)}")

        # Decide: init or track
        need_init = (poses_list[fidx - 1] is None) if fidx > 0 else True
        reinit_reason = ""
        if not need_init and fitness[max(fidx - 1, 0)] < _reinit_thr:
            need_init = True
            reinit_reason = f"prev_fit={fitness[max(fidx-1,0)]:.3f}"

        action = ""
        iou_val = -1.0
        ang_diff_deg = -1.0

        if need_init:
            # Use T_anchor (not prev frame, which may be garbage) as temporal bias
            T_prev_bias = T_anchor if T_anchor is not None else (poses_list[fidx - 1] if fidx > 0 else None)
            T_result, fit = full_rotation_init(model_pcd, target, T_prev=T_prev_bias, symmetries=symmetries)
            T_result = np.array(T_result)  # ensure writable
            canonicalize_rotation(T_result, T_anchor if T_anchor is not None else (poses_list[fidx-1] if fidx > 0 else None), symmetries)
            n_init += 1

            # Angular gate: same as IoU_REINIT — reject orientations far from anchor
            if T_anchor is not None and max_angular_diff is not None:
                ang_init = symmetry_aware_angular_distance(T_result, T_anchor, symmetries)
                if ang_init <= max_angular_diff:
                    clamp_count = 0
                    T_anchor = T_result
                    action = f"INIT({reinit_reason or 'first'} ang={np.degrees(ang_init):.1f}°)"
                else:
                    # Hold anchor rotation, keep INIT's translation (mask centroid fixes X/Y later)
                    T_result = np.array(T_result)
                    T_result[:3, :3] = T_anchor[:3, :3]
                    action = f"INIT_HOLD(ang={np.degrees(ang_init):.1f}°>5°)"
            else:
                clamp_count = 0
                T_anchor = T_result
                action = f"INIT({reinit_reason or 'first'})"
        else:
            # Velocity prediction + adaptive correspondence distance
            T_prev = poses_list[fidx - 1]
            T_prev2 = poses_list[fidx - 2] if fidx >= 2 and poses_list[fidx - 2] is not None else None

            if T_prev2 is not None:
                T_predicted = predict_pose(T_prev, T_prev2)
                corr_dist = adaptive_corr_dist(T_prev, T_prev2)
            else:
                T_predicted = T_prev
                corr_dist = MAX_CORR_TRACK_BASE

            T_result, fit = track_icp_frame(model_pcd, target, T_predicted, max_corr_dist=corr_dist)
            T_result = T_result.copy()  # make writable for potential rotation clamp
            canonicalize_rotation(T_result, T_prev, symmetries)
            n_track += 1
            action = "TRACK"

            # Rotation clamp: hold anchor rotation when ICP diverges
            # Compare against T_anchor (last good pose), NOT T_prev (which may be drifted)
            if max_angular_diff is not None and T_anchor is not None:
                ang_diff = symmetry_aware_angular_distance(T_result, T_anchor, symmetries)
                ang_diff_deg = np.degrees(ang_diff)
                if ang_diff > max_angular_diff:
                    clamp_count += 1
                    n_clamped += 1
                    _r_icp = Rot.from_matrix(T_result[:3, :3]).as_euler('xyz', degrees=True)
                    T_result[:3, :3] = T_anchor[:3, :3]  # simple anchor hold
                    # T_result[:3, 3] kept from ICP (translation grounded in point cloud)
                    action = f"CLAMP(ang={ang_diff_deg:.1f}° n={clamp_count} icp_rot=({_r_icp[0]:+.1f},{_r_icp[1]:+.1f},{_r_icp[2]:+.1f}) fit={fit:.3f})"
                else:
                    clamp_count = 0
                    T_anchor = T_result
                    action = "TRACK"

        # Silhouette IoU gating (use un-stretched pose for projection)
        if iou_gating_available and not need_init:
            mask_gt = obj_masks.get(fidx)
            if mask_gt is not None and mask_gt.any():
                T_iou = _unstretched_T(T_result)
                iou_val = compute_silhouette_iou(mesh_verts, T_iou, K, scale, mask_gt, H, W)
                if iou_val < IOU_REINIT_THRESHOLD:
                    T_prev_bias = poses_list[fidx - 1] if fidx > 0 else None
                    T_reinit, fit_reinit = full_rotation_init(model_pcd, target, T_prev=T_prev_bias, symmetries=symmetries)
                    T_reinit_iou = _unstretched_T(T_reinit)
                    iou_reinit = compute_silhouette_iou(mesh_verts, T_reinit_iou, K, scale, mask_gt, H, W)
                    if iou_reinit > iou_val:
                        # Angular gate: reject reinit if too far from anchor (symmetry-aware)
                        ang_reinit = symmetry_aware_angular_distance(T_reinit, T_anchor, symmetries) if T_anchor is not None else 0.0
                        if max_angular_diff is None or ang_reinit <= max_angular_diff:
                            T_result, fit = T_reinit, fit_reinit
                            T_result = np.array(T_result)  # ensure writable
                            canonicalize_rotation(T_result, T_anchor, symmetries)
                            clamp_count = 0
                            T_anchor = T_result  # reinit gives fresh anchor
                            n_iou_reinit += 1
                            action = f"IoU_REINIT(iou={iou_val:.3f}->{iou_reinit:.3f} ang={np.degrees(ang_reinit):.1f}°)"
                            iou_val = iou_reinit
                        else:
                            action = f"IoU_REJECT(iou={iou_val:.3f}->{iou_reinit:.3f} ang={np.degrees(ang_reinit):.1f}°>5°)"

        # Anchor X/Y translation to mask centroid (PoseCNN approach)
        # ICP depth (Z) is reliable; X/Y slides on planar surfaces.
        # Skip when fitness is too low — Z depth is unreliable.
        if obj_masks is not None and K is not None and fit > 0.1:
            _mask = obj_masks.get(fidx)
            if _mask is not None and _mask.sum() > 500:
                T_result = np.array(T_result)  # ensure writable
                _ys, _xs = np.where(_mask)
                _cx, _cy = _xs.mean(), _ys.mean()
                _tz = T_result[2, 3]
                _tx = (_cx - K[0, 2]) * _tz / K[0, 0]
                _ty = (_cy - K[1, 2]) * _tz / K[1, 1]
                if stretch_D is not None:
                    T_result[0, 3] = _tx * stretch_D[0]
                    T_result[1, 3] = _ty * stretch_D[1]
                else:
                    T_result[0, 3] = _tx
                    T_result[1, 3] = _ty

        poses_list[fidx] = T_result
        fitness[fidx] = fit

        # --- Per-frame debug line ---
        _t = T_result[:3, 3]
        _r = Rot.from_matrix(T_result[:3, :3]).as_euler('xyz', degrees=True)
        _a_r = Rot.from_matrix(T_anchor[:3, :3]).as_euler('xyz', degrees=True) if T_anchor is not None else [0,0,0]
        _parts = [f"F{fidx:04d}", f"{action:<55s}", f"fit={fit:.3f}", f"n={n_final:4d}",
                  f"pos=({_t[0]:+.3f},{_t[1]:+.3f},{_t[2]:+.3f})",
                  f"rot=({_r[0]:+6.1f},{_r[1]:+6.1f},{_r[2]:+6.1f})",
                  f"anchor=({_a_r[0]:+6.1f},{_a_r[1]:+6.1f},{_a_r[2]:+6.1f})"]
        if iou_val >= 0:
            _parts.append(f"iou={iou_val:.3f}")
        if ang_diff_deg >= 0:
            _parts.append(f"ang={ang_diff_deg:.1f}")
        print(" | ".join(_parts))

    # Convert to array
    poses = np.zeros((total_frames, 4, 4))
    for i in range(total_frames):
        if poses_list[i] is not None:
            poses[i] = poses_list[i]
        elif i > 0:
            poses[i] = poses[i - 1]
        else:
            poses[i] = np.eye(4)

    valid = fitness > 0
    print(f"[TrackObj] ICP tracking done.")
    print(f"  Mean fitness: {fitness[valid].mean():.3f}" if valid.any() else "  No valid frames")
    print(f"  Init/Track/IoU_reinit/Clamped: {n_init}/{n_track}/{n_iou_reinit}/{n_clamped}")
    return poses, fitness


# ---------------------------------------------------------------------------
# 5. Trajectory smoothing
# ---------------------------------------------------------------------------

def smooth_trajectory(poses, fps, window=21, polyorder=3):
    """Smooth position and rotation, compute velocities.

    Args:
        poses: (T, 4, 4) transformations
        fps: frames per second

    Returns:
        pos: (T, 3) smoothed positions
        quat: (T, 4) smoothed quaternions (xyzw for scipy)
        lin_vel: (T, 3)
        ang_vel: (T, 3)
    """
    T = poses.shape[0]
    dt = 1.0 / fps

    # Extract raw position and rotation
    raw_pos = poses[:, :3, 3]  # (T, 3)
    raw_rot = Rotation.from_matrix(poses[:, :3, :3])
    raw_quat = raw_rot.as_quat()  # (T, 4) xyzw

    # Ensure window <= T and is odd
    w = min(window, T)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 1:
        # Not enough frames to smooth, return raw
        pos = raw_pos
        quat = raw_quat
    else:
        # Smooth position
        pos = savgol_filter(raw_pos, w, polyorder, axis=0)

        # Smooth quaternion: sign-flip for continuity, savgol, renormalize
        quat = raw_quat.copy()
        for i in range(1, T):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        quat = savgol_filter(quat, w, polyorder, axis=0)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # Linear velocity: central differences
    lin_vel = np.zeros_like(pos)
    if T > 2:
        lin_vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
        lin_vel[0] = (pos[1] - pos[0]) / dt
        lin_vel[-1] = (pos[-1] - pos[-2]) / dt

    # Angular velocity: rotation difference
    ang_vel = np.zeros((T, 3))
    R_smooth = Rotation.from_quat(quat)
    for i in range(T - 1):
        delta_R = R_smooth[i + 1] * R_smooth[i].inv()
        ang_vel[i] = delta_R.as_rotvec() / dt
    if T > 1:
        ang_vel[-1] = ang_vel[-2]

    return pos, quat, lin_vel, ang_vel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Object 6DoF tracking from video + known mesh")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="GVHMR output dir (contains scene_data.pt)")
    parser.add_argument("--obj_mesh", type=str, required=True,
                        help="Path to object mesh (.obj, .ply, .stl, or .usd file)")
    parser.add_argument("--smooth_window", type=int, default=21,
                        help="Savitzky-Golay window length for smoothing")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("test")
    import cv2
    import numpy as np
    img = np.zeros((300,400,3), dtype=np.uint8)
    cv2.imshow('test1', img)
    cv2.waitKey(3)
    cv2.destroyAllWindows()

    # # Test 2: cv2 AFTER open3d — likely broken
    import open3d
    img2 = np.zeros((300,400,3), dtype=np.uint8)
    cv2.imshow('test2', img2)
    cv2.waitKey(3)
    cv2.destroyAllWindows()
    print("finished test")
     
    print("test select ROI")
    import cv2
    img = np.zeros((300,400,3), dtype=np.uint8)
    roi = cv2.selectROI("test_roi", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    print(f"roi = {roi}")

    # --- Load video frames ---
    from hmr4d.utils.video_io_utils import read_video_np
    video_path = str(out_dir / "0_input_video.mp4")
    assert Path(video_path).exists(), f"Video not found: {video_path}"
    frames = read_video_np(video_path)
    F, H, W, _ = frames.shape
    print(f"[TrackObj] Video: {F} frames, {W}x{H}")

    # --- Load scene_data ---
    scene_path = out_dir / "preprocess" / "scene_data.pt"
    assert scene_path.exists(), f"scene_data.pt not found: {scene_path}. Run 03_build_scene.py first."
    scene_data = torch.load(scene_path, weights_only=False)
    print(f"[TrackObj] Loaded scene_data.pt")

    # --- Load mesh (supports .obj, .ply, .stl, .usd) ---
    mesh = load_mesh(args.obj_mesh)
    print(f"[TrackObj] Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # --- Step 1: Object segmentation (user-interactive) ---
    obj_masks = run_object_segmentation(out_dir, frames)

    n_valid = sum(1 for m in obj_masks.values() if m.any())
    print(f"[TrackObj] Object detected in {n_valid}/{F} frames")
    if n_valid == 0:
        print("[TrackObj] ERROR: No object detected in any frame.")
        sys.exit(1)

    # --- Step 2: Extract object point clouds ---
    obj_pts, obj_norms = extract_object_pointclouds(scene_data, obj_masks)

    # --- Step 3: Auto-estimate depth scale ---
    # MoGe-2 metric depth has systematic scale error (~70% of real depth).
    # Estimate the correction by comparing MoGe-2 depth with the expected depth
    # from known mesh size + focal length + SAM2 mask pixel extent.
    mesh_verts = np.asarray(mesh.vertices)
    mesh_extents = mesh_verts.max(axis=0) - mesh_verts.min(axis=0)
    mesh_max_extent = float(mesh_extents.max())

    frame_names_sd = scene_data[0]
    im_K_dict = scene_data[3]
    K = np.array(im_K_dict[frame_names_sd[0]], dtype=np.float64)
    fx = K[0, 0]

    scale = estimate_scale_from_masks(obj_masks, obj_pts, fx, mesh_max_extent)

    # --- Step 4: Prepare model point cloud ---
    model_pcd = prepare_model_pcd(mesh, scale, n_points=5000)

    # --- Step 5: ICP tracking ---
    mesh_diagonal = float(np.linalg.norm(mesh_extents))
    clip_radius = CENTROID_CLIP_RADIUS_FACTOR * mesh_diagonal * scale
    print(f"[TrackObj] Centroid clip radius: {clip_radius:.4f}m")

    poses, fitness = run_icp_tracking(
        model_pcd, obj_pts, obj_norms, F,
        obj_masks=obj_masks, mesh_verts=mesh_verts, K=K, scale=scale, H=H, W=W,
        clip_radius=clip_radius)

    # --- Step 6: Smooth and compute velocities ---
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"[TrackObj] Video fps: {fps}")

    pos, quat, lin_vel, ang_vel = smooth_trajectory(
        poses, fps, window=args.smooth_window)

    # --- Step 7: Save results ---
    save_path = out_dir / "preprocess" / "obj_poses.pt"
    result = {
        "obj_pos": pos,           # (T, 3) world position
        "obj_quat": quat,         # (T, 4) xyzw quaternion
        "obj_lin_vel": lin_vel,    # (T, 3)
        "obj_ang_vel": ang_vel,    # (T, 3)
        "obj_scale": scale,        # float: depth scale correction (MoGe-2 depth / real depth)
        "obj_mesh_path": str(args.obj_mesh),
        "fitness": fitness,         # (T,) ICP fitness per frame
        "fps": fps,
    }
    torch.save(result, save_path)
    print(f"[TrackObj] Saved object poses to {save_path}")
    print(f"  Depth scale: {scale:.4f} (MoGe-2 depth / real depth)")
    print(f"  Mean fitness: {fitness[fitness > 0].mean():.3f}")
    print(f"  Frames tracked: {(fitness > 0).sum()}/{F}")


if __name__ == "__main__":
    main()
