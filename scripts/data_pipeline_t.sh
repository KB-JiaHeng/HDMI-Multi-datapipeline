#!/bin/bash
# HDMI-Multi end-to-end pipeline (scene-aligned + depth-scale version):
#   Video → GVHMR → SAM2 → MoGe-2 → Depth Scale → Grounding+XZ Kalman
#   → 100DOH contact → Object ICP tracking → GMR retarget → Visualization
#
# Usage:
#   bash scripts/data_pipeline_t.sh <video_path> <robot> <obj_mesh>
#
# Examples:
#   bash scripts/data_pipeline_t.sh path/to/video.mp4 unitree_g1 assets/box_real.usd

set -e

VIDEO="$1"
ROBOT="$2"
OBJ_MESH="$3"
if [ -z "$VIDEO" ] || [ -z "$ROBOT" ] || [ -z "$OBJ_MESH" ]; then
    echo "Usage: bash scripts/data_pipeline_t.sh <video_path> <robot> <obj_mesh>"
    echo "  robot: unitree_g1, engineai_pm01, etc."
    echo "  obj_mesh: path to real-world-scale mesh (.obj/.usd)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GVHMR_DIR="$REPO_ROOT/GVHMR"
OUTPUT_NAME="$(basename "${VIDEO%.*}")"
GVHMR_OUTPUT="$GVHMR_DIR/outputs/demo_test/$OUTPUT_NAME"
HDMI_OUTPUT="$REPO_ROOT/output/${OUTPUT_NAME}_${ROBOT}"

export PYTHONPATH="${GVHMR_DIR}:${GVHMR_DIR}/third-party/DPVO:${PYTHONPATH}"

eval "$(conda shell.bash hook 2>/dev/null)"

# ======================================================
# Steps 1-5: GVHMR scene alignment (conda: gvhmr)
# ======================================================
conda activate gvhmr
cd "$GVHMR_DIR"

echo "=========================================="
echo " Step 1/9: Tracking + Per-person GVHMR"
echo "=========================================="
python tools/demo/01_preprocess.py --static_cam --video "$VIDEO" --output_root outputs/demo_test

echo ""
echo "=========================================="
echo " Step 2/9: SAM2 segmentation"
echo "=========================================="
python tools/demo/02_segment.py --output_dir "$GVHMR_OUTPUT"

echo ""
echo "=========================================="
echo " Step 3/9: Build scene (MoGe-2 depth + normals + K)"
echo "=========================================="
python tools/demo/03_build_scene.py --output_dir "$GVHMR_OUTPUT"

echo ""
echo "=========================================="
echo " Step 4/9: Compute depth scale (GVHMR↔MoGe)"
echo "=========================================="
python tools/demo/04_compute_scale.py --output_dir "$GVHMR_OUTPUT"

echo ""
echo "=========================================="
echo " Step 5/9: Scaled grounding + XZ Kalman"
echo "=========================================="
python tools/demo/05_align.py --output_dir "$GVHMR_OUTPUT"

# ======================================================
# Step 5.5: Contact detection (conda: gvhmr)
# ======================================================
cd "$REPO_ROOT"

echo ""
echo "=========================================="
echo " Step 5.5/9: Auto contact detection (100DOH)"
echo "=========================================="
python scripts/detect_contact.py \
    --video "$VIDEO" \
    --output_dir "$GVHMR_OUTPUT"

# ======================================================
# Step 6: Object 6DoF tracking on scaled cloud (conda: gvhmr)
# ======================================================
cd "$GVHMR_DIR"
export DISPLAY=:0

echo ""
echo "=========================================="
echo " Step 6/9: Object 6DoF tracking (scaled ICP)"
echo "=========================================="
python tools/demo/06_track_object.py \
    --output_dir "$GVHMR_OUTPUT" \
    --obj_mesh "$OBJ_MESH"

# ======================================================
# Step 7: GMR retargeting → HDMI format (conda: gmr)
# ======================================================
conda activate gmr

echo ""
echo "=========================================="
echo " Step 7/9: Retarget to $ROBOT → HDMI format"
echo "=========================================="
AUTO_CONTACT="$GVHMR_OUTPUT/preprocess/contact_labels.pt"
CONTACT_ARGS=""
if [ -f "$AUTO_CONTACT" ]; then
    CONTACT_ARGS="--auto_contact $AUTO_CONTACT"
fi

OBJ_POSES="$GVHMR_OUTPUT/preprocess/obj_poses_scaled.pt"
OBJ_POSE_ARGS=""
if [ -f "$OBJ_POSES" ]; then
    OBJ_POSE_ARGS="--obj_pose_file $OBJ_POSES"
fi

python tools/demo/07_retarget.py \
    --gvhmr_pred_file "$GVHMR_OUTPUT/aligned_results_xz.pt" \
    --robot "$ROBOT" \
    --output_dir "$HDMI_OUTPUT" \
    $CONTACT_ARGS $OBJ_POSE_ARGS

# ======================================================
# Step 7.5: Compute contact target offsets (conda: gvhmr)
# ======================================================
conda activate gvhmr
cd "$REPO_ROOT"

echo ""
echo "=========================================="
echo " Step 8/9: Compute contact target offsets"
echo "=========================================="
python scripts/compute_contact_offset.py \
    --output_dir "$GVHMR_OUTPUT" \
    --obj_mesh "$OBJ_MESH"
cp "$GVHMR_OUTPUT/preprocess/contact_offsets.json" "$HDMI_OUTPUT/contact_offsets.json"

# ======================================================
# Step 9: Visualization (conda: gvhmr)
# ======================================================
conda activate gvhmr
cd "$GVHMR_DIR"

echo ""
echo "=========================================="
echo " Step 9/9: Visualization"
echo "=========================================="
python tools/demo/08_visualize.py \
    --output_dir "$GVHMR_OUTPUT" \
    --obj_mesh "$OBJ_MESH"

echo ""
echo "=========================================="
echo " Done!"
echo "  GVHMR results: $GVHMR_OUTPUT"
echo "  HDMI motion:   $HDMI_OUTPUT"
echo "=========================================="
