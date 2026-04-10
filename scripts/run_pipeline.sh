#!/bin/bash
# HDMI-Multi end-to-end pipeline:
#   Video → GVHMR multi-person → GMR retargeting → HDMI motion format
#
# Usage:
#   bash scripts/run_pipeline.sh <video_path> <robot> [task_name] [extra args]
#
# Example:
#   bash scripts/run_pipeline.sh path/to/video.mp4 unitree_g1
#   bash scripts/run_pipeline.sh path/to/video.mp4 unitree_g1 -s          # static camera
#   bash scripts/run_pipeline.sh path/to/video.mp4 engineai_pm01 dual_move_box_siheng_30fps -s  # with contact

set -e

VIDEO="$1"
ROBOT="$2"
if [ -z "$VIDEO" ] || [ -z "$ROBOT" ]; then
    echo "Usage: bash scripts/run_pipeline.sh <video_path> <robot> [task_name] [extra args...]"
    echo "  robot: unitree_g1, engineai_pm01, etc."
    echo "  task_name: (optional) task name in video_cfg/annotation.yaml for object contact"
    exit 1
fi
shift 2

# Check if next arg is a task_name (doesn't start with -)
TASK_NAME=""
if [ -n "$1" ] && [[ "$1" != -* ]]; then
    TASK_NAME="$1"
    shift
fi
EXTRA_ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GVHMR_DIR="$REPO_ROOT/GVHMR"
OUTPUT_NAME="$(basename "${VIDEO%.*}")"
GVHMR_OUTPUT="$GVHMR_DIR/outputs/demo_multi/$OUTPUT_NAME"
HDMI_OUTPUT="$REPO_ROOT/output/${OUTPUT_NAME}_${ROBOT}"

export PYTHONPATH="${GVHMR_DIR}/third-party/DPVO:${PYTHONPATH}"

eval "$(conda shell.bash hook 2>/dev/null)"

# ======================================================
# Step 1: Multi-person GVHMR (conda: gvhmr)
# ======================================================
conda activate gvhmr
cd "$GVHMR_DIR"

echo "=========================================="
echo " Step 1: Multi-person GVHMR"
echo "=========================================="
python tools/demo/demo_multi.py --video "$VIDEO" --use_dpvo --output_root outputs/demo_multi $EXTRA_ARGS

# ======================================================
# Step 2: GMR retargeting → HDMI format (conda: gmr)
# ======================================================
conda activate gmr
export DISPLAY=:0
cd "$REPO_ROOT"

echo ""
echo "=========================================="
echo " Step 2: Retarget to $ROBOT → HDMI format"
echo "=========================================="
CONTACT_ARGS=""
if [ -n "$TASK_NAME" ]; then
    CONTACT_ARGS="--contact_cfg $REPO_ROOT/video_cfg/annotation.yaml --task_name $TASK_NAME"
fi

python scripts/gvhmr_to_hdmi.py \
    --gvhmr_pred_file "$GVHMR_OUTPUT/aligned_results.pt" \
    --robot "$ROBOT" \
    --output_dir "$HDMI_OUTPUT" \
    $CONTACT_ARGS

echo ""
echo "=========================================="
echo " Done!"
echo "  GVHMR results: $GVHMR_OUTPUT"
echo "  HDMI motion:   $HDMI_OUTPUT"
echo "=========================================="
