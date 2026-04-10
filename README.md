# HDMI-Multi Data Pipeline

End-to-end pipeline: **monocular video → multi-person world-aligned motion → humanoid robot motion data**.

Produces `motion.npz` files directly consumable by [HDMI-Multi](https://github.com/KB-JiaHeng/HDMI-Multi) for humanoid RL training.

## Pipeline Overview

```
Video (MP4)
  │
  ├─ 01_preprocess ──→ Per-person SMPL params + tracking
  ├─ 02_segment ─────→ SAM2 instance masks
  ├─ 03_build_scene ─→ MoGe-2 metric depth + normals + K
  ├─ 04_align ───────→ World-aligned multi-person poses (Kalman + RANSAC)
  ├─ 05_render ──────→ Visualization videos
  ├─ detect_contact ─→ Auto hand-object contact labels (100DOH)
  ├─ vis_contact ────→ Contact visualization overlay
  └─ gvhmr_to_hdmi ──→ GMR retargeting → HDMI motion.npz
```

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `GVHMR/tools/demo/01_preprocess.py` | YOLO tracking + per-person GVHMR + ViTPose + DPVO SLAM |
| 2 | `GVHMR/tools/demo/02_segment.py` | SAM2 instance segmentation masks |
| 3 | `GVHMR/tools/demo/03_build_scene.py` | MoGe-2 single-pass metric depth + surface normals + camera K |
| 4 | `GVHMR/tools/demo/04_align.py` | Scene grounding: gravity refinement + sequential RANSAC multi-plane + bias-augmented Kalman filter with elevation mode-switch |
| 5 | `GVHMR/tools/demo/05_render.py` | In-camera overlay + global view rendering |
| 6 | `scripts/detect_contact.py` | 100DOH hand-object contact detection (per-person bbox crops) |
| 7 | `scripts/vis_contact.py` | Overlay contact detections on original video |
| 8 | `scripts/gvhmr_to_hdmi.py` | GMR IK retargeting → HDMI-format `motion.npz` |

## Setup

### 1. Clone with submodules

```bash
git clone --recursive git@github.com:KB-JiaHeng/HDMI-Multi-datapipeline.git
cd HDMI-Multi-datapipeline
```

### 2. GVHMR environment (Steps 1-7)

```bash
conda create -y -n gvhmr python=3.10
conda activate gvhmr
cd GVHMR && pip install -r requirements.txt && pip install -e . && cd ..
```

### 3. GMR environment (Step 8)

```bash
conda create -y -n gmr python=3.10
conda activate gmr
cd GMR && pip install -e . && cd ..
```

### 4. 100DOH hand-object contact detector (Step 6)

```bash
git clone https://github.com/ddshan/hand_object_detector.git /path/to/hand_object_detector
cd /path/to/hand_object_detector

# Build C/CUDA extensions (uses gvhmr env)
conda activate gvhmr
pip install easydict
cd lib && python setup.py build develop && cd ..

# Download pretrained model
mkdir -p models/res101_handobj_100K/pascal_voc
# Download faster_rcnn_1_8_132028.pth from:
#   https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XRyKprbby
# Place at: models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth
```

Update `HAND_DETECTOR_ROOT` in `scripts/detect_contact.py` to point to your clone location.

### 5. Checkpoints

Download GVHMR checkpoints from [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) and SMPL/SMPLX body models:

```
GVHMR/inputs/checkpoints/
├── body_models/smplx/SMPLX_{GENDER}.npz
├── body_models/smpl/SMPL_{GENDER}.pkl
├── dpvo/dpvo.pth
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

## Usage

### One-command full pipeline

```bash
bash scripts/data_pipeline_t.sh path/to/video.mp4 unitree_g1
```

This runs all 8 steps automatically. Output:
- `GVHMR/outputs/demo_test/<video_name>/` — intermediate results + visualizations
- `output/<video_name>_<robot>/agent_N/motion.npz` — HDMI-format motion data

### With manual contact annotation

For precise object contact timing (pickup/putdown frames), add a task name matching an entry in `video_cfg/annotation.yaml`:

```bash
bash scripts/data_pipeline_t.sh path/to/video.mp4 engineai_pm01 dual_move_box_siheng_30fps
```

This uses manual frame annotations instead of auto-detected contact.

### Step-by-step (manual)

```bash
# Steps 1-5: GVHMR scene alignment (conda: gvhmr)
conda activate gvhmr
cd GVHMR
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tools/demo/01_preprocess.py --use_dpvo --video path/to/video.mp4 --output_root outputs/demo_test
python tools/demo/02_segment.py --output_dir outputs/demo_test/video_name
python tools/demo/03_build_scene.py --output_dir outputs/demo_test/video_name
python tools/demo/04_align.py --output_dir outputs/demo_test/video_name
python tools/demo/05_render.py --output_dir outputs/demo_test/video_name

# Step 6-7: Contact detection (conda: gvhmr)
cd ..
python scripts/detect_contact.py --video path/to/video.mp4 --output_dir GVHMR/outputs/demo_test/video_name
python scripts/vis_contact.py --video path/to/video.mp4 --output_dir GVHMR/outputs/demo_test/video_name

# Step 8: Retargeting (conda: gmr)
conda activate gmr
python scripts/gvhmr_to_hdmi.py \
    --gvhmr_pred_file GVHMR/outputs/demo_test/video_name/aligned_results.pt \
    --robot unitree_g1 \
    --output_dir output/video_name_unitree_g1 \
    --auto_contact GVHMR/outputs/demo_test/video_name/preprocess/contact_labels.pt
```

## Output Format

Each agent's `motion.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `body_pos_w` | (T, B, 3) | World-frame body positions |
| `body_quat_w` | (T, B, 4) | World-frame body quaternions (wxyz) |
| `body_lin_vel_w` | (T, B, 3) | Linear velocities |
| `body_ang_vel_w` | (T, B, 3) | Angular velocities |
| `joint_pos` | (T, J) | Joint angles |
| `joint_vel` | (T, J) | Joint angular velocities |
| `object_contact` | (T, 1) | Binary contact flag (optional) |

`meta.json` contains `body_names`, `joint_names`, and `fps`.

## Supported Robots

All robots supported by GMR: `unitree_g1`, `engineai_pm01`, `unitree_h1`, `booster_t1`, `fourier_n1`, `stanford_toddy`, `kuavo_s45`, `hightorque_hi`, `galaxea_r1pro`, `booster_k1`, and more.

## Key Features

- **Multi-person world alignment**: Sequential RANSAC multi-plane + bias-augmented Kalman filter with gravity refinement and elevation mode-switch for stairs/ramps
- **Automatic contact detection**: 100DOH hand-object detector with per-person bbox crops from SAM2 tracking, ~35 fps
- **Scene-aware depth**: MoGe-2 single-pass metric depth + surface normals + camera intrinsics
- **Ground-plane anchoring**: Dual-foot contact gating with GVHMR static confidence + velocity scoring
- **End-to-end**: Single script from raw video to robot-ready motion data

## Acknowledgements

- [GVHMR](https://github.com/zju3dv/GVHMR) — World-Grounded Human Motion Recovery (SIGGRAPH Asia 2024)
- [GMR](https://github.com/YanjieZe/GMR) — General Motion Retargeting
- [MoGe-2](https://github.com/JiayiGuo3/MoGe-2) — Monocular Geometry Estimation
- [100DOH](https://github.com/ddshan/hand_object_detector) — Hand-Object Contact Detection (CVPR 2020)
- [SAM2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2
