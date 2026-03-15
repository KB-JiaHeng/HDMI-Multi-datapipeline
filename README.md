# HDMI-Multi Data Pipeline

Multi-person motion estimation and robot retargeting from monocular video.

## Pipeline

```
Video → GVHMR (multi-person alignment) → aligned_results.pt → GMR (retargeting) → robot_motion.pkl
```

| Stage | Tool | Output |
|-------|------|--------|
| 01_preprocess | GVHMR + ViTPose + DPVO | Per-person SMPL params |
| 02_segment | SAM2 | Instance segmentation masks |
| 03_build_scene | UniDepthV2 + CVD | Scene point cloud |
| 04_align | MegaHunter (JAX) | World-aligned multi-person translation |
| 05_render | PyTorch3D | Visualization videos |
| retarget | GMR | Robot joint trajectories |

## Setup

```bash
git clone --recursive git@github.com:KB-JiaHeng/HDMI-Multi-datapipeline.git
cd HDMI-Multi-datapipeline

# GVHMR environment
conda create -y -n gvhmr python=3.10
conda activate gvhmr
cd GVHMR && pip install -r requirements.txt && pip install -e . && cd ..

# GMR environment
conda create -y -n gmr python=3.10
conda activate gmr
cd GMR && pip install -e . && cd ..
```

### Checkpoints

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

```bash
conda activate gvhmr
cd GVHMR

# Full multi-person pipeline
python tools/demo/01_preprocess.py --video path/to/video.mp4 --output_root outputs/demo_multi
python tools/demo/02_segment.py --output_dir outputs/demo_multi/video_name
python tools/demo/03_build_scene.py --output_dir outputs/demo_multi/video_name
python tools/demo/04_align.py --output_dir outputs/demo_multi/video_name
python tools/demo/05_render.py --output_dir outputs/demo_multi/video_name

# Per-person retargeting
conda activate gmr
cd ../GMR
python scripts/gvhmr_to_robot.py --gvhmr_pred_file ../GVHMR/outputs/demo_multi/video_name/aligned_results.pt --robot unitree_g1
```

## Documentation

- [Technical Report: Multi-Person Alignment](docs/multi_person_alignment.md)

## Acknowledgements

- [GVHMR](https://github.com/zju3dv/GVHMR) — World-Grounded Human Motion Recovery
- [GMR](https://github.com/YanjieZe/GMR) — General Motion Retargeting
- [VideoMimic](https://github.com/video-mimic/VideoMimic) — MegaHunter scene optimization
