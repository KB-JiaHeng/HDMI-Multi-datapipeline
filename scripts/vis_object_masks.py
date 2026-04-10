"""
Visualize object masks by overlaying them on the original video.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline
    conda activate gvhmr
    python scripts/vis_object_masks.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path("/home/sihengzhao/research/GVHMR/outputs/demo_test/siheng/0_input_video.mp4")
DEFAULT_MASK_DIR = Path("/home/sihengzhao/research/GVHMR/outputs/demo_test/siheng/preprocess/masks/object_mask_data")


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay object masks on video for debugging")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="Path to source video")
    parser.add_argument("--mask_dir", type=str, default=str(DEFAULT_MASK_DIR), help="Path to object_mask_data directory")
    parser.add_argument("--out_video", type=str, default=None, help="Output video path")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha in [0,1]")
    parser.add_argument("--max_frames", type=int, default=0, help="Render at most N frames (0 = all)")
    return parser.parse_args()


def colorize_mask(frame_bgr, mask_bool, alpha):
    overlay = frame_bgr.copy()
    overlay[mask_bool] = (0, 255, 0)  # green on masked pixels
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)


def main():
    args = parse_args()

    video_path = Path(args.video)
    mask_dir = Path(args.mask_dir)
    out_path = Path(args.out_video) if args.out_video else video_path.parent / "vis_object_masks.mp4"
    alpha = float(args.alpha)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"--mask_dir must be a directory: {mask_dir}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0,1], got {alpha}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(mask_dir.glob("mask_*.npz"))
    if len(mask_files) == 0:
        raise FileNotFoundError(f"No mask files found in: {mask_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_frames = total_frames if args.max_frames <= 0 else min(total_frames, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output writer: {out_path}")

    masks_found = 0
    masks_missing = 0
    mask_area_sum = 0.0

    for fidx in range(render_frames):
        ok, frame = cap.read()
        if not ok:
            break

        mask_path = mask_dir / f"mask_{fidx:05d}.npz"
        if mask_path.exists():
            data = np.load(mask_path)
            if "mask" not in data:
                writer.release()
                cap.release()
                raise KeyError(f"'mask' key missing in {mask_path}")
            mask_bool = data["mask"].astype(bool)
            if mask_bool.shape != (height, width):
                writer.release()
                cap.release()
                raise ValueError(
                    f"Mask shape mismatch at frame {fidx}: got {mask_bool.shape}, expected {(height, width)}"
                )

            frame = colorize_mask(frame, mask_bool, alpha)
            area = int(mask_bool.sum())
            masks_found += 1
            mask_area_sum += area
            info = f"F{fidx} mask_pixels={area}"
            text_color = (0, 255, 0)
        else:
            masks_missing += 1
            info = f"F{fidx} MISSING_MASK"
            text_color = (0, 0, 255)

        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
        writer.write(frame)

    cap.release()
    writer.release()

    avg_area = (mask_area_sum / masks_found) if masks_found > 0 else 0.0
    print(f"[vis_object_masks] Saved to {out_path}")
    print(f"[vis_object_masks] Frames rendered: {render_frames}")
    print(f"[vis_object_masks] Masks found/missing: {masks_found}/{masks_missing}")
    print(f"[vis_object_masks] Avg mask area (pixels): {avg_area:.2f}")


if __name__ == "__main__":
    main()
