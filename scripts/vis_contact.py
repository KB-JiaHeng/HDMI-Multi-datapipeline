"""Visualize hand-object contact detections on the original video.

Draws red circles on hands detected as contacting objects (state 3=Portable, 4=Stationary),
green circles on hands with no object contact. Also shows per-person bbox crops.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline
    conda activate gvhmr
    python scripts/vis_contact.py \
        --video GVHMR/outputs/demo_test/siheng_dual/0_input_video.mp4 \
        --output_dir GVHMR/outputs/demo_test/siheng_dual
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

CONTACT_COLORS = {
    0: (0, 255, 0),    # No Contact — green
    1: (255, 255, 0),  # Self Contact — cyan
    2: (255, 0, 255),  # Another Person — magenta
    3: (0, 0, 255),    # Portable Object — RED
    4: (0, 0, 200),    # Stationary Object — dark red
}
CONTACT_NAMES = {0: 'None', 1: 'Self', 2: 'Person', 3: 'Object', 4: 'Furniture'}
SIDE_NAMES = {0: 'L', 1: 'R'}
BBOX_COLORS = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--out_video", type=str, default=None,
                        help="Output video path (default: <output_dir>/vis_contact.mp4)")
    args = parser.parse_args()

    preprocess_dir = os.path.join(args.output_dir, "preprocess")
    labels_path = os.path.join(preprocess_dir, "contact_labels.pt")
    assert os.path.exists(labels_path), f"Run detect_contact.py first: {labels_path}"

    labels = torch.load(labels_path, map_location='cpu')
    hand_dets_per_person = labels['hand_dets_per_person']
    contact_per_person = labels['contact_per_person']
    pids = sorted(contact_per_person.keys())

    # Load bboxes for drawing crop regions
    person_bboxes = {}
    for pd in sorted(glob.glob(os.path.join(preprocess_dir, "person_*"))):
        pid = int(os.path.basename(pd).split("_")[1])
        bbx_path = os.path.join(pd, "bbx.pt")
        if os.path.exists(bbx_path):
            bbx_data = torch.load(bbx_path, map_location='cpu')
            person_bboxes[pid] = bbx_data['bbx_xyxy'].numpy()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out_video or os.path.join(args.output_dir, "vis_contact.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in tqdm(range(num_frames), desc="Rendering contact vis"):
        ret, frame = cap.read()
        if not ret:
            break

        for i, pid in enumerate(pids):
            color = BBOX_COLORS[i % len(BBOX_COLORS)]
            is_contact = bool(contact_per_person[pid][f]) if f < len(contact_per_person[pid]) else False

            # Draw person bbox (thin)
            if pid in person_bboxes and f < len(person_bboxes[pid]):
                bx1, by1, bx2, by2 = person_bboxes[pid][f].astype(int)
                # Thicker border if contact
                thickness = 3 if is_contact else 1
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness)
                label = f"P{pid}" + (" CONTACT" if is_contact else "")
                cv2.putText(frame, label, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw hand detections
            dets = hand_dets_per_person[pid][f] if f < len(hand_dets_per_person[pid]) else None
            if dets is not None and pid in person_bboxes and f < len(person_bboxes[pid]):
                # Hand bbox coords are relative to crop — offset to full frame
                bx1, by1, bx2, by2 = person_bboxes[pid][f]
                bw, bh = bx2 - bx1, by2 - by1
                pad_ratio = 0.3
                ox = max(0, int(bx1 - bw * pad_ratio))
                oy = max(0, int(by1 - bh * pad_ratio))

                for det in dets:
                    hx1, hy1, hx2, hy2 = det[:4]
                    score = det[4]
                    contact_state = int(det[5])
                    side = int(det[9])

                    # Map crop coords to full frame
                    hx1_full = int(hx1 + ox)
                    hy1_full = int(hy1 + oy)
                    hx2_full = int(hx2 + ox)
                    hy2_full = int(hy2 + oy)

                    # Center of hand bbox
                    cx = (hx1_full + hx2_full) // 2
                    cy = (hy1_full + hy2_full) // 2
                    radius = max(10, int((hx2_full - hx1_full) * 0.3))

                    dot_color = CONTACT_COLORS.get(contact_state, (128, 128, 128))

                    # Filled circle for object contact, hollow for others
                    if contact_state in {3, 4}:
                        cv2.circle(frame, (cx, cy), radius, dot_color, -1)  # filled
                    else:
                        cv2.circle(frame, (cx, cy), radius, dot_color, 2)   # hollow

                    # Label
                    txt = f"{SIDE_NAMES.get(side, '?')} {CONTACT_NAMES.get(contact_state, '?')} {score:.2f}"
                    cv2.putText(frame, txt, (hx1_full, hy1_full - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1)

        # Frame counter + combined contact status
        combined = any(bool(contact_per_person[pid][f]) for pid in pids if f < len(contact_per_person[pid]))
        status = "CONTACT" if combined else "no contact"
        status_color = (0, 0, 255) if combined else (0, 200, 0)
        cv2.putText(frame, f"F{f} {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[vis_contact] Saved to {out_path}")


if __name__ == "__main__":
    main()
