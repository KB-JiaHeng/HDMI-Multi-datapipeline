"""
Auto-detect hand-object contact from video using 100DOH hand-object detector.

Outputs per-frame binary contact labels compatible with HDMI-Multi format.
For each frame, detects hands and their contact states (No Contact, Self Contact,
Another Person, Portable Object, Stationary Object). Object contact = state 3 or 4.

Usage:
    cd /home/sihengzhao/research/HDMI-Multi-datapipeline
    conda activate gvhmr
    python scripts/detect_contact.py \
        --video path/to/video.mp4 \
        --output_dir GVHMR/outputs/demo_test/video_name

Output:
    <output_dir>/preprocess/contact_labels.pt
    Dict with keys:
        "contact_per_frame": (T,) bool — any hand has object contact
        "hand_dets_per_frame": list of T elements, each is hand_dets array or None
        "fps": float
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

# 100DOH hand-object detector setup
HAND_DETECTOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hand_object_detector")

CONTACT_STATES = {
    0: 'No Contact',
    1: 'Self Contact',
    2: 'Another Person',
    3: 'Portable Object',
    4: 'Stationary Object',
}
# States that count as "object contact" for HDMI-Multi
OBJECT_CONTACT_STATES = {3, 4}


def _setup_detector():
    """Setup 100DOH detector: add lib to path, import modules."""
    lib_path = os.path.join(HAND_DETECTOR_ROOT, 'lib')
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    from model.utils.config import cfg, cfg_from_file, cfg_from_list
    from model.faster_rcnn.resnet import resnet

    cfg_path = os.path.join(HAND_DETECTOR_ROOT, "cfgs/res101.yml")
    model_path = os.path.join(HAND_DETECTOR_ROOT,
                              "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth")

    cfg_from_file(cfg_path)
    cfg.USE_GPU_NMS = True
    cfg.CUDA = True
    np.random.seed(cfg.RNG_SEED)

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    cfg_from_list(['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'])

    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    checkpoint = torch.load(model_path, map_location='cpu')
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint:
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    fasterRCNN.cuda().eval()

    # Persistent input tensors
    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()
    box_info = torch.FloatTensor(1).cuda()

    return fasterRCNN, pascal_classes, im_data, im_info, gt_boxes, num_boxes, box_info


def _detect_frame(im, fasterRCNN, pascal_classes,
                  im_data, im_info, gt_boxes, num_boxes, box_info,
                  thresh_hand=0.5):
    """Run 100DOH on a single BGR frame. Returns hand_dets (N, 10) or None."""
    from model.utils.config import cfg
    from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
    from model.roi_layers import nms
    from model.utils.blob import im_list_to_blob

    # Image blob
    im_orig = im.astype(np.float32, copy=True) - cfg.PIXEL_MEANS
    im_size_min = np.min(im_orig.shape[0:2])
    im_size_max = np.max(im_orig.shape[0:2])
    im_scale = float(cfg.TEST.SCALES[0]) / float(im_size_min)
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im_resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
    im_blob = im_list_to_blob([im_resized])
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scale]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_()

        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    contact_vector = loss_list[0][0]
    offset_vector = loss_list[1][0].detach()
    lr_vector = loss_list[2][0].detach()

    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    lr = (torch.sigmoid(lr_vector) > 0.5).squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scale
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    # Extract hand detections
    j = 2  # 'hand' class index
    inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
    if inds.numel() == 0:
        return None

    cls_scores = scores[:, j][inds]
    _, order = torch.sort(cls_scores, 0, True)
    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

    cls_dets = torch.cat((
        cls_boxes,
        cls_scores.unsqueeze(1),
        contact_indices[inds],
        offset_vector.squeeze(0)[inds],
        lr[inds]
    ), 1)
    cls_dets = cls_dets[order]
    from model.roi_layers import nms
    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
    cls_dets = cls_dets[keep.view(-1).long()]

    return cls_dets.cpu().numpy()


def _crop_frame(frame, bbox_xyxy, pad_ratio=0.3):
    """Crop frame around bbox with padding. Returns crop and offset (x0, y0)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    # Pad by ratio of bbox size
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    crop = frame[cy1:cy2, cx1:cx2]
    return crop, (cx1, cy1)


def detect_contact_video(video_path, preprocess_dir, thresh_hand=0.5, median_filter_size=5):
    """Run 100DOH on per-person bbox crops for each frame.

    Loads per-person bboxes from preprocess/person_N/bbx.pt, crops each
    person with padding, and runs hand-object contact detection independently.
    Per-person crops reduce inter-person occlusion and improve detection.

    Args:
        video_path: path to video file
        preprocess_dir: path to GVHMR preprocess dir with person_N/bbx.pt
        thresh_hand: detection confidence threshold
        median_filter_size: temporal median filter kernel size (odd, 0=disable)

    Returns:
        contact_per_person: dict {pid: (T,) bool} — per-person contact
        hand_dets_per_person: dict {pid: list of T arrays/None}
        fps: float
        combined: (T,) bool — any person has contact
    """
    print(f"[detect_contact] Loading 100DOH model...")
    detector_args = _setup_detector()
    print(f"[detect_contact] Model loaded.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[detect_contact] Video: {num_frames} frames, {fps:.1f} fps")

    # Load per-person bboxes
    import glob
    person_bboxes = {}
    for pd in sorted(glob.glob(os.path.join(preprocess_dir, "person_*"))):
        pid = int(os.path.basename(pd).split("_")[1])
        bbx_path = os.path.join(pd, "bbx.pt")
        if os.path.exists(bbx_path):
            bbx_data = torch.load(bbx_path, map_location='cpu')
            person_bboxes[pid] = bbx_data['bbx_xyxy'].numpy()  # (F, 4)

    assert person_bboxes, f"No person bboxes found in {preprocess_dir}/person_*/bbx.pt"
    pids = sorted(person_bboxes.keys())
    print(f"[detect_contact] {len(pids)} person(s): {pids}")

    contact_per_person = {pid: np.zeros(num_frames, dtype=bool) for pid in pids}
    hand_dets_per_person = {pid: [] for pid in pids}

    for f in tqdm(range(num_frames), desc="100DOH contact"):
        ret, frame = cap.read()
        if not ret:
            for pid in pids:
                hand_dets_per_person[pid].append(None)
            continue

        for pid in pids:
            if f >= len(person_bboxes[pid]):
                hand_dets_per_person[pid].append(None)
                continue

            crop, _ = _crop_frame(frame, person_bboxes[pid][f])
            if crop.size == 0:
                hand_dets_per_person[pid].append(None)
                continue

            hand_dets = _detect_frame(crop, *detector_args, thresh_hand=thresh_hand)
            hand_dets_per_person[pid].append(hand_dets)

            if hand_dets is not None:
                for det in hand_dets:
                    if int(det[5]) in OBJECT_CONTACT_STATES:
                        contact_per_person[pid][f] = True
                        break

    cap.release()

    # Temporal smoothing per person
    if median_filter_size > 0 and median_filter_size % 2 == 1:
        from scipy.ndimage import median_filter
        for pid in pids:
            raw = contact_per_person[pid]
            smoothed = median_filter(raw.astype(np.float32), size=median_filter_size) > 0.5
            n_changed = (smoothed != raw).sum()
            if n_changed > 0:
                print(f"[detect_contact] Person {pid}: median filter changed {n_changed} frames")
            contact_per_person[pid] = smoothed

    for pid in pids:
        n = contact_per_person[pid].sum()
        print(f"[detect_contact] Person {pid}: {n}/{num_frames} contact frames ({n/num_frames*100:.1f}%)")

    combined = np.zeros(num_frames, dtype=bool)
    for pid in pids:
        combined |= contact_per_person[pid]

    return contact_per_person, hand_dets_per_person, fps, combined


def main():
    parser = argparse.ArgumentParser(description="Auto-detect hand-object contact from video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="GVHMR output dir (saves to preprocess/contact_labels.pt)")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Hand detection confidence threshold")
    parser.add_argument("--median_filter", type=int, default=5,
                        help="Temporal median filter kernel size (odd, 0=disable)")
    args = parser.parse_args()

    preprocess_dir = os.path.join(args.output_dir, "preprocess")
    assert os.path.exists(preprocess_dir), f"Preprocess dir not found: {preprocess_dir}. Run 01_preprocess.py first."

    contact_per_person, hand_dets_per_person, fps, combined = detect_contact_video(
        args.video,
        preprocess_dir=preprocess_dir,
        thresh_hand=args.thresh,
        median_filter_size=args.median_filter,
    )

    # Save
    os.makedirs(preprocess_dir, exist_ok=True)
    save_path = os.path.join(preprocess_dir, "contact_labels.pt")

    torch.save({
        "contact_per_person": {pid: torch.from_numpy(c) for pid, c in contact_per_person.items()},
        "contact_combined": torch.from_numpy(combined),  # (T,) bool — any person
        "hand_dets_per_person": hand_dets_per_person,
        "fps": fps,
    }, save_path)
    print(f"[detect_contact] Saved to {save_path}")

    # Print per-second summary
    frames_per_sec = int(fps)
    pids = sorted(contact_per_person.keys())
    print(f"\nPer-second contact summary (combined):")
    for s in range(0, len(combined), frames_per_sec):
        e = min(s + frames_per_sec, len(combined))
        n = combined[s:e].sum()
        bar = '#' * int(n / max(1, e-s) * 20)
        print(f"  {s/fps:>5.1f}s-{e/fps:>5.1f}s: {n:>3d}/{e-s} {bar}")


if __name__ == "__main__":
    main()
