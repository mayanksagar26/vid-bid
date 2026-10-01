"""SAM 2: product cutout (image mode) and object tracking through the video (video mode)."""
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from . import models


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def product_cutout(img, category):
    """Return a tight BGRA cutout of the product in an uploaded image."""
    if img.ndim == 3 and img.shape[2] == 4 and (img[:, :, 3] < 250).mean() > 0.02:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    else:
        bgr = img[:, :, :3]
        h, w = bgr.shape[:2]
        box = _product_box(bgr, category)
        pred = models.sam2_image()
        with torch.inference_mode():
            pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = pred.predict(box=np.array(box, np.float32), multimask_output=True)
        # Prefer the mask that best fills the box (whole product, not just its label).
        bx = (box[2] - box[0]) * (box[3] - box[1])
        best = max(range(len(masks)), key=lambda i: scores[i] + 0.5 * min(1.0, masks[i].sum() / bx))
        m = _fill_holes(_largest_component(masks[best] > 0))
        alpha = (m * 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        raise ValueError("Could not find the product in that image.")
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return np.dstack([bgr[y1:y2, x1:x2], alpha[y1:y2, x1:x2]])


def _product_box(bgr, category):
    """Box around the product: detector proposals, re-ranked by CLIP's belief that the crop is `category`."""
    from . import compat
    h, w = bgr.shape[:2]
    model, _ = models.detector()
    r = model.predict(bgr, conf=0.05, agnostic_nms=True, verbose=False, device=models.DEVICE)[0]
    cands = sorted(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), reverse=True)[:8]
    m = 0.03
    full = [w * m, h * m, w * (1 - m), h * (1 - m)]
    if not cands:
        return full
    crops = [bgr[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for _, b in cands]
    best, best_s = full, -1.0
    for (conf, b), crop in zip(cands, crops):
        if crop.size == 0:
            continue
        p_cat = dict(compat.classify([crop])).get(category, 0.0)
        area = (b[2] - b[0]) * (b[3] - b[1]) / (w * h)
        s = p_cat * np.sqrt(conf) * (0.3 if area > 0.85 else 1.0)
        if s > best_s:
            best, best_s = b, s
    return best


# ------------------------------------------------------------------ video tracking

CHUNK = 150         # frames per SAM 2 session (bounded memory on 16 GB machines)
REPROMPT_EVERY = 2.0  # seconds between detector box re-prompts inside a chunk


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _mask_box(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def track_object(frames_dir: Path, frame_scale, segments, dets, fps, work: Path, progress):
    """Propagate the object's mask through every frame of each segment.

    frames_dir holds 00000.jpg... downscaled by frame_scale. dets: [[frame, box(full-res), conf], ...]
    Returns {frame_index: packed mask (at frames_dir resolution)} as a dict of np arrays.
    """
    pred = models.sam2_video()
    det_by_frame = {d[0]: (np.array(d[1]) * frame_scale, d[2]) for d in dets}
    det_frames = sorted(det_by_frame)
    masks = {}
    total = sum(e - s + 1 for s, e in segments)
    done = 0
    chunk_root = work / "sam_chunks"

    for seg_start, seg_end in segments:
        carry = None  # mask handed from the previous chunk's last frame
        start = seg_start
        while start <= seg_end:
            end = min(seg_end, start + CHUNK - 1)
            cdir = chunk_root / f"{start:05d}"
            if cdir.exists():
                shutil.rmtree(cdir)
            cdir.mkdir(parents=True)
            for i, f in enumerate(range(start, end + 1)):
                os.symlink(frames_dir / f"{f:05d}.jpg", cdir / f"{i:05d}.jpg")

            chunk_dets = [f for f in det_frames if start <= f <= end]
            with torch.inference_mode(), models.autocast():
                state = pred.init_state(video_path=str(cdir), offload_video_to_cpu=True,
                                        offload_state_to_cpu=True)
                prompted = False
                if carry is not None and carry.any():
                    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=carry)
                    prompted = True
                # Detector boxes keep SAM honest over long clips: re-prompt every few seconds,
                # but only where the detector is confident and agrees with the carried mask.
                last = -1e9
                for f in chunk_dets:
                    box, conf = det_by_frame[f]
                    if prompted and (conf < 0.45 or f - last < REPROMPT_EVERY * fps):
                        continue
                    if prompted and f == start and carry is not None:
                        mb = _mask_box(carry)
                        if mb is not None and _box_iou(mb, box) < 0.5:
                            continue
                    pred.add_new_points_or_box(state, frame_idx=f - start, obj_id=1, box=box)
                    prompted, last = True, f
                if not prompted:
                    carry = None
                    shutil.rmtree(cdir)
                    done += end - start + 1
                    start = end + 1
                    continue
                last_mask = None
                # start_frame_idx=None -> SAM 2 starts at the earliest prompted frame
                for fidx, _, logits in pred.propagate_in_video(state):
                    m = (logits[0, 0] > 0).cpu().numpy()
                    if m.any():
                        masks[start + fidx] = np.packbits(m)
                    last_mask = m
                    done += 1
                    if done % 10 == 0:
                        progress(done / total, f"Tracking object ({done}/{total} frames)")
                pred.reset_state(state)
                del state
            models.free_memory()
            shutil.rmtree(cdir)
            carry = last_mask
            if end == seg_end:
                break
            start = end  # overlap one frame so the hand-off mask lines up
    shutil.rmtree(chunk_root, ignore_errors=True)
    return masks
