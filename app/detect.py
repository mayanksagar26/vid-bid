"""Find replaceable objects: open-vocabulary detection + ByteTrack, then CLIP relabel and merge."""
import json
from pathlib import Path

import cv2
import numpy as np

from . import compat, media, models, vocab

ROOT = Path(__file__).resolve().parent.parent
TRACKER_CFG = ROOT / "app" / "bytetrack.yaml"

ANALYSIS_FPS = 15        # detection sample rate; SAM 2 later fills in every frame
DET_CONF = 0.12
MIN_TRACK_SECONDS = 1.0
SEGMENT_GAP_SECONDS = 1.5
MAX_OBJECTS = 12


def _crop(frame, box, pad=0.12, size=None):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw)); y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw)); y2 = int(min(h, y2 + pad * bh))
    c = frame[y1:y2, x1:x2].copy()
    if size and c.size:
        s = size / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return c


def _segments(frames, fps, gap_s=SEGMENT_GAP_SECONDS, stride=1):
    frames = sorted(frames)
    segs = []
    for f in frames:
        if segs and f - segs[-1][1] <= gap_s * fps:
            segs[-1][1] = f
        else:
            segs.append([f, f])
    return [[s, e + stride - 1] for s, e in segs]


def detect_objects(video: Path, work: Path, progress=lambda p, m: None):
    info = media.probe(video)
    fps = info["fps"]
    stride = max(1, round(fps / ANALYSIS_FPS))
    model, prompt_cats = models.detector()
    # Fresh tracker state for every video.
    if getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
        del model.predictor.trackers

    tracks = {}
    W, H = info["width"], info["height"]
    total = info["frames"]
    for fi, frame in enumerate(media.iter_frames(video)):
        if fi % stride:
            continue
        r = model.track(frame, persist=True, tracker=str(TRACKER_CFG), conf=DET_CONF, iou=0.5,
                        agnostic_nms=True, verbose=False, device=models.DEVICE, imgsz=640)[0]
        b = r.boxes
        if b is not None and b.id is not None:
            for tid, cls, conf, xyxy in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                            b.conf.tolist(), b.xyxy.tolist()):
                t = tracks.setdefault(tid, {"dets": [], "crops": [], "best": None, "votes": {}})
                box = [float(v) for v in xyxy]
                t["dets"].append([fi, box, float(conf)])
                cat = prompt_cats[cls]
                t["votes"][cat] = t["votes"].get(cat, 0) + conf
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 0.0005 * W * H or area > 0.9 * W * H:
                    continue
                touches = box[0] < 2 or box[1] < 2 or box[2] > W - 2 or box[3] > H - 2
                score = conf * np.sqrt(area) * (0.6 if touches else 1.0)
                if t["best"] is None or score > t["best"][0]:
                    t["best"] = (score, _crop(frame, box, size=360), fi)
                # Keep a handful of views spread over time for CLIP.
                if not t["crops"] or fi - t["crops"][-1][0] >= fps * 0.75:
                    t["crops"].append((fi, _crop(frame, box, pad=0.03, size=224)))
        if (fi // stride) % 15 == 0:
            progress(0.05 + 0.75 * fi / total, f"Scanning frames ({fi}/{total})")

    progress(0.82, "Classifying objects")
    min_dets = max(3, int(MIN_TRACK_SECONDS * fps / stride))
    cands = []
    for tid, t in tracks.items():
        if len(t["dets"]) < min_dets or t["best"] is None:
            continue
        confs = [d[2] for d in t["dets"]]
        if np.mean(confs) < 0.2 and max(confs) < 0.35:
            continue
        crops = [c for _, c in t["crops"]][:8] or [t["best"][1]]
        ranked = compat.classify(crops)
        top, top_p = ranked[0]
        if top not in vocab.CATEGORIES and top_p > 0.5:
            continue  # CLIP is fairly sure this is a person/scene/etc., not a product
        # Fuse CLIP (good at "can" vs "cup") with the detector's votes (good at "bottle" vs "can"
        # when the crop has clutter). Detector share is the fraction of its confidence mass.
        vtot = sum(t["votes"].values())
        fused = {c: pr + 0.5 * t["votes"].get(c, 0.0) / vtot
                 for c, pr in ranked if c in vocab.CATEGORIES}
        cat = max(fused, key=fused.get)
        p = dict(ranked)[cat]
        emb = compat.embed_images(crops).mean(0)
        cands.append({
            "tid": tid, "category": cat, "clip_p": p, "dets": t["dets"], "best": t["best"],
            "crops": crops, "emb": (emb / emb.norm()).cpu().numpy(),
            "start": t["dets"][0][0], "end": t["dets"][-1][0],
        })

    # Merge fragments of the same physical object (ByteTrack loses IDs on occlusion/blur).
    cands.sort(key=lambda c: c["start"])
    merged = []
    for c in cands:
        target = None
        for m in merged:
            if m["category"] != c["category"]:
                continue
            frames_m = {d[0] for d in m["dets"]}
            overlap = sum(1 for d in c["dets"] if d[0] in frames_m)
            if overlap > 2:
                continue
            sim = float(m["emb"] @ c["emb"])
            gap = (c["start"] - m["end"]) / fps
            if sim >= 0.86 or (gap <= 6.0 and sim >= 0.75):
                target = m
                break
        if target is None:
            merged.append(c)
        else:
            target["dets"] = sorted(target["dets"] + c["dets"], key=lambda d: d[0])
            if c["best"][0] > target["best"][0]:
                target["best"] = c["best"]
            target["crops"] = (target["crops"] + c["crops"])[:12]
            e = target["emb"] + c["emb"]
            target["emb"] = e / np.linalg.norm(e)
            target["start"] = min(target["start"], c["start"])
            target["end"] = max(target["end"], c["end"])

    objects = []
    (work / "objects").mkdir(exist_ok=True)
    for m in merged:
        segs = _segments([d[0] for d in m["dets"]], fps, stride=stride)
        visible = sum(e - s + 1 for s, e in segs) / fps
        conf = float(np.mean([d[2] for d in m["dets"]]))
        if visible < MIN_TRACK_SECONDS or (conf < 0.3 and visible < 3.0):
            continue  # brief, low-confidence detections are usually duplicates or noise
        objects.append({"m": m, "segs": segs, "visible": visible})
    objects.sort(key=lambda o: -o["visible"] * np.mean([d[2] for d in o["m"]["dets"]]))
    objects = objects[:MAX_OBJECTS]

    out = []
    for i, o in enumerate(objects):
        m = o["m"]
        oid = f"obj{i + 1}"
        odir = work / "objects" / oid
        odir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(odir / "thumb.jpg"), m["best"][1])
        for k, c in enumerate(m["crops"][:6]):
            cv2.imwrite(str(odir / f"ref{k}.jpg"), c)
        with open(odir / "dets.json", "w") as f:
            json.dump({"stride": stride, "dets": m["dets"]}, f)
        out.append({
            "id": oid,
            "category": m["category"],
            "label": m["category"],
            "thumb": f"objects/{oid}/thumb.jpg",
            "segments": [[round(s / fps, 2), round(e / fps, 2)] for s, e in o["segs"]],
            "segment_frames": o["segs"],
            "visible_seconds": round(o["visible"], 1),
            "confidence": round(float(np.mean([d[2] for d in m["dets"]])), 2),
            "best_frame": int(m["best"][2]),
            "shape": vocab.CATEGORIES.get(m["category"], {}).get("shape", "flat"),
        })
    progress(1.0, f"Found {len(out)} object(s)")
    return out
