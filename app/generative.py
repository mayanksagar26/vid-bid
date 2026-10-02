"""Generative replacement engines (cloud GPUs, paid per use).

vace   Wan 2.1 VACE 14B inpainting on fal.ai. vid-bid sends a crop around the object, a mask
       video from its own SAM 2 tracking and the product cutout as reference image. The model
       re-renders only the masked region (lighting, reflections, condensation, motion blur come
       out naturally) and the result is pasted back into the original full-resolution frames with
       a feathered edge, so everything outside the object stays pixel-identical.
aleph  Runway Aleph 2 video-to-video. Gets the whole clip, a text instruction and up to 5
       keyframes, edited stills taken from vid-bid's local render, as the target look.

Keys come from .env: FAL_KEY and RUNWAYML_API_SECRET.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import media, replace

FAL_ENDPOINT = "fal-ai/wan-vace-14b/inpainting"
FAL_PRICE_PER_SECOND_720P = 0.08    # billed per output second counted at 16 fps
ALEPH_PRICE_PER_SECOND = 0.28
MAX_CHUNK = 241                      # endpoint limit (frames per call)
MIN_CHUNK = 81
OVERLAP = 12                         # frames cross-faded between consecutive chunks
GEN_W, GEN_H = 1280, 720


class EngineError(RuntimeError):
    pass


def default_prompt(category, description):
    what = description.strip() or f"the {category} from the reference image"
    return (f"{what}. Photorealistic. The {category} matches the reference image exactly: same shape, "
            f"label artwork, logo, colours and text. Natural lighting, reflections, highlights and "
            f"condensation consistent with the scene; the hand and everything around it unchanged.")


def estimate_cost(engine, n_frames, fps):
    if engine == "vace":
        return round(n_frames / 16 * FAL_PRICE_PER_SECOND_720P, 2)
    if engine == "aleph":
        return round(max(2.0, n_frames / fps) * ALEPH_PRICE_PER_SECOND, 2)
    return 0.0


# ------------------------------------------------------------------ helpers

class MaskSeq:
    """Per-frame object masks kept at tracking resolution; full-res views made on demand."""

    def __init__(self, small, W, H):
        self.small, self.W, self.H = small, W, H

    def __contains__(self, f):
        return f in self.small

    def __iter__(self):
        return iter(sorted(self.small))

    def __bool__(self):
        return bool(self.small)

    def get(self, f):
        m = self.small.get(f)
        if m is None:
            return None
        return cv2.resize(m.astype(np.uint8) * 255, (self.W, self.H), interpolation=cv2.INTER_LINEAR) > 127

    def bbox(self, f):
        """(x0, y0, x1, y1) in full-res pixels."""
        m = self.small.get(f)
        ys, xs = np.nonzero(m)
        sx, sy = self.W / m.shape[1], self.H / m.shape[0]
        return xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy

    def area(self, f):
        return float(self.small[f].sum())


def _full_masks(work, obj, info, progress):
    """Cleaned per-frame masks from vid-bid's cached SAM 2 tracking, with short dropouts bridged."""
    packed, mshape, _ = replace._get_masks(work, obj, info, progress)
    small = {}
    for f, p in packed.items():
        m = replace._clean(replace._unpack(p, mshape))
        if m.sum() >= 20:
            small[f] = m
    if small:
        fs = sorted(small)
        for f in range(fs[0], fs[-1] + 1):
            if f not in small:
                near = min(fs, key=lambda x: abs(x - f))
                if abs(near - f) <= 8:
                    small[f] = small[near]
    return MaskSeq(small, info["width"], info["height"])


def _chunks(frames, n_total):
    """Split the object's frame span into windows of MIN_CHUNK..MAX_CHUNK frames with overlap."""
    f0, f1 = min(frames), max(frames)
    f0 = max(0, f0 - 4)
    f1 = min(n_total - 1, f1 + 4)
    span = f1 - f0 + 1
    if span < MIN_CHUNK:  # pad short spans (only the masked frames change anyway)
        extra = MIN_CHUNK - span
        f0 = max(0, f0 - extra // 2)
        f1 = min(n_total - 1, f0 + MIN_CHUNK - 1)
        f0 = max(0, f1 - MIN_CHUNK + 1)
        return [(f0, f1)]
    out, s = [], f0
    while True:
        e = min(f1, s + MAX_CHUNK - 1)
        if e - s + 1 < MIN_CHUNK:
            s = max(0, e - MIN_CHUNK + 1)
        out.append((s, e))
        if e >= f1:
            return out
        s = e - OVERLAP + 1


def _crop_window(masks, s, e, W, H):
    """Fixed 16:9 window around the object for this chunk (more pixels on the product)."""
    ys0, xs0, ys1, xs1 = H, W, 0, 0
    for f in range(s, e + 1):
        if f not in masks:
            continue
        bx0, by0, bx1, by1 = masks.bbox(f)
        xs0, ys0 = min(xs0, bx0), min(ys0, by0)
        xs1, ys1 = max(xs1, bx1), max(ys1, by1)
    if ys1 <= ys0:
        return 0, 0, W, H
    cx, cy = (xs0 + xs1) / 2, (ys0 + ys1) / 2
    bw, bh = (xs1 - xs0) * 1.5 + 32, (ys1 - ys0) * 1.5 + 32
    aspect = GEN_W / GEN_H
    bh = max(bh, bw / aspect, min(H, 360))
    bw = bh * aspect
    if bw > W or bh > H:  # can't fit a 16:9 window: use the whole frame
        return 0, 0, W, H
    x0 = int(np.clip(cx - bw / 2, 0, W - bw))
    y0 = int(np.clip(cy - bh / 2, 0, H - bh))
    return x0, y0, int(bw), int(bh)


def _dilate(m, px):
    k = max(3, int(px) | 1)
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _reference_png(cutout_path, out_path):
    """Product on white, centred on a square canvas: what VACE's reference branch expects."""
    c = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    a = c[:, :, 3:4].astype(np.float32) / 255.0
    rgb = (c[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    h, w = rgb.shape[:2]
    side = int(max(h, w) * 1.15)
    canvas = np.full((side, side, 3), 255, np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    canvas = cv2.resize(canvas, (1024, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _download(url, dst):
    with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst


def _mux_audio(video, audio_src, dst):
    cmd = [media.FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio_src),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy", "-shortest",
           "-movflags", "+faststart", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise EngineError("Could not add the original audio: " + p.stderr[-300:])


# ------------------------------------------------------------------ fal.ai Wan VACE

def run_vace(work: Path, obj, cutout_path: Path, description: str, progress, dry_run=False):
    """dry_run: do everything except the paid call (the source crop stands in for the generation)."""
    if not dry_run and not os.environ.get("FAL_KEY"):
        raise EngineError("FAL_KEY is missing. Add it to vid-bid/.env and restart ./run.sh.")
    if not dry_run:
        import fal_client

    t0 = time.time()
    info = media.probe(work / "input.mp4")
    W, H, fps, n = info["width"], info["height"], info["fps"], info["frames"]
    masks = _full_masks(work, obj, info, lambda p, m: progress(0.02 + 0.3 * p, m))
    if not masks:
        raise EngineError("Tracking lost the object in every frame.")
    chunks = _chunks(list(masks), n)
    gdir = work / "gen" / f"vace_{obj['id']}_{time.strftime('%H%M%S')}"
    gdir.mkdir(parents=True)
    ref_png = _reference_png(cutout_path, gdir / "reference.png")
    ref_url = None if dry_run else fal_client.upload_file(str(ref_png))
    prompt = default_prompt(obj["category"], description)

    # One pass over the input: write each part's source crop and mask videos as we go.
    plans = []
    for s_, e_ in chunks:
        x0, y0, cw, ch = _crop_window(masks, s_, e_, W, H)
        plans.append({"s": s_, "e": e_, "crop": (x0, y0, cw, ch), "dil": 0.05 * ch + 6,
                      "src": media.VideoWriter(gdir / f"src_{len(plans)}.mp4", GEN_W, GEN_H, fps, crf=14),
                      "msk": media.VideoWriter(gdir / f"mask_{len(plans)}.mp4", GEN_W, GEN_H, fps, crf=14)})
    progress(0.33, "Preparing source and mask videos")
    try:
        for f, frame in enumerate(media.iter_frames(work / "input.mp4")):
            for pl in plans:
                if not pl["s"] <= f <= pl["e"]:
                    continue
                x0, y0, cw, ch = pl["crop"]
                crop = frame[y0:y0 + ch, x0:x0 + cw]
                interp = cv2.INTER_AREA if ch > GEN_H else cv2.INTER_CUBIC
                pl["src"].write(cv2.resize(crop, (GEN_W, GEN_H), interpolation=interp))
                mm = _blend_mask(masks, f, pl)
                pl["msk"].write(cv2.cvtColor(cv2.resize(mm * 255, (GEN_W, GEN_H), interpolation=cv2.INTER_NEAREST),
                                             cv2.COLOR_GRAY2BGR))
            if f > chunks[-1][1]:
                break
    finally:
        for pl in plans:
            pl["src"].close()
            pl["msk"].close()

    billed = 0.0
    for ci, pl in enumerate(plans):
        s_, e_ = pl["s"], pl["e"]
        base = 0.36 + 0.54 * ci / len(plans)
        progress(base, f"Generating part {ci + 1}/{len(plans)} on fal.ai (frames {s_}-{e_})")
        if dry_run:
            pl["gen"] = gdir / f"src_{ci}.mp4"
            continue
        v_url = fal_client.upload_file(str(gdir / f"src_{ci}.mp4"))
        m_url = fal_client.upload_file(str(gdir / f"mask_{ci}.mp4"))
        args = {
            "prompt": prompt,
            "video_url": v_url,
            "mask_video_url": m_url,
            "ref_image_urls": [ref_url],
            "match_input_num_frames": True,
            "match_input_frames_per_second": True,
            "resolution": "720p",
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "seed": 1234,
        }
        with open(gdir / f"request_{ci}.json", "w") as fjs:
            json.dump(args, fjs, indent=1)

        def on_update(u, base=base, ci=ci):
            if isinstance(u, fal_client.InProgress):
                progress(base, f"Generating part {ci + 1}/{len(plans)} on fal.ai (rendering)")
            elif isinstance(u, fal_client.Queued):
                progress(base, f"Generating part {ci + 1}/{len(plans)} on fal.ai (queued #{u.position})")

        try:
            res = fal_client.subscribe(FAL_ENDPOINT, arguments=args, with_logs=False, on_queue_update=on_update)
        except Exception as ex:  # surface the provider's message (billing, validation...)
            raise EngineError(f"fal.ai rejected part {ci + 1}: {ex}") from ex
        pl["gen"] = _download(res["video"]["url"], gdir / f"gen_{ci}.mp4")
        billed += (e_ - s_ + 1) / 16 * FAL_PRICE_PER_SECOND_720P

    progress(0.92, "Compositing generated frames into the original")
    out_path = _composite(work, obj, masks, plans, info, "vace", feather=max(3.0, 0.012 * H))
    return {
        "engine": "vace",
        "video": f"outputs/{out_path.name}",
        "stamp": out_path.stem.split("_")[-1],
        "download_name": f"vid-bid-{obj['label']}-vace-{out_path.stem.split('_')[-1]}.mp4",
        "stats": f"Wan VACE 14B (fal.ai) · {len(plans)} part(s) · ~${billed:.2f} · {time.time() - t0:.0f}s",
        "cost": round(billed, 2),
    }


def _blend_mask(masks, f, pl):
    """0/1 mask (crop size) of where this part regenerates frame f: the object, dilated."""
    x0, y0, cw, ch = pl["crop"]
    m = masks.get(f)
    if m is None:
        return np.zeros((ch, cw), np.uint8)
    return _dilate(m[y0:y0 + ch, x0:x0 + cw], pl["dil"])


class _GenReader:
    """Sequential reader of one generated part, resampled to the part's frame count."""

    def __init__(self, pl):
        self.pl = pl
        cap = cv2.VideoCapture(str(pl["gen"]))
        n_gen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_need = pl["e"] - pl["s"] + 1
        self.idx = np.linspace(0, max(n_gen - 1, 0), n_need).round().astype(int)
        self.it = media.iter_frames(pl["gen"])
        self.pos, self.cur = -1, None

    def frame(self, f):
        want = self.idx[f - self.pl["s"]]
        while self.pos < want:
            nxt = next(self.it, None)
            if nxt is None:
                break
            self.cur, self.pos = nxt, self.pos + 1
        x0, y0, cw, ch = self.pl["crop"]
        return cv2.resize(self.cur, (cw, ch), interpolation=cv2.INTER_AREA if self.cur.shape[0] > ch else cv2.INTER_CUBIC)


def _composite(work, obj, masks, plans, info, tag, feather):
    """Paste generated crops back into the original frames (feathered), cross-fading overlaps."""
    W, H, fps = info["width"], info["height"], info["fps"]
    spans = [(pl["s"], pl["e"]) for pl in plans]
    for pl in plans:
        s, e = pl["s"], pl["e"]
        pl["has_prev"] = any(ps < s <= pe for ps, pe in spans if (ps, pe) != (s, e))
        pl["has_next"] = any(s < ns <= e for ns, ne in spans if (ns, ne) != (s, e))
        pl["reader"] = _GenReader(pl)

    stamp = time.strftime("%H%M%S")
    out_dir = work / "outputs"
    out_dir.mkdir(exist_ok=True)
    tmp = out_dir / f"_tmp_{tag}_{stamp}.mp4"
    final = out_dir / f"result_{obj['id']}_{tag}_{stamp}.mp4"
    wr = media.VideoWriter(tmp, W, H, fps, audio_src=None, crf=16)
    try:
        for i, frame in enumerate(media.iter_frames(work / "input.mp4")):
            active = [pl for pl in plans if pl["s"] <= i <= pl["e"]]
            if active:
                acc = np.zeros((H, W, 3), np.float32)
                wacc = np.zeros((H, W), np.float32)
                for pl in active:
                    k, L = i - pl["s"], pl["e"] - pl["s"] + 1
                    w = 1.0
                    if pl["has_prev"]:
                        w = min(w, (k + 1) / OVERLAP)
                    if pl["has_next"]:
                        w = min(w, (L - k) / OVERLAP)
                    x0, y0, cw, ch = pl["crop"]
                    a = cv2.GaussianBlur(_blend_mask(masks, i, pl).astype(np.float32), (0, 0), feather) * w
                    acc[y0:y0 + ch, x0:x0 + cw] += pl["reader"].frame(i).astype(np.float32) * a[..., None]
                    wacc[y0:y0 + ch, x0:x0 + cw] += a
                tot = np.clip(wacc, 0, 1)[..., None]
                mix = acc / np.maximum(wacc, 1e-6)[..., None]
                frame = np.clip(mix * tot + frame.astype(np.float32) * (1 - tot) + 0.5, 0, 255).astype(np.uint8)
            wr.write(frame)
    finally:
        wr.close()
    _mux_audio(tmp, work / "input.mp4", final)
    tmp.unlink(missing_ok=True)
    return final


# ------------------------------------------------------------------ Runway Aleph 2

def _keyframe_indices(masks, n, k=3):
    """Frames where the object is large and upright-ish, spread over the clip."""
    if not masks:
        return []
    fs = np.array(list(masks))
    area = np.array([masks.area(f) for f in fs], np.float64)
    good = fs[area >= np.percentile(area, 40)]
    if len(good) == 0:
        good = fs
    picks = [int(good[int(round(q))]) for q in np.linspace(0, len(good) - 1, k + 2)[1:-1]]
    return sorted(set(picks))


def run_aleph(work: Path, obj, cutout_path: Path, description: str, local_result: dict | None, progress):
    if not os.environ.get("RUNWAYML_API_SECRET"):
        raise EngineError("RUNWAYML_API_SECRET is missing. Add it to vid-bid/.env and restart ./run.sh.")
    from runwayml import RunwayML

    t0 = time.time()
    info = media.probe(work / "input.mp4")
    W, H, fps, n = info["width"], info["height"], info["fps"], info["frames"]
    if n / fps > 30.05:
        raise EngineError("Runway Aleph accepts clips up to 30 seconds.")
    masks = _full_masks(work, obj, info, lambda p, m: progress(0.02 + 0.2 * p, m))

    # Keyframes = edited stills. vid-bid's local render supplies them.
    if not local_result:
        progress(0.25, "Rendering keyframes locally")
        local_result = replace.run(work, obj, cutout_path, lambda p, m: progress(0.25 + 0.2 * p, m))
    local_video = work / local_result["video"]
    picks = _keyframe_indices(masks, n)
    gdir = work / "gen" / f"aleph_{obj['id']}_{time.strftime('%H%M%S')}"
    gdir.mkdir(parents=True)
    client = RunwayML()
    keyframes = []
    want = set(picks)
    for f, img in enumerate(media.iter_frames(local_video)):
        if f in want:
            p = gdir / f"key_{f}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(p, "rb") as fh:
                uri = client.uploads.create_ephemeral(file=fh).uri
            keyframes.append({"seconds": round(f / fps, 2), "uri": uri})
        if f >= max(picks):
            break

    progress(0.5, "Uploading video to Runway")
    with open(work / "input.mp4", "rb") as fh:
        video_uri = client.uploads.create_ephemeral(file=fh).uri
    what = description.strip() or "the product shown in the keyframes"
    prompt = (f"Replace the {obj['category']} with {what}, exactly as it looks in the keyframes: same shape, "
              f"label, logo and colours, following the same motion. Keep everything else identical: hands, "
              f"other objects, liquids, background, lighting and camera.")[:1000]
    ratio = f"{W}:{H}" if (W, H) in [(1920, 1080), (1280, 720), (1080, 1920), (720, 1280)] else "1280:720"
    try:
        task = client.video_to_video.create(model="aleph2", video_uri=video_uri, prompt_text=prompt,
                                            keyframes=keyframes, ratio=ratio, seed=1234)
    except Exception as ex:
        raise EngineError(f"Runway rejected the request: {ex}") from ex
    with open(gdir / "request.json", "w") as fjs:
        json.dump({"prompt": prompt, "keyframes": [k["seconds"] for k in keyframes], "ratio": ratio}, fjs)

    while True:
        t = client.tasks.retrieve(task.id)
        st = getattr(t, "status", "")
        if st in ("SUCCEEDED",):
            break
        if st in ("FAILED", "CANCELLED"):
            raise EngineError(f"Runway task {st.lower()}: {getattr(t, 'failure', '') or getattr(t, 'failure_code', '')}")
        prog = getattr(t, "progress", None) or 0
        progress(0.55 + 0.35 * float(prog), f"Runway Aleph is rendering ({st.lower()})")
        time.sleep(5)
    url = t.output[0]
    raw = _download(url, gdir / "aleph_raw.mp4")

    stamp = time.strftime("%H%M%S")
    final = work / "outputs" / f"result_{obj['id']}_aleph_{stamp}.mp4"
    (work / "outputs").mkdir(exist_ok=True)
    _mux_audio(raw, work / "input.mp4", final)
    cost = round(max(2.0, n / fps) * ALEPH_PRICE_PER_SECOND, 2)
    return {
        "engine": "aleph",
        "video": f"outputs/{final.name}",
        "stamp": stamp,
        "download_name": f"vid-bid-{obj['label']}-aleph-{stamp}.mp4",
        "stats": f"Runway Aleph 2 · {len(keyframes)} keyframes · ~${cost:.2f} · {time.time() - t0:.0f}s",
        "cost": cost,
    }
