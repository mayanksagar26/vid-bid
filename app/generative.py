"""Realistic replacement with a free, open-weights video model running on this machine.

vace   Wan 2.1 VACE (Apache-2.0) through diffusers. vid-bid sends a crop around the object, a
       mask video from its own SAM 2 tracking and the product cutout as reference image. The model
       re-renders only the masked region (lighting, reflections, condensation, motion blur come
       out naturally) and the result is pasted back into the original full-resolution frames with
       a feathered edge, so everything outside the object stays pixel-identical.

Needs an NVIDIA GPU (8 GB+; the model is offloaded to RAM between steps when VRAM is short) or
Apple Silicon. Weights download from Hugging Face on first use into models/hf (about 20 GB with
the text encoder). No accounts, keys or per-use costs. See models.vace_pipeline for the knobs.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from . import media, models, replace

GEN_W, GEN_H = 832, 480              # Wan 2.1's 480p size (16:9-ish, divisible by 16)
MAX_CHUNK = 81                       # frames per pass: the length Wan 2.1 was trained on
MIN_CHUNK = 33
OVERLAP = 12                         # frames cross-faded between consecutive chunks
NEGATIVE = ("blurry, low quality, worst quality, JPEG artifacts, deformed, distorted label, "
            "misspelled text, extra fingers, poorly drawn hands, fused fingers, flicker, "
            "overexposed, static, still picture, cartoon, painting, CGI")


class EngineError(RuntimeError):
    pass


def default_prompt(category, description):
    what = description.strip() or f"the {category} from the reference image"
    return (f"{what}. Photorealistic. The {category} matches the reference image exactly: same shape, "
            f"label artwork, logo, colours and text. Natural lighting, reflections, highlights and "
            f"condensation consistent with the scene; the hand and everything around it unchanged.")


def available():
    """(ok, reason) for the UI."""
    try:
        import diffusers  # noqa: F401
    except ImportError:
        return False, "Install the generative extras: .venv/bin/pip install -r requirements.txt"
    if models.DEVICE == "cpu" and not models.VACE_ALLOW_CPU:
        return False, "Needs an NVIDIA GPU or Apple Silicon. No GPU here? Use the free Colab notebook."
    return True, None


def plan_parts(n_frames_object):
    """How many generation passes the object's on-screen span needs (for the UI's estimate)."""
    if n_frames_object <= 0:
        return 0
    return len(_chunks([0, n_frames_object - 1], n_frames_object))


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


def _len4(n):
    """Wan's VAE packs 4 frames per latent: lengths must be 4k + 1."""
    return max(5, (int(n) - 1) // 4 * 4 + 1)


def _chunks(frames, n_total):
    """Windows of 4k+1 frames (MIN_CHUNK..MAX_CHUNK) covering the object's span, with overlap."""
    f0, f1 = min(frames), max(frames)
    f0 = max(0, f0 - 4)
    f1 = min(n_total - 1, f1 + 4)
    span = f1 - f0 + 1
    if span <= MAX_CHUNK:
        L = min(_len4(max(span + 3, MIN_CHUNK)), _len4(n_total))
        s = int(np.clip(f0 - (L - span) // 2, 0, n_total - L))
        return [(s, s + L - 1)]
    out, s = [], f0
    while True:
        e = s + MAX_CHUNK - 1
        if e >= f1:  # last window: full length, ending at the span's end
            s = max(0, f1 - MAX_CHUNK + 1)
            out.append((s, s + MAX_CHUNK - 1))
            return out
        out.append((s, e))
        s = e - OVERLAP + 1


def _crop_window(masks, s, e, W, H):
    """Fixed window around the object for this chunk, at the generator's aspect (more pixels on
    the product than the whole frame would give)."""
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
    bh = max(bh, bw / aspect, min(H, GEN_H))
    bw = bh * aspect
    if bw > W or bh > H:  # can't fit the window: use the whole frame
        return 0, 0, W, H
    x0 = int(np.clip(cx - bw / 2, 0, W - bw))
    y0 = int(np.clip(cy - bh / 2, 0, H - bh))
    return x0, y0, int(bw), int(bh)


def _dilate(m, px):
    k = max(3, int(px) | 1)
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _reference_image(cutout_path):
    """Product on white, the way VACE's reference branch expects it (RGB PIL image)."""
    from PIL import Image
    c = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    if c.shape[2] == 3:
        c = np.dstack([c, np.full(c.shape[:2], 255, np.uint8)])
    a = c[:, :, 3:4].astype(np.float32) / 255.0
    rgb = (c[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    h, w = rgb.shape[:2]
    pad = int(0.08 * max(h, w))
    rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))


def _blend_mask(masks, f, pl):
    """0/1 mask (crop size) of where this part regenerates frame f: the object, dilated."""
    x0, y0, cw, ch = pl["crop"]
    m = masks.get(f)
    if m is None:
        return np.zeros((ch, cw), np.uint8)
    return _dilate(m[y0:y0 + ch, x0:x0 + cw], pl["dil"])


# ------------------------------------------------------------------ Wan VACE (local)

def run_vace(work: Path, obj, cutout_path: Path, description: str, progress, dry_run=False):
    """dry_run: everything except the model (the source crop stands in for the generation)."""
    ok, why = available()
    if not ok and not dry_run:
        raise EngineError(why)
    from PIL import Image

    t0 = time.time()
    info = media.probe(work / "input.mp4")
    W, H, fps, n = info["width"], info["height"], info["fps"], info["frames"]
    masks = _full_masks(work, obj, info, lambda p, m: progress(0.02 + 0.2 * p, m))
    if not masks:
        raise EngineError("Tracking lost the object in every frame.")
    chunks = _chunks(list(masks), n)
    gdir = work / "gen" / f"vace_{obj['id']}_{time.strftime('%H%M%S')}"
    gdir.mkdir(parents=True)
    ref = _reference_image(cutout_path)
    ref.save(gdir / "reference.png")
    prompt = default_prompt(obj["category"], description)
    plans = [{"s": s, "e": e, "crop": _crop_window(masks, s, e, W, H)} for s, e in chunks]
    for pl in plans:
        pl["dil"] = 0.04 * pl["crop"][3] + 6

    if not dry_run:
        progress(0.23, "Encoding the prompt (first run downloads the model, ~20 GB)")
        embeds = models.vace_prompt_embeds(prompt, NEGATIVE)
        progress(0.26, "Loading Wan VACE")
        pipe = models.vace_pipeline()
    steps = models.VACE_STEPS

    for ci, pl in enumerate(plans):
        s_, e_ = pl["s"], pl["e"]
        x0, y0, cw, ch = pl["crop"]
        base = 0.28 + 0.64 * ci / len(plans)
        progress(base, f"Generating part {ci + 1}/{len(plans)} (frames {s_}-{e_})")
        src, msk = [], []
        for f, frame in enumerate(media.iter_frames(work / "input.mp4")):
            if f < s_:
                continue
            if f > e_:
                break
            crop = frame[y0:y0 + ch, x0:x0 + cw]
            crop = cv2.resize(crop, (GEN_W, GEN_H), interpolation=cv2.INTER_AREA if ch > GEN_H else cv2.INTER_CUBIC)
            mm = cv2.resize(_blend_mask(masks, f, pl) * 255, (GEN_W, GEN_H), interpolation=cv2.INTER_NEAREST)
            # VACE inpainting convention: the region to regenerate is neutral grey in the source.
            crop[mm > 127] = 128
            src.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            msk.append(Image.fromarray(mm))
        cv2.imwrite(str(gdir / f"mask_{ci}_first.png"), np.array(msk[0]))
        out_path = gdir / f"gen_{ci}.mp4"
        if dry_run:
            gen = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in src]
        else:
            import torch

            def on_step(_pipe, i, _t, kw, base=base, ci=ci):
                progress(base + 0.64 / len(plans) * (i + 1) / steps,
                         f"Generating part {ci + 1}/{len(plans)} · step {i + 1}/{steps}")
                return kw

            try:
                with torch.inference_mode():
                    out = pipe(video=src, mask=msk, reference_images=[ref],
                               prompt_embeds=embeds[0], negative_prompt_embeds=embeds[1],
                               height=GEN_H, width=GEN_W, num_frames=len(src),
                               num_inference_steps=steps, guidance_scale=5.0,
                               generator=torch.Generator("cpu").manual_seed(1234),
                               callback_on_step_end=on_step, output_type="np").frames[0]
            except Exception as ex:
                raise EngineError(f"Wan VACE failed on part {ci + 1}: {ex}") from ex
            gen = [cv2.cvtColor((np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR) for fr in out]
        wr = media.VideoWriter(out_path, GEN_W, GEN_H, fps, crf=12)
        try:
            for fr in gen:
                wr.write(np.ascontiguousarray(fr))
        finally:
            wr.close()
        pl["gen"] = out_path

    progress(0.93, "Compositing generated frames into the original")
    out_path = _composite(work, obj, masks, plans, info, "vace", feather=max(3.0, 0.012 * H))
    stamp = out_path.stem.split("_")[-1]
    return {
        "engine": "vace",
        "video": f"outputs/{out_path.name}",
        "stamp": stamp,
        "download_name": f"vid-bid-{obj['label']}-vace-{stamp}.mp4",
        "stats": f"Wan VACE ({models.VACE_MODEL.split('/')[-1]}, local) · {len(plans)} part(s) · "
                 f"{steps} steps · {time.time() - t0:.0f}s",
    }


class _GenReader:
    """Sequential reader of one generated part, resampled to the part's frame count."""

    def __init__(self, pl):
        self.pl = pl
        n_gen = media.probe(pl["gen"])["frames"]
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
    plans = sorted(plans, key=lambda pl: pl["s"])
    for j, pl in enumerate(plans):  # overlap (frames) with the previous / next part
        pl["ov_prev"] = max(0, plans[j - 1]["e"] - pl["s"] + 1) if j > 0 else 0
        pl["ov_next"] = max(0, pl["e"] - plans[j + 1]["s"] + 1) if j + 1 < len(plans) else 0
        pl["reader"] = _GenReader(pl)

    stamp = time.strftime("%H%M%S")
    out_dir = work / "outputs"
    out_dir.mkdir(exist_ok=True)
    final = out_dir / f"result_{obj['id']}_{tag}_{stamp}.mp4"
    wr = media.VideoWriter(final, W, H, fps, audio_src=work / "input.mp4", crf=16)
    try:
        for i, frame in enumerate(media.iter_frames(work / "input.mp4")):
            active = [pl for pl in plans if pl["s"] <= i <= pl["e"]]
            if active:
                acc = np.zeros((H, W, 3), np.float32)
                wacc = np.zeros((H, W), np.float32)
                for pl in active:
                    k, j = i - pl["s"], pl["e"] - i
                    w = 1.0  # linear cross-fade over each overlap; the two weights sum to 1
                    if k < pl["ov_prev"]:
                        w = min(w, (k + 1) / (pl["ov_prev"] + 1))
                    if j < pl["ov_next"]:
                        w = min(w, (j + 1) / (pl["ov_next"] + 1))
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
    return final
