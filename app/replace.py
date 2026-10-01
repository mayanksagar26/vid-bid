"""Replace a tracked object with a product cutout, frame by frame.

Pipeline
  1. SAM 2 masks for every frame the object is visible (cached per object).
  2. Geometry per frame from the mask: axis, width and, for cylinders (cans, bottles, cups), the
     top and bottom rim ellipses fitted to the mask outline. Ends cut off by the frame edge are
     reconstructed from frames where they're visible. Outliers are dropped, then everything is
     smoothed over time.
  3. Render: the product's label is wrapped onto the body between the rims with a cylindrical
     remap. The original lid/top stays. Shading follows a cylinder plus a lighting gradient fitted
     from the original object; white balance/exposure come from the surroundings; blur is matched
     to the footage (focus + motion).
  4. Composite: pixels inside the body that SAM says are *not* the object (fingers, foam, straws)
     stay in front. Slivers of the old object outside the new body are inpainted.
  5. Encode with the original audio.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from . import media, segment

GAP_FILL = 6            # frames: reuse a neighbouring mask when SAM drops the object briefly
TAIL_PAD_SECONDS = 0.5  # keep tracking a little past the last detection
SEG_JOIN_SECONDS = 2.0  # join detection segments separated by short gaps
MAX_STRETCH = 1.2       # max anisotropic stretch of the label before cropping instead
# Typical visible body height / diameter between the rims, used when no frame shows both ends.
DEFAULT_BODY_RATIO = {"can": 1.7, "bottle": 2.8, "cup": 1.1, "jar": 1.2}


# ------------------------------------------------------------------ masks

def _clean(mask):
    m = segment._largest_component(mask)
    if not m.any():
        return m
    # Fill small holes (specular highlights, compression noise), keep big ones (real occluders).
    filled = segment._fill_holes(m)
    holes = filled & ~m
    if holes.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
        small = np.zeros(n, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < 0.02 * m.sum()
        m = m | small[lab]
    return m


def _get_masks(work, obj, info, progress):
    odir = work / "objects" / obj["id"]
    cache = odir / "masks.npz"
    frames_dir = work / "frames"
    if cache.exists():
        z = np.load(cache)
        shape = tuple(int(v) for v in z["shape"])
        masks = {int(k[1:]): z[k] for k in z.files if k.startswith("f")}
        progress(0.7, "Using cached tracking")
        return masks, shape, float(z["scale"])

    progress(0.01, "Extracting frames")
    if not (frames_dir / "00000.jpg").exists():
        media.extract_frames_jpg(work / "input.mp4", frames_dir)
    first = cv2.imread(str(frames_dir / "00000.jpg"))
    shape = first.shape[:2]
    scale = shape[1] / info["width"]

    with open(odir / "dets.json") as f:
        dets = json.load(f)["dets"]
    fps, n = info["fps"], info["frames"]
    segs = []
    for s, e in obj["segment_frames"]:
        e = min(n - 1, e + int(TAIL_PAD_SECONDS * fps))
        if segs and s - segs[-1][1] <= SEG_JOIN_SECONDS * fps:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])

    masks = segment.track_object(frames_dir, scale, segs, dets, fps, work,
                                 lambda p, m: progress(0.02 + 0.68 * p, m))
    np.savez_compressed(cache, shape=np.array(shape), scale=np.array(scale),
                        **{f"f{k}": v for k, v in masks.items()})
    return masks, shape, scale


def _unpack(packed, shape):
    return np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)


# ------------------------------------------------------------------ geometry

def _robust_fit(A, y, iters=2):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(iters):
        res = np.abs(A @ coef - y)
        keep = res <= 2.5 * np.median(res) + 1.0
        if keep.sum() < A.shape[1] + 3:
            break
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return coef


def _measure(mask, prev_up, cylinder):
    """Silhouette of one mask (mask pixel units).

    Seen from above or below, a cylinder's sides converge (perspective), so the left and right
    edges are fitted as lines along the axis: centre c(u) = c0 + c1*u, half-width h(u) = h0 + h1*u.
    For cylinders the top/bottom outlines are fitted as rim ellipses (half-height e_t / e_b).
    Returns up/right axes, a_t/a_b (rim centres along up), the line coefficients, e_t/e_b and
    top_ok/bot_ok (that end is visible, not cut by the frame edge)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    hull = cv2.convexHull(pts)
    (_, _), (_, _), ang = cv2.minAreaRect(hull)
    a = np.deg2rad(ang)
    e1 = np.array([np.cos(a), np.sin(a)])
    e2 = np.array([-np.sin(a), np.cos(a)])
    if prev_up is None:
        up = e1 if abs(e1[1]) > abs(e2[1]) else e2  # products are usually upright
        if up[1] > 0:
            up = -up
    else:
        up = e1 if abs(e1 @ prev_up) > abs(e2 @ prev_up) else e2
        if up @ prev_up < 0:
            up = -up
    right = np.array([-up[1], up[0]])
    pu, pr = pts @ up, pts @ right
    u0, u1 = np.percentile(pu, [0.3, 99.7])
    r0, r1 = np.percentile(pr, [0.3, 99.7])
    g = {"up": up, "right": right, "a_t": u1, "a_b": u0,
         "c": np.array([(r0 + r1) / 2, 0.0]), "hw": np.array([(r1 - r0) / 2, 0.0]),
         "e_t": 0.0, "e_b": 0.0, "top_ok": True, "bot_ok": True, "area": float(mask.sum())}

    on_border = (xs <= 1) | (ys <= 1) | (xs >= w - 2) | (ys >= h - 2)
    if on_border.any():
        bu = pu[on_border]
        g["top_ok"] = not (bu > u1 - 0.15 * (u1 - u0)).any()
        g["bot_ok"] = not (bu < u0 + 0.15 * (u1 - u0)).any()

    # Side lines from the middle of the body (rims excluded), bins along the axis.
    L = u1 - u0
    edges = np.linspace(u0 + 0.2 * L, u1 - 0.2 * L, 25)
    idx = np.digitize(pu, edges)
    uc, lft, rgt = [], [], []
    for i in range(1, len(edges)):
        sel = idx == i
        if sel.sum() < 4:
            continue
        uc.append((edges[i - 1] + edges[i]) / 2)
        lft.append(np.percentile(pr[sel], 1))
        rgt.append(np.percentile(pr[sel], 99))
    if len(uc) >= 8:
        uc, lft, rgt = map(np.asarray, (uc, lft, rgt))
        A = np.stack([np.ones_like(uc), uc], 1)
        lc = _robust_fit(A, lft)
        rc = _robust_fit(A, rgt)
        c, hw = (lc + rc) / 2, (rc - lc) / 2
        # Taper limited to what perspective can plausibly do (+-35% over the body length).
        hw[1] = np.clip(hw[1], -0.35 * hw[0] / max(L, 1), 0.35 * hw[0] / max(L, 1))
        g["c"], g["hw"] = c, hw
    if not cylinder:
        return g

    # Rim ellipses: top outline a_t + e_t*sqrt(1-t^2), bottom outline a_b - e_b*sqrt(1-t^2),
    # with t measured against the local (tapered) centre and half-width.
    hloc = np.maximum(g["hw"][0] + g["hw"][1] * pu, 1.0)
    t = (pr - (g["c"][0] + g["c"][1] * pu)) / hloc
    tedges = np.linspace(-0.85, 0.85, 36)
    idx = np.digitize(t, tedges)
    tc, top, bot = [], [], []
    for i in range(1, len(tedges)):
        sel = idx == i
        if sel.sum() < 3:
            continue
        tc.append((tedges[i - 1] + tedges[i]) / 2)
        top.append(pu[sel].max())
        bot.append(pu[sel].min())
    if len(tc) < 10:
        return g
    tc, top, bot = map(np.asarray, (tc, top, bot))
    sq = np.sqrt(1 - tc ** 2)
    if g["top_ok"]:
        a_t, e_t = _robust_fit(np.stack([np.ones_like(sq), sq], 1), top)
        emax = 0.75 * (g["hw"][0] + g["hw"][1] * a_t)
        g["e_t"] = float(np.clip(e_t, 0, emax))
        g["a_t"] = float(a_t + (e_t - g["e_t"]))
    if g["bot_ok"]:
        a_b, e_b = _robust_fit(np.stack([np.ones_like(sq), -sq], 1), bot)
        emax = 0.75 * (g["hw"][0] + g["hw"][1] * a_b)
        g["e_b"] = float(np.clip(e_b, 0, emax))
        g["a_b"] = float(a_b - (e_b - g["e_b"]))
    return g


def _runs(frames):
    frames = np.asarray(frames)
    breaks = np.where(np.diff(frames) > 1)[0] + 1
    return np.split(np.arange(len(frames)), breaks)


def _smooth(frames, vals, sigmas):
    out = vals.copy()
    for run in _runs(frames):
        if len(run) < 3:
            continue
        for j, s in enumerate(sigmas):
            if s <= 0:
                continue
            v = median_filter(vals[run, j], size=min(5, len(run) | 1), mode="nearest")
            out[run, j] = gaussian_filter1d(v, s, mode="nearest")
    return out


def _geometry(masks, shape, inv_scale, shape_kind, category):
    """Per-frame geometry in full-resolution pixels:
    T (top rim centre xy), up angle, W, H (body height between rim centres), e_t, e_b."""
    cyl = shape_kind == "cylinder"
    frames, gs, prev_up = [], [], None
    for f in sorted(masks):
        m = _clean(_unpack(masks[f], shape))
        if m.sum() < 30:
            continue
        g = _measure(m, prev_up, cyl)
        prev_up = g["up"]
        frames.append(f)
        gs.append(g)
    if not gs:
        return {}

    # Drop outliers: masks far smaller/larger than their neighbours (SAM grabbing a fragment).
    area = np.array([g["area"] for g in gs])
    med = median_filter(area, size=min(31, len(area) | 1), mode="nearest")
    ok = (area > 0.35 * med) & (area < 2.5 * med)
    frames = [f for f, k in zip(frames, ok) if k]
    gs = [g for g, k in zip(gs, ok) if k]
    if not gs:
        return {}

    def hw_at(g, u):
        return max(1.0, g["hw"][0] + g["hw"][1] * u)

    W = np.array([hw_at(g, (g["a_t"] + g["a_b"]) / 2) * 2 for g in gs])
    both = np.array([g["top_ok"] and g["bot_ok"] for g in gs])
    H_raw = np.array([g["a_t"] - g["a_b"] for g in gs])
    if both.sum() >= 5:
        ratio = float(np.median(H_raw[both] / W[both]))
    else:
        ratio = DEFAULT_BODY_RATIO.get(category, 1.6)
    e_ok_t = np.array([g["e_t"] / (2 * hw_at(g, g["a_t"])) for g in gs if g["top_ok"]])
    e_ok_b = np.array([g["e_b"] / (2 * hw_at(g, g["a_b"])) for g in gs if g["bot_ok"]])
    e_rel_t = float(np.median(e_ok_t)) if len(e_ok_t) else 0.15 * cyl
    e_rel_b = float(np.median(e_ok_b)) if len(e_ok_b) else e_rel_t * 1.1

    rows = []
    for g, Wg in zip(gs, W):
        a_t, a_b, e_t, e_b = g["a_t"], g["a_b"], g["e_t"], g["e_b"]
        if not g["top_ok"] and not g["bot_ok"]:
            mid = (a_t + a_b) / 2
            a_t, a_b = mid + ratio * Wg / 2, mid - ratio * Wg / 2
            e_t, e_b = e_rel_t * 2 * hw_at(g, a_t), e_rel_b * 2 * hw_at(g, a_b)
        elif not g["top_ok"]:
            a_t = a_b + ratio * Wg
            e_t = e_rel_t * 2 * hw_at(g, a_t)
        elif not g["bot_ok"]:
            a_b = a_t - ratio * Wg
            e_b = e_rel_b * 2 * hw_at(g, a_b)
        c = g["c"]
        T = g["up"] * a_t + g["right"] * (c[0] + c[1] * a_t)
        Bc = g["up"] * a_b + g["right"] * (c[0] + c[1] * a_b)
        rows.append([T[0], T[1], Bc[0], Bc[1], hw_at(g, a_t), hw_at(g, a_b), e_t, e_b])
    vals = np.array(rows, np.float64) * inv_scale
    sm = _smooth(frames, vals, sigmas=[1.5, 1.5, 1.5, 1.5, 3.0, 3.0, 5.0, 5.0])
    return {f: sm[i] for i, f in enumerate(frames)}, ratio


# ------------------------------------------------------------------ appearance

class Look:
    """Exponentially smoothed per-frame appearance parameters."""

    def __init__(self):
        self.vals = {}

    def ema(self, name, val, a=0.15):
        val = np.asarray(val, np.float64)
        cur = self.vals.get(name)
        self.vals[name] = val if cur is None else (1 - a) * cur + a * val
        return self.vals[name]


def _fit_gradient(lum, t, sel):
    """Quadratic fit of luminance across the object's width -> lighting direction/falloff."""
    if sel.sum() < 150:
        return np.array([1.0, 0.0, 0.0])
    tt, ll = t[sel], lum[sel]
    A = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, ll, rcond=None)
    base = max(coef[0] + coef[2] / 3, 8.0)  # mean of the fitted curve over [-1, 1]
    return coef / base


def _sharpness(gray, sel):
    """High/mid frequency energy ratio: roughly content-independent blur measure."""
    if sel.sum() < 100:
        return None
    g = gray.astype(np.float32)
    hf = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))
    mf = np.abs(cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 3.0))
    return float(hf[sel].mean() / (mf[sel].mean() + 1e-3))


def _motion_kernel(vx, vy):
    speed = float(np.hypot(vx, vy))
    n = int(round(speed * 0.5))  # ~180 degree shutter
    if n < 2:
        return None
    n = min(n, 31)
    k = np.zeros((2 * n + 1, 2 * n + 1), np.float32)
    dx, dy = vx / speed, vy / speed
    for s in np.linspace(-n / 2, n / 2, 2 * n + 1):
        k[int(round(n + s * dy)), int(round(n + s * dx))] = 1
    return k / k.sum()


# ------------------------------------------------------------------ product texture

def _label_texture(cut, shape_kind):
    """Crop the cutout to its solid label area (drop transparent corners, rims and lid)."""
    a = cut[:, :, 3].astype(np.float32) / 255.0
    h, w = a.shape
    if shape_kind != "cylinder":
        return cut
    cols = np.where(a[int(h * 0.3):int(h * 0.7)].mean(0) > 0.9)[0]
    if len(cols) < 4:
        return cut
    c0, c1 = cols.min(), cols.max() + 1
    inner = a[:, c0 + int(0.1 * (c1 - c0)): c1 - int(0.1 * (c1 - c0))]
    rows = np.where(inner.min(1) > 0.9)[0]
    if len(rows) < 4:
        return cut[:, c0:c1]
    r0, r1 = rows.min(), rows.max() + 1
    # Skip the first/last few % (shoulder/rim of the can, where the photo curves away).
    m = int(0.03 * (r1 - r0))
    return cut[r0 + m: r1 - m, c0:c1]


def _texture_window(tex_h, tex_w, body_ratio):
    """Window (x0, x1, y0, y1) of the texture to show so the label is stretched at most MAX_STRETCH;
    beyond that, crop instead."""
    tex_ar = tex_h / tex_w
    stretch = np.clip(body_ratio / tex_ar, 1 / MAX_STRETCH, MAX_STRETCH)
    need = body_ratio / stretch
    if need < tex_ar:          # product is relatively taller: show a central band
        hw = need * tex_w
        y0 = (tex_h - hw) / 2
        return 0.0, float(tex_w), y0, y0 + hw
    ww = tex_h / need          # product is relatively wider: show its central columns
    x0 = (tex_w - ww) / 2
    return x0, x0 + ww, 0.0, float(tex_h)


# ------------------------------------------------------------------ main

def run(work: Path, obj, cutout_path: Path, progress):
    t0 = time.time()
    info = media.probe(work / "input.mp4")
    W_img, H_img, fps = info["width"], info["height"], info["fps"]
    shape_kind = obj.get("shape", "flat")

    masks, mshape, scale = _get_masks(work, obj, info, progress)
    if not masks:
        raise RuntimeError("Tracking lost the object in every frame.")
    progress(0.71, "Fitting geometry")
    res = _geometry(masks, mshape, 1.0 / scale, shape_kind, obj.get("category"))
    if not res:
        raise RuntimeError("Could not fit the object's shape in any frame.")
    geo, body_ratio = res

    # Fill brief dropouts with the nearest frame so the old object doesn't flash through.
    mframes = np.array(sorted(geo))
    fill = {}
    for f in range(int(mframes.min()), int(mframes.max()) + 1):
        if f in geo:
            continue
        j = np.searchsorted(mframes, f)
        near = min((mframes[k] for k in (j - 1, j) if 0 <= k < len(mframes)), key=lambda x: abs(x - f))
        if abs(near - f) <= GAP_FILL:
            fill[f] = int(near)

    cut = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    if cut.shape[2] == 3:
        cut = np.dstack([cut, np.full(cut.shape[:2], 255, np.uint8)])
    tex = _label_texture(cut, shape_kind)
    # Pre-shrink to ~2x the largest on-screen size so remap doesn't alias.
    maxH = max(float(np.hypot(g[0] - g[2], g[1] - g[3])) for g in geo.values())
    s0 = min(1.0, 2.0 * maxH / tex.shape[0])
    if s0 < 1.0:
        tex = cv2.resize(tex, (max(2, int(tex.shape[1] * s0)), max(2, int(tex.shape[0] * s0))),
                         interpolation=cv2.INTER_AREA)
    tex = tex.astype(np.float32) / 255.0
    if shape_kind != "cylinder":
        tex[:, :, :3] *= tex[:, :, 3:4]  # premultiplied; flat products keep their silhouette
    window = _texture_window(tex.shape[0], tex.shape[1], body_ratio)

    out_dir = work / "outputs"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    out_path = out_dir / f"result_{obj['id']}_{stamp}.mp4"
    writer = media.VideoWriter(out_path, W_img, H_img, fps, audio_src=work / "input.mp4")
    look = Look()
    replaced = 0
    try:
        for fi, frame in enumerate(media.iter_frames(work / "input.mp4")):
            src = fi if fi in geo else fill.get(fi)
            if src is not None:
                frame = _render(frame, src, geo, masks, mshape, tex, window, shape_kind, look)
                replaced += 1
            writer.write(frame)
            if fi % 15 == 0:
                progress(0.72 + 0.27 * fi / info["frames"], f"Rendering ({fi}/{info['frames']})")
    finally:
        writer.close()

    took = time.time() - t0
    return {
        "video": f"outputs/{out_path.name}",
        "stamp": stamp,
        "download_name": f"vid-bid-{obj['label']}-{stamp}.mp4",
        "replaced_frames": replaced,
        "stats": f"Replaced in {replaced} of {info['frames']} frames · {took:.0f}s",
    }


def _render(frame, src, geo, masks, mshape, tex, window, shape_kind, look):
    H_img, W_img = frame.shape[:2]
    Tx, Ty, Bx, By, h_t, h_b, e_t, e_b = geo[src]
    T, B = np.array([Tx, Ty]), np.array([Bx, By])
    axis = T - B
    Hb = float(np.linalg.norm(axis))
    if Hb < 4:
        return frame
    up = axis / Hb
    right = np.array([-up[1], up[0]])
    Wd = h_t + h_b  # mean diameter

    m_small = _clean(_unpack(masks[src], mshape))
    mask = cv2.resize(m_small.astype(np.uint8) * 255, (W_img, H_img), interpolation=cv2.INTER_LINEAR) > 127
    if not mask.any():
        return frame

    # ROI: old mask + new body (incl. bottom ellipse) + padding for blur/inpaint context.
    quad = np.array([T + right * h_t + up * e_t, T - right * h_t + up * e_t,
                     B + right * h_b - up * e_b, B - right * h_b - up * e_b])
    ys, xs = np.nonzero(mask)
    pad = int(0.15 * Wd) + 8
    x0 = int(max(0, min(quad[:, 0].min(), xs.min()) - pad))
    y0 = int(max(0, min(quad[:, 1].min(), ys.min()) - pad))
    x1 = int(min(W_img, max(quad[:, 0].max(), xs.max()) + pad))
    y1 = int(min(H_img, max(quad[:, 1].max(), ys.max()) + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    roi_u8 = frame[y0:y1, x0:x1]
    roi = roi_u8.astype(np.float32) / 255.0
    mroi = mask[y0:y1, x0:x1]

    # Cylinder coordinates of every ROI pixel: t across the width, s down the body.
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    rx, ry = gx - T[0], gy - T[1]
    ut = rx * up[0] + ry * up[1]
    lam = np.clip(-ut / Hb, -0.3, 1.3)                  # 0 at the top rim, 1 at the bottom rim
    hloc = h_t + (h_b - h_t) * lam                      # perspective taper
    t = (rx * right[0] + ry * right[1]) / hloc
    sq = np.sqrt(np.clip(1 - t * t, 0, 1))
    front_top = -e_t * sq                        # front edge of the top rim, relative to T
    s = (front_top - ut) / (Hb + (e_b - e_t) * sq)
    inside_t = np.abs(t) <= 1
    lid = inside_t & (ut > front_top) & (ut <= e_t * sq + 1)   # visible top ellipse: keep original

    # Body coverage, antialiased at the silhouette and rims.
    a_w = np.clip((1 - np.abs(t)) * hloc + 0.5, 0, 1)
    a_s = np.clip(s * Hb + 0.5, 0, 1) * np.clip((1 - s) * Hb + 0.5, 0, 1)
    body_a = (a_w * a_s).astype(np.float32)

    # Sample the label texture.
    tx0, tx1, ty0, ty1 = window
    map_x = (tx0 + (t + 1) / 2 * (tx1 - tx0)).astype(np.float32)
    map_y = (ty0 + s * (ty1 - ty0)).astype(np.float32)
    samp = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if shape_kind == "cylinder":
        color = samp[:, :, :3]
        alpha = body_a
    else:
        alpha = body_a * samp[:, :, 3]
        color = samp[:, :, :3] / np.maximum(samp[:, :, 3:4], 1e-3)

    # --- lighting
    lum = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2GRAY).astype(np.float64)
    body_sel = mroi & (s > 0.1) & (s < 0.9) & inside_t
    grad = look.ema("grad", _fit_gradient(lum, t, body_sel))
    g = np.clip(grad[0] + grad[1] * t + grad[2] * t * t, 0.55, 1.2)
    shade = 0.4 + 0.6 * g
    if shape_kind == "cylinder":
        shade = shade * (1.0 - 0.38 * np.abs(np.clip(t, -1, 1)) ** 3.5)
    ring = cv2.dilate(mroi.astype(np.uint8), np.ones((pad, pad), np.uint8)).astype(bool) & ~mroi
    ring_bgr = roi[ring].mean(0) if ring.sum() > 50 else roi.reshape(-1, 3).mean(0)
    tint = look.ema("tint", ring_bgr / max(ring_bgr.mean(), 1e-3))
    ring_lum = float(ring_bgr @ np.array([0.114, 0.587, 0.299]))
    gain = float(look.ema("gain", np.clip(0.62 + 0.5 * ring_lum, 0.7, 1.05)))
    color = color * shade[..., None].astype(np.float32)
    color = color * ((1 + 0.35 * (tint - 1)) * gain).astype(np.float32)[None, None, :]
    color = np.clip(color, 0, 1)

    # --- blur: match the footage's focus, then add motion blur
    prem = np.dstack([color * alpha[..., None], alpha]).astype(np.float32)
    target = _sharpness(lum, cv2.erode(mroi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    if target is not None:
        interior = cv2.erode((body_a > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        gray_new = cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        best_sig, best_err = 0.5, 1e9
        for sig in (0.5, 0.9, 1.4, 2.0, 2.8, 3.8):
            sh = _sharpness(cv2.GaussianBlur(gray_new, (0, 0), sig), interior)
            if sh is not None and abs(sh - target) < best_err:
                best_sig, best_err = sig, abs(sh - target)
        sig = float(look.ema("blur", best_sig, a=0.2))
    else:
        sig = 0.6
    prem = cv2.GaussianBlur(prem, (0, 0), sig)
    prev = geo.get(src - 1)
    if prev is not None:
        k = _motion_kernel(Tx - prev[0], Ty - prev[1])
        if k is not None:
            prem = cv2.filter2D(prem, -1, k)
    a = np.clip(prem[:, :, 3], 0, 1)

    # --- occluders: inside the new body but not part of the tracked object (fingers, foam...)
    # Only inside the mask's convex hull: where the fitted body merely overshoots the mask
    # (fit error, motion blur) we draw the product rather than cut a hole in it.
    hull_pts = cv2.convexHull(np.stack(np.nonzero(mroi)[::-1], 1).astype(np.int32))
    hull = np.zeros(mroi.shape, np.uint8)
    cv2.fillConvexPoly(hull, hull_pts, 1)
    occ = (body_a > 0.5) & hull.astype(bool) & ~mroi
    kk = max(3, int(0.04 * Wd)) | 1
    occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN, np.ones((kk, kk), np.uint8))
    occ_soft = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 1.0)
    keep = 1 - occ_soft
    a = a * keep

    # --- slivers of the old object outside the new body/lid get inpainted
    old = cv2.dilate(mroi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    leftover = old & (a < 0.5) & ~lid & ~occ.astype(bool)
    if shape_kind == "cylinder":
        leftover &= ~(inside_t & (s > 1))   # below the bottom rim: the can's base, keep it
    base_img = roi_u8.copy()
    if leftover.any():
        base_img = cv2.inpaint(base_img, leftover.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    base_f = base_img.astype(np.float32) / 255.0

    out = prem[:, :, :3] * keep[..., None] + base_f * (1 - a[..., None])
    frame[y0:y1, x0:x1] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    return frame
