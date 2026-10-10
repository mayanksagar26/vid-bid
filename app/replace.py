"""Replace a tracked object with a product cutout, frame by frame.

Pipeline
  1. SAM 2 masks for every frame the object is visible (cached per object).
  2. Geometry per frame from the mask: axis and body width with perspective taper. For cylinders
     (cans, bottles, cups) also the lid and base ellipses, fitted to the ends of the outline, and the
     outline in between (neck, shoulder, base bevel) learnt over the whole clip. Ends cut off by the
     frame edge are reconstructed from frames where they're visible. Outliers are dropped, then
     everything is smoothed over time.
  3. Render: the product photo is unrolled (the front half of its surface, row by row) and wrapped
     back onto the object as a surface of revolution, landmark to landmark: neck to neck, body to
     body, base to base. The original lid stays. Lighting is read off the original object (shade
     map, highlights) plus a Fresnel rim reflecting the surroundings; white balance and exposure
     come from the surroundings; blur is matched to the footage (focus + motion); the original
     surface's fine detail (condensation, grain) goes back on top, and the surroundings wrap a
     little light over the new edge.
  4. Composite: pixels inside the body that SAM says are *not* the object (fingers, foam, straws)
     stay in front. Slivers of the old object outside the new body are inpainted.
  5. Encode with the original audio.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d

from . import media, segment

GAP_FILL = 6            # frames: reuse a neighbouring mask when SAM drops the object briefly
TAIL_PAD_SECONDS = 0.5  # keep tracking a little past the last detection
SEG_JOIN_SECONDS = 5.0  # bridge detection gaps (e.g. a can tilted mid-pour) with SAM tracking
MAX_STRETCH = 1.2       # max anisotropic stretch of the label before cropping instead
HIGHLIGHT_GAIN = 0.6    # how much of the original's specular highlights goes onto the new label
DETAIL_GAIN = 0.9       # condensation / grain transferred from the original surface
WRAP_GAIN = 0.35        # light wrap from the surroundings at the silhouette edge
# Typical visible body height / diameter between the rims, used when no frame shows both ends.
DEFAULT_BODY_RATIO = {"can": 1.7, "bottle": 2.8, "cup": 1.1, "jar": 1.2}
# Outline between the ends, as radius / body radius at LAMS (0 = lid centre, 1 = base centre).
PROFILE_BINS = 65
LAMS = np.linspace(0.0, 1.0, PROFILE_BINS)
SHOULDER = 0.97         # radius ratio where the straight body starts/ends
SEAM = 0.04             # height of a can's lid seam / its diameter (bare metal, kept from the footage)


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


def _end_fit(uc, wid, u_end, hw_end, sign, thr=0.0):
    """Ellipse at one end of a cylinder from the silhouette's half-width profile.

    Near the end the outline is half an ellipse (half-width r, half-height e) centred e inside the
    extreme point: wid(d) = r * sqrt(1 - ((e - d) / e)^2), d = distance in from the end. Fitting the
    end on its own keeps a can's shoulder (neck narrower than body) from inflating the ellipse.
    u_end may already be trimmed to where the outline is thr wide; the tip it cut is added back.
    Returns (e, r, tip) with the true end at u_end + sign * tip, or None."""
    d = sign * (u_end - uc)
    ok = np.isfinite(wid) & (d >= 0)
    step = float(np.median(np.abs(np.diff(uc)))) if len(uc) > 1 else 1.0
    best, best_err = None, np.inf
    for e in np.linspace(0.04, 0.6, 29) * hw_end:
        near = ok & (np.abs(d - e) <= max(0.12 * e, step))
        if not near.any():
            continue
        r = float(np.median(wid[near]))
        tip = e * (1 - np.sqrt(max(0.0, 1 - (thr / r) ** 2))) if r > thr else 0.0
        dd = d + tip
        zone = ok & (dd <= e)
        if zone.sum() < 3:
            continue
        z = (e - dd[zone]) / e
        pred = r * np.sqrt(np.clip(1 - z * z, 0, 1))
        err = float(np.mean((wid[zone] - pred) ** 2)) / max(r, 1.0) ** 2
        if err < best_err:
            best, best_err = (float(e), r, tip), err
    return best


def _measure(mask, prev_up, cylinder):
    """Silhouette of one mask (mask pixel units).

    Seen from above or below, a cylinder's sides converge (perspective), so the left and right
    edges are fitted as lines along the axis: centre c(u) = c0 + c1*u, half-width h(u) = h0 + h1*u.
    For cylinders the lid and base are fitted as ellipses (erel_* = half-height / half-width, i.e.
    how much the camera looks down on that end) and the outline in between is sampled as a profile
    relative to the body width. Returns up/right axes, a_t/a_b (end centres along up), the line
    coefficients, erel_t/erel_b, prof and top_ok/bot_ok (that end is visible, not cut by the frame
    edge)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    hull = cv2.convexHull(pts)
    (_, _), (rw, rh), ang = cv2.minAreaRect(hull)
    a = np.deg2rad(ang)
    e1 = np.array([np.cos(a), np.sin(a)])
    e2 = np.array([-np.sin(a), np.cos(a)])
    if prev_up is None:
        if max(rw, rh) > 1.25 * min(rw, rh):  # clearly elongated: the axis runs along it
            up = e1 if rw > rh else e2
        else:
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
         "erel_t": 0.0, "erel_b": 0.0, "prof": None,
         "top_ok": True, "bot_ok": True, "area": float(mask.sum()),
         "solid": float(mask.sum()) / max(1.0, cv2.contourArea(hull))}  # < ~0.97: something covers it

    on_border = (xs <= 1) | (ys <= 1) | (xs >= w - 2) | (ys >= h - 2)
    if on_border.any():
        bu = pu[on_border]
        g["top_ok"] = not (bu > u1 - 0.15 * (u1 - u0)).any()
        g["bot_ok"] = not (bu < u0 + 0.15 * (u1 - u0)).any()

    # Side lines from the middle of the body (ends excluded), bins along the axis.
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
    if not cylinder or L < 8:
        return g

    # Half-width of the outline along the axis, ~1 px bins over the whole mask (ends included).
    umin, umax = float(pu.min()), float(pu.max())
    nb = int(np.clip(umax - umin, 16, 512))
    ucs = umin + (np.arange(nb) + 0.5) * (umax - umin) / nb
    bi = np.clip(((pu - umin) / max(umax - umin, 1e-6) * nb).astype(int), 0, nb - 1)
    lo, hi = np.full(nb, np.inf), np.full(nb, -np.inf)
    np.minimum.at(lo, bi, pr)
    np.maximum.at(hi, bi, pr)
    # Half-width from the body's centreline, wider side: a hand or thumb on one side only ever
    # narrows that side, and the outline is symmetric about the axis.
    cc = g["c"][0] + g["c"][1] * ucs
    wid = np.where(hi >= lo, np.maximum(cc - lo, hi - cc), np.nan)

    def hw_at(u):
        return max(1.0, g["hw"][0] + g["hw"][1] * u)

    # The ends: last rows at least 30% of the body wide, so thin things attached to the mask
    # (a pouring stream, a straw) don't pass for the lid. _end_fit adds the trimmed tip back.
    thr = 0.3 * g["hw"][0]
    wide = np.where(np.nan_to_num(wid) >= thr)[0]
    if len(wide) >= 2:
        half = 0.5 * (umax - umin) / nb
        umin, umax = ucs[wide[0]] - half, ucs[wide[-1]] + half
    if g["top_ok"]:
        fit = _end_fit(ucs, wid, umax, hw_at(umax), +1, thr)
        if fit:
            e, r, tip = fit
            g["a_t"], g["erel_t"] = umax + tip - e, min(0.75, e / max(r, 1.0))
    if g["bot_ok"]:
        fit = _end_fit(ucs, wid, umin, hw_at(umin), -1, thr)
        if fit:
            e, r, tip = fit
            g["a_b"], g["erel_b"] = umin - tip + e, min(0.75, e / max(r, 1.0))
    if g["top_ok"] and g["bot_ok"] and g["a_t"] - g["a_b"] > 4:
        lam = (g["a_t"] - ucs) / (g["a_t"] - g["a_b"])
        rel = wid / np.maximum(g["hw"][0] + g["hw"][1] * ucs, 1.0)
        ok = np.isfinite(rel) & (lam >= -0.02) & (lam <= 1.02)
        if ok.sum() >= 8:
            g["prof"] = np.interp(LAMS, lam[ok][::-1], rel[ok][::-1])
    return g


def _runs(frames):
    frames = np.asarray(frames)
    breaks = np.where(np.diff(frames) > 1)[0] + 1
    return np.split(np.arange(len(frames)), breaks)


def _smooth(frames, vals, sigmas, medians=None):
    out = vals.copy()
    for run in _runs(frames):
        if len(run) < 3:
            continue
        for j, s in enumerate(sigmas):
            if s <= 0:
                continue
            k = medians[j] if medians else 5
            v = median_filter(vals[run, j], size=min(k, len(run)) | 1, mode="nearest")
            out[run, j] = gaussian_filter1d(v, s, mode="nearest")
    return out


def _default_profile(category):
    if category == "can":  # neck ~0.8 of the body, shoulder ~12% of the height, base bevel
        return np.interp(LAMS, [0.0, 0.13, 0.91, 1.0], [0.8, 1.0, 1.0, 0.8])
    return np.ones(PROFILE_BINS)


def _clip_profile(gs, category):
    """Outline between the ends over the clip, from frames where nothing covers the object
    (fingers notch the silhouette) when there are enough of them."""
    profs = [g["prof"] for g in gs if g["prof"] is not None]
    clean = [g["prof"] for g in gs if g["prof"] is not None and g["solid"] >= 0.975]
    if len(clean) >= 5:
        profs = clean
    if len(profs) < 5:
        return _default_profile(category)
    p = np.nanmedian(np.array(profs), 0)
    p = gaussian_filter1d(np.nan_to_num(p, nan=1.0), 1.0, mode="nearest")
    p = np.clip(p, 0.3, 1.05)
    mid = (LAMS > 0.3) & (LAMS < 0.7)
    return np.clip(p / max(float(np.median(p[mid])), 1e-3), 0.3, 1.0)


def _landmarks(prof):
    """lambda where the straight body starts and ends."""
    body = np.where(prof >= SHOULDER)[0]
    if len(body) < 2:
        return 0.0, 1.0
    return float(LAMS[body[0]]), float(LAMS[body[-1]])


def _geometry(masks, shape, inv_scale, shape_kind, category):
    """Per-frame geometry in full-resolution pixels:
    T (lid centre xy), B (base centre xy), body half-widths at T and B, erel_t, erel_b.
    Plus clip-wide shape: body height/width ratio and the outline profile."""
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
    er_t = [g["erel_t"] for g in gs if g["top_ok"] and g["erel_t"] > 0]
    er_b = [g["erel_b"] for g in gs if g["bot_ok"] and g["erel_b"] > 0]
    erel_t = float(np.median(er_t)) if er_t else 0.2 * cyl
    erel_b = float(np.median(er_b)) if er_b else erel_t * 1.1
    erel_b = max(erel_b, erel_t)  # a can's base bevel makes its own fit unreliable
    prof = _clip_profile(gs, category) if cyl else np.ones(PROFILE_BINS)

    rows = []
    for g, Wg in zip(gs, W):
        a_t, a_b = g["a_t"], g["a_b"]
        et = g["erel_t"] if g["top_ok"] and g["erel_t"] > 0 else erel_t
        eb = g["erel_b"] if g["bot_ok"] and g["erel_b"] > 0 else erel_b
        if not g["top_ok"] and not g["bot_ok"]:
            mid = (a_t + a_b) / 2
            a_t, a_b = mid + ratio * Wg / 2, mid - ratio * Wg / 2
        elif not g["top_ok"]:
            a_t = a_b + ratio * Wg
        elif not g["bot_ok"]:
            a_b = a_t - ratio * Wg
        c = g["c"]
        T = g["up"] * a_t + g["right"] * (c[0] + c[1] * a_t)
        Bc = g["up"] * a_b + g["right"] * (c[0] + c[1] * a_b)
        rows.append([T[0], T[1], Bc[0], Bc[1], hw_at(g, a_t), hw_at(g, a_b), et, max(eb, et)])
    vals = np.array(rows, np.float64)
    vals[:, :6] *= inv_scale
    # The end ellipses change slowly but their fits are noisy under hands: ~0.5 s median.
    sm = _smooth(frames, vals, sigmas=[1.5, 1.5, 1.5, 1.5, 3.0, 3.0, 5.0, 5.0],
                 medians=[5, 5, 5, 5, 5, 5, 13, 13])
    return {f: sm[i] for i, f in enumerate(frames)}, {"ratio": ratio, "prof": prof,
                                                     "keep_ends": category == "can"}


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


def _scene_light(roi, sel, Wd, look):
    """Light on the original object's surface: a shade map (multiplies the new label) and a
    highlight map (added on top, like the gloss on a printed can).

    Read from V = max(R, G, B), which tracks the light on bare metal and on saturated ink alike,
    so most of the old label's artwork cancels out (red text on a silver can disappears)."""
    if sel.sum() < 150:
        return None
    V = roi.max(2)
    sig = max(2.0, 0.05 * Wd)
    w = sel.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    Lf = cv2.GaussianBlur(V * w, (0, 0), sig) / np.maximum(den, 1e-3)
    vals = Lf[sel]
    ref = max(float(look.ema("v_ref", np.percentile(vals, 55))), 0.05)
    top = float(look.ema("v_top", np.percentile(vals, 93)))
    conf = np.clip(den / 0.5, 0, 1)  # far from visible surface (fingers, rims): use the reference
    Lf = conf * Lf + (1 - conf) * ref
    shade = np.clip(Lf / ref, 0.35, 1.25)
    thr = max(top, 1.08 * ref)
    hl = np.clip((Lf - thr) / max(1.0 - thr, 0.05), 0, 1)
    return shade.astype(np.float32), hl.astype(np.float32)


def _surface_detail(roi, sel, Wd):
    """Fine detail of the original surface (condensation, scuffs, sensor grain, compression noise)
    to lay over the new label so it sits in the footage instead of looking printed on top.

    High-pass of V, gated off wherever the old label has colour edges (its artwork)."""
    V = roi.max(2)
    sd = max(1.2, 0.01 * Wd)
    hp = V - cv2.GaussianBlur(V, (0, 0), sd)
    C = V - roi.min(2)
    m1 = cv2.GaussianBlur(C, (0, 0), 2 * sd)
    m2 = cv2.GaussianBlur(C * C, (0, 0), 2 * sd)
    cstd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    k = max(3, int(4 * sd) | 1)
    inner = cv2.erode(sel.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    gate = np.clip(1 - cstd / 0.07, 0, 1) * cv2.GaussianBlur(inner, (0, 0), sd)
    return (np.clip(hp, -0.12, 0.12) * gate).astype(np.float32)


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

def _unwrap(cut):
    """Unroll the front half of a cylindrical product photo: row y, column j <-> sin(angle) in
    [-1, 1], each row scaled by the photo's own outline there (neck, shoulder, body, base).
    Returns BGR float texture and landmark rows (top, body start, body end, bottom)."""
    a = cut[:, :, 3].astype(np.float32) / 255.0
    solid = a > 0.5
    rows = np.where(solid.sum(1) >= 3)[0]
    r0, r1 = int(rows.min()), int(rows.max())
    n = r1 - r0 + 1
    sub = solid[r0:r1 + 1]
    has = sub.any(1)
    lo = np.where(has, sub.argmax(1), np.nan)
    hi = np.where(has, sub.shape[1] - 1 - sub[:, ::-1].argmax(1), np.nan)
    yy = np.arange(n)
    lo = np.interp(yy, yy[has], lo[has])
    hi = np.interp(yy, yy[has], hi[has])
    k = max(3, (n // 100) | 1)
    lo = gaussian_filter1d(median_filter(lo, k, mode="nearest"), 1.0, mode="nearest")
    hi = gaussian_filter1d(median_filter(hi, k, mode="nearest"), 1.0, mode="nearest")
    cen, rad = (lo + hi) / 2, np.maximum((hi - lo) / 2 - 1.0, 0.5)  # inset: skip the matte's edge
    body_r = float(np.percentile(rad, 75))
    body = np.where(rad >= SHOULDER * body_r)[0]
    ps, pe = (int(body[0]), int(body[-1])) if len(body) >= 2 else (0, n - 1)

    N = max(8, int(round(2 * body_r)))
    st = np.linspace(-1, 1, N, dtype=np.float32)
    map_x = (cen[:, None] + st[None, :] * rad[:, None]).astype(np.float32)
    map_y = np.repeat((yy + r0)[:, None], N, 1).astype(np.float32)
    prem = np.dstack([cut[:, :, :3].astype(np.float32) / 255.0 * a[..., None], a])
    U = cv2.remap(prem, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = U[:, :, :3] / np.maximum(U[:, :, 3:4], 1e-3)
    rgb = np.where(U[:, :, 3:4] > 0.02, rgb, 0).astype(np.float32)
    return np.clip(rgb, 0, 1), (0, ps, pe, n - 1), body_r


def _delight(tex, rows, strength=0.75):
    """Remove most of the studio reflections baked into a can/bottle product photo: bright vertical
    bands sitting on the label's darkest level. The scene's own highlights go on instead."""
    V = tex.max(2)
    n = V.shape[1]
    body = slice(rows[0], rows[1] + 1)
    prof = np.median(V[body], 0)
    base = gaussian_filter1d(minimum_filter1d(prof, max(3, n // 4), mode="nearest"), max(1.0, n / 40))
    sat = np.median((V - tex.min(2))[body], 0)
    band = np.clip(prof - base, 0, None) * np.clip(1 - sat / 0.25, 0, 1)
    band = gaussian_filter1d(band, max(1.0, n / 80))
    # Only on the printed body; fade in over a few % so the shoulder doesn't get a seam.
    y = np.arange(len(V), dtype=np.float32)
    ramp = max(2.0, 0.03 * len(V))
    wy = np.clip((y - rows[0]) / ramp, 0, 1) * np.clip((rows[1] - y) / ramp, 0, 1)
    return np.clip(tex - strength * wy[:, None, None] * band[None, :, None], 0, 1).astype(np.float32)


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


def _cylinder_texture(cut, shape_info, max_h):
    """Unrolled, de-lit product texture plus the mapping lambda -> texture row (landmark to
    landmark) and the column window."""
    U, (p0, ps, pe, p1), body_r = _unwrap(cut)
    U = _delight(U, (ps, pe))
    ls, le = _landmarks(shape_info["prof"])
    if le - ls < 0.2:
        ls, le = 0.0, 1.0
        ps, pe = p0, p1
    if shape_info.get("keep_ends") and ps > p0:
        # A can's shoulder: the photo's own neck carries studio highlights and matte edges at a
        # steep angle, so continue the label's top colour (column-wise median) up to the seam
        # instead; the scene's shading gives it the shoulder's form.
        k = max(3, int(0.05 * (pe - ps)))
        strip = np.median(U[ps:ps + k], 0)
        U = U.copy()
        U[p0:ps + 1] = strip[None]
    # The straight body: same aspect as the original's (stretch <= MAX_STRETCH, then crop).
    target_ar = shape_info["ratio"] * (le - ls)
    x0, x1, y0, y1 = _texture_window(max(pe - ps, 1), U.shape[1], target_ar)
    h0, w0 = U.shape[:2]
    s0 = min(1.0, 2.0 * max_h / h0)  # ~2x the largest on-screen size so remap doesn't alias
    if s0 < 1.0:
        U = cv2.resize(U, (max(2, int(w0 * s0)), max(2, int(h0 * s0))), interpolation=cv2.INTER_AREA)
    sx, sy = U.shape[1] / w0, U.shape[0] / h0
    lm = {"ls": ls, "le": le,
          "rows": np.array([p0, ps, ps + y0, ps + y1, pe, p1], np.float64) * sy,
          "cols": (x0 * sx, x1 * sx)}
    return {"img": U, "lm": lm}


def _flat_texture(cut, ratio, max_h):
    tex = cut
    s0 = min(1.0, 2.0 * max_h / tex.shape[0])
    if s0 < 1.0:
        tex = cv2.resize(tex, (max(2, int(tex.shape[1] * s0)), max(2, int(tex.shape[0] * s0))),
                         interpolation=cv2.INTER_AREA)
    tex = tex.astype(np.float32) / 255.0
    tex[:, :, :3] *= tex[:, :, 3:4]  # premultiplied; flat products keep their silhouette
    return {"img": tex, "window": _texture_window(tex.shape[0], tex.shape[1], ratio)}


def _tex_rows(lam, lm, l0=0.0):
    """Texture row for each lambda: neck/shoulder, body window, base bevel mapped piecewise
    (the top part starting at l0)."""
    ls, le = lm["ls"], lm["le"]
    p0, ps, wy0, wy1, pe, p1 = lm["rows"]
    top = p0 + (ps - p0) * np.clip(lam - l0, 0, None) / max(ls - l0, 1e-6)
    mid = wy0 + (wy1 - wy0) * (lam - ls) / max(le - ls, 1e-6)
    bot = pe + (p1 - pe) * (lam - le) / max(1 - le, 1e-6)
    out = np.where(lam < ls, top, np.where(lam <= le, mid, bot))
    return out.astype(np.float32)


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
    geo, shape_info = res

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
    maxH = max(float(np.hypot(g[0] - g[2], g[1] - g[3])) for g in geo.values())
    if shape_kind == "cylinder":
        tex = _cylinder_texture(cut, shape_info, maxH)
    else:
        tex = _flat_texture(cut, shape_info["ratio"], maxH)

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
                frame = _render(frame, src, geo, masks, mshape, tex, shape_info, shape_kind, look)
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


def _render(frame, src, geo, masks, mshape, tex, shape_info, shape_kind, look):
    H_img, W_img = frame.shape[:2]
    Tx, Ty, Bx, By, h_t, h_b, er_t, er_b = geo[src]
    T, B = np.array([Tx, Ty]), np.array([Bx, By])
    axis = T - B
    Hb = float(np.linalg.norm(axis))
    if Hb < 4:
        return frame
    up = axis / Hb
    right = np.array([-up[1], up[0]])
    Wd = h_t + h_b  # mean diameter
    cyl = shape_kind == "cylinder"
    prof = shape_info["prof"] if cyl else np.ones(PROFILE_BINS)
    if not cyl:
        er_t = er_b = 0.0

    m_small = _clean(_unpack(masks[src], mshape))
    mask = cv2.resize(m_small.astype(np.uint8) * 255, (W_img, H_img), interpolation=cv2.INTER_LINEAR) > 127
    if not mask.any():
        return frame

    # ROI: old mask + new body (incl. both end ellipses) + padding for blur/inpaint context.
    e_t, e_b = er_t * h_t, er_b * h_b
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

    # Surface coordinates of every ROI pixel, for a surface of revolution seen from above:
    # lam along the axis (0 = lid centre, 1 = base centre), st = sin(angle) across the front.
    # A point (lam, st) projects to  ut = -lam*Hb - erel(lam)*r(lam)*sqrt(1-st^2),  pr = r(lam)*st,
    # inverted by fixed-point iteration (r changes slowly with lam).
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    rx, ry = gx - T[0], gy - T[1]
    ut = rx * up[0] + ry * up[1]
    pr = rx * right[0] + ry * right[1]
    lam = np.clip(-ut / Hb, -0.3, 1.3)
    for _ in range(4):
        lc = np.clip(lam, 0, 1)
        r = (h_t + (h_b - h_t) * lc) * np.interp(lc, LAMS, prof).astype(np.float32)
        st = pr / np.maximum(r, 1.0)
        ct = np.sqrt(np.clip(1 - st * st, 0, 1))
        lam = np.clip((-ut - (er_t + (er_b - er_t) * lc) * r * ct) / Hb, -0.3, 1.3)
    tc = np.clip(st, -1, 1)
    inside_t = np.abs(st) <= 1
    # Range of lam that gets the product. A can's lid seam and base bevel are bare metal on any can:
    # the footage's own stay (with its real reflections) and only the printed part is replaced.
    l_top, l_bot = 0.0, 1.0
    if cyl and shape_info.get("keep_ends"):
        ls, le = _landmarks(prof)
        l_top = min(SEAM * (h_t + h_b) / Hb, 0.5 * ls) if ls > 0 else 0.0
        l_bot = le if le < 1 else 1.0
    r_lid = h_t * float(prof[0])
    lid = inside_t & (lam < l_top) & (ut <= e_t * float(prof[0]) * ct + 1) & (np.abs(pr) <= r_lid)

    # Body coverage, antialiased at the silhouette and ends.
    a_w = np.clip((1 - np.abs(st)) * r + 0.5, 0, 1)
    a_s = np.clip((lam - l_top) * Hb + 0.5, 0, 1) * np.clip((l_bot - lam) * Hb + 0.5, 0, 1)
    body_a = (a_w * a_s).astype(np.float32)

    # Sample the product.
    if cyl:
        lm = tex["lm"]
        cx0, cx1 = lm["cols"]
        map_x = (cx0 + (tc + 1) / 2 * (cx1 - cx0 - 1)).astype(np.float32)
        map_y = _tex_rows(lam, lm, l_top)
        color = cv2.remap(tex["img"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = body_a
    else:
        tx0, tx1, ty0, ty1 = tex["window"]
        map_x = (tx0 + (st + 1) / 2 * (tx1 - tx0)).astype(np.float32)
        map_y = (ty0 + lam * (ty1 - ty0)).astype(np.float32)
        samp = cv2.remap(tex["img"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = body_a * samp[:, :, 3]
        color = samp[:, :, :3] / np.maximum(samp[:, :, 3:4], 1e-3)

    # --- lighting, read off the original object where possible
    lum = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2GRAY).astype(np.float64)
    body_sel = mroi & (lam > 0.1) & (lam < 0.9) & inside_t
    ring = cv2.dilate(mroi.astype(np.uint8), np.ones((pad, pad), np.uint8)).astype(bool) & ~mroi
    ring_bgr = roi[ring].mean(0) if ring.sum() > 50 else roi.reshape(-1, 3).mean(0)
    tint = look.ema("tint", ring_bgr / max(ring_bgr.mean(), 1e-3))
    ring_lum = float(ring_bgr @ np.array([0.114, 0.587, 0.299]))
    gain = float(look.ema("gain", np.clip(0.62 + 0.5 * ring_lum, 0.7, 1.05)))
    env = look.ema("env", ring_bgr).astype(np.float32)
    light = _scene_light(roi, body_sel, Wd, look)
    if light is not None:
        shade, hl = light
    else:  # nothing visible to read: quadratic fit across the width + cylinder falloff
        grad = look.ema("grad", _fit_gradient(lum, st, body_sel))
        g = np.clip(grad[0] + grad[1] * st + grad[2] * st * st, 0.55, 1.2)
        shade = (0.4 + 0.6 * g).astype(np.float32)
        if cyl:
            shade = shade * (1.0 - 0.38 * np.abs(tc) ** 3.5).astype(np.float32)
        hl = np.zeros_like(shade)
    color = color * shade[..., None]
    color = color * ((1 + 0.35 * (tint - 1)) * gain).astype(np.float32)[None, None, :]
    if cyl:
        # Glossy coating: highlights stay bright over any ink, and the rim reflects the surroundings
        # more and more towards grazing angles (Schlick's Fresnel, F0 = 0.04 for lacquer).
        fres = (0.04 + 0.96 * (1 - ct) ** 5).astype(np.float32)
        color = color + HIGHLIGHT_GAIN * hl[..., None] * (env / max(float(env.max()), 1e-3)) \
            + fres[..., None] * env[None, None, :]
    # Nothing in real footage is pure black: lens flare lifts the darkest ink a little.
    lift = 0.012 + 0.03 * ring_lum
    color = np.clip(lift + (1 - lift) * color, 0, 1)

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

    # --- the old object is painted out from its surroundings (at half resolution: it's smooth).
    # Only slivers outside the new silhouette and the new edge's antialiasing show it, but filling
    # all of it keeps the old label's colours from bleeding into those slivers.
    old = cv2.dilate(mroi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    fill = old & ~lid & ~occ.astype(bool)
    if cyl:
        fill &= ~(inside_t & (lam > l_bot))   # the can's base, keep it
    base_img = roi_u8.copy()
    if fill.any():
        hs = 0.5 if min(fill.shape) >= 64 else 1.0
        small = cv2.resize(roi_u8, None, fx=hs, fy=hs, interpolation=cv2.INTER_AREA)
        fm = cv2.resize(fill.astype(np.uint8) * 255, (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        fm = cv2.dilate(fm, np.ones((3, 3), np.uint8))
        painted = cv2.inpaint(small, fm, 3, cv2.INPAINT_TELEA)
        painted = cv2.resize(painted, (fill.shape[1], fill.shape[0]), interpolation=cv2.INTER_LINEAR)
        base_img[fill] = painted[fill]
    base_f = base_img.astype(np.float32) / 255.0
    if cyl and shape_info.get("keep_ends"):
        # The lid, seam and base stay from the footage because they are bare metal on any can. Where
        # the end fit overshoots onto the old print (obliquely seen ends), take its colour out:
        # V = max(R, G, B) keeps the metal's shading and droplets, while saturated ink turns into
        # the same light grey as the aluminium around it. Tinted like the real lid.
        metal = (lid | (inside_t & (lam > l_bot))) & mroi
        if metal.any():
            Vb = base_f.max(2)
            neutral = metal & (Vb - base_f.min(2) < 0.08)
            tm = base_f[neutral].mean(0) if neutral.sum() > 20 else np.ones(3, np.float32)
            tm = (tm / max(float(tm.max()), 1e-3)).astype(np.float32)
            soft = cv2.GaussianBlur(metal.astype(np.float32), (0, 0), 1.0)[..., None]
            base_f = base_f * (1 - soft) + Vb[..., None] * tm[None, None, :] * soft

    out = prem[:, :, :3] * keep[..., None] + base_f * (1 - a[..., None])

    # --- the footage's own surface detail (condensation, grain) goes back on, at its sharpness
    det_sel = mroi & (body_a > 0.5) & ~lid
    out = out + DETAIL_GAIN * (_surface_detail(roi, det_sel, Wd) * a)[..., None]

    # --- light wrap: the surroundings bleed a little over the new silhouette's edge
    bgw = (~cv2.dilate(mroi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.float32)
    sw = max(2.0, 0.03 * Wd)
    den = cv2.GaussianBlur(bgw, (0, 0), sw)
    bg = cv2.GaussianBlur(base_f * bgw[..., None], (0, 0), sw) / np.maximum(den, 1e-3)[..., None]
    edge = np.clip(1 - cv2.GaussianBlur(a, (0, 0), max(1.5, 0.012 * Wd)), 0, 1) * a * np.clip(den * 3, 0, 1)
    out = out + WRAP_GAIN * edge[..., None] * np.maximum(bg - out, 0)

    frame[y0:y1, x0:x1] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    return frame
