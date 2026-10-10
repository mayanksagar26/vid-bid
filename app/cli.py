"""vid-bid from the command line: the web app's pipeline without the browser.

  .venv/bin/python -m app.cli --url https://youtu.be/8ivQOS4r-K8 --product samples/pepsi_black.jpg --object can
  .venv/bin/python -m app.cli --video clip.mp4 --product pepsi.png --object can --engine vace \\
      --describe "a black Pepsi Black can" --out pepsi.mp4

Each run is a normal project in data/<id>/: with ./run.sh running, open
http://localhost:8000/#p=<id> to compare before/after in the browser.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2


class _Progress:
    def __init__(self):
        self.last_msg, self.last_t = None, 0.0

    def __call__(self, frac, msg):
        """Print each new stage, and progress within a stage every few seconds."""
        now = time.time()
        stage = msg.split("(")[0].split("·")[0]
        if stage != (self.last_msg or "").split("(")[0].split("·")[0] or now - self.last_t > 5:
            print(f"  [{frac * 100:5.1f}%] {msg}", flush=True)
            self.last_msg, self.last_t = msg, now


def _pick(objects, want):
    if want:
        cands = [o for o in objects if want.lower() in (o["category"].lower(), o["label"].lower())]
        if not cands:
            found = ", ".join(sorted({o["category"] for o in objects})) or "nothing"
            sys.exit(f"No '{want}' found in the video (found: {found}). Try --object-id.")
    else:
        cands = objects
    return max(cands, key=lambda o: o["visible_seconds"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.cli", description="Replace an object in a video with your product.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="local video file")
    src.add_argument("--url", help="YouTube (or any yt-dlp) link; the first 60 s are used")
    ap.add_argument("--product", required=True, help="photo of the new product (PNG with transparency is best)")
    ap.add_argument("--object", help="kind of object to replace, e.g. can (default: the one on screen longest)")
    ap.add_argument("--object-id", help="exact object id from the detection list (obj1, obj2...)")
    ap.add_argument("--engine", choices=["local", "vace"], default="local",
                    help="local: fast, any computer (default). vace: Wan VACE on your GPU, slower, re-rendered")
    ap.add_argument("--describe", default="", help="short description of the product (used by vace)")
    ap.add_argument("--out", help="also copy the result here")
    ap.add_argument("--force", action="store_true", help="skip the 'same kind of object' check")
    args = ap.parse_args(argv)

    product_path = Path(args.product)
    img = cv2.imread(str(product_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read the product image {product_path}.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    from . import compat, detect, generative, media, replace, segment, server
    progress = _Progress()
    t0 = time.time()

    # 1. input
    if args.url:
        p = server.new_project({"kind": "youtube", "url": args.url})
        work = server.DATA / p["id"]
        print(f"Downloading {args.url}")
        raw, title = media.download_youtube(args.url, work)
        p["source"]["title"] = title
        server.save(p)
    else:
        video = Path(args.video)
        if not video.is_file():
            sys.exit(f"No such video: {video}")
        p = server.new_project({"kind": "upload", "name": video.name})
        work = server.DATA / p["id"]
        raw = work / f"upload{video.suffix.lower()}"
        shutil.copyfile(video, raw)
    pid = p["id"]
    print(f"Project {pid} · preparing video")
    info = media.normalize(raw, work / "input.mp4")
    raw.unlink(missing_ok=True)
    server.update(pid, info=info, video="input.mp4", status="detecting")

    # 2. detect
    print(f"Detecting objects ({info['width']}x{info['height']}, {info['duration']:.1f} s)")
    objects = detect.detect_objects(work / "input.mp4", work, progress)
    server.update(pid, objects=objects, status="ready" if objects else "no_objects")
    if not objects:
        sys.exit("No replaceable objects found in the video.")
    for o in objects:
        segs = ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in o["segments"])
        print(f"  {o['id']:6} {o['label']:10} on screen {o['visible_seconds']:5.1f}s  ({segs})")
    if args.object_id:
        obj = next((o for o in objects if o["id"] == args.object_id), None)
        if obj is None:
            sys.exit(f"No object {args.object_id}.")
    else:
        obj = _pick(objects, args.object)
    print(f"Replacing {obj['id']} ({obj['label']})")

    # 3. product: same-kind check + cutout
    refs = [cv2.imread(str(x)) for x in sorted((work / "objects" / obj["id"]).glob("ref*.jpg"))]
    check = compat.check_same_kind(obj["category"], img, [r for r in refs if r is not None])
    print(f"  check: {check['message']}")
    if not check["ok"] and not args.force:
        sys.exit("Stopped: the product doesn't look like the same kind of object (use --force to override).")
    prod_dir = work / "product"
    prod_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    cv2.imwrite(str(prod_dir / f"product_{stamp}.png"), img)
    cut = segment.product_cutout(img, obj["category"])
    cutout = prod_dir / f"cutout_{stamp}.png"
    cv2.imwrite(str(cutout), cut)
    check["ok"] = True
    server.update(pid, product={"object_id": obj["id"], "file": f"product/product_{stamp}.png", "check": check,
                                "cutout": f"product/cutout_{stamp}.png"}, selected=obj["id"], results={}, result=None)

    # 4. replace
    print(f"Rendering with the {args.engine} engine")
    if args.engine == "local":
        out = replace.run(work, obj, cutout, progress)
        out["engine"] = "local"
        out["stats"] = "Fast (local) · " + out["stats"]
    else:
        out = generative.run_vace(work, obj, cutout, args.describe, progress)
    out["object_id"] = obj["id"]
    server.update(pid, status="done", result=out, results={args.engine: out}, description=args.describe)

    result = work / out["video"]
    if args.out:
        shutil.copyfile(result, args.out)
        result = Path(args.out)
    print(f"Done in {time.time() - t0:.0f}s · {out['stats']}")
    print(f"Result: {result}")
    print(f"Compare in the browser (with ./run.sh running): http://localhost:8000/#p={pid}")


if __name__ == "__main__":
    main()
