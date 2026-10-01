"""vid-bid web server: upload/YouTube -> detect -> pick object -> upload product -> replace."""
import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import compat, detect, media, models

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="vid-bid")
JOBS: dict[str, dict] = {}
_state_lock = threading.Lock()

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------- project state

def pdir(pid):
    if not pid.isalnum():
        raise HTTPException(404, "No such project")
    d = DATA / pid
    if not d.is_dir():
        raise HTTPException(404, "No such project")
    return d


def load(pid):
    with open(pdir(pid) / "project.json") as f:
        return json.load(f)


def save(p):
    with _state_lock:
        path = DATA / p["id"] / "project.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(p, f, indent=1)
        tmp.replace(path)


def update(pid, **kw):
    p = load(pid)
    p.update(kw)
    save(p)
    return p


def new_project(source):
    pid = uuid.uuid4().hex[:10]
    (DATA / pid).mkdir()
    p = {"id": pid, "created": time.time(), "source": source, "status": "new",
         "objects": [], "product": None, "result": None, "job": None, "error": None}
    save(p)
    return p


# ---------------------------------------------------------------- jobs

def start_job(pid, kind, fn):
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "project": pid, "kind": kind, "status": "queued", "progress": 0.0,
           "message": "Waiting for the GPU", "error": None, "started": time.time()}
    JOBS[jid] = job
    update(pid, job=jid)

    def progress(frac, msg):
        job["progress"] = round(float(min(max(frac, 0.0), 1.0)), 4)
        job["message"] = msg

    def run():
        with models.GPU_LOCK:
            job["status"] = "running"
            try:
                fn(progress)
                job["status"] = "done"
                job["progress"] = 1.0
            except Exception as e:  # surface every failure to the UI
                traceback.print_exc()
                job["status"] = "error"
                job["error"] = str(e) or e.__class__.__name__
                try:
                    update(pid, status="error", error=job["error"])
                except Exception:
                    pass
            finally:
                models.free_memory()

    threading.Thread(target=run, daemon=True).start()
    return job


def ingest_and_detect(pid, raw: Path | None = None, url: str | None = None):
    work = DATA / pid

    def fn(progress):
        nonlocal raw
        if url:
            update(pid, status="downloading")
            progress(0.02, "Downloading from YouTube")
            raw, title = media.download_youtube(url, work)
            p = load(pid)
            p["source"]["title"] = title
            save(p)
        update(pid, status="preparing")
        progress(0.04, "Preparing video")
        info = media.normalize(raw, work / "input.mp4")
        if raw.name != "input.mp4":
            raw.unlink(missing_ok=True)
        update(pid, info=info, video="input.mp4", status="detecting")
        objects = detect.detect_objects(work / "input.mp4", work, progress)
        update(pid, objects=objects, status="ready" if objects else "no_objects", error=None)

    return start_job(pid, "detect", fn)


# ---------------------------------------------------------------- API

@app.post("/api/projects/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Use mp4, mov, webm or mkv.")
    p = new_project({"kind": "upload", "name": file.filename})
    raw = DATA / p["id"] / f"upload{ext}"
    with open(raw, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job = ingest_and_detect(p["id"], raw=raw)
    return {"project": p["id"], "job": job["id"]}


class YouTubeReq(BaseModel):
    url: str


@app.post("/api/projects/youtube")
def from_youtube(req: YouTubeReq):
    p = new_project({"kind": "youtube", "url": req.url})
    job = ingest_and_detect(p["id"], url=req.url)
    return {"project": p["id"], "job": job["id"]}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = load(pid)
    job = JOBS.get(p.get("job") or "")
    p["job_state"] = job
    return p


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "No such job")
    return JOBS[jid]


@app.post("/api/projects/{pid}/product")
async def upload_product(pid: str, object_id: str = Form(...), file: UploadFile = File(...)):
    p = load(pid)
    obj = next((o for o in p["objects"] if o["id"] == object_id), None)
    if obj is None:
        raise HTTPException(404, "Pick an object first.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, "Upload a PNG, JPG or WEBP image.")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "Could not read that image.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    work = pdir(pid)
    refs = [cv2.imread(str(x)) for x in sorted((work / "objects" / object_id).glob("ref*.jpg"))]
    with models.GPU_LOCK:
        check = compat.check_same_kind(obj["category"], img, [r for r in refs if r is not None])
    stamp = uuid.uuid4().hex[:6]
    prod_dir = work / "product"
    prod_dir.mkdir(exist_ok=True)
    fname = f"product_{stamp}.png"
    cv2.imwrite(str(prod_dir / fname), img)
    product = {"object_id": object_id, "file": f"product/{fname}", "check": check, "cutout": None}
    if check["ok"]:
        from . import segment
        with models.GPU_LOCK:
            cut = segment.product_cutout(img, obj["category"])
        cv2.imwrite(str(prod_dir / f"cutout_{stamp}.png"), cut)
        product["cutout"] = f"product/cutout_{stamp}.png"
    update(pid, product=product, selected=object_id)
    status = 200 if check["ok"] else 422
    return JSONResponse(product, status_code=status)


class ProcessReq(BaseModel):
    object_id: str


@app.post("/api/projects/{pid}/process")
def process(pid: str, req: ProcessReq):
    p = load(pid)
    prod = p.get("product")
    if not prod or prod["object_id"] != req.object_id or not prod["check"]["ok"]:
        raise HTTPException(400, "Upload a matching product image for this object first.")
    job = JOBS.get(p.get("job") or "")
    if job and job["status"] in ("queued", "running"):
        raise HTTPException(409, "This project is already busy.")
    obj = next(o for o in p["objects"] if o["id"] == req.object_id)
    work = pdir(pid)

    def fn(progress):
        from . import replace
        update(pid, status="processing", result=None)
        out = replace.run(work, obj, work / prod["cutout"], progress)
        update(pid, status="done", result=out)

    job = start_job(pid, "process", fn)
    return {"job": job["id"]}


@app.get("/files/{pid}/{path:path}")
def files(pid: str, path: str):
    base = pdir(pid).resolve()
    f = (base / path).resolve()
    if base not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"ok": True, "device": models.DEVICE}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
