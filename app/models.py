"""Lazy, process-wide model singletons. Everything runs locally (MPS on Apple Silicon)."""
import os
import threading
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from . import vocab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

YOLO_WEIGHTS = os.environ.get("VIDBID_YOLO", "yolov8m-worldv2.pt")
SAM2_SIZE = os.environ.get("VIDBID_SAM2", "small")  # tiny | small | base_plus | large
# SAM 2's memory attention is quadratic in tokens; at 1024px it runs ~1 fps on an M-series GPU,
# at 512px + fp16 ~10 fps with near-identical masks for product-sized objects.
SAM2_VIDEO_RES = int(os.environ.get("VIDBID_SAM2_RES", "512"))
_SAM2 = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
_SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

# One heavy job at a time: they all share the GPU and 16 GB of unified memory.
GPU_LOCK = threading.Lock()
_load_lock = threading.Lock()
_cache = {}


def _once(key, fn):
    with _load_lock:
        if key not in _cache:
            _cache[key] = fn()
        return _cache[key]


def detector():
    """YOLO-World with the vid-bid vocabulary baked in. Returns (model, prompt->category list)."""
    def load():
        from ultralytics import YOLOWorld
        cwd = os.getcwd()
        os.chdir(MODELS)  # ultralytics downloads weights into the cwd
        try:
            model = YOLOWorld(YOLO_WEIGHTS)
            prompts, cats = vocab.detector_prompts()
            model.set_classes(prompts)
        finally:
            os.chdir(cwd)
        return model, cats
    return _once("detector", load)


def clip():
    """open_clip model + preprocess + tokenizer."""
    def load():
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", cache_dir=str(MODELS / "clip"))
        model.eval().to(DEVICE)
        tok = open_clip.get_tokenizer("ViT-B-16")
        return model, preprocess, tok
    return _once("clip", load)


def _sam2_checkpoint():
    name, cfg = _SAM2[SAM2_SIZE]
    path = MODELS / name
    if not path.exists() or path.stat().st_size < 10_000_000:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_SAM2_URL + name, tmp)
        tmp.rename(path)
    return str(path), cfg


def sam2_image():
    def load():
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt, cfg = _sam2_checkpoint()
        return SAM2ImagePredictor(build_sam2(cfg, ckpt, device=DEVICE))
    return _once("sam2_image", load)


def sam2_video():
    def load():
        from sam2.build_sam import build_sam2_video_predictor
        ckpt, cfg = _sam2_checkpoint()
        return build_sam2_video_predictor(
            cfg, ckpt, device=DEVICE, hydra_overrides_extra=[f"++model.image_size={SAM2_VIDEO_RES}"])
    return _once("sam2_video", load)


def autocast():
    """fp16 autocast on Apple GPUs (about 1.4x faster for SAM 2); a no-op elsewhere."""
    import contextlib
    if DEVICE == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def free_memory():
    import gc
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
