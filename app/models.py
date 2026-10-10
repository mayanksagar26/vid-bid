"""Lazy, process-wide model singletons. Everything runs locally (MPS on Apple Silicon, CUDA on
NVIDIA, CPU otherwise), with open weights only."""
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
_SAM2_URLS = [  # Meta's CDN, then the same checkpoints re-hosted on GitHub (Ultralytics assets)
    ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/", None),
    ("https://github.com/ultralytics/assets/releases/download/v8.3.0/",
     {"tiny": "sam2.1_t.pt", "small": "sam2.1_s.pt", "base_plus": "sam2.1_b.pt", "large": "sam2.1_l.pt"}),
]

# Generative engine: Wan 2.1 VACE (Apache-2.0) via diffusers, weights from Hugging Face.
VACE_MODEL = os.environ.get("VIDBID_VACE_MODEL", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")  # or ...-14B-diffusers
VACE_STEPS = int(os.environ.get("VIDBID_VACE_STEPS", "25"))
VACE_ALLOW_CPU = os.environ.get("VIDBID_VACE_ALLOW_CPU") == "1"  # hours per clip; for testing only
HF_CACHE = MODELS / "hf"

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
        errors = []
        for base, names in _SAM2_URLS:
            try:
                urllib.request.urlretrieve(base + (names[SAM2_SIZE] if names else name), tmp)
                tmp.rename(path)
                break
            except OSError as e:
                errors.append(f"{base}: {e}")
        else:
            raise RuntimeError("Could not download SAM 2 weights: " + "; ".join(errors))
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


def matting():
    """BiRefNet (via rembg's ONNX export) for product background removal."""
    def load():
        os.environ.setdefault("U2NET_HOME", str(MODELS / "rembg"))
        from rembg import new_session
        return new_session("birefnet-general")
    return _once("matting", load)


def _vace_dtype():
    name = os.environ.get("VIDBID_VACE_DTYPE")  # e.g. float16 / bfloat16
    if name:
        return getattr(torch, name)
    if DEVICE == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if DEVICE == "mps":
        return torch.bfloat16
    return torch.float32


def _cuda_mem():
    return torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" else 0


def vace_prompt_embeds(prompt, negative):
    """UMT5 embeddings of (prompt, negative prompt), cached on disk.

    The text encoder is ~10 GB, five times the 1.3B video model, so it's loaded only for a new
    prompt, straight onto the GPU when it fits (else the CPU: slower, same result), and freed
    before the video model loads."""
    import hashlib
    key = hashlib.sha1(f"{VACE_MODEL}|{prompt}|{negative}".encode()).hexdigest()[:16]
    path = HF_CACHE / "embeds" / f"{key}.pt"
    if path.exists():
        e = torch.load(path, map_location="cpu")
        return e["pos"], e["neg"]
    from diffusers import WanVACEPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    dtype = _vace_dtype()
    dev = DEVICE if DEVICE == "mps" or _cuda_mem() >= 14e9 else "cpu"
    tok = AutoTokenizer.from_pretrained(VACE_MODEL, subfolder="tokenizer", cache_dir=HF_CACHE)
    te = UMT5EncoderModel.from_pretrained(VACE_MODEL, subfolder="text_encoder", cache_dir=HF_CACHE,
                                          torch_dtype=dtype if dev != "cpu" else torch.float32,
                                          device_map=dev if dev != "cpu" else None)
    enc = WanVACEPipeline(tokenizer=tok, text_encoder=te, vae=None, scheduler=None, transformer=None)
    with torch.inference_mode():
        pos, neg = enc.encode_prompt(prompt, negative, do_classifier_free_guidance=True,
                                     max_sequence_length=512, device=torch.device(dev), dtype=dtype)
    pos, neg = pos.cpu(), neg.cpu()
    del enc, te
    free_memory()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pos": pos, "neg": neg}, path)
    return pos, neg


def vace_pipeline():
    """Wan 2.1 VACE without its text encoder (prompts come in as embeddings)."""
    def load():
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        vae = AutoencoderKLWan.from_pretrained(VACE_MODEL, subfolder="vae", torch_dtype=torch.float32,
                                               cache_dir=HF_CACHE)
        pipe = WanVACEPipeline.from_pretrained(VACE_MODEL, vae=vae, text_encoder=None, tokenizer=None,
                                               torch_dtype=_vace_dtype(), cache_dir=HF_CACHE)
        # flow_shift 3.0 is Wan's setting for 480p
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        if DEVICE == "cuda" and _cuda_mem() < 20e9:
            pipe.enable_model_cpu_offload()  # 8-16 GB cards: weights wait in RAM between stages
        else:
            pipe.to(DEVICE)
        return pipe
    return _once("vace", load)


def free_memory():
    import gc
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
