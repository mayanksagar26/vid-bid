"""CLIP-based object classification and the "is this the same kind of object?" check."""
import numpy as np
import torch
from PIL import Image

from . import models, vocab

_text_cache = {}


def _text_bank():
    """(labels, is_product, normalized text embeddings [L, D]) with prompt ensembling per label."""
    if "bank" in _text_cache:
        return _text_cache["bank"]
    model, _, tok = models.clip()
    labels, is_product, embs = [], [], []
    groups = [(k, v["clip"], True) for k, v in vocab.CATEGORIES.items()]
    groups += [(k, v, False) for k, v in vocab.DISTRACTORS.items()]
    with torch.no_grad():
        for label, captions, prod in groups:
            e = model.encode_text(tok(captions).to(models.DEVICE)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            embs.append(e / e.norm())
            labels.append(label)
            is_product.append(prod)
    bank = (labels, is_product, torch.stack(embs))
    _text_cache["bank"] = bank
    return bank


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if img.ndim == 3 and img.shape[2] == 4:  # BGRA with alpha: composite on white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return Image.fromarray(img[:, :, ::-1].copy())  # BGR -> RGB


def embed_images(imgs):
    """BGR/BGRA numpy images (or PIL) -> normalized CLIP embeddings [N, D]."""
    model, preprocess, _ = models.clip()
    batch = torch.stack([preprocess(_to_pil(i)) for i in imgs]).to(models.DEVICE)
    with torch.no_grad():
        e = model.encode_image(batch).float()
    return e / e.norm(dim=-1, keepdim=True)


def classify(imgs):
    """Average class probabilities over one or more views of the same object.

    Returns a list of (label, prob) sorted by prob, over product categories and distractors."""
    labels, _, text = _text_bank()
    e = embed_images(imgs)
    probs = (100.0 * e @ text.T).softmax(dim=-1).mean(0).cpu().numpy()
    order = np.argsort(-probs)
    return [(labels[i], float(probs[i])) for i in order]


def product_category(ranked):
    """Best product category from a classify() ranking, ignoring distractors."""
    for label, p in ranked:
        if label in vocab.CATEGORIES:
            return label, p
    return None, 0.0


def check_same_kind(target_cat, product_img, reference_crops):
    """Decide whether the uploaded product image shows the same kind of object as target_cat.

    Two signals:
      * zero-shot CLIP class of the upload (must be target_cat, or a close runner-up)
      * image-image similarity between the upload and crops of the original object (tie-breaker)
    """
    ranked = classify([product_img])
    probs = dict(ranked)
    top_label, top_p = ranked[0]
    p_target = probs.get(target_cat, 0.0)
    sim = 0.0
    if reference_crops:
        e = embed_images([product_img] + list(reference_crops))
        sim = float((e[1:] @ e[0]).mean().cpu())

    ok = (top_label == target_cat
          or (p_target >= 0.25 and p_target >= 0.5 * top_p)
          or (p_target >= 0.10 and sim >= 0.80))

    what = top_label if top_label in vocab.CATEGORIES else {
        "person": "a person", "animal": "an animal", "food": "food", "vehicle": "a vehicle",
        "scene": "a scene", "text": "a logo or text", "plant": "a plant", "furniture": "furniture",
    }.get(top_label, top_label)
    if top_label in vocab.CATEGORIES:
        what = vocab.with_article(top_label)

    if ok:
        msg = f"Looks like {vocab.with_article(target_cat)}. Good to go."
    else:
        msg = (f"That isn't the same object. It looks like {what}. "
               f"Upload an image of {vocab.with_article(target_cat)}.")
    return {
        "ok": bool(ok),
        "message": msg,
        "target": target_cat,
        "predicted": top_label,
        "target_prob": round(p_target, 3),
        "similarity": round(sim, 3),
        "top": [(l, round(p, 3)) for l, p in ranked[:5]],
    }
