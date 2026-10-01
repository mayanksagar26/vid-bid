"""Object categories vid-bid knows how to find and replace.

Each category has:
  prompts  - text prompts fed to the open-vocabulary detector (YOLO-World)
  clip     - captions used by CLIP to classify an uploaded product image
  shape    - "cylinder" (cans, bottles, cups) gets cylindrical shading;
             "flat" objects get a plain planar warp
  article  - for messages ("an image of a can")
"""

CATEGORIES = {
    "can": {
        "prompts": ["can", "soda can", "beer can", "drink can"],
        "clip": ["a photo of a soda can", "a photo of an aluminum drink can", "a photo of a beer can"],
        "shape": "cylinder",
        "article": "a",
    },
    "bottle": {
        "prompts": ["bottle", "water bottle", "plastic bottle", "glass bottle"],
        "clip": ["a photo of a plastic bottle", "a photo of a glass bottle", "a photo of a water bottle"],
        "shape": "cylinder",
        "article": "a",
    },
    "cup": {
        "prompts": ["cup", "coffee cup", "mug", "paper cup"],
        "clip": ["a photo of a coffee mug", "a photo of a paper coffee cup", "a photo of a cup"],
        "shape": "cylinder",
        "article": "a",
    },
    "jar": {
        "prompts": ["jar"],
        "clip": ["a photo of a jar"],
        "shape": "cylinder",
        "article": "a",
    },
    "box": {
        "prompts": ["box", "cereal box", "product box", "package"],
        "clip": ["a photo of a cardboard box", "a photo of a cereal box", "a photo of a product package box"],
        "shape": "flat",
        "article": "a",
    },
    "phone": {
        "prompts": ["cell phone", "smartphone"],
        "clip": ["a photo of a smartphone", "a photo of a mobile phone"],
        "shape": "flat",
        "article": "a",
    },
    "laptop": {
        "prompts": ["laptop"],
        "clip": ["a photo of a laptop computer"],
        "shape": "flat",
        "article": "a",
    },
    "book": {
        "prompts": ["book"],
        "clip": ["a photo of a book", "a photo of a book cover"],
        "shape": "flat",
        "article": "a",
    },
    "snack bag": {
        "prompts": ["chip bag", "snack bag"],
        "clip": ["a photo of a bag of chips", "a photo of a snack packet"],
        "shape": "flat",
        "article": "a",
    },
    "shoe": {
        "prompts": ["shoe", "sneaker"],
        "clip": ["a photo of a shoe", "a photo of a sneaker"],
        "shape": "flat",
        "article": "a",
    },
    "watch": {
        "prompts": ["wristwatch"],
        "clip": ["a photo of a wristwatch"],
        "shape": "flat",
        "article": "a",
    },
    "sunglasses": {
        "prompts": ["sunglasses"],
        "clip": ["a photo of sunglasses"],
        "shape": "flat",
        "article": "",
    },
    "headphones": {
        "prompts": ["headphones"],
        "clip": ["a photo of headphones"],
        "shape": "flat",
        "article": "",
    },
    "remote": {
        "prompts": ["remote control"],
        "clip": ["a photo of a TV remote control"],
        "shape": "flat",
        "article": "a",
    },
}

# Captions for things that are none of the above, so CLIP can say "this is not a can"
# instead of being forced to pick the closest product category.
DISTRACTORS = {
    "person": ["a photo of a person", "a photo of a face"],
    "animal": ["a photo of an animal", "a photo of a dog", "a photo of a cat"],
    "food": ["a photo of a plate of food", "a photo of fruit"],
    "vehicle": ["a photo of a car"],
    "scene": ["a photo of a landscape", "a photo of a room", "a photo of a building"],
    "text": ["a screenshot of text", "a logo on a plain background"],
    "plant": ["a photo of a plant", "a photo of flowers"],
    "furniture": ["a photo of a chair", "a photo of a table"],
}


def detector_prompts():
    """Flat prompt list for the detector plus a parallel list mapping prompt index -> category."""
    prompts, cats = [], []
    for cat, spec in CATEGORIES.items():
        for p in spec["prompts"]:
            prompts.append(p)
            cats.append(cat)
    return prompts, cats


def with_article(cat):
    art = CATEGORIES.get(cat, {}).get("article", "a")
    if art == "a" and cat[:1] in "aeiou":
        art = "an"
    return f"{art} {cat}".strip()
