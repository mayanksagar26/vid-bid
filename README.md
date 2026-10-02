# vid-bid

Replace an object in a video with your product. It runs locally on Apple Silicon. No paid APIs.

1. **Input:** upload a video, or paste a YouTube link (the first 60 s are used).
2. **Detect:** the app lists the replaceable objects it finds (cans, bottles, cups, boxes, phones…), each with a thumbnail and the times it's on screen.
3. **Replacement image:** upload a photo of the new item.
4. **Compatibility check:** if the photo isn't the same kind of object, it's refused with a message like *"That isn't the same object. It looks like a shoe. Upload an image of a can."*
5. **Process:** the object is tracked through every frame and the product is composited in its place. The original audio is kept.
6. **Output:** preview the result, compare before and after with a slider or side by side, and download the MP4.

## Run

```bash
./run.sh
```

Then open <http://localhost:8000>.

The first run creates `.venv` with Python 3.12 or 3.11 and installs `requirements.txt`. Model weights download on first use into `models/` (about 1.5 GB in total). `PORT=9000 ./run.sh` changes the port.

There are no global installs: ffmpeg comes from the `imageio-ffmpeg` wheel, and `yt-dlp` is installed in the venv.

## Engines

When you replace an object you pick one of three engines:

| Engine | What it does | Cost |
|---|---|---|
| **Fast preview** (local) | Wraps the product label onto the tracked shape on your Mac. Exact logo, but it can look pasted on. | Free |
| **Realistic: Wan VACE 14B** ([fal.ai](https://fal.ai/models/fal-ai/wan-vace-14b/inpainting)) | Open-source video model. vid-bid sends a crop around the object, a mask video from its own tracking, and the product cutout as a reference. The model re-renders only that region, so lighting, reflections, condensation and motion look natural. The result is pasted back into the original full-res frames; everything else stays bit-identical. | $0.08 per output second at 720p, counted at 16 fps (about $3.60 for a 30 s clip) |
| **Runway Aleph 2** ([Runway API](https://dev.runwayml.com)) | Premium video-to-video editor. Gets the clip plus up to 3 keyframes, which are stills from the fast preview, as the target look. Clips up to 30 s. | $0.28 per second (about $8.40 for 30 s) |

The paid engines need API keys in `vid-bid/.env`, which git ignores:

```
FAL_KEY=...
RUNWAYML_API_SECRET=...
```

Restart `./run.sh` after adding them. An engine without a key shows as disabled in the UI.

## How it works

| Stage | Method |
|---|---|
| Detect + track | YOLO-World v2 (open vocabulary, `app/vocab.py`) with ByteTrack at about 15 fps. Each track is relabelled by fusing CLIP zero-shot probabilities with the detector's class votes, and fragments of the same object are merged by CLIP similarity. |
| Compatibility check | CLIP ViT-B/16 zero-shot over the product categories plus "distractor" classes (person, animal, scene…). Image-to-image similarity with crops of the original object is used as a tie-breaker. |
| Product cutout | BiRefNet background removal (via `rembg`). A PNG with real transparency is used as-is. |
| Segmentation through time | SAM 2.1 (small) video predictor over every frame at 512 px with fp16 autocast, which runs at about 10 fps on MPS (1024 px manages only about 1 fps). It runs in 150-frame chunks so memory stays bounded, handing the mask from one chunk to the next, and is re-prompted with confident detector boxes every 2 s to prevent drift. Masks are cached per object, so re-running with another product image skips tracking. |
| Geometry | For each mask, the left and right silhouette edges are fitted as lines, so perspective taper is captured. For cylinders (cans, bottles, cups), the top and bottom rim ellipses are fitted to the outline. Ends cut off by the frame edge are reconstructed from frames where they're visible, outlier masks are dropped, and everything is smoothed over time. |
| Replacement | The product's label (its cutout cropped to the solid body) is wrapped onto the tapered cylinder between the rims with a per-pixel remap. The label can stretch by at most 1.2×; beyond that it's cropped rather than distorted. The original lid and anything on top of it stays as it was. Shading uses a cylinder falloff plus a lighting gradient fitted from the original object; white balance and exposure come from the surroundings; blur is matched to the footage's focus, and motion blur to the object's velocity. Pixels inside the object's outline that SAM excludes (fingers, foam, straws) stay in front. Slivers of the old object are inpainted. |
| Audio | The original track is copied unchanged into the output. |

Environment knobs: `VIDBID_SAM2=tiny|small|base_plus|large` (default `small`), `VIDBID_SAM2_RES=512` (SAM input size; 1024 is sharper but about 10× slower), and `VIDBID_YOLO=yolov8l-worldv2.pt`.

## Layout

```
app/server.py    FastAPI app + background jobs (one GPU job at a time)
app/detect.py    detection, tracking, CLIP relabel/merge
app/compat.py    CLIP classification + same-kind check
app/segment.py   SAM 2 cutout + chunked video tracking
app/replace.py   geometry, rendering, compositing, encoding
app/media.py     ffmpeg / yt-dlp helpers
app/vocab.py     object categories
static/          single-page UI
```

Project state lives in `data/<project-id>/` (git-ignored).

## Benchmark clip

"Slowly fizzing coke zero", <https://www.youtube.com/watch?v=mp60AngR5tA> (first 60 s). It shows one Coke Zero can on a desk with a mostly static camera; foam spilling over the top is the only occlusion.

No video is included in this repo. Paste the URL into the app, or fetch the clip yourself:

```bash
mkdir -p samples
.venv/bin/yt-dlp -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 \
  --ffmpeg-location bin/ffmpeg --download-sections "*0-60" \
  -o samples/coke_zero.mp4 "https://www.youtube.com/watch?v=mp60AngR5tA"
```

For the replacement image, use any photo of a different can. A front-on product shot on a plain background works best.

## Performance

On the benchmark (848×480, 60 s, 1,800 frames) on an Apple Silicon Mac with 16 GB:
- detection: about 70 s;
- tracking: about 3 min;
- rendering and encoding: about 70 s.

A re-run with a different product image reuses the cached tracking and takes about 70 s.

## Limits

- **Rotation.** The label is wrapped as if the object keeps the same side to the camera. If the real can is spun around, the new label doesn't spin with it.
- **Attached occluders.** Things SAM counts as part of the object, such as foam dripping down the side of the benchmark can, get covered by the new label. Separate occluders (fingers, foam on top, other objects) stay in front.
- **Shape mismatch.** A slim can replacing a standard can is fitted to the original's silhouette, with the label cropped to fit.
- **Cuts.** Hard scene cuts aren't detected; each detected appearance is tracked separately.
- **Detector licence.** See Licence below.

## Licence

vid-bid's own code is MIT (see `LICENSE`). Its dependencies have their own licences:

- **Ultralytics / YOLO-World is AGPL-3.0.** vid-bid imports it for detection, so if you distribute vid-bid or offer it as a network service, the combined work is subject to the AGPL. To avoid that, swap the detector for an Apache/MIT one, such as OWLv2 or Grounding DINO via `transformers`.
- SAM 2 (code and weights): Apache-2.0.
- open_clip: MIT. The LAION CLIP weights are MIT.
- yt-dlp: Unlicense. ffmpeg (bundled via imageio-ffmpeg): GPL build.

Only download videos you have the right to use.
