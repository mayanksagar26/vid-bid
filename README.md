# vid-bid

Replace an object in a video with your product: swap the can in an ad for yours. Free and open source end to end. It runs on your own machine (Apple Silicon, an NVIDIA PC, or CPU only for the fast engine) or on a free Colab GPU. There are no paid APIs, keys or accounts.

1. **Input:** upload a video, or paste a YouTube link (the first 60 s are used).
2. **Detect:** the app lists the replaceable objects it finds (cans, bottles, cups, boxes, phones…), each with a thumbnail and the times it's on screen.
3. **Replacement image:** upload a photo of the new item.
4. **Compatibility check:** if the photo isn't the same kind of object, it's refused with a message like *"That isn't the same object. It looks like a shoe. Upload an image of a can."*
5. **Process:** the object is tracked through every frame and the product is put in its place, with the scene's own lighting, condensation and grain. The original audio is kept.
6. **Output:** preview the result, compare before and after with a slider or side by side, and download the MP4.

## Run

```bash
./run.sh
```

Then open <http://localhost:8000>.

The first run creates `.venv` with Python 3.12 or 3.11 and installs `requirements.txt`. Model weights download on first use into `models/`: about 1.5 GB for detection, tracking and cutouts, and about 20 GB more if you use the Wan VACE engine. `PORT=9000 ./run.sh` changes the port.

There are no global installs: ffmpeg comes from the `imageio-ffmpeg` wheel, and `yt-dlp` is installed in the venv.

**No GPU?** Open [`notebooks/vid-bid-free-gpu.ipynb`](notebooks/vid-bid-free-gpu.ipynb) in Google Colab (File → Open notebook → GitHub), pick the free T4 GPU and run the cells. It also runs on Kaggle.

## Command line

The same pipeline without the browser:

```bash
.venv/bin/python -m app.cli --url https://youtu.be/8ivQOS4r-K8 \
    --product samples/pepsi_black.jpg --object can --out pepsi.mp4
```

Options:

- `--video clip.mp4` instead of `--url`.
- `--engine vace --describe "a black Pepsi Black can"` for the generative engine.
- `--object-id obj2` to pick an exact object from the printed list.
- `--force` to skip the same-kind check.

Each run is a normal project in `data/`, so with `./run.sh` running you can compare it in the browser at the link the CLI prints.

## Engines

| Engine | What it does | Hardware | Speed (30 s clip) |
|---|---|---|---|
| **Fast** (default) | Wraps the product photo onto the tracked shape, lit by the scene (see below). Exact logo, frame-accurate. | Any computer; CPU is fine | Minutes |
| **Realistic: Wan VACE** | [Wan 2.1 VACE 1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers), an open-weights video model, runs through `diffusers`. vid-bid sends a crop around the object, a mask video from its own tracking, and the product cutout as the reference image. The model re-renders only that region, so reflections, the hand's grip and motion interact naturally. The result is pasted back into the original frames; everything else is untouched apart from re-encoding. | NVIDIA GPU with 8 GB or more, or Apple Silicon | One pass per ~3 s part: a few minutes per part on a recent GPU, longer on a free T4 or a Mac |

The generative model can drift on fine label text, so when the logo must be pixel-exact, use Fast. Knobs for VACE:

- `VIDBID_VACE_MODEL=Wan-AI/Wan2.1-VACE-14B-diffusers` uses the 14B model (better, needs a large GPU).
- `VIDBID_VACE_STEPS=25` sets the number of denoising steps.
- `VIDBID_VACE_DTYPE=float16` forces half precision.

## Example: Diet Coke → Pepsi Black

"Need a Diet Coke? Take a Break to Sip and Refresh", <https://youtu.be/8ivQOS4r-K8>, a 30 s, 1080p ad. A silver, condensation-covered can slides in, gets picked up (revealing a glass behind it), is tilted to pour, and is put back down beside the glass.

1. Save a front-on photo of a Pepsi Black can as `samples/pepsi_black.jpg`. A plain background is fine: the cutout is automatic. `samples/` is git-ignored.
2. Run `.venv/bin/python -m app.cli --url https://youtu.be/8ivQOS4r-K8 --product samples/pepsi_black.jpg --object can --out pepsi.mp4`, or use the web UI with the same link and photo.

What makes the result look real rather than pasted on:

- **Shape.** The new can follows the old one's silhouette, including the neck and shoulder, which are learnt from the clip.
- **Ends.** The original lid, seam and base bevel stay, since they're bare aluminium on any can.
- **Light.** The black label picks up the silver can's lighting, its specular highlights and a Fresnel rim.
- **Surface detail.** The footage's own condensation droplets and grain are laid back over the label.
- **Hands.** Fingers stay in front.

## How it works

| Stage | Method |
|---|---|
| Detect + track | YOLO-World v2 (open vocabulary, `app/vocab.py`) with ByteTrack at about 15 fps. Each track is relabelled by fusing CLIP zero-shot probabilities with the detector's class votes, and fragments of the same object are merged by CLIP similarity. |
| Compatibility check | CLIP ViT-B/16 zero-shot over the product categories plus "distractor" classes (person, animal, scene…). Image-to-image similarity with crops of the original object is used as a tie-breaker. |
| Product cutout | BiRefNet background removal (via `rembg`). A PNG with real transparency is used as-is. |
| Segmentation through time | SAM 2.1 (small) video predictor over every frame at 512 px. On MPS it uses fp16 autocast and runs at about 10 fps; 1024 px manages only about 1 fps. It runs in 150-frame chunks so memory stays bounded, handing the mask from one chunk to the next, and is re-prompted with confident detector boxes every 2 s to prevent drift. Masks are cached per object, so re-running with another product image skips tracking. |
| Geometry | For each mask, the left and right silhouette edges are fitted as lines, so perspective taper is captured. For cylinders (cans, bottles, cups), the lid and base ellipses are fitted from the ends of the outline, which gives the camera's angle onto each end. The outline in between (neck, shoulder, base bevel) is learnt over the whole clip as a radius profile. Ends cut off by the frame edge are reconstructed from frames where they're visible, outlier masks are dropped, and everything is smoothed over time. |
| Replacement | The product photo is unrolled row by row (the front half of its surface) and wrapped back onto the object as a surface of revolution, landmark to landmark. The label can stretch by at most 1.2×; beyond that it's cropped. For cans, the footage's own lid seam and base bevel stay, and the shoulder continues the label's top colour. |
| Lighting | Read off the original object through V = max(R, G, B), where most of the old artwork cancels out. That gives a shade map (multiplied), its brightest highlights (added, like gloss on lacquer) and a Fresnel rim reflecting the surroundings. Studio highlights baked into the product photo are mostly removed first. White balance and exposure come from the surroundings, and blacks get a little lens flare. |
| Compositing | Blur is matched to the footage's focus, and motion blur to the object's velocity. The original surface's fine detail (condensation, scuffs, grain, compression noise) is laid back on top, gated off the old label's colour edges. The surroundings wrap a little light over the new edge. Pixels inside the object's outline that SAM excludes (fingers, foam, straws) stay in front. The old object is painted out from its surroundings wherever the new one doesn't cover it. |
| Audio | The original track is copied unchanged into the output. |

Environment knobs:

- `VIDBID_SAM2=tiny|small|base_plus|large` (default `small`).
- `VIDBID_SAM2_RES=512`: SAM input size. 1024 is sharper but about 10× slower.
- `VIDBID_YOLO=yolov8l-worldv2.pt` changes the detector.
- The `VIDBID_VACE_*` settings above.

Put them in the shell or in `vid-bid/.env`.

## Layout

```
app/server.py      FastAPI app + background jobs (one GPU job at a time)
app/cli.py         the same pipeline from the command line
app/detect.py      detection, tracking, CLIP relabel/merge
app/compat.py      CLIP classification + same-kind check
app/segment.py     SAM 2 cutout + chunked video tracking
app/replace.py     geometry, rendering, compositing, encoding (Fast engine)
app/generative.py  Wan VACE engine (crops, masks, chunked generation, paste-back)
app/models.py      model loading (all open weights)
app/media.py       ffmpeg / yt-dlp helpers
app/vocab.py       object categories
static/            single-page UI
notebooks/         free-GPU (Colab/Kaggle) notebook
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

CPU only (4 cores, no GPU) on the Diet Coke ad (1920×1080, 30 s, the can on screen for 636 of 720 frames):
- product cutout (BiRefNet): about 1 min;
- tracking (SAM 2.1 small at 512 px): about 11 min;
- Fast rendering and encoding: about 4 min.

## Limits

- **Rotation.** The label is wrapped as if the object keeps the same side to the camera. If the real can is spun around, the new label doesn't spin with it.
- **Attached occluders.** Things SAM counts as part of the object, such as foam dripping down the side of the benchmark can, get covered by the new label. Separate occluders (fingers, foam on top, other objects) stay in front.
- **Shape mismatch.** A slim can replacing a standard can is fitted to the original's silhouette, with the label cropped to fit.
- **Viewpoint.** Cylinders are assumed to be seen from above or level; a can held high above the camera gets its label rows curved the wrong way.
- **Cuts.** Hard scene cuts aren't detected; each detected appearance is tracked separately.
- **Detector licence.** See Licence below.

## Licence

vid-bid's own code is MIT (see `LICENSE`). Its dependencies have their own licences:

- **Ultralytics / YOLO-World is AGPL-3.0.** vid-bid imports it for detection, so if you distribute vid-bid or offer it as a network service, the combined work is subject to the AGPL. To avoid that, swap the detector for an Apache/MIT one, such as OWLv2 or Grounding DINO via `transformers`.
- SAM 2 (code and weights): Apache-2.0.
- Wan 2.1 VACE weights and diffusers: Apache-2.0. The UMT5 text encoder is Apache-2.0.
- open_clip: MIT. The LAION CLIP weights are MIT. BiRefNet: MIT.
- yt-dlp: Unlicense. ffmpeg (bundled via imageio-ffmpeg): GPL build.

Only download videos you have the right to use, and only put brands in footage you have the right to edit.
