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

## How it works

| Stage | Method |
|---|---|
| Detect + track | YOLO-World v2 (open vocabulary, `app/vocab.py`) with ByteTrack at about 15 fps. Each track is relabelled by fusing CLIP zero-shot probabilities with the detector's class votes, and fragments of the same object are merged by CLIP similarity. |
| Compatibility check | CLIP ViT-B/16 zero-shot over the product categories plus "distractor" classes (person, animal, scene…). Image-to-image similarity with crops of the original object is used as a tie-breaker. |
| Product cutout | The detector proposes boxes, CLIP picks the best one for the target category, and SAM 2.1 segments it. A PNG with an alpha channel is used as-is. |
| Segmentation through time | SAM 2.1 video predictor over every frame. It runs in 96-frame chunks so a minute of video fits in 16 GB, handing the mask from one chunk to the next, and is re-prompted with confident detector boxes every 2 s to prevent drift. |
| Replacement | Per-frame oriented geometry from the mask, temporally smoothed. The cutout gets an affine warp onto it, cylindrical shading plus a lighting gradient fitted from the original object, white balance and brightness matched to the scene, motion blur from object velocity, and occluder-aware compositing so fingers and foam in front of the object stay in front. Any leftover original pixels are inpainted. |
| Audio | The original track is copied unchanged into the output. |

Environment knobs: `VIDBID_SAM2=tiny|small|base_plus|large` (default `small`) and `VIDBID_YOLO=yolov8l-worldv2.pt`.

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

## Limits

- The object is treated as a rigid label-facing-camera item. If the real object rotates (for example, a can turned to show its back), the new label does not rotate with it.
- Objects that leave the frame partially get a truncated fit.
- Processing time is roughly 2–4× real time on an M-series Mac for a 480p–720p minute.
