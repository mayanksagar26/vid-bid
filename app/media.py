"""Video I/O: ffmpeg (bundled via imageio-ffmpeg), yt-dlp downloads, frame encoding."""
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parent.parent
FFMPEG_DIR = ROOT / "bin"  # contains an `ffmpeg` symlink so yt-dlp can find it

MAX_SECONDS = 120   # longer inputs are trimmed
MAX_HEIGHT = 1080


class MediaError(RuntimeError):
    pass


def _run(cmd, err):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-6:]
        raise MediaError(f"{err}: " + " | ".join(tail))
    return p


def ensure_ffmpeg_link():
    FFMPEG_DIR.mkdir(exist_ok=True)
    link = FFMPEG_DIR / "ffmpeg"
    if not link.exists():
        link.symlink_to(FFMPEG)


def download_youtube(url, out_dir: Path, max_seconds=60):
    """Download the first `max_seconds` of a YouTube (or any yt-dlp supported) URL."""
    if not re.match(r"^https?://", url.strip()):
        raise MediaError("That doesn't look like a URL.")
    ensure_ffmpeg_link()
    out_tmpl = str(out_dir / "download.%(ext)s")
    cmd = [
        str(ROOT / ".venv" / "bin" / "yt-dlp"), "--no-playlist", "--no-warnings", "-q",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b",
        "--merge-output-format", "mp4",
        "--ffmpeg-location", str(FFMPEG_DIR / "ffmpeg"),
        "--download-sections", f"*0-{max_seconds}",
        "--print-json", "--no-simulate",
        "-o", out_tmpl, url.strip(),
    ]
    p = _run(cmd, "Download failed")
    title = None
    try:
        title = json.loads(p.stdout.strip().splitlines()[-1]).get("title")
    except Exception:
        pass
    files = sorted(out_dir.glob("download.*"))
    if not files:
        raise MediaError("Download finished but no video file was produced.")
    return files[0], title


def normalize(src: Path, dst: Path, max_seconds=MAX_SECONDS):
    """Re-encode to constant-frame-rate H.264/AAC mp4 that browsers, OpenCV and SAM all agree on."""
    info = probe(src)
    fps = info["fps"] if 1 <= info["fps"] <= 60 else 30
    vf = f"scale=-2:'min({MAX_HEIGHT},ih)':flags=lanczos,fps={fps:.6f},format=yuv420p"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-t", str(max_seconds),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    _run(cmd, "Could not read that video")
    return probe(dst)


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or w <= 0:
        raise MediaError("The video has no readable frames.")
    return {"fps": float(fps), "frames": n, "width": w, "height": h, "duration": n / fps}


def iter_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def extract_frames_jpg(path: Path, out_dir: Path, max_side=1024):
    """Dump every frame as 00000.jpg, 00001.jpg ... (SAM 2's video loader format), downscaled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(path)
    scale = min(1.0, max_side / max(info["width"], info["height"]))
    vf = f"scale={int(info['width'] * scale) // 2 * 2}:{int(info['height'] * scale) // 2 * 2}"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(path), "-vf", vf,
           "-q:v", "2", "-start_number", "0", str(out_dir / "%05d.jpg")]
    _run(cmd, "Frame extraction failed")
    return scale


class VideoWriter:
    """Pipe BGR frames into ffmpeg, muxing the audio track from `audio_src`."""

    def __init__(self, dst: Path, width, height, fps, audio_src: Path | None = None, crf=17):
        cmd = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
               "-i", "pipe:0"]
        if audio_src is not None:
            cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-shortest", str(dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="ignore")
        if self.proc.wait() != 0:
            raise MediaError("Encoding failed: " + err[-400:])
