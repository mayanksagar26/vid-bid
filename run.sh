#!/usr/bin/env bash
# Start vid-bid on http://localhost:8000 (first run sets up .venv and downloads models on demand).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv with $PY"
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

mkdir -p bin data models
ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" bin/ffmpeg

export PYTORCH_ENABLE_MPS_FALLBACK=1
PORT="${PORT:-8000}"
echo "vid-bid → http://localhost:$PORT"
exec .venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
