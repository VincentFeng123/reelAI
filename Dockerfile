# syntax=docker/dockerfile:1
# Railway deploy image for the FastAPI backend.
#
# We use an explicit Dockerfile rather than Railpack because the reel ingestion
# pipeline in `backend/app/ingestion/` needs ffmpeg + ffprobe on PATH for
# silencedetect / audio extraction / Whisper preprocessing, and Railpack's
# package declaration syntax is not reliable for system binaries. Every time
# someone has tried to "fix" this with railpack.toml / railpack.json, the
# result has been a silent no-op and ingest_search started bouncing with a
# DownloadError. Do not revert this to Railpack without a working test.
#
# The `# syntax=docker/dockerfile:1` header tells BuildKit to use the latest
# stable Dockerfile frontend (features like heredoc, RUN --mount=, etc).
# It's also a strong signal to Railway's builder auto-detection that this
# is a "real" Dockerfile and should not be bypassed for Railpack.
FROM python:3.12-slim

# System deps:
# - ffmpeg: required by backend/app/ingestion/ffmpeg_tools.py
# - ca-certificates, curl: TLS + debugging
# - git: yt-dlp sometimes needs it for extractor fallbacks
# Slim image keeps build context small; `--no-install-recommends` avoids pulling
# in X11 / doc packages we don't need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version \
    && ffprobe -version

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer when only
# application code changes. `requirements.txt` lives under `backend/` so we
# copy just that file, install, then copy the rest of the source.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Application source — the main.py startup sweeps the ingestion package on
# import, so we need the whole backend tree present before uvicorn boots.
COPY backend ./backend

# Railway injects $PORT. Bind to 0.0.0.0 so the container is reachable from
# the Railway edge proxy. `backend.app.main:app` is the FastAPI entrypoint.
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
