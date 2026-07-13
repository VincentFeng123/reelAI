# syntax=docker/dockerfile:1
# Railway deploy image for the FastAPI backend.
#
# The backend is a durable Railway process. Supadata supplies hosted timestamped
# transcript cues (native when available, generated otherwise). ffmpeg is used
# only for bounded acoustic checks around selected clip edges; no local Whisper
# runtime or full-media download is installed.
#
# The `# syntax=docker/dockerfile:1` header tells BuildKit to use the latest
# stable Dockerfile frontend (features like heredoc, RUN --mount=, etc).
# It's also a strong signal to Railway's builder auto-detection that this
# is a "real" Dockerfile and should not be bypassed for Railpack.
FROM denoland/deno:bin-2.9.2 AS deno_runtime

FROM python:3.12-slim

COPY --from=deno_runtime /deno /usr/local/bin/deno

# System deps:
# - ca-certificates, curl: TLS + debugging
# - ffmpeg: decode two bounded audio edge windows for silence verification
# Slim image keeps build context small; `--no-install-recommends` avoids pulling
# in X11 / doc packages we don't need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl --version

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer when only
# application code changes. `requirements.txt` lives under `backend/` so we
# copy just that file, install, then copy the rest of the source.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && deno --version \
    && python -m yt_dlp --version \
    && ffmpeg -version >/dev/null

# Application source — the main.py startup sweeps the ingestion package on
# import, so we need the whole backend tree present before uvicorn boots.
COPY backend ./backend

# Railway injects $PORT. Bind to 0.0.0.0 so the container is reachable from
# the Railway edge proxy. `backend.app.main:app` is the FastAPI entrypoint.
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Use shell form (not JSON exec form) so ${PORT} is expanded by /bin/sh at
# container start. Railway injects $PORT at runtime; exec form with sh -c
# has historically been mangled by Railway's runner, causing uvicorn to see
# the literal string "$PORT" and fail with "not a valid integer".
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
