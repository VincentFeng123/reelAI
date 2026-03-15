#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/.logs"
PID_DIR="$ROOT_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

is_listening() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

backend_healthy() {
  curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local attempts="${2:-20}"
  local delay="${3:-0.5}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if is_listening "$port"; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

wait_for_backend_health() {
  local attempts="${1:-30}"
  local delay="${2:-0.5}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if backend_healthy; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

spawn_detached() {
  local workdir="$1"
  local log_file="$2"
  shift 2

  python3 - "$workdir" "$log_file" "$@" <<'PY'
import subprocess
import sys

workdir = sys.argv[1]
log_file = sys.argv[2]
cmd = sys.argv[3:]

with open(log_file, "ab", buffering=0) as log:
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
    )

print(proc.pid)
PY
}

stop_backend_port() {
  local pids
  pids="$(lsof -t -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids >/dev/null 2>&1 || true
    sleep 1
  fi

  pids="$(lsof -t -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids >/dev/null 2>&1 || true
    sleep 1
  fi
}

start_backend() {
  if is_listening 8000; then
    if backend_healthy; then
      echo "Backend already running on localhost:8000"
      return
    fi
    echo "Backend on localhost:8000 is unresponsive; replacing it"
    stop_backend_port
  fi

  spawn_detached \
    "$ROOT_DIR/backend" \
    "$LOG_DIR/backend.log" \
    "$ROOT_DIR/backend/.venv/bin/python" \
    -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    >"$PID_DIR/backend.pid"
  if wait_for_port 8000 30 0.5 && wait_for_backend_health 30 0.5; then
    echo "Backend started on localhost:8000"
  else
    echo "Backend failed to start. See $LOG_DIR/backend.log"
    exit 1
  fi
}

start_frontend() {
  if is_listening 3001; then
    echo "Frontend already running on localhost:3001"
    return
  fi

  # Prevent stale Next runtime chunk errors from previous dev sessions.
  rm -rf "$ROOT_DIR/.next"

  spawn_detached \
    "$ROOT_DIR" \
    "$LOG_DIR/frontend.log" \
    npm run dev -- --hostname 0.0.0.0 --port 3001 \
    >"$PID_DIR/frontend.pid"
  if wait_for_port 3001 40 0.5; then
    echo "Frontend started on localhost:3001"
  else
    echo "Frontend failed to start. See $LOG_DIR/frontend.log"
    exit 1
  fi
}

start_backend
start_frontend

echo "Frontend: http://localhost:3001 (or http://127.0.0.1:3001)"
echo "Backend:  http://localhost:8000 (or http://127.0.0.1:8000)"
