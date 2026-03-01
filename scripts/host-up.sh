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

start_backend() {
  if is_listening 8000; then
    echo "Backend already running on localhost:8000"
    return
  fi

  nohup bash -lc "cd '$ROOT_DIR/backend' && source .venv/bin/activate && exec uvicorn app.main:app --host 0.0.0.0 --port 8000" \
    >"$LOG_DIR/backend.log" 2>&1 &
  echo $! >"$PID_DIR/backend.pid"
  if wait_for_port 8000 30 0.5; then
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

  nohup bash -lc "cd '$ROOT_DIR' && exec npm run dev -- --hostname 0.0.0.0 --port 3001" \
    >"$LOG_DIR/frontend.log" 2>&1 &
  echo $! >"$PID_DIR/frontend.pid"
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
