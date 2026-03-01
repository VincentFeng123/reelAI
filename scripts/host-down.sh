#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/.pids"

kill_pid_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$file" || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$file"
}

kill_by_port() {
  local port="$1"
  local pids
  pids="$(lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids >/dev/null 2>&1 || true
  fi
}

kill_pid_file "$PID_DIR/backend.pid"
kill_pid_file "$PID_DIR/frontend.pid"

kill_by_port 8000
kill_by_port 3001

# Also stop dev reloader parent processes that may respawn children.
pkill -f "uvicorn app.main:app --reload --host 127.0.0.1 --port 8000" >/dev/null 2>&1 || true
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000" >/dev/null 2>&1 || true
pkill -f "uvicorn app.main:app --reload --host localhost --port 8000" >/dev/null 2>&1 || true
pkill -f "uvicorn app.main:app --host localhost --port 8000" >/dev/null 2>&1 || true
pkill -f "npm run dev -- --hostname 127.0.0.1 --port 3001" >/dev/null 2>&1 || true
pkill -f "next dev --hostname 127.0.0.1 --port 3001" >/dev/null 2>&1 || true
pkill -f "npm run dev -- --hostname localhost --port 3001" >/dev/null 2>&1 || true
pkill -f "next dev --hostname localhost --port 3001" >/dev/null 2>&1 || true

echo "Stopped backend (8000) and frontend (3001)."
