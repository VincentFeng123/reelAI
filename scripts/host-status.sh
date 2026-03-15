#!/usr/bin/env bash
set -euo pipefail

show_port() {
  local name="$1"
  local port="$2"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "$name: RUNNING on localhost:$port"
  else
    echo "$name: DOWN"
  fi
}

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  if curl --silent --show-error --fail --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "Backend: RUNNING and healthy on localhost:8000"
  else
    echo "Backend: LISTENING but unresponsive on localhost:8000"
  fi
else
  echo "Backend: DOWN"
fi

show_port "Frontend" 3001
