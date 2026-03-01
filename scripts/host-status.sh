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

show_port "Backend" 8000
show_port "Frontend" 3001
