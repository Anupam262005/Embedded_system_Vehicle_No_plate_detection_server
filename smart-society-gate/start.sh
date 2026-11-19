#!/usr/bin/env bash
set -e

# Start AI service in background on port 8001
echo "[start] Starting AI service on 127.0.0.1:8001..."
uvicorn ai_service.main:app --host 127.0.0.1 --port 8001 &

# Start Gateway in foreground on Render's allocated port (default 8000)
echo "[start] Starting Gateway on 0.0.0.0:${PORT:-8000}..."
exec uvicorn gateway.main:app --host 0.0.0.0 --port "${PORT:-8000}"
