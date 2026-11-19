#!/usr/bin/env bash
set -e

# If AI_URL points to an external service (like Google Colab / HuggingFace), don't start local AI
if [[ "$AI_URL" == *"127.0.0.1"* || "$AI_URL" == *"localhost"* || -z "$AI_URL" ]]; then
    echo "[start] Starting AI service locally on 127.0.0.1:8001..."
    uvicorn ai_service.main:app --host 127.0.0.1 --port 8001 &
else
    echo "[start] External AI URL detected ($AI_URL). Skipping local AI service to save RAM."
fi

# Start Gateway in foreground on Render's allocated port (default 8000)
echo "[start] Starting Gateway on 0.0.0.0:${PORT:-8000}..."
exec uvicorn gateway.main:app --host 0.0.0.0 --port "${PORT:-8000}"
