"""Smart Society Gate - HTTP Gateway (ESP32-facing, port 8000).

Run:  uvicorn gateway.main:app --host 0.0.0.0 --port 8000
"""
import threading
from contextlib import asynccontextmanager
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from gateway import db as dbmod
from gateway.config import settings
from gateway.decision import (
    ALLOWED, DECISION_NAMES, SYSTEM_ERROR, evaluate_ai, evaluate_vehicle,
)
from gateway.image_input import ImageError, to_ai_jpeg
from gateway.security import verify_key

_ai_slots = threading.BoundedSemaphore(settings.MAX_CONCURRENT_AI)
_device_locks: dict[str, threading.Lock] = {}
_device_locks_guard = threading.Lock()
_recent: dict[tuple[str, str], float] = {}  # (device, plate) -> last scan time
_recent_guard = threading.Lock()


def _err(status: int, detail: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail, **extra})


def _device_lock(device_id: str) -> threading.Lock:
    with _device_locks_guard:
        return _device_locks.setdefault(device_id, threading.Lock())


def call_ai(jpeg: bytes) -> dict:
    """POST raw JPEG to the AI service. Raises on any failure."""
    resp = httpx.post(
        f"{settings.AI_URL}/v1/read-plate/raw",
        content=jpeg,
        headers={"Content-Type": "image/jpeg", "X-API-Key": settings.AI_API_KEY},
        timeout=settings.AI_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def _is_duplicate(device_id: str, plate: str) -> bool:
    if not plate or settings.COOLDOWN_S <= 0:
        return False
    now = time.monotonic()
    with _recent_guard:
        last = _recent.get((device_id, plate))
        _recent[(device_id, plate)] = now
        return last is not None and (now - last) < settings.COOLDOWN_S


@asynccontextmanager
async def lifespan(_app):
    try:
        dbmod.ensure_indexes(dbmod.get_db(settings))
    except Exception as exc:  # gateway still starts; /health shows db down
        print(f"[gateway] WARNING: could not reach MongoDB at startup: {exc}")
    yield


app = FastAPI(title="Smart Society Gate - Gateway", version="2.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Smart Society Gate - Gateway",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    status = {"gateway": "ok", "ai": "down", "db": "down"}
    try:
        r = httpx.get(f"{settings.AI_URL}/health", timeout=3)
        if r.status_code == 200:
            status["ai"] = "ok"
    except Exception:
        pass
    try:
        dbmod.get_db(settings).client.admin.command("ping")
        status["db"] = "ok"
    except Exception:
        pass
    return status


@app.post("/api/v1/gate/scan")
async def scan_endpoint(request: Request):
    # Read the raw body here, then run the blocking work in a worker thread.
    # Reject obviously oversized uploads before reading them (FR-GW-3)
    try:
        declared = int(request.headers.get("content-length", "0"))
    except ValueError:
        declared = 0
    if declared > settings.MAX_IMAGE_BYTES:
        return _err(413, "payload too large")
    body = await request.body()
    return await run_in_threadpool(process_scan, dict(request.headers), body)


def process_scan(headers: dict, body: bytes) -> JSONResponse:
    t0 = time.perf_counter()
    request_id = f"scan-{uuid.uuid4().hex[:12]}"
    device_id = headers.get("x-device-id", "").strip()
    device_key = headers.get("x-device-key", "")

    # 1. Authenticate device (FR-GW-2)
    if not device_id or not device_key:
        return _err(401, "missing device credentials")
    try:
        db = dbmod.get_db(settings)
        device = db.devices.find_one({"device_id": device_id})
    except Exception:
        return _err(503, "database unavailable", code=SYSTEM_ERROR,
                    decision=DECISION_NAMES[SYSTEM_ERROR], request_id=request_id)
    if not device or not device.get("enabled", False) or not verify_key(
        device_key, device.get("key_hash", "")
    ):
        return _err(401, "invalid or disabled device credentials")

    # 2. Validate request + image (FR-GW-3, FR-GW-4, FR-GW-5)
    if not body:
        return _err(400, "empty request body")
    if len(body) > settings.MAX_IMAGE_BYTES:
        return _err(413, "payload too large")
    try:
        jpeg, _w, _h = to_ai_jpeg(
            headers.get("content-type", ""), body, headers, settings.ALLOWED_IMAGE_FORMATS
        )
    except ImageError as exc:
        return _err(400, str(exc))

    # One scan at a time per device (FR-ESP-5 / NFR-PERF-2)
    lock = _device_lock(device_id)
    if not lock.acquire(blocking=False):
        return _err(429, "previous scan from this device still in progress")
    try:
        return _decide_and_log(db, device_id, request_id, jpeg, t0)
    finally:
        lock.release()


def _decide_and_log(db, device_id, request_id, jpeg, t0) -> JSONResponse:
    code, plate, det_conf, ocr_conf, error, vehicle = SYSTEM_ERROR, "", None, None, None, None
    try:
        # 3. Call AI (bounded concurrency, timeout)
        if not _ai_slots.acquire(timeout=2):
            return _err(429, "server busy, try again")
        try:
            ai = call_ai(jpeg)
        except Exception as exc:
            ai, error = None, f"AI service error: {type(exc).__name__}: {exc}"
        finally:
            _ai_slots.release()

        if ai is None:
            code = SYSTEM_ERROR
        else:
            # 4-6. Inspect + validate + normalize
            r = evaluate_ai(ai, settings.CONF_MIN, settings.MIN_PLATE_LEN, settings.STRICT_FORMAT)
            plate, det_conf, ocr_conf, error = r.plate, r.det_conf, r.ocr_conf, r.error
            if r.code is not None:
                code = r.code
            else:
                # 7-9. Exact lookup + status/date checks
                try:
                    vehicle = db.vehicles.find_one({"plate_number": plate})
                    today = datetime.now(ZoneInfo(settings.TIMEZONE)).date()
                    code = evaluate_vehicle(vehicle, today)
                except Exception as exc:
                    code, error = SYSTEM_ERROR, f"database error: {exc}"
    except Exception as exc:  # anything unexpected -> fail closed
        code, error = SYSTEM_ERROR, f"internal error: {exc}"

    duplicate = _is_duplicate(device_id, plate) if code != SYSTEM_ERROR else False
    processing_ms = int((time.perf_counter() - t0) * 1000)

    # 10. Access log (FR-GW-10)
    log = {
        "timestamp": datetime.now(timezone.utc),
        "request_id": request_id,
        "device_id": device_id,
        "plate_text": plate or None,
        "detection_confidence": det_conf,
        "ocr_confidence": ocr_conf,
        "code": code,
        "decision": DECISION_NAMES[code],
        "vehicle_id": str(vehicle["_id"]) if vehicle else None,
        "processing_ms": processing_ms,
        "duplicate": duplicate,
        "error": error,
    }
    try:
        db.access_logs.insert_one(log)
    except Exception as exc:
        print(f"[gateway] could not write access log {request_id}: {exc}")

    # 11. Minimal response for the ESP32 (NFR-PRIV-1)
    granted = code == ALLOWED
    return JSONResponse(
        status_code=200,
        content={
            "code": code,
            "decision": DECISION_NAMES[code],
            "plate": plate or None,
            "owner": vehicle.get("owner_name") if granted else None,
            "flat": vehicle.get("flat_no") if granted else None,
            "duplicate": duplicate,
            "request_id": request_id,
            "processing_ms": processing_ms,
        },
    )
