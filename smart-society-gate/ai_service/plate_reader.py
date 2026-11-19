"""YOLO plate detection + EasyOCR text reading (Section 5.3, 10.3).

If ai_service/models/best.pt is missing, the reader falls back to running
EasyOCR on the whole image. That is only for quick testing with close-up
plate photos; use the trained best.pt for the real demo.
"""
import os
import re
import threading
import time

import cv2
import numpy as np

ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Crop quality gates (reject before OCR -> "unreadable" instead of bad text)
MIN_CROP_W, MIN_CROP_H = 60, 15
MIN_BRIGHTNESS = 15
MIN_SHARPNESS = 10.0  # variance of Laplacian
RETRY_BELOW_CONF = 0.60  # run the extra preprocessing pass below this


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def valid_format(plate: str) -> bool:
    return bool(_STANDARD.match(plate) or _BH.match(plate))


def strip_ind(plate: str) -> str:
    """Remove HSRP 'IND' marking and border line artifacts (e.g. trailing '1') only if what remains is a valid plate."""
    if valid_format(plate):
        return plate
    for cand in (plate[3:] if plate.startswith("IND") else None,
                 plate[:-3] if plate.endswith("IND") else None,
                 plate.replace("IND", "", 1)):
        if cand and valid_format(cand):
            return cand
    # Remove single trailing border artifact read as character (e.g. HR26DK83371 -> HR26DK8337)
    if len(plate) > 1 and valid_format(plate[:-1]):
        return plate[:-1]
    # Remove single leading border artifact
    if len(plate) > 1 and valid_format(plate[1:]):
        return plate[1:]
    return plate


class PlateReader:
    def __init__(self, weights: str, det_conf_min: float = 0.25, gpu: bool = False):
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)
        except Exception:
            pass
        cv2.setNumThreads(0)
        import easyocr

        self.det_conf_min = det_conf_min
        self.lock = threading.Lock()  # serialize inference (NFR-PERF-2)
        self.model = None
        if weights and os.path.isfile(weights):
            from ultralytics import YOLO

            self.model = YOLO(weights)
            print(f"[ai] YOLO weights loaded: {weights}")
        else:
            print(f"[ai] WARNING: '{weights}' not found -> whole-image OCR fallback mode")
        self.ocr = easyocr.Reader(["en"], gpu=gpu)

    @property
    def mode(self) -> str:
        return "yolo+easyocr" if self.model else "ocr-only-fallback"

    def warm_up(self):
        try:
            import gc
            blank = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.putText(blank, "UP70AB1234", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            self.read(blank)
            gc.collect()
        except Exception as exc:
            print(f"[ai] warm_up skipped: {exc}")

    # ---------- OCR helpers ----------
    def _ocr_once(self, img):
        segs = self.ocr.readtext(img, allowlist=ALNUM, detail=1, paragraph=False)
        segs = [s for s in segs if normalize(s[1])]
        if not segs:
            return "", None, None
        segs = [s for s in segs if normalize(s[1]) != "IND"] or segs
        # Order segments: two-row plates -> top row first, then left to right
        ys = [np.mean([p[1] for p in s[0]]) for s in segs]
        hs = [max(p[1] for p in s[0]) - min(p[1] for p in s[0]) for s in segs]
        row_gap = max(np.median(hs) * 0.6, 1)
        top = min(ys)
        rows = [0 if (y - top) < row_gap else 1 for y in ys]
        order = sorted(range(len(segs)),
                       key=lambda i: (rows[i], min(p[0] for p in segs[i][0])))
        text = strip_ind(normalize("".join(segs[i][1] for i in order)))
        confs = [float(segs[i][2]) for i in order]
        return text, min(confs), sum(confs) / len(confs)

    def _read_crop(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w < MIN_CROP_W or h < MIN_CROP_H:
            return "", 0.0, 0.0, "crop too small"
        if gray.mean() < MIN_BRIGHTNESS:
            return "", 0.0, 0.0, "crop too dark"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
            return "", 0.0, 0.0, "crop too blurry"

        if w < 300:  # upscale small plates for OCR
            scale = 300 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        best = self._ocr_once(gray)
        # Conditional extra pass only when first pass is weak (10.3 item 3)
        if best[1] is None or best[1] < RETRY_BELOW_CONF or not valid_format(best[0]):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            alt = self._ocr_once(clahe)
            if alt[1] is not None and (
                best[1] is None
                or (valid_format(alt[0]), alt[1]) > (valid_format(best[0]), best[1])
            ):
                best = alt
        text, min_conf, avg_conf = best
        return text, (min_conf or 0.0), (avg_conf or 0.0), None

    # ---------- main entry ----------
    def read(self, img) -> dict:
        t0 = time.perf_counter()
        with self.lock:
            detections = []
            if self.model is not None:
                res = self.model(img, conf=self.det_conf_min, verbose=False)[0]
                boxes = sorted(
                    ((b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes),
                    key=lambda x: -x[1],
                )[:3]
                H, W = img.shape[:2]
                for (x1, y1, x2, y2), conf in boxes:
                    px, py = int((x2 - x1) * 0.04), int((y2 - y1) * 0.08)
                    x1, y1 = max(int(x1) - px, 0), max(int(y1) - py, 0)
                    x2, y2 = min(int(x2) + px, W), min(int(y2) + py, H)
                    text, mn, avg, reason = self._read_crop(img[y1:y2, x1:x2])
                    detections.append({
                        "bbox": [x1, y1, x2, y2], "det_conf": round(conf, 4),
                        "text": text, "ocr_conf": round(mn, 4),
                        "ocr_conf_avg": round(avg, 4),
                        "valid_format": valid_format(text), "reject_reason": reason,
                    })
                # Fallback: if YOLO found no plate, test if the image itself is already a cropped plate photo
                if not detections or not any(d.get("text") for d in detections):
                    text, mn, avg, reason = self._read_crop(img)
                    if text:
                        detections.append({
                            "bbox": [0, 0, W, H], "det_conf": 1.0, "text": text,
                            "ocr_conf": round(mn, 4), "ocr_conf_avg": round(avg, 4),
                            "valid_format": valid_format(text), "reject_reason": reason,
                        })
            else:
                text, mn, avg, reason = self._read_crop(img)
                if text:
                    H, W = img.shape[:2]
                    detections.append({
                        "bbox": [0, 0, W, H], "det_conf": None, "text": text,
                        "ocr_conf": round(mn, 4), "ocr_conf_avg": round(avg, 4),
                        "valid_format": valid_format(text), "reject_reason": reason,
                    })

        if not detections:
            return {
                "plate_found": False, "plate_text": "", "detection_confidence": None,
                "ocr_confidence": None, "ocr_confidence_avg": None,
                "valid_format": False, "detections": [], "mode": self.mode,
                "processing_ms": int((time.perf_counter() - t0) * 1000),
            }
        # Prefer a valid-format read, then the highest OCR confidence
        best = max(detections, key=lambda d: (d["valid_format"], d["ocr_conf"]))
        return {
            "plate_found": True,
            "plate_text": best["text"],
            "detection_confidence": best["det_conf"],
            "ocr_confidence": best["ocr_conf"],          # minimum segment confidence
            "ocr_confidence_avg": best["ocr_conf_avg"],
            "valid_format": best["valid_format"],
            "detections": detections,
            "mode": self.mode,
            "processing_ms": int((time.perf_counter() - t0) * 1000),
        }
