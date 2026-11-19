"""Pure access-decision logic (Sections 4 and 8). No I/O here, so it is easy to test."""
import re
from datetime import date, datetime

NO_PLATE, UNREADABLE, ALLOWED, NOT_REGISTERED, BLOCKED_EXPIRED, SYSTEM_ERROR = range(6)

DECISION_NAMES = {
    NO_PLATE: "NO_PLATE",
    UNREADABLE: "UNREADABLE",
    ALLOWED: "ALLOWED",
    NOT_REGISTERED: "DENIED_NOT_REGISTERED",
    BLOCKED_EXPIRED: "DENIED_BLOCKED_OR_EXPIRED",
    SYSTEM_ERROR: "SYSTEM_ERROR",
}

# Indian registration formats: standard (UP70AB1234, DL3CAB1234, MH12A1234)
# and Bharat series (22BH1234AA).
_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
_BH = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


def normalize_plate(text) -> str:
    """Uppercase, keep only A-Z and 0-9 (FR-GW-8), and strip HSRP badge/border noise."""
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if is_indian_format(clean):
        return clean
    candidates = []
    # Strip trailing border artifacts (1 to 2 chars)
    for i in range(1, min(3, len(clean))):
        if is_indian_format(clean[:-i]):
            candidates.append(clean[:-i])
    # Strip leading badge (IND, YND, ID, 1ND, etc.) (1 to 4 chars)
    for i in range(1, min(5, len(clean))):
        if is_indian_format(clean[i:]):
            candidates.append(clean[i:])
    # Strip both
    for left in range(1, 4):
        for right in range(1, 3):
            cand = clean[left:-right]
            if is_indian_format(cand):
                candidates.append(cand)
    if candidates:
        candidates.sort(key=lambda c: len(c), reverse=True)
        return candidates[0]
    return clean


def is_indian_format(plate: str) -> bool:
    return bool(_STANDARD.match(plate) or _BH.match(plate))


def _num(value):
    """Return a float in 0..1, or None if missing/invalid (never assume high)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


class AIResult:
    def __init__(self, code=None, plate="", det_conf=None, ocr_conf=None, error=None):
        self.code = code          # None means "plate is usable, go to DB lookup"
        self.plate = plate
        self.det_conf = det_conf
        self.ocr_conf = ocr_conf
        self.error = error


# def evaluate_ai(ai, conf_min: float, min_len: int, strict: bool) -> AIResult:
#     """Steps 3-6 of Section 8. Adapter rule (T15):
#     - response not a dict / plate_found not a bool         -> code 5
#     - plate found but ocr_confidence missing or invalid    -> code 5 (malformed)
#     - no plate                                             -> code 0
#     - text empty/short/bad format/low OCR confidence       -> code 1
#     """
    
#     if not isinstance(ai, dict) or not isinstance(ai.get("plate_found"), bool):
#         return AIResult(SYSTEM_ERROR, error="malformed AI response")

#     if not ai["plate_found"]:
#         return AIResult(NO_PLATE)

#     det_conf = _num(ai.get("detection_confidence"))
#     ocr_conf = _num(ai.get("ocr_confidence"))
#     plate = normalize_plate(ai.get("plate_text"))

#     if ocr_conf is None:
#         return AIResult(SYSTEM_ERROR, plate, det_conf, None,
#                         "AI response missing/invalid ocr_confidence")
#     if (
#         not plate
#         or len(plate) < min_len
#         or ocr_conf < conf_min
#         or (strict and not is_indian_format(plate))
#     ):
#         return AIResult(UNREADABLE, plate, det_conf, ocr_conf)
#     return AIResult(None, plate, det_conf, ocr_conf)



def evaluate_ai(ai, conf_min: float, min_len: int, strict: bool) -> AIResult:
    """Steps 3-6 of Section 8."""

    if not isinstance(ai, dict) or not isinstance(ai.get("plate_found"), bool):
        return AIResult(SYSTEM_ERROR, error="malformed AI response")

    if not ai["plate_found"]:
        return AIResult(NO_PLATE)

    det_conf = _num(ai.get("detection_confidence"))
    ocr_conf = _num(ai.get("ocr_confidence"))
    plate = normalize_plate(ai.get("plate_text"))

    # Debug information
    print("\n========== AI DEBUG ==========")
    print("Plate found:", ai["plate_found"])
    print("Detected plate:", plate)
    print("Detection confidence:", det_conf)
    print("OCR confidence:", ocr_conf)
    print("Required confidence:", conf_min)
    print("Minimum length:", min_len)
    print("Actual length:", len(plate))
    print("Indian format valid:", is_indian_format(plate))
    print("Strict mode:", strict)
    print("==============================")

    if ocr_conf is None:
        return AIResult(
            SYSTEM_ERROR,
            plate,
            det_conf,
            None,
            "AI response missing/invalid ocr_confidence"
        )

    if (
        not plate
        or len(plate) < min_len
        or ocr_conf < conf_min
        or (strict and not is_indian_format(plate))
    ):
        return AIResult(UNREADABLE, plate, det_conf, ocr_conf)

    return AIResult(None, plate, det_conf, ocr_conf)


def _to_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def evaluate_vehicle(vehicle, today: date) -> int:
    """Steps 7-9 of Section 8. Validity dates are inclusive (Section 8.3)."""
    if vehicle is None:
        return NOT_REGISTERED
    if str(vehicle.get("status", "")).lower() != "active":
        return BLOCKED_EXPIRED
    try:
        valid_from = _to_date(vehicle.get("valid_from"))
        valid_to = _to_date(vehicle.get("valid_to"))
    except ValueError:
        return BLOCKED_EXPIRED  # bad dates in DB -> fail closed
    if valid_from and today < valid_from:
        return BLOCKED_EXPIRED
    if valid_to and today > valid_to:
        return BLOCKED_EXPIRED
    return ALLOWED
