from __future__ import annotations

import re
from datetime import date


def _digits_only(x: str) -> str:
    return "".join(ch for ch in x if ch.isdigit())


def normalize_phone(value: str | None) -> str | None:
    if value is None:
        return None
    digits = _digits_only(value)
    if not digits:
        return None

    if len(digits) == 10:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    return digits


def normalize_email(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip().lower()
    return s or None


def parse_money_usd(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip().lower()
    if not s:
        return None

    s = s.replace("usd", "").replace("$", "").replace(",", "").strip()

    mult = 1.0
    if s.endswith("k"):
        mult = 1000.0
        s = s[:-1].strip()

    s = re.sub(r"[^0-9.]+", "", s)
    if not s:
        return None

    try:
        return float(s) * mult
    except Exception:
        return None


def normalize_date(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None

    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if not m:
        return s

    y = int(m.group(1))
    mo = int(m.group(2))
    d = int(m.group(3))
    try:
        _ = date(y, mo, d)
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except Exception:
        return s


def normalize_person_name(value: str | None) -> str | None:
    if value is None:
        return None
    s = " ".join(value.strip().split())
    if not s:
        return None

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            last = parts[0]
            first = parts[1]
            s = f"{first} {last}"

    s = s.strip('"').strip("'").strip()
    return s or None


def normalize_policy_number(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None

    s = re.sub(r"\s+", " ", s)
    if " " in s and "-" not in s:
        tokens = [t for t in s.split(" ") if t]
        if len(tokens) >= 2:
            s = "-".join(tokens)

    s = s.replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s)
    return s or None


def normalize_police_report_number(value: str | None) -> str | None:
    if value is None:
        return None
    s = " ".join(value.strip().split())
    if not s:
        return None

    candidates = re.findall(r"[A-Za-z0-9-]+", s)
    best: str | None = None
    best_score = -1
    for c in candidates:
        digit_count = sum(ch.isdigit() for ch in c)
        if digit_count < 3:
            continue
        score = digit_count * 10 + len(c)
        if score > best_score:
            best_score = score
            best = c

    return best or s


def normalize_location(value: str | None) -> str | None:
    if value is None:
        return None
    s = " ".join(value.strip().split())
    if not s:
        return None

    s = re.sub(r"^\s*near\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(parts) if parts else s


def normalize_loss_description(value: str | None) -> str | None:
    if value is None:
        return None
    s = " ".join(value.strip().split())
    if not s:
        return None

    s = re.sub(r"^the insured\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\binsured\s+reports\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthe vehicle\s+", "vehicle ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthe car\s+", "car ", s, flags=re.IGNORECASE)
    s = s.replace(". ", "; ").replace(".", "")
    s = re.sub(r"\s*;\s*", "; ", s).strip("; ").strip()
    return s or None


def location_city_state(value: str | None) -> str | None:
    if value is None:
        return None
    s = normalize_location(value)
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 3 and re.fullmatch(r"[A-Z]{2}", parts[-1]):
        return f"{parts[-2]}, {parts[-1]}"
    return s
