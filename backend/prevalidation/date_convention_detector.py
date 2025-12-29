"""Utilities for detecting the month language in a run's Two-Year Payment History."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


HEBREW_MONTHS = (
    "ינו׳",
    "פבר׳",
    "מרץ",
    "אפר׳",
    "מאי",
    "יוני",
    "יולי",
    "אוג׳",
    "ספט׳",
    "אוק׳",
    "נוב׳",
    "דצמ׳",
)

ENGLISH_MONTH_TOKENS = frozenset(
    (
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
)

logger = logging.getLogger(__name__)

DATE_CONVENTION_SCOPE = os.getenv("DATE_CONVENTION_SCOPE", "global")


def _iter_account_raw_texts(raw_lines_path: str) -> Iterable[str]:
    """Yield the text field for each entry in a raw_lines.json file."""

    with open(raw_lines_path, "r", encoding="utf-8") as handle:
        in_object = False
        buffer_lines: list[str] = []

        for line in handle:
            stripped = line.lstrip()

            if not in_object:
                if stripped.startswith("{"):
                    in_object = True
                    buffer_lines = [line]
                continue

            buffer_lines.append(line)

            if stripped.startswith("}") or stripped.startswith("},"):
                in_object = False
                json_blob = "".join(buffer_lines).rstrip().rstrip(",")
                if not json_blob:
                    continue

                try:
                    record = json.loads(json_blob)
                except json.JSONDecodeError:
                    continue

                text = record.get("text")
                if isinstance(text, str):
                    yield text


def _analyze_text_for_months(text: str) -> Tuple[int, int, list[str], list[str]]:
    """Return hit counts and evidence tokens for Hebrew and English months."""

    he_hits = 0
    he_samples: list[str] = []
    for month in HEBREW_MONTHS:
        occurrences = text.count(month)
        if not occurrences:
            continue
        he_hits += occurrences
        remaining_slots = 3 - len(he_samples)
        if remaining_slots > 0:
            he_samples.extend([f"token={month}"] * min(occurrences, remaining_slots))

    en_hits = 0
    en_samples: list[str] = []
    tokens: list[str] = []
    current: list[str] = []

    for char in text:
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))

    for token in tokens:
        lower = token.lower()
        if lower in ENGLISH_MONTH_TOKENS:
            en_hits += 1
            if len(en_samples) < 3:
                en_samples.append(f"token={lower}")

    return he_hits, en_hits, he_samples, en_samples


def detect_month_language_for_run(run_dir: str) -> Dict[str, object]:
    """Scans all accounts' raw_lines.json files to infer the month language."""

    accounts_dir = os.path.join(run_dir, "cases", "accounts")

    total_he_hits = 0
    total_en_hits = 0
    accounts_scanned = 0
    he_evidence_samples: list[str] = []
    en_evidence_samples: list[str] = []

    if os.path.isdir(accounts_dir):
        for entry in sorted(os.scandir(accounts_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue

            raw_lines_path = os.path.join(entry.path, "raw_lines.json")
            if not os.path.isfile(raw_lines_path):
                continue

            accounts_scanned += 1
            marker_found = False

            for text in _iter_account_raw_texts(raw_lines_path):
                if not marker_found:
                    if "Two-Year Payment History" in text:
                        marker_found = True
                    else:
                        continue

                he_hits, en_hits, he_samples, en_samples = _analyze_text_for_months(text)
                total_he_hits += he_hits
                total_en_hits += en_hits

                if he_samples and len(he_evidence_samples) < 3:
                    slots = 3 - len(he_evidence_samples)
                    he_evidence_samples.extend(he_samples[:slots])
                if en_samples and len(en_evidence_samples) < 3:
                    slots = 3 - len(en_evidence_samples)
                    en_evidence_samples.extend(en_samples[:slots])

    if total_he_hits > total_en_hits:
        month_language = "he"
        convention = "DMY"
        confidence = 1.0
    elif total_en_hits > total_he_hits:
        month_language = "en"
        convention = "MDY"
        confidence = 1.0
    else:
        month_language = "unknown"
        convention = None
        confidence = 0.0

    if os.environ.get("VALIDATION_DEBUG") == "1":
        for idx, sample in enumerate(he_evidence_samples):
            logger.debug("DATE_DETECT evidence[he][%d]=%s", idx, sample)
        for idx, sample in enumerate(en_evidence_samples):
            logger.debug("DATE_DETECT evidence[en][%d]=%s", idx, sample)

    return {
        "date_convention": {
            "scope": DATE_CONVENTION_SCOPE,
            "convention": convention,
            "month_language": month_language,
            "confidence": confidence,
            "evidence_counts": {
                "he_hits": total_he_hits,
                "en_hits": total_en_hits,
                "accounts_scanned": accounts_scanned,
            },
            "detector_version": "1.0",
        }
    }


def read_date_convention(run_dir: os.PathLike[str] | str) -> Optional[dict[str, Any]]:
    """Return the persisted ``date_convention`` block for ``run_dir`` if present."""

    base_dir = Path(run_dir)
    if not base_dir.exists():
        return None

    scope_path = os.getenv("DATE_CONVENTION_PATH", "traces/date_convention.json")
    target_path = base_dir / scope_path

    try:
        raw = target_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug(
            "DATE_CONVENTION_READ_FAILED run_dir=%s path=%s",
            run_dir,
            target_path,
            exc_info=True,
        )
        return None

    if not raw.strip():
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug(
            "DATE_CONVENTION_INVALID_JSON run_dir=%s path=%s",
            run_dir,
            target_path,
            exc_info=True,
        )
        return None

    block = payload.get("date_convention")
    if isinstance(block, dict):
        return dict(block)
    return None
