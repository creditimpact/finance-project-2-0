from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from backend.core.metrics.field_coverage import metrics
from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

logger = logging.getLogger(__name__)

_REGISTRY_CACHE: Dict[str, Any] | None = None
_PRIORITY = ["EQ", "TU", "EX"]

_PLACEHOLDER_VALUES = {"--", "-", "N/A", "NA"}
_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})[.\-/\s](\d{4})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_DATEISH_RE = re.compile(r"^[\d\s./-]+$")


def load_registry(path: str | None = None) -> Dict[str, Any]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    p = Path(path) if path else Path(__file__).with_name("registry.yaml")
    try:
        with open(p) as fh:
            _REGISTRY_CACHE = yaml.safe_load(fh) or {}
    except Exception:
        logger.exception("normalized_registry_load_failed")
        _REGISTRY_CACHE = {}
    return _REGISTRY_CACHE


def _coerce_number(val: Any) -> Tuple[float | None, bool]:
    try:
        text = str(val).replace(",", "")
        return float(text), False
    except Exception:
        return None, False


def _coerce_string(val: Any) -> Tuple[str, bool]:
    text = str(val)
    stripped = text.strip()
    return stripped, stripped != text


def _coerce_date_iso(val: Any) -> Tuple[str | None, bool]:
    if val is None:
        return None, False

    text = str(val).strip()
    if not text:
        return None, False

    if text.upper() in _PLACEHOLDER_VALUES:
        return None, False

    parsed = parse_date_any(text)
    if parsed:
        return parsed, False

    month_year = _MONTH_YEAR_RE.match(text)
    if month_year:
        month, year = month_year.groups()
        try:
            d = datetime(int(year), int(month), 1)
        except ValueError:
            return None, False
        return d.strftime("%Y-%m-01"), True

    year_only = _YEAR_ONLY_RE.match(text)
    if year_only:
        year = year_only.group(1)
        try:
            d = datetime(int(year), 1, 1)
        except ValueError:
            return None, False
        return d.strftime("%Y-01-01"), True

    if _DATEISH_RE.match(text):
        return None, False

    return text, False


def build_normalized(by_bureau: Dict[str, Dict[str, Any]], registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    overlay: Dict[str, Dict[str, Any]] = {}
    fields = registry.get("fields", {})
    bureaus = registry.get("bureaus", [])
    for field_name, cfg in fields.items():
        sources_raw: Dict[str, Any] = {}
        coerced_vals: Dict[str, Any] = {}
        derived_flags: Dict[str, bool] = {}
        labels_map = cfg.get("bureau_labels", {})
        for bureau in bureaus:
            raw_dict = by_bureau.get(bureau, {}) or {}
            candidates = labels_map.get(bureau, [])
            key = next((l for l in candidates if l in raw_dict), None)
            if key is None:
                continue
            raw_value = raw_dict.get(key)
            sources_raw[bureau] = raw_value
            coerce_type = cfg.get("coerce", "passthrough")
            derived = False
            value = raw_value
            if coerce_type == "number":
                value, _ = _coerce_number(raw_value)
            elif coerce_type == "string":
                value, derived = _coerce_string(raw_value)
            elif coerce_type == "date_iso":
                value, derived = _coerce_date_iso(raw_value)
            elif coerce_type == "passthrough":
                value = raw_value
            norm_map = cfg.get("normalize_map", {})
            if isinstance(value, str):
                lowered = value.lower()
                for canon, variants in norm_map.items():
                    var_lows = [v.lower() for v in variants]
                    if lowered == canon.lower() or lowered in var_lows:
                        if value != canon:
                            derived = True
                        value = canon
                        break
            coerced_vals[bureau] = value
            derived_flags[bureau] = derived
        if not coerced_vals:
            overlay[field_name] = {"sources": {}, "status": "missing"}
            continue
        # determine value and status
        val_map: Dict[Any, list[str]] = {}
        for b, v in coerced_vals.items():
            val_map.setdefault(v, []).append(b)
        if len(val_map) == 1:
            chosen_value = next(iter(val_map.keys()))
            bureaus_present = list(coerced_vals.keys())
            if len(bureaus_present) == 1 and derived_flags[bureaus_present[0]]:
                status = "derived"
            else:
                status = "agreed"
        else:
            status = "conflict"
            max_count = max(len(bs) for bs in val_map.values())
            tied = [v for v, bs in val_map.items() if len(bs) == max_count]
            if len(tied) == 1:
                chosen_value = tied[0]
            else:
                chosen_value = None
                for b in _PRIORITY:
                    if b in coerced_vals and coerced_vals[b] in tied:
                        chosen_value = coerced_vals[b]
                        break
                if chosen_value is None:
                    chosen_value = tied[0]
        field_entry = {"sources": sources_raw, "status": status}
        if status != "missing":
            field_entry["value"] = chosen_value
        overlay[field_name] = field_entry
    return overlay


def compute_mapping_coverage(
    by_bureau: Dict[str, Dict[str, Any]],
    registry: Dict[str, Any],
) -> Tuple[float, Dict[str, int]]:
    mapping: Dict[str, set[str]] = {}
    for cfg in registry.get("fields", {}).values():
        for bureau, labels in cfg.get("bureau_labels", {}).items():
            mapping.setdefault(bureau, set()).update(labels)
    mapped = 0
    total = 0
    unmapped: Dict[str, int] = {}
    for bureau, data in by_bureau.items():
        for label in data.keys():
            total += 1
            if label in mapping.get(bureau, set()):
                mapped += 1
            else:
                unmapped[label] = unmapped.get(label, 0) + 1
    percent = round(100.0 * mapped / total, 2) if total else 0.0
    return percent, unmapped


def emit_mapping_coverage_metrics(
    session_id: str,
    account_id: str,
    by_bureau: Dict[str, Dict[str, Any]],
    registry: Dict[str, Any],
) -> None:
    try:
        percent, unmapped = compute_mapping_coverage(by_bureau, registry)
        metrics.gauge(
            "stage1.normalized.registry.coverage",
            percent,
            tags={"session_id": session_id, "account_id": account_id},
        )
        top = sorted(unmapped.items(), key=lambda kv: kv[1], reverse=True)[:20]
        for label, count in top:
            metrics.count(
                "stage1.normalized.registry.unmapped",
                count,
                tags={"session_id": session_id, "label": label},
            )
        if top:
            logger.info(
                "normalized.registry_unmapped %s",
                {
                    "session_id": session_id,
                    "account_id": account_id,
                    "top_unmapped": top,
                },
            )
    except Exception:
        logger.exception("normalized_registry_metrics_failed")
