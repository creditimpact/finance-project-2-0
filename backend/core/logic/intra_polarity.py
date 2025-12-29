"""Per-account polarity analysis based on bureau data."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from backend.core.io.json_io import update_json_in_place
from backend.core.io.tags import upsert_tag
from backend.core.logic.polarity import classify_field_value
from backend.core.logic.summary_compact import compact_merge_sections

logger = logging.getLogger(__name__)

_BUREAU_KEYS: tuple[str, ...] = ("transunion", "experian", "equifax")
_POLARITY_SCHEMA_VERSION = 1


def _load_bureaus(account_path: Path, sid: str) -> Dict[str, Any]:
    bureaus_path = account_path / "bureaus.json"
    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "POLARITY_BUREAUS_MISSING sid=%s path=%s",
            sid,
            bureaus_path,
        )
        return {}
    except OSError:
        logger.exception("POLARITY_BUREAUS_READ_FAILED path=%s", bureaus_path)
        return {}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.exception("POLARITY_BUREAUS_PARSE_FAILED path=%s", bureaus_path)
        return {}

    if not isinstance(data, Mapping):
        logger.warning("POLARITY_BUREAUS_INVALID root_type=%s", type(data).__name__)
        return {}

    return dict(data)


def _should_write_probes() -> bool:
    flag = os.getenv("WRITE_POLARITY_PROBES")
    if not flag:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _format_bureau_list(bureaus: tuple[str, ...] | list[str]) -> str:
    if not bureaus:
        return "[]"
    return "[" + ",".join(bureaus) + "]"


def _maybe_write_probe_tags(
    account_dir: Path,
    *,
    sid: str,
    idx: str,
    payload: Mapping[str, Any],
) -> None:
    if not _should_write_probes():
        return

    tag_payload = {"source": "intra_polarity"}
    tag_payload.update(payload)
    upsert_tag(account_dir, tag_payload, unique_keys=("kind", "field", "bureau"))
    logger.info(
        "POLARITY_TAG_WRITTEN sid=%s idx=%s field=%s bureau=%s p=%s s=%s",
        sid,
        idx,
        payload.get("field"),
        payload.get("bureau"),
        payload.get("polarity"),
        payload.get("severity"),
    )


def _collect_account_fields(bureaus_data: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for bureau_key in _BUREAU_KEYS:
        bureau_values = bureaus_data.get(bureau_key)
        if not isinstance(bureau_values, Mapping):
            continue
        for raw_field in bureau_values.keys():
            field = str(raw_field)
            if field in seen:
                continue
            seen.add(field)
            fields.append(field)
    return fields


def analyze_account_polarity(sid: str, account_dir: "os.PathLike[str]") -> Dict[str, Any]:
    """Analyze polarity for bureau fields and persist results."""

    account_path = Path(account_dir)
    account_idx = account_path.name
    bureaus_data = _load_bureaus(account_path, sid)
    fields = _collect_account_fields(bureaus_data)

    bureaus_block: Dict[str, Dict[str, Dict[str, Any]]] = {}

    present_bureaus: list[str] = [
        bureau
        for bureau in _BUREAU_KEYS
        if isinstance(bureaus_data.get(bureau), Mapping)
    ]

    logger.info(
        "POLARITY_START sid=%s idx=%s bureaus=%s",
        sid,
        account_idx,
        _format_bureau_list(present_bureaus),
    )

    include_vals = os.getenv("POLARITY_INCLUDE_VALUES") == "1"
    include_rules = os.getenv("POLARITY_INCLUDE_RULES") == "1"

    for field in fields:
        log_values: Dict[str, str] = {bureau: "-" for bureau in _BUREAU_KEYS}
        for bureau_key in _BUREAU_KEYS:
            bureau_values = bureaus_data.get(bureau_key)
            if not isinstance(bureau_values, Mapping):
                continue
            if field not in bureau_values:
                continue
            raw_value = bureau_values.get(field)
            classification = classify_field_value(field, raw_value)
            if classification.get("reason") == "field not configured":
                classification = dict(classification)
                classification["polarity"] = "neutral"
                classification["severity"] = "low"
            polarity_value = classification.get("polarity")
            polarity = str(polarity_value) if polarity_value else "unknown"
            severity_value = classification.get("severity")
            severity = str(severity_value) if severity_value else "low"
            bureau_results = bureaus_block.setdefault(bureau_key, {})
            cell: Dict[str, Any] = {
                "polarity": polarity,
                "severity": severity,
            }
            if include_vals:
                cell["value_raw"] = raw_value
                cell["value_norm"] = classification.get("value_norm")

            if include_rules:
                rule_hit = classification.get("rule_hit")
                reason = classification.get("reason")
                if rule_hit is not None:
                    cell["rule_hit"] = rule_hit
                if reason is not None:
                    cell["reason"] = reason
                cell["source"] = f"bureaus.json:{bureau_key}.{field}"

            bureau_results[field] = cell
            _maybe_write_probe_tags(
                account_path,
                sid=sid,
                idx=account_idx,
                payload={
                    "kind": "polarity_probe",
                    "field": field,
                    "bureau": bureau_key,
                    "polarity": polarity,
                    "severity": severity,
                },
            )
            log_values[bureau_key] = f"{polarity}:{severity}"

        logger.info(
            "POLARITY_FIELD sid=%s idx=%s field=%s TU=%s EX=%s EQ=%s",
            sid,
            account_idx,
            field,
            log_values.get("transunion", "-"),
            log_values.get("experian", "-"),
            log_values.get("equifax", "-"),
        )

    polarity_block: Dict[str, Any] = {
        "schema_version": _POLARITY_SCHEMA_VERSION,
        "bureaus": bureaus_block,
    }

    summary_path = account_path / "summary.json"

    def _update_summary(existing: object) -> Dict[str, Any]:
        if isinstance(existing, Mapping):
            summary: Dict[str, Any] = dict(existing)
        else:
            summary = {}

        if summary.get("polarity_check") == polarity_block:
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary)
            return summary

        summary["polarity_check"] = polarity_block
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary)
        return summary

    try:
        update_json_in_place(summary_path, _update_summary)
    except ValueError:
        logger.exception("POLARITY_SUMMARY_PARSE_FAILED path=%s", summary_path)
    except OSError:
        logger.exception("POLARITY_SUMMARY_WRITE_FAILED path=%s", summary_path)

    return polarity_block


__all__ = ["analyze_account_polarity"]
