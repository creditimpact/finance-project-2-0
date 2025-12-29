"""Utilities for applying validation AI decisions to account summaries."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.core.io.json_io import _atomic_write_json
from backend.pipeline.runs import RUNS_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "apply_validation_ai_decisions_for_account",
    "apply_validation_ai_decisions_for_all_accounts",
]


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else RUNS_ROOT
    return Path(runs_root)


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        logger.warning("VALIDATION_AI_SUMMARY_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("VALIDATION_AI_SUMMARY_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload
    logger.warning("VALIDATION_AI_SUMMARY_INVALID_TYPE path=%s type=%s", path, type(payload).__name__)
    return None


def _load_result_lines(path: Path) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return results
    except OSError:  # pragma: no cover - defensive logging
        logger.warning("VALIDATION_AI_RESULTS_READ_FAILED path=%s", path, exc_info=True)
        return results

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - defensive logging
            logger.warning(
                "VALIDATION_AI_RESULTS_PARSE_FAILED path=%s line_snippet=%s",
                path,
                line[:200],
            )
            continue
        if not isinstance(payload, Mapping):
            continue

        field_value = payload.get("field")
        if not isinstance(field_value, str):
            continue

        normalized_key = field_value.strip().lower()
        if not normalized_key:
            continue

        results[normalized_key] = dict(payload)

    return results


def _normalize_citations(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    citations: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                citations.append(text)
    return citations


def _assign_if_changed(target: MutableMapping[str, Any], key: str, value: Any) -> bool:
    if value is None:
        if key in target:
            target.pop(key, None)
            return True
        return False

    existing = target.get(key)
    if existing == value:
        return False
    target[key] = value
    return True


def apply_validation_ai_decisions_for_account(
    sid: str,
    runs_root: Path | str | None,
    account_id: int,
    *,
    results_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Merge AI validation decisions back into an account summary."""

    runs_root_path = _resolve_runs_root(runs_root)
    results_path = (
        runs_root_path
        / sid
        / "ai_packs"
        / "validation"
        / "results"
        / f"acc_{account_id:03d}.result.jsonl"
    )

    if results_override is not None:
        results_map = {
            key: dict(value)
            for key, value in results_override.items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
    else:
        results_map = _load_result_lines(results_path)

    if not results_map:
        return False

    summary_path = (
        runs_root_path
        / sid
        / "cases"
        / "accounts"
        / f"{account_id}"
        / "summary.json"
    )

    summary_payload = _load_json(summary_path)
    if summary_payload is None:
        return False

    validation_block = summary_payload.get("validation_requirements")
    if not isinstance(validation_block, Mapping):
        return False

    findings = validation_block.get("findings")
    if not isinstance(findings, list):
        return False

    changed = False

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if not finding.get("send_to_ai"):
            continue

        field_name = finding.get("field")
        if not isinstance(field_name, str):
            continue

        field_key = field_name.strip().lower()
        if not field_key:
            continue

        result_entry = results_map.get(field_key)
        if not result_entry:
            continue

        decision_raw = result_entry.get("decision")
        decision = decision_raw.strip() if isinstance(decision_raw, str) else None
        rationale_raw = result_entry.get("rationale")
        rationale = rationale_raw if isinstance(rationale_raw, str) else None
        citations = _normalize_citations(result_entry.get("citations"))
        legacy_raw = result_entry.get("legacy_decision")
        legacy_decision = legacy_raw.strip() if isinstance(legacy_raw, str) else None

        if decision is not None:
            if (
                "decision" in finding
                and isinstance(finding.get("decision"), str)
                and finding.get("decision") != decision
            ):
                finding.setdefault("pre_ai_decision", finding.get("decision"))
            changed |= _assign_if_changed(finding, "decision", decision)
            changed |= _assign_if_changed(finding, "decision_source", "ai")

        if decision is not None:
            changed |= _assign_if_changed(finding, "default_decision", decision)

        if finding.get("default_decision_hint") is not None:
            changed |= _assign_if_changed(finding, "default_decision_hint", None)

        changed |= _assign_if_changed(finding, "ai_decision", decision)
        changed |= _assign_if_changed(finding, "ai_rationale", rationale)

        if citations:
            changed |= _assign_if_changed(finding, "ai_citations", citations)
        else:
            changed |= _assign_if_changed(finding, "ai_citations", None)

        if legacy_decision is not None:
            changed |= _assign_if_changed(finding, "ai_legacy_decision", legacy_decision)

        ai_payload: dict[str, Any] = {}
        if decision is not None:
            ai_payload["decision"] = decision
        if rationale is not None:
            ai_payload["rationale"] = rationale
        if citations:
            ai_payload["citations"] = citations
        if legacy_decision is not None:
            ai_payload["legacy_decision"] = legacy_decision
        ai_payload["source"] = "validation_ai"

        changed |= _assign_if_changed(
            finding,
            "validation_ai",
            ai_payload if ai_payload else None,
        )

    if not changed:
        return False

    try:
        _atomic_write_json(summary_path, summary_payload)
    except OSError:  # pragma: no cover - defensive logging
        logger.warning(
            "VALIDATION_AI_SUMMARY_WRITE_FAILED sid=%s account_id=%s path=%s",
            sid,
            account_id,
            summary_path,
            exc_info=True,
        )
        return False

    return True


def _discover_result_maps(
    sid: str, runs_root_path: Path
) -> dict[int, dict[str, Mapping[str, Any]]]:
    results_dir = runs_root_path / sid / "ai_packs" / "validation" / "results"
    if not results_dir.is_dir():
        return {}

    account_results: dict[int, dict[str, Mapping[str, Any]]] = {}
    for result_path in sorted(results_dir.glob("acc_*.result.jsonl"), key=lambda p: p.name):
        name = result_path.name
        parts = name.split(".")
        if not parts:
            continue
        stem = parts[0]
        if not stem.startswith("acc_"):
            continue
        numeric = stem[len("acc_") :]
        try:
            account_id = int(numeric)
        except ValueError:
            continue

        payload = _load_result_lines(result_path)
        if payload:
            account_results[account_id] = payload

    return account_results


def apply_validation_ai_decisions_for_all_accounts(
    sid: str,
    runs_root: Path | str | None = None,
) -> Mapping[str, Any]:
    """Apply AI validation decisions for every account in ``sid``.

    Returns a mapping with counters that summarize the merge activity.
    """

    runs_root_path = _resolve_runs_root(runs_root)
    accounts_dir = runs_root_path / sid / "cases" / "accounts"
    if not accounts_dir.is_dir():
        return {
            "accounts_discovered": 0,
            "accounts_updated": 0,
            "fields_total": 0,
            "fields_updated": 0,
            "results_files": 0,
        }

    account_results = _discover_result_maps(sid, runs_root_path)

    stats = {
        "accounts_discovered": len(account_results),
        "accounts_updated": 0,
        "fields_total": sum(len(results) for results in account_results.values()),
        "fields_updated": 0,
        "results_files": len(account_results),
        "accounts": sorted(str(account_id) for account_id in account_results),
    }

    if not account_results:
        return stats

    for account_id, results_map in account_results.items():
        applied = apply_validation_ai_decisions_for_account(
            sid,
            runs_root_path,
            account_id,
            results_override=results_map,
        )
        if applied:
            stats["accounts_updated"] += 1
            stats["fields_updated"] += len(results_map)

    return stats


def _finding_has_ai_decision(finding: Mapping[str, Any]) -> bool:
    if not isinstance(finding, Mapping):
        return False

    validation_ai_block = finding.get("validation_ai")
    if isinstance(validation_ai_block, Mapping):
        decision_value = validation_ai_block.get("decision")
        if isinstance(decision_value, str) and decision_value.strip():
            return True

    ai_decision = finding.get("ai_decision")
    if isinstance(ai_decision, str) and ai_decision.strip():
        return True

    decision_value = finding.get("decision")
    decision_source = finding.get("decision_source")
    if (
        isinstance(decision_value, str)
        and decision_value.strip()
        and isinstance(decision_source, str)
        and decision_source.strip().lower() == "ai"
    ):
        return True

    default_decision = finding.get("default_decision")
    if isinstance(default_decision, str) and default_decision.strip():
        source = finding.get("decision_source")
        if isinstance(source, str) and source.strip().lower() == "ai":
            return True

    return False


def summarize_validation_ai_state(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> Mapping[str, Any]:
    """Return aggregate status about AI decisions applied to account summaries."""

    runs_root_path = _resolve_runs_root(runs_root)
    accounts_dir = runs_root_path / sid / "cases" / "accounts"

    accounts_total = 0
    accounts_pending: list[str] = []
    accounts_complete = 0
    fields_total = 0
    fields_applied = 0

    if not accounts_dir.is_dir():
        return {
            "accounts_total": 0,
            "accounts_with_send": 0,
            "accounts_complete": 0,
            "accounts_pending": [],
            "fields_total": 0,
            "fields_applied": 0,
            "fields_pending": 0,
            "completed": False,
        }

    for account_path in sorted(accounts_dir.iterdir(), key=lambda path: path.name):
        if not account_path.is_dir():
            continue

        summary_payload = _load_json(account_path / "summary.json")
        if not isinstance(summary_payload, Mapping):
            continue

        validation_block = summary_payload.get("validation_requirements")
        if not isinstance(validation_block, Mapping):
            continue

        findings = validation_block.get("findings")
        if not isinstance(findings, list):
            continue

        account_fields_total = 0
        account_fields_applied = 0

        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            if not finding.get("send_to_ai"):
                continue

            account_fields_total += 1
            fields_total += 1

            if _finding_has_ai_decision(finding):
                account_fields_applied += 1
                fields_applied += 1

        if account_fields_total == 0:
            continue

        accounts_total += 1

        if account_fields_applied >= account_fields_total:
            accounts_complete += 1
        else:
            accounts_pending.append(account_path.name)

    fields_pending = max(fields_total - fields_applied, 0)
    accounts_with_send = accounts_total
    completed = fields_total > 0 and fields_pending == 0

    return {
        "accounts_total": accounts_total,
        "accounts_with_send": accounts_with_send,
        "accounts_complete": accounts_complete,
        "accounts_pending": sorted(accounts_pending),
        "fields_total": fields_total,
        "fields_applied": fields_applied,
        "fields_pending": fields_pending,
        "completed": completed,
    }
