"""Filesystem helpers for reading planner inputs and resolving outputs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from .exceptions import StrategyPlannerError, UnsupportedDurationUnitError
from .planner import PlannerOutputs
from .types import Finding

log = logging.getLogger(__name__)

_DEFAULT_ACCOUNT_SUBDIR = "strategy"
_DEFAULT_MASTER_FILENAME = "plan.json"
_DEFAULT_WEEKDAY_PREFIX = "plan_wd"
_LOG_FILENAME = "logs.txt"

_TRUE_LITERALS = {"1", "true", "yes", "on"}
_FALSE_LITERALS = {"0", "false", "no", "off"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class WrittenPaths(TypedDict):
    dir: str
    master: str
    weekdays: Dict[str, str]
    log: str


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def resolve_strategy_dir_for_account(account_dir: Path, account_subdir: Optional[str] = None) -> Path:
    """Ensure and return the strategy directory for ``account_dir``."""

    subdir = (account_subdir or _get_env("PLANNER_ACCOUNT_SUBDIR", _DEFAULT_ACCOUNT_SUBDIR))
    subdir = subdir or _DEFAULT_ACCOUNT_SUBDIR
    target = account_dir / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


def strategy_dir_from_summary(summary_path: Path) -> Path:
    """Return the account-scoped strategy directory for ``summary_path``."""

    account_dir = summary_path.parent
    return resolve_strategy_dir_for_account(account_dir)


def master_plan_path(summary_path: Path) -> Path:
    filename = _get_env("PLANNER_MASTER_FILENAME", _DEFAULT_MASTER_FILENAME) or _DEFAULT_MASTER_FILENAME
    return strategy_dir_from_summary(summary_path) / filename


def weekday_plan_path(summary_path: Path, weekday: int) -> Path:
    prefix = _get_env("PLANNER_WEEKDAY_FILENAME_PREFIX", _DEFAULT_WEEKDAY_PREFIX) or _DEFAULT_WEEKDAY_PREFIX
    filename = f"{prefix}{weekday}.json"
    return strategy_dir_from_summary(summary_path) / filename


def _normalize_documents(raw: Iterable) -> List[str] | None:
    docs: List[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            docs.append(text)
    return docs or None


def load_findings_from_summary(summary_path: Path) -> List[Finding]:
    """Read validation findings from ``summary.json`` and normalize."""

    try:
        text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.debug("PLANNER_SUMMARY_MISSING path=%s", summary_path)
        return []
    except OSError as exc:  # pragma: no cover - defensive IO guard
        log.warning("PLANNER_SUMMARY_READ_FAILED path=%s", summary_path, exc_info=exc)
        raise StrategyPlannerError(f"Unable to read summary: {summary_path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("PLANNER_SUMMARY_INVALID_JSON path=%s", summary_path, exc_info=exc)
        raise StrategyPlannerError(f"Invalid summary JSON at {summary_path}") from exc

    block = data.get("validation_requirements") if isinstance(data, dict) else None
    if not isinstance(block, dict):
        log.debug("PLANNER_SUMMARY_NO_REQUIREMENTS path=%s", summary_path)
        return []

    items = block.get("findings")
    if not isinstance(items, list):
        return []

    findings: List[Finding] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue

        field = entry.get("field")
        if not field:
            continue

        duration_unit = str(entry.get("duration_unit") or "").strip() or "business_days"
        if duration_unit != "business_days":
            raise UnsupportedDurationUnitError(
                f"Unsupported duration_unit '{duration_unit}' for field '{field}'"
            )

        if "min_days" not in entry:
            raise StrategyPlannerError(f"Finding '{field}' is missing min_days")
        try:
            min_days = int(entry.get("min_days"))
        except (TypeError, ValueError) as exc:
            raise StrategyPlannerError(
                f"Finding '{field}' has non-numeric min_days: {entry.get('min_days')}"
            ) from exc

        category = str(entry.get("category") or "terms")
        documents_raw = entry.get("documents")
        documents: List[str] | None = None
        if isinstance(documents_raw, list):
            documents = _normalize_documents(documents_raw)
        elif documents_raw is not None:
            documents = _normalize_documents([documents_raw])

        bureaus_raw = entry.get("bureaus")
        bureaus: List[str] | None = None
        if isinstance(bureaus_raw, list):
            bureaus = []
            for bureau in bureaus_raw:
                text = str(bureau).strip()
                if text:
                    bureaus.append(text.lower())
        if not bureaus:
            bureaus = None

        present_count_raw = entry.get("present_count")
        present_count: Optional[int]
        try:
            present_count = int(present_count_raw) if present_count_raw is not None else None
        except (TypeError, ValueError):
            present_count = None

        missing_count_raw = entry.get("missing_count")
        try:
            missing_count = int(missing_count_raw) if missing_count_raw is not None else None
        except (TypeError, ValueError):
            missing_count = None

        is_missing = _coerce_bool(entry.get("is_missing"))
        is_mismatch = _coerce_bool(entry.get("is_mismatch"))

        bureau_dispute_state_raw = entry.get("bureau_dispute_state")
        bureau_dispute_state: Optional[Dict[str, str]] = None
        if isinstance(bureau_dispute_state_raw, dict):
            bureau_dispute_state = {}
            for bureau_name, state_value in bureau_dispute_state_raw.items():
                key = str(bureau_name).strip().lower()
                if not key:
                    continue
                bureau_dispute_state[key] = str(state_value).strip().lower() if state_value is not None else ""
            if not bureau_dispute_state:
                bureau_dispute_state = None

        validation_ai_block = entry.get("validation_ai") if isinstance(entry.get("validation_ai"), dict) else None

        ai_decision: Optional[str] = None
        ai_rationale: Optional[str] = None
        ai_citations: Optional[List[str]] = None
        ai_legacy_decision: Optional[str] = None

        if isinstance(validation_ai_block, dict):
            raw_decision = validation_ai_block.get("decision")
            if isinstance(raw_decision, str) and raw_decision.strip():
                ai_decision = raw_decision.strip()

            raw_rationale = validation_ai_block.get("rationale")
            if isinstance(raw_rationale, str) and raw_rationale.strip():
                ai_rationale = raw_rationale.strip()

            raw_citations = validation_ai_block.get("citations")
            if isinstance(raw_citations, list):
                citations_list = [
                    str(item).strip()
                    for item in raw_citations
                    if isinstance(item, (str, int, float)) and str(item).strip()
                ]
                if citations_list:
                    ai_citations = citations_list

            raw_legacy = validation_ai_block.get("legacy_decision")
            if isinstance(raw_legacy, str) and raw_legacy.strip():
                ai_legacy_decision = raw_legacy.strip()

        if ai_decision is None:
            raw_ai_decision = entry.get("ai_decision")
            if isinstance(raw_ai_decision, str) and raw_ai_decision.strip():
                ai_decision = raw_ai_decision.strip()

        # NEW: Support validation outputs that use ai_validation_decision (and siblings)
        # This block intentionally occurs AFTER the existing recognized keys so we only
        # fall back when the prior sources were absent.
        if ai_decision is None:
            raw_ai_validation_decision = entry.get("ai_validation_decision")
            if isinstance(raw_ai_validation_decision, str) and raw_ai_validation_decision.strip():
                ai_decision = raw_ai_validation_decision.strip()

        # Mirror rationale/citations if the legacy loader fields are still empty
        if ai_rationale is None:
            raw_ai_validation_rationale = entry.get("ai_validation_rationale")
            if isinstance(raw_ai_validation_rationale, str) and raw_ai_validation_rationale.strip():
                ai_rationale = raw_ai_validation_rationale.strip()
        if ai_citations is None:
            raw_ai_validation_citations = entry.get("ai_validation_citations")
            if isinstance(raw_ai_validation_citations, list):
                filtered = [
                    str(item).strip()
                    for item in raw_ai_validation_citations
                    if isinstance(item, (str, int, float)) and str(item).strip()
                ]
                if filtered:
                    ai_citations = filtered

        if ai_rationale is None:
            raw_ai_rationale = entry.get("ai_rationale")
            if isinstance(raw_ai_rationale, str) and raw_ai_rationale.strip():
                ai_rationale = raw_ai_rationale.strip()

        if ai_citations is None:
            raw_ai_citations = entry.get("ai_citations")
            if isinstance(raw_ai_citations, list):
                citations_list = [
                    str(item).strip()
                    for item in raw_ai_citations
                    if isinstance(item, (str, int, float)) and str(item).strip()
                ]
                if citations_list:
                    ai_citations = citations_list

        if ai_legacy_decision is None:
            raw_legacy = entry.get("ai_legacy_decision")
            if isinstance(raw_legacy, str) and raw_legacy.strip():
                ai_legacy_decision = raw_legacy.strip()

        default_decision_value = entry.get("default_decision")
        if isinstance(default_decision_value, str):
            default_decision_value = default_decision_value.strip() or None
        else:
            default_decision_value = None

        if not default_decision_value:
            if isinstance(ai_decision, str) and ai_decision.strip():
                default_decision_value = ai_decision.strip()
            else:
                raw_decision_value = entry.get("decision")
                if isinstance(raw_decision_value, str) and raw_decision_value.strip():
                    default_decision_value = raw_decision_value.strip()

        finding_obj = Finding(
            field=str(field),
            category=category,
            min_days=max(min_days, 0),
            duration_unit="business_days",
            default_decision=default_decision_value,
            reason_code=(entry.get("reason_code") or None),
            documents=documents,
            bureaus=bureaus,
            present_count=present_count,
            is_missing=is_missing,
            is_mismatch=is_mismatch,
            bureau_dispute_state=bureau_dispute_state,
            missing_count=missing_count,
            ai_decision=ai_decision,
            ai_rationale=ai_rationale,
            ai_citations=ai_citations,
            ai_legacy_decision=ai_legacy_decision,
        )

        if isinstance(bureau_dispute_state_raw, dict):
            setattr(finding_obj, "bureau_dispute_state_raw", dict(bureau_dispute_state_raw))

        findings.append(finding_obj)

    return findings


def _write_text_atomically(target: Path, payload: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix:
        tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    else:
        tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(target)


def _dumps_pretty_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _project_plan_for_output(
    plan: Dict[str, Any], 
    omit_summary_and_constraints: bool = False
) -> Dict[str, Any]:
    """Project plan for output serialization, optionally omitting noise keys.
    
    Args:
        plan: The plan dict to project
        omit_summary_and_constraints: If True, remove 'summary' and 'constraints' keys
        
    Returns:
        Projected plan (shallow copy with keys removed if needed)
    """
    if not omit_summary_and_constraints:
        return plan
    
    # Shallow copy to avoid modifying original
    projected = dict(plan)
    projected.pop("summary", None)
    projected.pop("constraints", None)
    return projected


def _serialize_log_events(
    events: Iterable[Dict[str, Any]], *, account_id: Optional[str]
) -> List[str]:
    serialized: List[str] = []
    for raw in events:
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        payload.setdefault("account", account_id)
        payload.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        serialized.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return serialized


def append_strategy_logs(
    summary_path: Path,
    events: Iterable[Dict[str, Any]],
    *,
    account_subdir: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Path:
    """Append ``events`` to the strategy logs JSONL file."""

    strategy_dir = resolve_strategy_dir_for_account(summary_path.parent, account_subdir)
    log_path = strategy_dir / _LOG_FILENAME

    serialized = _serialize_log_events(events, account_id=account_id)
    if not serialized:
        return log_path

    existing_lines: list[str] = []
    try:
        if log_path.exists():
            existing_text = log_path.read_text(encoding="utf-8")
            if existing_text.strip():
                existing_lines.extend(existing_text.strip().splitlines())
    except OSError:  # pragma: no cover - defensive read guard
        existing_lines = []

    payload = "\n".join(existing_lines + serialized) + "\n"
    _write_text_atomically(log_path, payload)
    return log_path


def write_plan_files_atomically(
    *,
    plan: PlannerOutputs,
    summary_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    master_name: str,
    weekday_prefix: str,
    log_lines: Iterable[Dict[str, Any]] = (),
    account_id: Optional[str] = None,
    account_subdir: Optional[str] = None,
    omit_summary_and_constraints: bool = False,
) -> WrittenPaths:
    """Persist planner output bundle to disk atomically.
    
    Args:
        plan: Planner output bundle
        summary_path: Path to account's summary.json
        out_dir: Output directory (if summary_path not provided)
        master_name: Filename for master plan
        weekday_prefix: Prefix for weekday plan filenames
        log_lines: Additional log lines to append
        account_id: Account ID for logging
        account_subdir: Subdirectory for strategy output
        omit_summary_and_constraints: If True, remove 'summary' and 'constraints' from output
    """

    if summary_path is None and out_dir is None:
        raise ValueError("summary_path or out_dir must be provided")

    if summary_path is not None:
        out_dir = resolve_strategy_dir_for_account(summary_path.parent, account_subdir)
    else:
        out_dir = Path(out_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)

    master_payload = dict(plan["master"])
    by_weekday_payload = dict(master_payload.get("by_weekday", {}))

    master_path = out_dir / master_name
    log_path = out_dir / _LOG_FILENAME

    weekdays_written: Dict[str, str] = {}
    for weekday, payload in sorted(plan["weekday_plans"].items(), key=lambda item: item[0]):
        if weekday < 0 or weekday > 4:
            continue
        weekday_path = out_dir / f"{weekday_prefix}{weekday}.json"
        log.info("PLANNER_WRITE_WEEKDAY path=%s weekday=%s", weekday_path, weekday)
        projected_payload = _project_plan_for_output(payload, omit_summary_and_constraints)
        _write_text_atomically(weekday_path, _dumps_pretty_json(projected_payload))
        weekdays_written[str(weekday)] = str(weekday_path)
        by_weekday_payload[str(weekday)] = str(weekday_path)

    master_payload["by_weekday"] = by_weekday_payload

    log.info("PLANNER_WRITE_MASTER path=%s", master_path)
    projected_master = _project_plan_for_output(master_payload, omit_summary_and_constraints)
    _write_text_atomically(master_path, _dumps_pretty_json(projected_master))
    plan["master"] = master_payload

    schedule_events = list(plan.get("schedule_logs", []))
    provided_events = list(log_lines)
    if not provided_events:
        best_overall = master_payload.get("best_overall", {})
        events = schedule_events + [
            {
                "event": "planner_written",
                "mode": master_payload.get("mode_used"),
                "best_start": best_overall.get("start_weekday"),
                "total_span_days": best_overall.get("calendar_span_days"),
            }
        ]
    else:
        events = schedule_events + provided_events
    serialized = _serialize_log_events(events, account_id=account_id)

    existing_lines: list[str] = []
    try:
        if log_path.exists():
            existing_text = log_path.read_text(encoding="utf-8")
            if existing_text.strip():
                existing_lines.extend(existing_text.strip().splitlines())
    except OSError:  # pragma: no cover - defensive read guard
        existing_lines = []

    log_payload = "\n".join(existing_lines + serialized) + "\n"
    log.info("PLANNER_WRITE_LOG path=%s entries=%s", log_path, len(serialized))
    _write_text_atomically(log_path, log_payload)

    return WrittenPaths(
        dir=str(out_dir),
        master=str(master_path),
        weekdays=weekdays_written,
        log=str(log_path),
    )
