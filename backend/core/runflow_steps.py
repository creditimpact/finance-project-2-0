from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import json
import os
import uuid

from backend.core.io.json_io import _atomic_write_json as _shared_atomic_write_json
from backend.runflow.counters import (
    runflow_validation_findings_total as _validation_findings_total_from_runflow,
    stage_counts as _stage_counts_from_disk,
    validation_findings_count as _validation_findings_count,
)


def _utcnow_iso() -> str:
    """Return the current UTC timestamp encoded as an ISO string."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _iso_from_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class _RunsRootProxy:
    def _path(self) -> Path:
        return Path(os.getenv("RUNS_ROOT", "runs"))

    def __truediv__(self, other: object) -> Path:
        return self._path() / other

    def __rtruediv__(self, other: object) -> Path:
        return Path(other) / self._path()

    def __fspath__(self) -> str:
        return os.fspath(self._path())

    def __str__(self) -> str:
        return str(self._path())

    def __repr__(self) -> str:
        return f"RunsRootProxy({self._path()!r})"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._path(), name)


RUNS_ROOT = _RunsRootProxy()


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


_SCHEMA_VERSION = os.getenv("RUNFLOW_STEPS_SCHEMA_VERSION") or "2.2"
_VERIFY_STEPS = _env_enabled("RUNFLOW_STEPS_VERIFY", default=True)
_ENABLE_SPANS = _env_enabled("RUNFLOW_STEPS_ENABLE_SPANS", default=True)


def _steps_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_steps.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _shared_atomic_write_json(path, payload)


def _normalise_steps(entries: Any, default_t: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    max_seq = 0
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            data = {str(k): v for k, v in dict(entry).items()}
            name = data.get("name")
            status = data.get("status")
            if not isinstance(name, str):
                continue
            if not isinstance(status, str):
                status = "unknown"
            seq_raw = data.get("seq")
            try:
                seq_val = int(seq_raw)
            except (TypeError, ValueError):
                seq_val = 0
            if seq_val <= 0:
                seq_val = max_seq + 1
            max_seq = max(max_seq, seq_val)
            t_value = data.get("t")
            if not isinstance(t_value, str):
                t_value = default_t
            record: dict[str, Any] = {
                "seq": seq_val,
                "name": name,
                "status": status,
                "t": t_value,
            }
            for field in (
                "account",
                "metrics",
                "out",
                "reason",
                "span_id",
                "parent_span_id",
                "error",
            ):
                if field in data:
                    record[field] = data[field]
            result.append(record)

    result.sort(key=lambda item: item["seq"])
    return result


def _normalise_aggregate_entry(entry: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    stage = entry.get("stage")
    if not isinstance(stage, str) or not stage:
        return None

    summary_raw = entry.get("summary")
    summary: dict[str, int] = {}
    if isinstance(summary_raw, Mapping):
        for key, value in summary_raw.items():
            try:
                summary[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    return {"stage": stage, "summary": summary}


def _normalise_aggregates(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []

    normalised: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        parsed = _normalise_aggregate_entry(entry)
        if parsed is not None:
            normalised.append(parsed)
    return normalised


def _legacy_substage_steps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []
    latest: dict[str, dict[str, Any]] = {}
    for item in entries:
        name = item.get("name")
        if not isinstance(name, str):
            continue
        prev = latest.get(name)
        if prev is None or int(item.get("seq", 0)) >= int(prev.get("seq", 0)):
            legacy_entry = {
                key: value
                for key, value in item.items()
                if key in {"name", "status", "t", "account", "metrics", "out", "reason", "span_id", "parent_span_id", "error", "seq"}
            }
            latest[name] = legacy_entry
    ordered = sorted(latest.values(), key=lambda entry: int(entry.get("seq", 0)))
    return ordered


def _normalise_stage(stage: str, payload: Mapping[str, Any], now: str) -> dict[str, Any]:
    data = {str(k): v for k, v in dict(payload).items()}
    status = data.get("status")
    if not isinstance(status, str):
        status = "running"

    started_at = data.get("started_at")
    if not isinstance(started_at, str):
        started_at = now

    ended_at = data.get("ended_at")
    if isinstance(ended_at, str):
        ended_val: Optional[str] = ended_at
    else:
        ended_val = None

    summary = data.get("summary")
    if isinstance(summary, Mapping):
        summary_payload = {str(k): v for k, v in dict(summary).items()}
    else:
        summary_payload = None

    empty_ok = bool(data.get("empty_ok")) if data.get("empty_ok") is not None else False

    steps_raw = data.get("steps")
    steps_list = _normalise_steps(steps_raw, started_at)

    if not steps_list:
        substages = data.get("substages")
        if isinstance(substages, Mapping):
            extracted: list[dict[str, Any]] = []
            seq = 0
            for substage_payload in substages.values():
                if not isinstance(substage_payload, Mapping):
                    continue
                steps = substage_payload.get("steps")
                if not isinstance(steps, list):
                    continue
                for entry in steps:
                    if not isinstance(entry, Mapping):
                        continue
                    name = entry.get("name")
                    if not isinstance(name, str):
                        continue
                    seq += 1
                    merged = {
                        "seq": seq,
                        "name": name,
                        "status": str(entry.get("status") or "unknown"),
                        "t": str(entry.get("t") or started_at),
                    }
                    for field in (
                        "account",
                        "metrics",
                        "out",
                        "reason",
                        "span_id",
                        "parent_span_id",
                        "error",
                    ):
                        if field in entry:
                            merged[field] = entry[field]
                    extracted.append(merged)
            steps_list = extracted

    steps_list.sort(key=lambda entry: entry["seq"])
    next_seq = steps_list[-1]["seq"] + 1 if steps_list else 1

    substages_raw = data.get("substages")
    substages: dict[str, Any]
    if isinstance(substages_raw, Mapping):
        substages = {str(k): dict(v) for k, v in substages_raw.items() if isinstance(v, Mapping)}
    else:
        substages = {}

    default_substage = substages.get("default")
    if isinstance(default_substage, Mapping):
        default_substage = dict(default_substage)
    else:
        default_substage = {}

    default_substage.setdefault("started_at", started_at)
    legacy_steps = _legacy_substage_steps(steps_list)
    if legacy_steps:
        default_substage["steps"] = legacy_steps
        default_substage["status"] = legacy_steps[-1].get("status", "running")
    else:
        default_substage.setdefault("steps", [])
        default_substage.setdefault("status", "running")

    substages["default"] = default_substage

    stage_payload: dict[str, Any] = {
        k: v
        for k, v in data.items()
        if k
        not in {
            "steps",
            "next_seq",
            "substages",
            "summary",
            "status",
            "started_at",
            "ended_at",
            "empty_ok",
        }
    }

    stage_payload["status"] = status
    stage_payload["started_at"] = started_at
    if ended_val is not None:
        stage_payload["ended_at"] = ended_val
    if summary_payload is not None:
        stage_payload["summary"] = summary_payload
    if empty_ok:
        stage_payload["empty_ok"] = True

    stage_payload["steps"] = steps_list
    stage_payload["next_seq"] = next_seq
    stage_payload["substages"] = substages

    return stage_payload


def _derive_summary_error(stage_payload: Mapping[str, Any]) -> str:
    steps = stage_payload.get("steps")
    if isinstance(steps, list):
        for entry in reversed(steps):
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("status")) != "error":
                continue
            error_payload = entry.get("error")
            if isinstance(error_payload, Mapping):
                message = error_payload.get("message")
                error_type = error_payload.get("type")
                if isinstance(message, str) and message:
                    if isinstance(error_type, str) and error_type:
                        return f"{error_type}: {message}"
                    return message
            reason = entry.get("reason")
            if isinstance(reason, str) and reason:
                return reason
            name = entry.get("name")
            if isinstance(name, str) and name:
                return f"{name} failed"
            break
    return "stage failed"


def _load_steps_payload(sid: str, schema_version: str) -> dict[str, Any]:
    path = _steps_path(sid)
    now = _utcnow_iso()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        payload = {}

    if not isinstance(payload, Mapping):
        payload = {}

    stages_raw = payload.get("stages")
    stages: dict[str, Any]
    if isinstance(stages_raw, Mapping):
        stages = {
            str(stage): _normalise_stage(stage, value, now)
            for stage, value in stages_raw.items()
            if isinstance(value, Mapping)
        }
    else:
        stages = {}

    aggregates_raw = payload.get("aggregates")
    aggregates = _normalise_aggregates(aggregates_raw)

    result: dict[str, Any] = {
        "sid": str(payload.get("sid") or sid),
        "schema_version": schema_version,
        "stages": stages,
        "updated_at": str(payload.get("updated_at") or now),
    }

    if aggregates:
        result["aggregates"] = aggregates

    _update_updated_at(result)
    return result


def _update_updated_at(payload: MutableMapping[str, Any], *candidates: str) -> None:
    timestamps = []
    for value in candidates:
        parsed = _parse_iso(value)
        if parsed is not None:
            timestamps.append(parsed)

    existing = _parse_iso(payload.get("updated_at"))
    if existing is not None:
        timestamps.append(existing)

    for stage_payload in payload.get("stages", {}).values():
        if not isinstance(stage_payload, Mapping):
            continue
        for field in ("started_at", "ended_at"):
            parsed = _parse_iso(stage_payload.get(field))
            if parsed is not None:
                timestamps.append(parsed)
        steps = stage_payload.get("steps")
        if isinstance(steps, list):
            for entry in steps:
                if not isinstance(entry, Mapping):
                    continue
                parsed = _parse_iso(entry.get("t"))
                if parsed is not None:
                    timestamps.append(parsed)

    timestamps.append(datetime.now(timezone.utc))
    payload["updated_at"] = _iso_from_datetime(max(timestamps))


def _dump_steps_payload(sid: str, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(_steps_path(sid), payload)


def steps_update_aggregate(sid: str, stage: str, summary: Mapping[str, Any]) -> None:
    now = _utcnow_iso()
    payload = _load_steps_payload(sid, _SCHEMA_VERSION)
    aggregates_existing = payload.get("aggregates")
    aggregates = _normalise_aggregates(aggregates_existing)

    stage_name = str(stage)
    summary_payload: dict[str, int] = {}
    for key, value in summary.items():
        try:
            summary_payload[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    updated = False
    new_entries: list[dict[str, Any]] = []
    for entry in aggregates:
        if entry.get("stage") == stage_name:
            new_entries.append({"stage": stage_name, "summary": dict(summary_payload)})
            updated = True
        else:
            new_entries.append(entry)

    if not updated:
        new_entries.append({"stage": stage_name, "summary": dict(summary_payload)})

    if new_entries:
        payload["aggregates"] = new_entries
    else:
        payload.pop("aggregates", None)

    _update_updated_at(payload, now)
    _dump_steps_payload(sid, payload)


def _clear_steps_verify_hint(stage_payload: MutableMapping[str, Any]) -> None:
    existing_error = stage_payload.get("error")

    if isinstance(existing_error, Mapping):
        error_payload = dict(existing_error)
        hint_value = error_payload.get("hint")
        if isinstance(hint_value, str) and hint_value.startswith("steps_verify:"):
            error_payload.pop("hint", None)
        if error_payload:
            stage_payload["error"] = error_payload
        else:
            stage_payload.pop("error", None)
    elif isinstance(existing_error, str) and existing_error.startswith("steps_verify:"):
        stage_payload.pop("error", None)


def _verify_validation_summary(sid: str, stage_payload: MutableMapping[str, Any]) -> None:
    run_dir = RUNS_ROOT / sid
    expected = _validation_findings_total_from_runflow(run_dir)
    # Compare the number of findings written to disk, not just the number of
    # packs, so multi-field packs do not trigger false mismatches.
    actual = _validation_findings_count(run_dir)

    if expected is None or actual is None:
        return

    try:
        expected_int = int(expected)
    except (TypeError, ValueError):
        return

    try:
        actual_int = int(actual)
    except (TypeError, ValueError):
        return

    expected_int = max(expected_int, 0)
    actual_int = max(actual_int, 0)

    if actual_int == expected_int:
        _clear_steps_verify_hint(stage_payload)
        return

    existing_error = stage_payload.get("error")
    if isinstance(existing_error, Mapping):
        error_payload = dict(existing_error)
    else:
        error_payload = {}

    error_payload["hint"] = (
        "steps_verify: findings_count expected="
        f"{expected_int} actual={actual_int}"
    )
    stage_payload["error"] = error_payload


def _verify_stage_summary(
    sid: str, stage: str, stage_payload: MutableMapping[str, Any]
) -> None:
    if not _VERIFY_STEPS:
        return

    if stage == "validation":
        _verify_validation_summary(sid, stage_payload)
        return

    summary = stage_payload.get("summary")
    if not isinstance(summary, Mapping):
        return

    disk_counts = _stage_counts_from_disk(stage, RUNS_ROOT / sid)
    if not disk_counts:
        return

    mismatches: list[str] = []
    for key, expected in disk_counts.items():
        actual = summary.get(key)
        try:
            actual_int = int(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_int = None
        if actual_int != expected:
            mismatches.append(f"{key} expected={expected} actual={actual!r}")

    if not mismatches:
        return

    existing_error = stage_payload.get("error")
    if isinstance(existing_error, Mapping):
        error_payload = dict(existing_error)
    else:
        error_payload = {}

    error_payload["hint"] = "steps_verify: " + "; ".join(mismatches)
    stage_payload["error"] = error_payload


def steps_init(sid: str, schema_version: str | None = None) -> None:
    schema = schema_version or _SCHEMA_VERSION
    payload = _load_steps_payload(sid, schema)
    _update_updated_at(payload)
    _dump_steps_payload(sid, payload)


def _ensure_stage(
    payload: MutableMapping[str, Any], stage: str, started_at: Optional[str]
) -> tuple[dict[str, Any], bool]:
    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        payload["stages"] = stages

    stage_payload = stages.get(stage)
    created = False
    if not isinstance(stage_payload, Mapping):
        created = True
        ts = started_at or _utcnow_iso()
        stage_payload = {
            "status": "running",
            "started_at": ts,
            "steps": [],
            "next_seq": 1,
            "substages": {
                "default": {"status": "running", "started_at": ts, "steps": []}
            },
        }
    else:
        stage_payload = _normalise_stage(stage, stage_payload, started_at or _utcnow_iso())

    stages[stage] = stage_payload
    return stage_payload, created


def steps_stage_start(
    sid: str,
    stage: str,
    started_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> bool:
    payload = _load_steps_payload(sid, _SCHEMA_VERSION)
    stage_payload, created = _ensure_stage(payload, stage, started_at)
    ts = started_at or stage_payload.get("started_at") or _utcnow_iso()

    stage_payload["status"] = "running"
    stage_payload.setdefault("started_at", ts)
    stage_payload.setdefault("substages", {}).setdefault(
        "default", {"status": "running", "started_at": ts, "steps": []}
    )

    if extra:
        for key, value in extra.items():
            stage_payload[str(key)] = value

    payload.setdefault("stages", {})[stage] = stage_payload
    _update_updated_at(payload, ts)
    _dump_steps_payload(sid, payload)
    return created


def steps_stage_finish(
    sid: str,
    stage: str,
    status: str,
    summary: Optional[Mapping[str, Any]],
    ended_at: Optional[str] = None,
    *,
    empty_ok: bool = False,
) -> None:
    payload = _load_steps_payload(sid, _SCHEMA_VERSION)
    stage_payload, _ = _ensure_stage(payload, stage, ended_at)
    ts = ended_at or _utcnow_iso()

    stage_payload["status"] = status
    stage_payload["ended_at"] = ts
    if summary:
        existing_summary = stage_payload.get("summary")
        if isinstance(existing_summary, Mapping):
            merged = dict(existing_summary)
            merged.update({str(k): v for k, v in summary.items()})
            stage_payload["summary"] = merged
        else:
            stage_payload["summary"] = {str(k): v for k, v in summary.items()}
    elif status == "error":
        stage_payload["summary"] = {}
    if empty_ok:
        stage_payload["empty_ok"] = True

    if status == "error":
        summary_payload = stage_payload.get("summary")
        if isinstance(summary_payload, Mapping):
            summary_dict = dict(summary_payload)
        else:
            summary_dict = {}
        summary_dict.setdefault("error", _derive_summary_error(stage_payload))
        stage_payload["summary"] = summary_dict

    _verify_stage_summary(sid, stage, stage_payload)

    stages = payload.setdefault("stages", {})
    stages[stage] = stage_payload
    _update_updated_at(payload, ts)
    _dump_steps_payload(sid, payload)


def _prepare_step_entry(
    stage_payload: MutableMapping[str, Any],
    name: str,
    status: str,
    t_value: str,
    seq: Optional[int],
    account: Optional[str],
    metrics: Optional[Mapping[str, Any]],
    out: Optional[Mapping[str, Any]],
    reason: Optional[str],
    span_id: Optional[str],
    parent_span_id: Optional[str],
    error: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    steps = stage_payload.setdefault("steps", [])
    if not isinstance(steps, list):
        steps = []
        stage_payload["steps"] = steps

    next_seq = stage_payload.get("next_seq")
    try:
        next_seq_int = int(next_seq)
    except (TypeError, ValueError):
        next_seq_int = 1

    if seq is None or seq < next_seq_int:
        seq_value = next_seq_int
    else:
        seq_value = seq

    for existing in steps:
        if isinstance(existing, Mapping):
            try:
                existing_seq = int(existing.get("seq"))
            except (TypeError, ValueError):
                continue
            if existing_seq >= seq_value:
                seq_value = existing_seq + 1

    stage_payload["next_seq"] = seq_value + 1

    entry: dict[str, Any] = {
        "seq": seq_value,
        "name": name,
        "status": status,
        "t": t_value,
    }

    if account is not None:
        entry["account"] = account
    if metrics:
        entry["metrics"] = {str(k): v for k, v in metrics.items()}
    if out:
        entry["out"] = {str(k): v for k, v in out.items()}
    if reason is not None:
        entry["reason"] = reason
    if _ENABLE_SPANS and span_id is not None:
        entry["span_id"] = span_id
    if _ENABLE_SPANS and parent_span_id is not None:
        entry["parent_span_id"] = parent_span_id
    if error is not None:
        entry["error"] = dict(error)

    steps.append(entry)
    steps.sort(key=lambda item: item["seq"])
    return entry


def steps_append(
    sid: str,
    stage: str,
    name: str,
    status: str,
    *,
    t: Optional[str] = None,
    seq: Optional[int] = None,
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = _load_steps_payload(sid, _SCHEMA_VERSION)
    stage_payload, _ = _ensure_stage(payload, stage, t)
    ts = t or _utcnow_iso()

    entry = _prepare_step_entry(
        stage_payload,
        name,
        status,
        ts,
        seq,
        account,
        metrics,
        out,
        reason,
        span_id,
        parent_span_id,
        error,
    )

    substages = stage_payload.setdefault("substages", {})
    if not isinstance(substages, dict):
        substages = {}
        stage_payload["substages"] = substages

    default_substage = substages.get("default")
    if isinstance(default_substage, Mapping):
        default_substage = dict(default_substage)
    else:
        default_substage = {}

    default_substage.setdefault("started_at", stage_payload.get("started_at") or ts)
    default_substage["status"] = status
    steps_list = default_substage.setdefault("steps", [])
    if not isinstance(steps_list, list):
        steps_list = []
        default_substage["steps"] = steps_list

    legacy_entry = {
        key: value
        for key, value in entry.items()
        if key
        in {
            "seq",
            "name",
            "status",
            "t",
            "account",
            "metrics",
            "out",
            "reason",
            "span_id",
            "parent_span_id",
            "error",
        }
    }

    replaced = False
    for existing in steps_list:
        if isinstance(existing, Mapping) and existing.get("name") == name:
            existing.clear()
            existing.update(legacy_entry)
            replaced = True
            break
    if not replaced:
        steps_list.append(legacy_entry)
    steps_list.sort(key=lambda item: int(item.get("seq", 0)))

    substages["default"] = default_substage
    stage_payload["substages"] = substages

    payload.setdefault("stages", {})[stage] = stage_payload
    _update_updated_at(payload, ts)
    _dump_steps_payload(sid, payload)


__all__ = [
    "RUNS_ROOT",
    "steps_init",
    "steps_stage_start",
    "steps_stage_finish",
    "steps_append",
    "steps_update_aggregate",
]

