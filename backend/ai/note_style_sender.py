"""Execution helpers for the note_style AI stage."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from backend import config
from backend.ai.note_style.io import note_style_snapshot
from backend.ai.note_style_ingest import ingest_note_style_result
from backend.ai.note_style_results import record_note_style_failure
from backend.ai.note_style.parse import parse_note_style_response_payload
from backend.ai.note_style_logging import log_structured_event
from backend.ai.note_style.schema import (
    NOTE_STYLE_TOOL_FUNCTION_NAME,
    NOTE_STYLE_TOOL_PARAMETERS_SCHEMA,
    build_note_style_tool,
)
from backend.ai.note_style.prompt import build_base_system_prompt, build_response_instruction
from backend.config.note_style import NoteStyleResponseMode
from backend.config import note_style as note_cfg
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    normalize_note_style_account_id,
)
from backend.core.paths import normalize_worker_path
from backend.core.services.ai_client import get_ai_client
from backend.runflow.manifest import resolve_note_style_stage_paths
from backend.pipeline import runs as pipeline_runs


log = logging.getLogger(__name__)


_INDEX_THIN_THRESHOLD_BYTES = 128


_PATH_LOG_CACHE: set[str] = set()


_PARSE_MODE_COUNTERS: defaultdict[str, dict[str, int]] = defaultdict(
    lambda: {"success": 0, "failure": 0}
)


def _normalize_parse_mode_label(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return "unknown"


def _response_payload_mode(response_payload: Any) -> str:
    if isinstance(response_payload, Mapping):
        candidate = response_payload.get("mode")
        return _normalize_parse_mode_label(candidate)
    return "unknown"


def _record_parse_outcome(*, response_payload: Any, success: bool) -> None:
    mode_label = _response_payload_mode(response_payload)
    bucket = "success" if success else "failure"
    counters = _PARSE_MODE_COUNTERS[mode_label]
    counters[bucket] += 1
    log.info(
        "NOTE_STYLE_PARSE_TALLY mode=%s success_total=%s failure_total=%s outcome=%s",
        mode_label,
        counters["success"],
        counters["failure"],
        bucket,
    )
    log_structured_event(
        "NOTE_STYLE_PARSE_TALLY",
        logger=log,
        mode=mode_label,
        success_total=counters["success"],
        failure_total=counters["failure"],
        outcome=bucket,
    )


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False
    return True


def _log_sender_paths(sid: str, paths: NoteStylePaths) -> None:
    signature = "|".join(
        [
            sid,
            paths.base.as_posix(),
            paths.packs_dir.as_posix(),
            paths.results_dir.as_posix(),
            paths.index_file.as_posix(),
            paths.log_file.as_posix(),
        ]
    )

    if signature in _PATH_LOG_CACHE:
        return

    _PATH_LOG_CACHE.add(signature)
    log.info(
        "NOTE_STYLE_SEND_PATHS sid=%s base=%s packs=%s results=%s index=%s logs=%s manifest_paths=%s",
        sid,
        paths.base,
        paths.packs_dir,
        paths.results_dir,
        paths.index_file,
        paths.log_file,
        config.NOTE_STYLE_USE_MANIFEST_PATHS,
    )


def _build_normalized_lookup(accounts: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for account in accounts:
        normalized = normalize_note_style_account_id(account)
        if not normalized:
            continue
        mapping.setdefault(normalized, account)
    return mapping


def _normalize_account_id_set(accounts: set[str]) -> set[str]:
    return set(_build_normalized_lookup(accounts).keys())


def _coerce_attr(payload: Any, name: str) -> Any:
    if hasattr(payload, name):
        return getattr(payload, name)
    if isinstance(payload, Mapping):
        return payload.get(name)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except (TypeError, ValueError):
            return None
    return None


def _extract_usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return (None, None)

    prompt = _coerce_attr(usage, "prompt_tokens")
    if prompt is None:
        prompt = _coerce_attr(usage, "input_tokens")

    response = _coerce_attr(usage, "completion_tokens")
    if response is None:
        response = _coerce_attr(usage, "output_tokens")
    if response is None:
        response = _coerce_attr(usage, "response_tokens")

    return (_coerce_int(prompt), _coerce_int(response))


def _describe_response_format(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        type_value = value.get("type")
        if isinstance(type_value, str) and type_value.strip():
            return type_value.strip()
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _describe_tool_choice(tool_choice: Any, tools: Any) -> str | None:
    if isinstance(tool_choice, Mapping):
        function_payload = tool_choice.get("function")
        if isinstance(function_payload, Mapping):
            name = function_payload.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    if isinstance(tools, Sequence):
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function_payload = tool.get("function")
            if not isinstance(function_payload, Mapping):
                continue
            name = function_payload.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _resolve_request_metadata(
    response_kwargs: Mapping[str, Any], *, mode: str | None = None
) -> dict[str, Any]:
    response_format = None
    tool_choice = None
    if "response_format" in response_kwargs:
        response_format = _describe_response_format(response_kwargs.get("response_format"))
    if "tools" in response_kwargs or "tool_choice" in response_kwargs:
        tool_choice = _describe_tool_choice(
            response_kwargs.get("tool_choice"),
            response_kwargs.get("tools"),
        )
    return {
        "response_format": response_format,
        "tool_choice": tool_choice,
        "mode": mode,
    }


def _build_response_metrics(
    response_payload: Any,
    *,
    response_format: str | None,
    tool_choice: str | None,
    mode: str | None,
) -> dict[str, Any]:
    request_id = _coerce_attr(response_payload, "id")
    usage = _coerce_attr(response_payload, "usage")
    prompt_tokens, response_tokens = _extract_usage_tokens(usage)
    response_mode = _response_payload_mode(response_payload)

    normalized_request_id: str | None
    if isinstance(request_id, str):
        normalized_request_id = request_id.strip() or None
    elif request_id is None:
        normalized_request_id = None
    else:
        normalized_request_id = str(request_id)

    return {
        "request_id": normalized_request_id,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "response_format": response_format,
        "tool_choice": tool_choice,
        "mode": mode,
        "response_mode": response_mode,
    }


def _log_pack_request_metrics(
    *,
    sid: str,
    account_id: str,
    model: str,
    metrics: Mapping[str, Any] | None,
    parse_ok: bool,
    retry_count: int,
) -> None:
    payload = {
        "sid": sid,
        "account_id": account_id,
        "request_id": None,
        "model": model,
        "response_format": None,
        "tool_choice": None,
        "prompt_tokens": None,
        "response_tokens": None,
        "parse_ok": bool(parse_ok),
        "retry_count": max(0, retry_count),
    }

    if metrics:
        for key in ("request_id", "response_format", "tool_choice"):
            value = metrics.get(key)
            if value is not None:
                payload[key] = value
        mode_value = metrics.get("mode")
        if mode_value is not None:
            payload["mode"] = mode_value
        response_mode_value = metrics.get("response_mode")
        if response_mode_value is not None:
            payload["response_mode"] = response_mode_value
        for key in ("prompt_tokens", "response_tokens"):
            value = metrics.get(key)
            if value is not None:
                payload[key] = value

    log_structured_event("NOTE_STYLE_MODEL_METRICS", logger=log, **payload)


def _extract_pack_note_text(pack_payload: Mapping[str, Any]) -> str | None:
    candidate = pack_payload.get("note_text")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    context = pack_payload.get("context")
    if isinstance(context, Mapping):
        context_note = context.get("note_text")
        if isinstance(context_note, str) and context_note.strip():
            return context_note.strip()

    return None


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    def _coerce(value: Path | str) -> Path:
        if isinstance(value, Path):
            return value.resolve()

        text = str(value or "").strip()
        if not text:
            return Path("runs").resolve()

        sanitized = text.replace("\\", "/")
        try:
            return normalize_worker_path(Path.cwd(), sanitized)
        except ValueError:
            return Path("runs").resolve()

    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        if env_value:
            return _coerce(env_value)
        return Path("runs").resolve()

    return _coerce(runs_root)


def _resolve_packs_dir(paths: NoteStylePaths) -> Path:
    override = os.getenv("NOTE_STYLE_PACKS_DIR")
    if override:
        run_dir = paths.base.parent.parent
        try:
            candidate = normalize_worker_path(run_dir, override)
        except ValueError:
            candidate = paths.packs_dir
        return candidate
    return paths.packs_dir


def _is_within_directory(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _load_pack_records(pack_path: Path) -> list[Mapping[str, Any]]:
    try:
        raw = pack_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Pack file not found: {pack_path}") from None
    except OSError as exc:
        raise RuntimeError(f"Failed to read pack file: {pack_path}") from exc

    payloads: list[Mapping[str, Any]] = []
    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in pack file: {pack_path}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Pack payload must be an object: {pack_path}")
        payloads.append(parsed)

    if not payloads:
        raise ValueError(f"Pack file is empty: {pack_path}")
    return payloads


def _load_pack_payload(pack_path: Path) -> Mapping[str, Any]:
    return _load_pack_records(pack_path)[0]


_DISALLOWED_MESSAGE_KEY_SUBSTRINGS: tuple[str, ...] = (
    "debug",
    "snapshot",
    "raw",
    "blob",
)


_STRICT_SYSTEM_MESSAGE_CONTENT = build_base_system_prompt()

_CORRECTIVE_SYSTEM_MESSAGE = (
    "Your previous output was not valid JSON. Follow the system instructions "
    "and reply with exactly one JSON object that matches the schema."
)

# Generous headroom to avoid truncation of the strict JSON payload the model
# returns. The schema tops out well below this value, but the additional buffer
# prevents partial responses when packs include richer context.
_NOTE_STYLE_RESPONSE_MAX_TOKENS = 1600

def _normalize_message_content(value: Any) -> str | Sequence[Any]:
    """Return content compatible with the chat API."""

    if isinstance(value, str):
        return value

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, (bytes, bytearray)):
                normalized.append(item.decode("utf-8", "ignore"))
            elif isinstance(item, Mapping):
                normalized.append({str(key): _safe_json_payload(val) for key, val in item.items()})
            else:
                normalized.append(str(item))
        return normalized

    if isinstance(value, Mapping):
        safe_payload = _safe_json_payload(value)
        try:
            return json.dumps(safe_payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps(str(safe_payload), ensure_ascii=False)

    if value is None:
        return ""

    return str(value)


def _sanitize_message_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in entry.items():
        key_text = str(key)
        lower_key = key_text.lower()
        if any(fragment in lower_key for fragment in _DISALLOWED_MESSAGE_KEY_SUBSTRINGS):
            continue
        sanitized[key_text] = value

    if "role" not in sanitized or "content" not in sanitized:
        raise ValueError("Pack message missing required role/content fields")

    sanitized["content"] = _normalize_message_content(sanitized["content"])

    return sanitized


def _coerce_response_format(pack_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a strict JSON response format for the model call.

    Older packs occasionally specified custom ``response_format`` payloads which
    allowed free-form text responses. For the note_style stage we want to enforce
    structured JSON output regardless of the pack configuration, so we coerce any
    provided value back into the "json_object" type. This keeps the contract
    simple and prevents accidental regressions when older packs are replayed.
    """

    candidate = pack_payload.get("response_format")
    if isinstance(candidate, Mapping):
        extras: dict[str, Any] = {}
        for key, value in candidate.items():
            key_text = str(key)
            if key_text == "type":
                continue
            extras[key_text] = value
        return {"type": "json_object", **extras}

    if isinstance(candidate, str) and candidate.strip():
        if candidate.strip().lower() != "json_object":
            log.debug(
                "NOTE_STYLE_FORCE_JSON_RESPONSE format=%s", candidate.strip()
            )

    return {"type": "json_object"}


def _build_tool_payload() -> Mapping[str, Any]:
    return build_note_style_tool()


def _build_tool_choice() -> Mapping[str, Any]:
    return {
        "type": "function",
        "function": {"name": NOTE_STYLE_TOOL_FUNCTION_NAME},
    }


def _coerce_messages(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, Sequence):
        raise ValueError("Pack payload missing messages sequence")
    normalized: list[Mapping[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, Mapping):
            raise ValueError("Pack messages must be mapping objects")
        sanitized = _sanitize_message_entry(entry)
        if str(sanitized.get("role", "")).lower() == "system":
            continue
        normalized.append(sanitized)

    system_msg = {"role": "system", "content": _STRICT_SYSTEM_MESSAGE_CONTENT}
    return [system_msg, *normalized]


def _messages_for_attempt(
    base_messages: Sequence[Mapping[str, Any]],
    attempt_index: int,
) -> list[Mapping[str, Any]]:
    messages = copy.deepcopy(list(base_messages))
    if attempt_index <= 0:
        return messages

    corrective = {"role": "system", "content": _CORRECTIVE_SYSTEM_MESSAGE}
    if messages and str(messages[0].get("role", "")).lower() == "system":
        messages.insert(1, corrective)
    else:
        messages.insert(0, corrective)
    return messages


def _append_instruction_text(base: str, instruction: str) -> str:
    base_text = base.rstrip()
    if instruction in base_text.splitlines():
        return base_text if base_text else instruction
    if not base_text:
        return instruction
    return f"{base_text}\n{instruction}"


def _apply_mode_response_instruction(
    messages: list[Mapping[str, Any]], request_mode: str
) -> None:
    use_tools = request_mode == "tool"
    instruction = build_response_instruction(use_tools=use_tools)
    if not instruction:
        return

    for message in messages:
        role = str(message.get("role", "")).lower()
        if role != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _append_instruction_text(content, instruction)
        elif isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            serialized = [str(part) for part in content]
            if instruction not in serialized:
                serialized.append(instruction)
            message["content"] = serialized
        else:
            message["content"] = instruction
        break


def _normalize_response_mode(
    value: NoteStyleResponseMode | str | None,
) -> NoteStyleResponseMode:
    if isinstance(value, NoteStyleResponseMode):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == NoteStyleResponseMode.TOOL.value:
            return NoteStyleResponseMode.TOOL
        if normalized in {
            NoteStyleResponseMode.CONTENT.value,
            "json",
            "json_object",
        }:
            return NoteStyleResponseMode.CONTENT

    return NoteStyleResponseMode.CONTENT


def _tooling_configured() -> bool:
    try:
        return bool(NOTE_STYLE_TOOL_PARAMETERS_SCHEMA)
    except Exception:  # pragma: no cover - defensive
        return False


def _determine_request_mode(
    *,
    base_mode: NoteStyleResponseMode,
    attempt_index: int,
    enable_tool_call_retry: bool,
    has_response_format: bool,
) -> str:
    _ = enable_tool_call_retry
    _ = has_response_format

    response_mode_value = getattr(base_mode, "value", str(base_mode)).strip().lower()
    allow_flag = bool(note_cfg.NOTE_STYLE_ALLOW_TOOL_CALLS)
    env_allow = bool(note_cfg.NOTE_STYLE_ALLOW_TOOLS)
    allow_tools = allow_flag and env_allow

    if allow_tools and response_mode_value == "tool":
        return "tool"

    if attempt_index <= 0 and response_mode_value == "tool":
        log.info(
            "NOTE_STYLE_TOOL_MODE_DISABLED forcing content mode base_mode=%s allow_tools=%s env_allow_tools=%s",
            response_mode_value or "",
            allow_flag,
            env_allow,
        )

    return "content"


def _response_kwargs_for_attempt(
    response_format: Mapping[str, Any] | None,
    attempt_index: int,
    *,
    response_mode: NoteStyleResponseMode | str | None,
    enable_tool_call_retry: bool,
) -> tuple[dict[str, Any], str]:
    _ = response_format
    _ = attempt_index
    _ = response_mode
    _ = enable_tool_call_retry

    kwargs: dict[str, Any] = {}
    kwargs["response_format"] = {"type": "json_object"}
    return (kwargs, "content")


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return str(PurePosixPath(relative))


def _coerce_result_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)
    try:
        return Path(text)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json_payload(item) for item in value]
    if hasattr(value, "model_dump_json"):
        try:
            return json.loads(value.model_dump_json())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return value.dict()  # type: ignore[call-arg]
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _account_id_from_pack_path(pack_path: Path) -> str:
    stem = pack_path.stem
    if stem.startswith("acc_"):
        return stem[4:]
    return stem


def _load_index_account_map(paths: NoteStylePaths) -> dict[str, str]:
    index_path = getattr(paths, "index_file", None)
    if not isinstance(index_path, Path):
        return {}

    try:
        raw = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        log.warning(
            "STYLE_SEND_INDEX_READ_FAILED path=%s", index_path, exc_info=True
        )
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("STYLE_SEND_INDEX_INVALID_JSON path=%s", index_path)
        return {}

    mapping: dict[str, str] = {}
    if isinstance(payload, Mapping):
        entries = payload.get("packs")
        if isinstance(entries, Sequence):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                account_id = str(entry.get("account_id") or "").strip()
                pack_path_value = entry.get("pack_path")
                if not account_id:
                    continue
                if isinstance(pack_path_value, str):
                    normalized = pack_path_value.strip()
                    if normalized:
                        mapping[normalized] = account_id
    return mapping


@dataclass(frozen=True)
class _PackCandidate:
    pack_path: Path
    result_path: Path | None = None
    account_id: str | None = None
    normalized_account_id: str | None = None


def _normalize_manifest_entry_path(
    value: Any,
    *,
    paths: NoteStylePaths,
    default_dir: Path,
) -> Path | None:
    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)

    sanitized = str(text).strip()
    if not sanitized:
        return None

    sanitized = sanitized.replace("\\", "/")
    run_dir = paths.base.parent.parent

    try:
        candidate = normalize_worker_path(run_dir, sanitized)
    except ValueError:
        return None

    if not _is_within_directory(candidate, run_dir):
        return None

    stage_base = paths.base.resolve()
    stage_name = stage_base.name.lower()
    sanitized_relative = sanitized.lstrip("./")
    if stage_name and sanitized_relative.lower().startswith(f"{stage_name}/"):
        sanitized_for_stage = sanitized_relative.split("/", 1)[1]
    else:
        sanitized_for_stage = sanitized

    if not _is_within_directory(candidate, stage_base):
        try:
            candidate = normalize_worker_path(
                default_dir.parent, sanitized_for_stage
            )
        except ValueError:
            candidate = default_dir / Path(sanitized_for_stage)

        if not _is_within_directory(candidate, run_dir):
            return None

    return candidate.resolve()


def _load_manifest_pack_entries(
    paths: NoteStylePaths, *, sid: str | None = None
) -> list[_PackCandidate]:
    index_path = paths.index_file
    try:
        raw = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        log.warning(
            "STYLE_SEND_MANIFEST_INDEX_READ_FAILED sid=%s path=%s",
            sid,
            index_path,
            exc_info=True,
        )
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(
            "STYLE_SEND_MANIFEST_INDEX_INVALID_JSON sid=%s path=%s",
            sid,
            index_path,
            exc_info=True,
        )
        return []

    pack_entries = payload.get("packs")
    if not isinstance(pack_entries, Sequence):
        return []

    candidates: list[_PackCandidate] = []
    for entry in pack_entries:
        if not isinstance(entry, Mapping):
            continue

        pack_path_value = entry.get("pack_path") or entry.get("pack")
        pack_path = _normalize_manifest_entry_path(
            pack_path_value,
            paths=paths,
            default_dir=paths.packs_dir,
        )
        if pack_path is None:
            log.warning(
                "STYLE_SEND_MANIFEST_PACK_INVALID sid=%s value=%s",
                sid,
                pack_path_value,
            )
            continue

        result_path_value = entry.get("result_path") or entry.get("result")
        result_path = _normalize_manifest_entry_path(
            result_path_value,
            paths=paths,
            default_dir=paths.results_dir,
        )

        account_raw = entry.get("account_id")
        account_id: str | None
        if account_raw is None:
            account_id = None
        else:
            account_text = str(account_raw).strip()
            account_id = account_text or None

        normalized = (
            normalize_note_style_account_id(account_id)
            if account_id is not None
            else None
        )

        candidates.append(
            _PackCandidate(
                pack_path=pack_path,
                result_path=result_path,
                account_id=account_id,
                normalized_account_id=normalized,
            )
        )

    return candidates


def _warn_if_index_thin(paths: NoteStylePaths, *, sid: str) -> None:
    if not config.NOTE_STYLE_WAIT_FOR_INDEX:
        return

    index_path = getattr(paths, "index_file", None)
    if not isinstance(index_path, Path):
        return

    display_path = _relativize(index_path, paths.base)

    try:
        size = index_path.stat().st_size
    except FileNotFoundError:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s path=%s reason=missing",
            sid,
            display_path,
        )
        return
    except OSError:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s path=%s reason=stat_failed",
            sid,
            display_path,
            exc_info=True,
        )
        return

    if size < _INDEX_THIN_THRESHOLD_BYTES:
        log.warning(
            "NOTE_STYLE_INDEX_THIN sid=%s bytes=%s threshold=%s path=%s",
            sid,
            size,
            _INDEX_THIN_THRESHOLD_BYTES,
            display_path,
        )


def _account_paths_for_candidate(
    paths: NoteStylePaths,
    account_id: str,
    candidate: _PackCandidate | None,
) -> NoteStyleAccountPaths:
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    if candidate is None:
        return account_paths

    pack_file = candidate.pack_path
    result_file = candidate.result_path or account_paths.result_file

    if candidate.result_path is not None:
        candidate.result_path.parent.mkdir(parents=True, exist_ok=True)

    return NoteStyleAccountPaths(
        account_id=account_paths.account_id,
        pack_file=pack_file,
        result_file=result_file,
        result_raw_file=account_paths.result_raw_file,
        debug_file=account_paths.debug_file,
    )


def _load_result_payload(result_path: Path) -> Mapping[str, Any] | None:
    try:
        raw = result_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning(
            "STYLE_SEND_EXISTING_READ_FAILED path=%s",
            result_path,
            exc_info=True,
        )
        return None

    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            log.warning(
                "STYLE_SEND_EXISTING_INVALID_JSON path=%s",
                result_path,
                exc_info=True,
            )
            return None
        if not isinstance(payload, Mapping):
            log.warning(
                "STYLE_SEND_EXISTING_INVALID_PAYLOAD path=%s",
                result_path,
            )
            return None
        return payload
    return None


def _payload_has_completed_analysis(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    analysis = payload.get("analysis")
    if isinstance(analysis, Mapping) and bool(analysis):
        return True
    return False


def _result_skip_reason(payload: Mapping[str, Any] | None) -> str | None:
    if _payload_has_completed_analysis(payload):
        return "existing_analysis"
    if isinstance(payload, Mapping):
        status = str(payload.get("status") or "").strip().lower()
        if status == "failed":
            return "existing_failure"
    return None


def _result_has_completed_analysis(result_path: Path) -> bool:
    payload = _load_result_payload(result_path)
    return _payload_has_completed_analysis(payload)


def _result_is_terminal(
    result_path: Path, payload: Mapping[str, Any] | None = None
) -> bool:
    if payload is None:
        payload = _load_result_payload(result_path)
    if _payload_has_completed_analysis(payload):
        return True
    if isinstance(payload, Mapping):
        status = str(payload.get("status") or "").strip().lower()
        if status == "failed":
            return True
    return False


def _extract_response_text(response_payload: Any) -> str:
    def _choices_from_payload(payload: Any) -> Sequence[Any] | None:
        if payload is None:
            return None
        if hasattr(payload, "choices"):
            return getattr(payload, "choices")
        if isinstance(payload, Mapping):
            value = payload.get("choices")  # type: ignore[assignment]
            if isinstance(value, Sequence):
                return value
        return None

    def _message_from_choice(choice: Any) -> Any:
        if choice is None:
            return None
        if hasattr(choice, "message"):
            return getattr(choice, "message")
        if isinstance(choice, Mapping):
            return choice.get("message")
        return None

    def _coerce_content_from_message(message: Any) -> str:
        content: Any
        if message is None:
            return ""
        if hasattr(message, "content"):
            content = getattr(message, "content")
        elif isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = message

        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
            pieces: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    pieces.append(chunk)
                elif isinstance(chunk, Mapping):
                    text_piece = chunk.get("text")
                    if isinstance(text_piece, str):
                        pieces.append(text_piece)
            return "".join(pieces)
        if content is not None:
            return str(content)
        return ""

    def _extract_from_choices(choices: Sequence[Any] | None) -> str:
        if not isinstance(choices, Sequence) or not choices:
            return ""
        message = _message_from_choice(choices[0])
        return _coerce_content_from_message(message)

    primary_text = _extract_from_choices(_choices_from_payload(response_payload))
    if primary_text:
        return primary_text

    if isinstance(response_payload, Mapping):
        raw_candidate = response_payload.get("raw") or response_payload.get("openai")
        fallback_text = _extract_from_choices(_choices_from_payload(raw_candidate))
        if fallback_text:
            return fallback_text

    return ""


def _ensure_valid_json_response(response_payload: Any) -> None:
    parse_note_style_response_payload(response_payload)


def _write_raw_response(
    *,
    account_paths: NoteStyleAccountPaths,
    response_payload: Any,
    preserve_existing: bool = False,
) -> Path:
    """Persist the raw model content for debugging."""

    raw_path = account_paths.result_raw_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if preserve_existing and raw_path.exists():
        return raw_path

    text_content = _extract_response_text(response_payload)
    if isinstance(text_content, str) and text_content.strip():
        to_write = text_content
    else:
        safe_payload = _safe_json_payload(response_payload)
        to_write = json.dumps(safe_payload, ensure_ascii=False, indent=2)
    if not to_write.endswith("\n"):
        to_write = f"{to_write}\n"

    raw_path.write_text(to_write, encoding="utf-8")
    return raw_path


def _write_invalid_result_marker(
    *, account_paths: NoteStyleAccountPaths, reason: str
) -> Path:
    marker = {"error": "invalid_result", "reason": reason, "at": _now_iso()}
    result_path = account_paths.result_file
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False)
        handle.write("\n")
    return result_path


def _extract_error_payload(exc: Exception) -> dict[str, Any]:
    error_type = exc.__class__.__name__ or "Exception"
    message = str(exc).strip()
    code_value: Any | None = None
    for attr in ("status_code", "http_status", "code"):
        candidate = getattr(exc, attr, None)
        if candidate is None:
            continue
        if isinstance(candidate, (int, str)):
            code_value = candidate
            break
        try:
            code_value = int(candidate)  # type: ignore[arg-type]
            break
        except (TypeError, ValueError):
            code_value = str(candidate)
            break

    error_payload: dict[str, Any] = {"type": error_type}
    if message:
        error_payload["message"] = message
    if code_value is not None:
        error_payload["code"] = code_value
    return error_payload


_SENSITIVE_SNAPSHOT_KEYS = (
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "auth",
    "token",
    "secret",
)

_RAW_OPENAI_EXCERPT_LIMIT = 4096


def _should_redact_snapshot_key(key: str) -> bool:
    lowered = key.strip().lower()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in _SENSITIVE_SNAPSHOT_KEYS)


def _redact_sensitive_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _should_redact_snapshot_key(key_text):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _redact_sensitive_payload(item)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_sensitive_payload(item) for item in value]
    return value


def _summarize_openai_response(response_payload: Any) -> tuple[str | None, str | None]:
    mode: str | None = None
    if isinstance(response_payload, Mapping):
        mode_candidate = response_payload.get("mode")
        if isinstance(mode_candidate, str) and mode_candidate.strip():
            mode = mode_candidate.strip()

    if mode is None:
        choices = _coerce_attr(response_payload, "choices")
        if isinstance(choices, Sequence) and choices:
            first_choice = choices[0]
            message = _coerce_attr(first_choice, "message")
            if message is not None:
                content = _coerce_attr(message, "content")
                if isinstance(content, str) and content.strip():
                    mode = "content"
                else:
                    tool_calls = _coerce_attr(message, "tool_calls")
                    if isinstance(tool_calls, Sequence) and tool_calls:
                        mode = "tool"

    if mode is None and isinstance(response_payload, Mapping):
        if response_payload.get("tool_json") is not None or response_payload.get(
            "raw_tool_arguments"
        ) is not None:
            mode = "tool"
        elif response_payload.get("content_json") is not None or response_payload.get(
            "raw_content"
        ) is not None:
            mode = "content"

    snapshot_target: Any = response_payload
    if isinstance(response_payload, Mapping):
        for key in ("openai", "raw"):
            if key in response_payload:
                snapshot_target = response_payload[key]
                break

    safe_payload = _safe_json_payload(snapshot_target)
    sanitized_payload = _redact_sensitive_payload(safe_payload)

    excerpt: str | None = None
    try:
        serialized = json.dumps(
            sanitized_payload, ensure_ascii=False, sort_keys=True, indent=None
        )
    except TypeError:
        serialized = str(sanitized_payload)

    serialized = serialized.strip()
    if serialized:
        if len(serialized) > _RAW_OPENAI_EXCERPT_LIMIT:
            excerpt = serialized[: _RAW_OPENAI_EXCERPT_LIMIT]
        else:
            excerpt = serialized

    return mode, excerpt


def _write_failure_result(
    *,
    sid: str,
    account_paths: NoteStyleAccountPaths,
    paths: NoteStylePaths,
    error_payload: Mapping[str, Any],
    raw_openai_mode: str | None = None,
    raw_openai_payload_excerpt: str | None = None,
) -> Path:
    failure_path = account_paths.result_file
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "account": account_paths.account_id,
        "sid": sid,
        "error": dict(error_payload),
        "completed_at": _now_iso(),
    }
    if isinstance(raw_openai_mode, str) and raw_openai_mode.strip():
        payload["raw_openai_mode"] = raw_openai_mode.strip()
    if isinstance(raw_openai_payload_excerpt, str) and raw_openai_payload_excerpt.strip():
        payload["raw_openai_payload_excerpt"] = raw_openai_payload_excerpt.strip()
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return failure_path


def _handle_invalid_response(
    *,
    sid: str,
    account_id: str,
    account_paths: NoteStyleAccountPaths,
    response_payload: Any,
    runs_root_path: Path,
    reason: str,
    write_raw: bool = True,
    preserve_existing_raw: bool = False,
) -> None:
    raw_path: Path | None = None
    if write_raw:
        raw_path = _write_raw_response(
            account_paths=account_paths,
            response_payload=response_payload,
            preserve_existing=preserve_existing_raw,
        )
    else:
        candidate = account_paths.result_raw_file
        if candidate.exists():
            raw_path = candidate

    marker_path = _write_invalid_result_marker(
        account_paths=account_paths, reason=reason
    )
    log.warning(
        "NOTE_STYLE_RESULT_INVALID sid=%s account_id=%s reason=%s raw_path=%s result_path=%s",
        sid,
        account_id,
        reason,
        raw_path.resolve().as_posix() if raw_path else "<not-written>",
        marker_path.resolve().as_posix(),
    )
    raw_mode, raw_excerpt = _summarize_openai_response(response_payload)
    record_note_style_failure(
        sid,
        account_id,
        runs_root=runs_root_path,
        error="invalid_result",
        parser_reason=reason,
        raw_path=raw_path,
        raw_openai_mode=(raw_mode or "content"),
        raw_openai_payload_excerpt=raw_excerpt,
    )


def _send_pack_payload(
    *,
    sid: str,
    account_id: str,
    pack_payload: Mapping[str, Any],
    pack_relative: str,
    pack_path: Path,
    account_paths: NoteStyleAccountPaths,
    paths: NoteStylePaths,
    runs_root_path: Path,
    client: Any,
) -> bool:
    log.info(
        "STYLE_SEND_ACCOUNT_START sid=%s account_id=%s pack=%s",
        sid,
        account_id,
        pack_path,
    )

    skip_existing_results = config.NOTE_STYLE_SKIP_IF_RESULT_EXISTS
    idempotent_hash_enabled = _env_flag_enabled(
        "NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH",
        default=config.NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH,
    )

    existing_payload: Mapping[str, Any] | None = None
    if skip_existing_results or idempotent_hash_enabled:
        existing_payload = _load_result_payload(account_paths.result_file)

    skip_reason: str | None = None
    if skip_existing_results:
        skip_reason = _result_skip_reason(existing_payload)

    if skip_reason:
        result_relative = _relativize(account_paths.result_file, paths.base)
        log.info(
            "STYLE_SEND_SKIP_EXISTING sid=%s account_id=%s result=%s reason=%s",
            sid,
            account_id,
            result_relative,
            skip_reason,
        )
        log_structured_event(
            "NOTE_STYLE_SKIP_EXISTING",
            logger=log,
            sid=sid,
            account_id=account_id,
            reason=skip_reason,
            result_path=result_relative,
        )
        return False

    if idempotent_hash_enabled:
        note_text = _extract_pack_note_text(pack_payload)
        if isinstance(note_text, str) and note_text:
            candidate_hash = hashlib.sha256(
                note_text.encode("utf-8", "ignore")
            ).hexdigest()
            existing_hash: str | None = None
            if isinstance(existing_payload, Mapping):
                stored_hash = existing_payload.get("note_hash")
                if isinstance(stored_hash, str):
                    trimmed = stored_hash.strip()
                    if trimmed:
                        existing_hash = trimmed
            if existing_hash and existing_hash == candidate_hash:
                result_relative = _relativize(account_paths.result_file, paths.base)
                log.info(
                    "STYLE_SEND_SKIP_NOTE_HASH sid=%s account_id=%s hash=%s result=%s",
                    sid,
                    account_id,
                    candidate_hash,
                    result_relative,
                )
                log_structured_event(
                    "NOTE_STYLE_SEND_SKIPPED",
                    logger=log,
                    sid=sid,
                    account_id=account_id,
                    reason="note_hash_match",
                    note_hash=candidate_hash,
                    result_path=result_relative,
                )
                return False

    model = str(pack_payload.get("model") or "").strip()
    if not model:
        model = config.NOTE_STYLE_MODEL
    messages = _coerce_messages(pack_payload)

    response_format = _coerce_response_format(pack_payload)
    response_mode = _normalize_response_mode(config.NOTE_STYLE_RESPONSE_MODE)

    retry_attempts = max(0, int(config.NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS))
    enable_tool_call_retry = bool(config.NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL)
    total_attempts = max(1, 1 + retry_attempts)
    failure_records: list[Mapping[str, Any]] = []
    raw_response_written = False
    final_latency = 0.0

    def _handle_invalid_attempt(
        *,
        attempt_index: int,
        reason: str,
        response_payload: Any,
        response_metrics: Mapping[str, Any] | None = None,
    ) -> bool:
        nonlocal raw_response_written
        failure_records.append({"attempt": attempt_index + 1, "reason": reason})
        _record_parse_outcome(response_payload=response_payload, success=False)
        if not raw_response_written:
            _write_raw_response(
                account_paths=account_paths,
                response_payload=response_payload,
            )
            raw_response_written = True
        else:
            _write_raw_response(
                account_paths=account_paths,
                response_payload=response_payload,
                preserve_existing=True,
            )

        remaining_attempts = total_attempts - attempt_index - 1
        if remaining_attempts <= 0:
            _log_pack_request_metrics(
                sid=sid,
                account_id=account_id,
                model=model or "",
                metrics=response_metrics,
                parse_ok=False,
                retry_count=attempt_index,
            )
            _handle_invalid_response(
                sid=sid,
                account_id=account_id,
                account_paths=account_paths,
                response_payload=response_payload,
                runs_root_path=runs_root_path,
                reason=reason,
                write_raw=not raw_response_written,
                preserve_existing_raw=True,
            )
            log_structured_event(
                "NOTE_STYLE_RESULT_INVALID_FINAL",
                logger=log,
                sid=sid,
                account_id=account_id,
                reason=reason,
                attempts=failure_records,
                total_attempts=total_attempts,
            )
            return False

        log.warning(
            "NOTE_STYLE_RESULT_INVALID_RETRY sid=%s account_id=%s attempt=%d remaining=%d reason=%s",
            sid,
            account_id,
            attempt_index + 1,
            remaining_attempts,
            reason,
        )
        log_structured_event(
            "NOTE_STYLE_INVALID_RESULT_RETRY",
            logger=log,
            sid=sid,
            account_id=account_id,
            attempt=attempt_index + 1,
            remaining_attempts=remaining_attempts,
            reason=reason,
        )
        return True

    for attempt_index in range(total_attempts):
        response_kwargs, request_mode = _response_kwargs_for_attempt(
            response_format,
            attempt_index,
            response_mode=response_mode,
            enable_tool_call_retry=enable_tool_call_retry,
        )
        response_kwargs.setdefault("max_tokens", _NOTE_STYLE_RESPONSE_MAX_TOKENS)

        temperature = 0
        top_p = 1
        max_tokens = response_kwargs.get(
            "max_tokens", _NOTE_STYLE_RESPONSE_MAX_TOKENS
        )

        response_kwargs = {
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        request_metadata = _resolve_request_metadata(response_kwargs, mode=request_mode)

        attempt_messages = _messages_for_attempt(messages, attempt_index)
        _apply_mode_response_instruction(attempt_messages, request_mode)

        response_metrics: Mapping[str, Any] | None = None
        start = time.perf_counter()
        try:
            response = client.chat_completion(
                model=model or None,
                messages=list(attempt_messages),
                **response_kwargs,
            )
            latency = time.perf_counter() - start
            final_latency = latency
            response_format_payload = response_kwargs.get("response_format")
            if (
                isinstance(response_format_payload, Mapping)
                and response_format_payload.get("type") == "json_object"
                and not isinstance(response.get("content_json"), Mapping)
            ):
                log.warning(
                    "NOTE_STYLE: json_object expected but wrapper returned no content_json keys=%s",
                    sorted(response.keys()),
                )
            log.info(
                "STYLE_SEND_MODEL_CALL sid=%s account_id=%s model=%s status=success attempt=%d latency=%.3fs",
                sid,
                account_id,
                model or "",
                attempt_index + 1,
                latency,
            )
            response_metrics = _build_response_metrics(
                response,
                response_format=request_metadata.get("response_format"),
                tool_choice=request_metadata.get("tool_choice"),
                mode=request_mode,
            )
            response_payload_mode = _response_payload_mode(response)
            has_content_payload = bool(
                response.get("content_json")
                or (isinstance(response.get("raw_content"), str) and response.get("raw_content").strip())
            )
            has_tool_payload = bool(
                response.get("tool_json")
                or response.get("raw_tool_arguments")
            )
            log.info(
                "NOTE_STYLE_RESPONSE_MODE sid=%s account_id=%s configured_mode=%s request_mode=%s response_mode=%s has_content=%s has_tool=%s",
                sid,
                account_id,
                response_mode.value,
                request_mode,
                response_payload_mode,
                has_content_payload,
                has_tool_payload,
            )
            log_structured_event(
                "NOTE_STYLE_RESPONSE_MODE",
                logger=log,
                sid=sid,
                account_id=account_id,
                configured_mode=response_mode.value,
                request_mode=request_mode,
                response_mode=response_payload_mode,
                has_content=bool(has_content_payload),
                has_tool=bool(has_tool_payload),
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            final_latency = latency
            log.exception(
                "STYLE_SEND_MODEL_CALL sid=%s account_id=%s model=%s status=error attempt=%d latency=%.3fs",
                sid,
                account_id,
                model or "",
                attempt_index + 1,
                latency,
            )
            error_payload = _extract_error_payload(exc)
            failure_path = _write_failure_result(
                sid=sid,
                account_paths=account_paths,
                paths=paths,
                error_payload=error_payload,
            )
            log.warning(
                "NOTE_STYLE_RESULT_FAILED_ARTIFACT sid=%s account_id=%s path=%s",
                sid,
                account_id,
                _relativize(failure_path, paths.base),
            )
            _log_pack_request_metrics(
                sid=sid,
                account_id=account_id,
                model=model or "",
                metrics=request_metadata,
                parse_ok=False,
                retry_count=attempt_index,
            )
            try:
                record_note_style_failure(
                    sid,
                    account_id,
                    runs_root=runs_root_path,
                    error=str(error_payload.get("message") or error_payload.get("type")),
                )
            except Exception:  # pragma: no cover - defensive logging
                log.warning(
                    "NOTE_STYLE_FAILURE_RECORD_FAILED sid=%s account_id=%s",
                    sid,
                    account_id,
                    exc_info=True,
                )
            return False

        try:
            _ensure_valid_json_response(response)
        except ValueError as exc:
            reason_text = str(exc).strip() or exc.__class__.__name__
            if not _handle_invalid_attempt(
                attempt_index=attempt_index,
                reason=reason_text,
                response_payload=response,
                response_metrics=response_metrics,
            ):
                return False
            continue

        try:
            written_path = ingest_note_style_result(
                sid=sid,
                account_id=account_id,
                runs_root=runs_root_path,
                account_paths=account_paths,
                pack_payload=pack_payload,
                response_payload=response,
            )
        except ValueError as exc:
            log.warning(
                "STYLE_SEND_RESULTS_FAILED sid=%s account_id=%s reason=parse_error",
                sid,
                account_id,
                exc_info=True,
            )
            reason_text = str(exc).strip() or exc.__class__.__name__
            if not _handle_invalid_attempt(
                attempt_index=attempt_index,
                reason=reason_text,
                response_payload=response,
                response_metrics=response_metrics,
            ):
                return False
            continue
        except NotImplementedError:
            written_path = account_paths.result_file
        except Exception as exc:
            log.exception(
                "STYLE_SEND_RESULTS_FAILED sid=%s account_id=%s",
                sid,
                account_id,
            )
            raw_mode, raw_excerpt = _summarize_openai_response(response)
            raw_path = _write_raw_response(
                account_paths=account_paths,
                response_payload=response,
                preserve_existing=raw_response_written,
            )
            _log_pack_request_metrics(
                sid=sid,
                account_id=account_id,
                model=model or "",
                metrics=response_metrics,
                parse_ok=False,
                retry_count=attempt_index,
            )
            record_note_style_failure(
                sid,
                account_id,
                runs_root=runs_root_path,
                error=str(exc),
                raw_path=raw_path,
                raw_openai_mode=(raw_mode or "content"),
                raw_openai_payload_excerpt=raw_excerpt,
            )
            raise

        _log_pack_request_metrics(
            sid=sid,
            account_id=account_id,
            model=model or "",
            metrics=response_metrics,
            parse_ok=True,
            retry_count=attempt_index,
        )
        result_path = _coerce_result_path(written_path) or account_paths.result_file
        log.info(
            "NOTE_STYLE_SENT sid=%s account_id=%s result=%s attempts=%d",
            sid,
            account_id,
            result_path,
            attempt_index + 1,
        )

        result_relative = _relativize(result_path, paths.base)
        log_structured_event(
            "NOTE_STYLE_SENT",
            logger=log,
            sid=sid,
            account_id=account_id,
            model=model or "",
            latency_seconds=final_latency,
            pack_path=pack_relative,
            result_path=result_relative,
            attempts=attempt_index + 1,
            retries_used=max(0, attempt_index),
        )

        _record_parse_outcome(response_payload=response, success=True)

        log.info(
            "STYLE_SEND_ACCOUNT_END sid=%s account_id=%s status=completed attempts=%d",
            sid,
            account_id,
            attempt_index + 1,
        )
        return True

    return False


def send_note_style_packs_for_sid(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> list[str]:
    """Send all note_style packs with ``status="built"`` for ``sid``.

    Returns a list of account identifiers that were processed. Raises any
    exception encountered while reading packs, invoking the model, or persisting
    results so the caller can handle retries.
    """

    runs_root_path = _resolve_runs_root(runs_root)
    paths = resolve_note_style_stage_paths(runs_root_path, sid, create=False)
    _log_sender_paths(sid, paths)
    _warn_if_index_thin(paths, sid=sid)
    snapshot = note_style_snapshot(sid, runs_root=runs_root_path)
    expected_lookup = _build_normalized_lookup(set(snapshot.packs_expected))
    expected_normalized = set(expected_lookup.keys())
    completed_normalized = _normalize_account_id_set(set(snapshot.packs_completed))
    failed_normalized = _normalize_account_id_set(set(snapshot.packs_failed))
    pending_normalized = expected_normalized - (completed_normalized | failed_normalized)
    log_structured_event(
        "NOTE_STYLE_SNAPSHOT_COUNTS",
        logger=log,
        sid=sid,
        packs_built=len(snapshot.packs_built),
        packs_completed=len(snapshot.packs_completed),
        packs_failed=len(snapshot.packs_failed),
    )
    packs_dir = _resolve_packs_dir(paths)
    debug_dir = getattr(paths, "debug_dir", paths.base / "debug")
    response_mode = _normalize_response_mode(config.NOTE_STYLE_RESPONSE_MODE)
    env_glob_raw = os.getenv("NOTE_STYLE_PACK_GLOB")
    env_glob = None
    if env_glob_raw:
        sanitized_glob = env_glob_raw.strip().replace("\\", "/")
        if sanitized_glob:
            env_glob = sanitized_glob
    fallback_glob = "acc_*.jsonl"
    default_glob = env_glob or fallback_glob

    log.info(
        "NOTE_STYLE_SEND_START sid=%s packs_dir=%s glob=%s use_manifest=%s",
        sid,
        packs_dir,
        "<manifest>" if config.NOTE_STYLE_USE_MANIFEST_PATHS else default_glob,
        config.NOTE_STYLE_USE_MANIFEST_PATHS,
    )
    log.info(
        "NOTE_STYLE_SEND_CONFIG sid=%s response_mode=%s retry_count=%s strict_schema=%s",
        sid,
        response_mode.value,
        config.NOTE_STYLE_RETRY_COUNT,
        int(bool(config.NOTE_STYLE_STRICT_SCHEMA)),
    )

    manifest_candidates: list[_PackCandidate] = []
    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        manifest_candidates = _load_manifest_pack_entries(paths, sid=sid)

    pack_candidates: list[_PackCandidate]
    if manifest_candidates:
        pack_candidates = manifest_candidates
        log.info(
            "NOTE_STYLE_PACK_DISCOVERY sid=%s glob=%s matches=%s manifest=%s",
            sid,
            "<manifest>",
            len(pack_candidates),
            True,
        )
    else:
        glob_pattern = default_glob

        def _collect_candidates(pattern: str) -> list[Path]:
            try:
                raw_matches = sorted(packs_dir.glob(pattern))
            except ValueError:
                log.warning(
                    "NOTE_STYLE_PACK_GLOB_INVALID sid=%s glob=%s", sid, pattern
                )
                return []
            filtered: list[Path] = []
            packs_dir_resolved = packs_dir.resolve()
            debug_dir_resolved = debug_dir.resolve()
            for match in raw_matches:
                if match.is_dir():
                    continue
                if _is_within_directory(match, debug_dir_resolved):
                    continue
                try:
                    relative_parts = match.resolve().relative_to(packs_dir_resolved).parts
                except ValueError:
                    relative_parts = match.parts
                if any(part.startswith("results") for part in relative_parts):
                    continue
                if match.name.startswith("results"):
                    continue
                filtered.append(match)
            return filtered

        file_candidates = _collect_candidates(glob_pattern)
        effective_glob = glob_pattern

        if env_glob and not file_candidates and glob_pattern != fallback_glob:
            fallback_candidates = _collect_candidates(fallback_glob)
            if fallback_candidates:
                log.info(
                    "NOTE_STYLE_PACK_GLOB_FALLBACK sid=%s glob=%s fallback=%s matches=%s",
                    sid,
                    glob_pattern,
                    fallback_glob,
                    len(fallback_candidates),
                )
                file_candidates = fallback_candidates
                effective_glob = fallback_glob

        pack_candidates = [_PackCandidate(pack_path=path) for path in file_candidates]
        log.info(
            "NOTE_STYLE_PACK_DISCOVERY sid=%s glob=%s matches=%s",
            sid,
            effective_glob,
            len(pack_candidates),
        )

    sample_candidates = [
        _relativize(candidate.pack_path, paths.base)
        for candidate in pack_candidates[:5]
    ]
    log.info(
        "NOTE_STYLE_PACKS_FOUND sid=%s count=%s sample=%s",
        sid,
        len(pack_candidates),
        sample_candidates,
    )

    pending_tracker = set(pending_normalized)

    if not pending_normalized:
        log.info(
            "NOTE_STYLE_NO_PENDING sid=%s expected=%s completed=%s failed=%s",
            sid,
            len(expected_normalized),
            len(completed_normalized),
            len(failed_normalized),
        )
        return []

    if not pack_candidates:
        log.info("NOTE_STYLE_NO_PACKS sid=%s", sid)
        if pending_tracker:
            for missing_account in sorted(pending_tracker):
                display_account = expected_lookup.get(missing_account, missing_account)
                log.warning(
                    "NOTE_STYLE_PENDING_NO_PACK sid=%s account_id=%s",
                    sid,
                    display_account,
                )
        return []

    client = get_ai_client()
    processed: list[str] = []
    index_account_map = _load_index_account_map(paths)
    seen_accounts: set[str] = set()

    for candidate in pack_candidates:
        pack_path = candidate.pack_path

        if manifest_candidates:
            run_dir = paths.base.parent.parent
            if not _is_within_directory(pack_path, run_dir):
                log.warning(
                    "STYLE_SEND_PACK_OUTSIDE_MANIFEST sid=%s path=%s run_dir=%s",
                    sid,
                    pack_path,
                    run_dir,
                )
                continue
        else:
            if not _is_within_directory(pack_path, packs_dir):
                log.warning(
                    "STYLE_SEND_PACK_OUTSIDE_DIR sid=%s path=%s packs_dir=%s",
                    sid,
                    pack_path,
                    packs_dir,
                )
                continue

        if _is_within_directory(pack_path, debug_dir):
            log.info(
                "STYLE_SEND_SKIP_DEBUG sid=%s path=%s",
                sid,
                pack_path,
            )
            continue
        if not pack_path.is_file():
            continue
        try:
            pack_records = _load_pack_records(pack_path)
        except Exception:
            log.exception(
                "STYLE_SEND_PACK_LOAD_FAILED sid=%s path=%s", sid, pack_path
            )
            raise

        pack_relative = _relativize(pack_path, paths.base)

        for pack_payload in pack_records:
            account_id = str(pack_payload.get("account_id") or "").strip()
            if not account_id and candidate.account_id:
                account_id = candidate.account_id
            if not account_id:
                account_id = index_account_map.get(pack_relative, "")
            if not account_id and candidate.normalized_account_id:
                account_id = candidate.normalized_account_id
            if not account_id:
                account_id = _account_id_from_pack_path(pack_path)
            if not account_id:
                log.warning(
                    "STYLE_SEND_ACCOUNT_UNKNOWN sid=%s pack=%s", sid, pack_path
                )
                continue
            account_paths = _account_paths_for_candidate(paths, account_id, candidate)

            normalized_account_id = account_paths.account_id

            if normalized_account_id in seen_accounts:
                log.info(
                    "NOTE_STYLE_SKIP_DUPLICATE sid=%s account_id=%s pack=%s",
                    sid,
                    account_id,
                    pack_relative,
                )
                continue

            seen_accounts.add(normalized_account_id)

            if normalized_account_id not in expected_normalized:
                log.info(
                    "NOTE_STYLE_SKIP_UNEXPECTED sid=%s account_id=%s pack=%s",
                    sid,
                    account_id,
                    pack_relative,
                )
                continue

            if normalized_account_id not in pending_normalized:
                log.info(
                    "NOTE_STYLE_SKIP_NOT_PENDING sid=%s account_id=%s pack=%s",
                    sid,
                    account_id,
                    pack_relative,
                )
                continue

            if pipeline_runs.account_result_ready(
                sid,
                normalized_account_id,
                runs_root=runs_root_path,
            ):
                result_relative = _relativize(account_paths.result_file, paths.base)
                display_account = account_id or normalized_account_id
                log.info(
                    "STYLE_SEND_SKIP_EXISTING sid=%s account_id=%s result=%s reason=%s",
                    sid,
                    display_account,
                    result_relative,
                    "terminal_result",
                )
                log_structured_event(
                    "NOTE_STYLE_SKIP_EXISTING",
                    logger=log,
                    sid=sid,
                    account_id=display_account,
                    reason="terminal_result",
                    result_path=result_relative,
                )
                pending_tracker.discard(normalized_account_id)
                continue

            log.info(
                "NOTE_STYLE_SENDING sid=%s account_id=%s file=%s",
                sid,
                account_id,
                pack_relative,
            )

            if _send_pack_payload(
                sid=sid,
                account_id=account_id,
                pack_payload=pack_payload,
                pack_relative=pack_relative,
                pack_path=pack_path,
                account_paths=account_paths,
                paths=paths,
                runs_root_path=runs_root_path,
                client=client,
            ):
                processed.append(account_id)
            pending_tracker.discard(normalized_account_id)

    log.info(
        "NOTE_STYLE_SEND_DONE sid=%s processed=%s",
        sid,
        processed,
    )

    if pending_tracker:
        for missing_account in sorted(pending_tracker):
            display_account = expected_lookup.get(missing_account, missing_account)
            log.warning(
                "NOTE_STYLE_PENDING_NO_PACK sid=%s account_id=%s",
                sid,
                display_account,
            )

    return processed


def send_note_style_pack_for_account(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
) -> bool:
    runs_root_path = _resolve_runs_root(runs_root)
    paths = resolve_note_style_stage_paths(runs_root_path, sid, create=False)
    _log_sender_paths(sid, paths)
    _warn_if_index_thin(paths, sid=sid)
    snapshot = note_style_snapshot(sid, runs_root=runs_root_path)
    expected_normalized = set(
        _build_normalized_lookup(set(snapshot.packs_expected)).keys()
    )
    completed_normalized = _normalize_account_id_set(set(snapshot.packs_completed))
    failed_normalized = _normalize_account_id_set(set(snapshot.packs_failed))

    target_normalized = normalize_note_style_account_id(account_id)
    if not target_normalized or target_normalized not in expected_normalized:
        log.info(
            "NOTE_STYLE_SKIP_UNEXPECTED sid=%s account_id=%s pack=<direct>",
            sid,
            account_id,
        )
        return False

    if target_normalized in completed_normalized | failed_normalized:
        log.info(
            "NOTE_STYLE_SKIP_NOT_PENDING sid=%s account_id=%s pack=<direct>",
            sid,
            account_id,
        )
        return False

    candidate: _PackCandidate | None = None
    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        target = normalize_note_style_account_id(account_id)
        for entry in _load_manifest_pack_entries(paths, sid=sid):
            normalized_entry = entry.normalized_account_id
            if normalized_entry is None and entry.account_id is not None:
                normalized_entry = normalize_note_style_account_id(entry.account_id)
            if normalized_entry == target:
                candidate = entry
                break

    account_paths = _account_paths_for_candidate(paths, account_id, candidate)

    pack_path = account_paths.pack_file
    if not pack_path.exists():
        log.info(
            "STYLE_SEND_PACK_MISSING sid=%s account_id=%s path=%s",
            sid,
            account_id,
            pack_path,
        )
        return False

    try:
        pack_records = _load_pack_records(pack_path)
    except Exception:
        log.exception(
            "STYLE_SEND_PACK_LOAD_FAILED sid=%s path=%s", sid, pack_path
        )
        raise

    pack_relative = _relativize(pack_path, paths.base)
    client = get_ai_client()

    for pack_payload in pack_records:
        normalized_account_id = account_paths.account_id
        if pipeline_runs.account_result_ready(
            sid,
            normalized_account_id,
            runs_root=runs_root_path,
        ):
            result_relative = _relativize(account_paths.result_file, paths.base)
            display_account = account_id or normalized_account_id
            log.info(
                "STYLE_SEND_SKIP_EXISTING sid=%s account_id=%s result=%s reason=%s",
                sid,
                display_account,
                result_relative,
                "terminal_result",
            )
            log_structured_event(
                "NOTE_STYLE_SKIP_EXISTING",
                logger=log,
                sid=sid,
                account_id=display_account,
                reason="terminal_result",
                result_path=result_relative,
            )
            continue
        if _send_pack_payload(
            sid=sid,
            account_id=account_id,
            pack_payload=pack_payload,
            pack_relative=pack_relative,
            pack_path=pack_path,
            account_paths=account_paths,
            paths=paths,
            runs_root_path=runs_root_path,
            client=client,
        ):
            return True

    return False


__all__ = ["send_note_style_packs_for_sid", "send_note_style_pack_for_account"]
