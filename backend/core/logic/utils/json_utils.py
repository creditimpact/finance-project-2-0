import json
import logging
import re
from pathlib import Path

try:
    from json_repair import repair_json as _jsonrepair
except Exception:  # pragma: no cover - optional dependency
    _jsonrepair = None

_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")
_DOUBLE_COMMA_RE = re.compile(r",\s*,+")
_SINGLE_QUOTE_KEY_RE = re.compile(r"'([^']*)'(?=\s*:)")
_SINGLE_QUOTE_VALUE_RE = re.compile(r":\s*'([^']*)'")
_UNQUOTED_KEY_RE = re.compile(
    r"(?P<prefix>^|[,{])\s*(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*(?=:)"
)
_MISSING_COMMA_VALUE_RE = re.compile(
    r"(:\s*)(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(\s+)(?=\")"
)
_LOG_PATH = Path("invalid_ai_json.log")


def _log_invalid_json(raw: str, repaired: str, error: str) -> None:
    """Persist the raw and repaired AI output for debugging."""
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("Raw:\n")
            f.write(raw)
            f.write("\nRepaired:\n")
            f.write(repaired)
            f.write("\nError:\n")
            f.write(error)
            f.write("\n---\n")
    except Exception:  # pragma: no cover - best effort only
        logging.debug("Unable to log invalid JSON output.")


def _basic_clean(content: str) -> str:
    """Apply regex-based fixes for common JSON issues."""
    cleaned = _TRAILING_COMMA_RE.sub("", content)
    cleaned = _DOUBLE_COMMA_RE.sub(",", cleaned)
    cleaned = _SINGLE_QUOTE_KEY_RE.sub(r'"\1"', cleaned)
    cleaned = _SINGLE_QUOTE_VALUE_RE.sub(r': "\1"', cleaned)
    cleaned = _UNQUOTED_KEY_RE.sub(
        lambda m: f'{m.group("prefix")}"{m.group("key")}"', cleaned
    )
    cleaned = _MISSING_COMMA_VALUE_RE.sub(r"\1\2,\3", cleaned)
    return cleaned


def _repair_json(content: str) -> str:
    """Attempt to repair malformed JSON using json_repair or basic regex fixes."""
    if _jsonrepair is not None:
        try:
            content = _jsonrepair(content)
        except Exception:  # pragma: no cover - if repair fails, fall back
            logging.debug("json_repair failed to process input")
    return _basic_clean(content)


def parse_json(text: str):
    """Parse ``text`` as JSON, attempting repairs on failure.

    Returns a tuple of ``(data, error_reason)`` where ``error_reason`` is ``None``
    on success or ``"invalid_json"`` when parsing fails even after repair.
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        logging.warning("Initial JSON parse error: %s", e)
        repaired = _repair_json(text)
        logging.debug("Raw JSON before repair: %s", text)
        logging.debug("Repaired JSON string: %s", repaired)
        try:
            return json.loads(repaired), None
        except json.JSONDecodeError as e2:
            logging.error("Repaired JSON parse error: %s", e2)
            _log_invalid_json(text, repaired, str(e2))
            return {}, "invalid_json"
