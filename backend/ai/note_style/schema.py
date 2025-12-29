"""Schema definitions and validation helpers for note_style results."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Mapping, Sequence

from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict, Field, confloat, conint, constr


NOTE_STYLE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "tone",
        "context_hints",
        "emphasis",
        "confidence",
        "risk_flags",
    ],
    "additionalProperties": False,
    "properties": {
        "tone": {"type": "string", "minLength": 1},
        "context_hints": {
            "type": "object",
            "required": ["timeframe", "topic", "entities"],
            "additionalProperties": False,
            "properties": {
                "timeframe": {
                    "type": "object",
                    "required": ["month", "relative"],
                    "additionalProperties": False,
                    "properties": {
                        "month": {"type": ["string", "null"], "minLength": 1},
                        "relative": {"type": ["string", "null"], "minLength": 1},
                    },
                },
                "topic": {"type": "string", "minLength": 1},
                "entities": {
                    "type": "object",
                    "required": ["creditor", "amount"],
                    "additionalProperties": False,
                    "properties": {
                        "creditor": {"type": ["string", "null"], "minLength": 1},
                        "amount": {"type": ["number", "null"]},
                    },
                },
            },
        },
        "emphasis": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

# Exported for tool-calling parameter definitions.
NOTE_STYLE_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = NOTE_STYLE_ANALYSIS_SCHEMA
NOTE_STYLE_TOOL_FUNCTION_NAME = "submit_note_style_analysis"
NOTE_STYLE_TOOL_DESCRIPTION = (
    "Return the strict note_style analysis JSON object that satisfies the schema."
)


def build_note_style_tool() -> dict[str, Any]:
    """Return a fresh tool definition for note_style requests."""

    return {
        "type": "function",
        "function": {
            "name": NOTE_STYLE_TOOL_FUNCTION_NAME,
            "description": NOTE_STYLE_TOOL_DESCRIPTION,
            "parameters": copy.deepcopy(NOTE_STYLE_TOOL_PARAMETERS_SCHEMA),
        },
    }

_ANALYSIS_VALIDATOR = Draft7Validator(NOTE_STYLE_ANALYSIS_SCHEMA)


class NoteStyleTimeframe(BaseModel):
    """Timeframe context extracted from the note analysis."""

    model_config = ConfigDict(extra="forbid")

    month: constr(strip_whitespace=True, min_length=1) | None = Field(default=None)
    relative: constr(strip_whitespace=True, min_length=1) | None = None


class NoteStyleEntities(BaseModel):
    """Entity hints referenced by the AI analysis."""

    model_config = ConfigDict(extra="forbid")

    creditor: constr(strip_whitespace=True, min_length=1) | None = None
    amount: float | None = None


class NoteStyleContextHints(BaseModel):
    """Helpful context derived from the note."""

    model_config = ConfigDict(extra="forbid")

    timeframe: NoteStyleTimeframe
    topic: constr(strip_whitespace=True, min_length=1)
    entities: NoteStyleEntities


class NoteStyleAnalysis(BaseModel):
    """Structured analysis of a generated note."""

    model_config = ConfigDict(extra="forbid")

    tone: constr(strip_whitespace=True, min_length=1)
    context_hints: NoteStyleContextHints
    emphasis: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list
    )
    confidence: confloat(ge=0.0, le=1.0)
    risk_flags: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list
    )


class NoteStyleMetrics(BaseModel):
    """Summary metrics for the generated note."""

    model_config = ConfigDict(extra="forbid")

    char_len: conint(ge=0)
    word_len: conint(ge=0)


class NoteStyleResult(BaseModel):
    """Canonical persisted result for note_style runs."""

    model_config = ConfigDict(extra="forbid")

    sid: constr(strip_whitespace=True, min_length=1)
    account_id: constr(strip_whitespace=True, min_length=1)
    analysis: NoteStyleAnalysis
    note_metrics: NoteStyleMetrics
    note_hash: constr(strip_whitespace=True, min_length=1)
    evaluated_at: datetime


def validate_note_style_analysis(payload: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    """Validate ``payload`` against the strict note_style analysis schema."""

    if not isinstance(payload, Mapping):
        return False, ["payload_not_mapping"]

    errors = sorted(_ANALYSIS_VALIDATOR.iter_errors(payload), key=_error_sort_key)
    messages = [error.message for error in errors]
    return (not messages), messages


def validate_result(obj: Any) -> NoteStyleResult:
    """Validate and coerce ``obj`` into a :class:`NoteStyleResult`."""

    if isinstance(obj, NoteStyleResult):
        return obj

    return NoteStyleResult.model_validate(obj)


def to_json(obj: NoteStyleResult | Mapping[str, Any]) -> str:
    """Serialize a validated :class:`NoteStyleResult` to a JSON string."""

    result = obj if isinstance(obj, NoteStyleResult) else validate_result(obj)
    return result.model_dump_json(by_alias=False, exclude_none=False) + "\n"


def _error_sort_key(error: Any) -> tuple[Sequence[Any], str]:
    path = tuple(error.path) if isinstance(error.path, Sequence) else ()
    return path, error.message


__all__ = [
    "NOTE_STYLE_ANALYSIS_SCHEMA",
    "NOTE_STYLE_TOOL_PARAMETERS_SCHEMA",
    "NOTE_STYLE_TOOL_FUNCTION_NAME",
    "NOTE_STYLE_TOOL_DESCRIPTION",
    "build_note_style_tool",
    "validate_note_style_analysis",
    "NoteStyleResult",
    "validate_result",
    "to_json",
]
