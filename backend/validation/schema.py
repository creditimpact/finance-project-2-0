"""Schema definitions and validation helpers for validation AI decisions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft7Validator

VALIDATION_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "sid",
        "account_id",
        "id",
        "field",
        "decision",
        "rationale",
        "citations",
        "reason_code",
        "reason_label",
        "modifiers",
        "confidence",
    ],
    "properties": {
        "sid": {"type": "string"},
        "account_id": {"type": "integer"},
        "id": {"type": "string"},
        "field": {"type": "string"},
        "decision": {"enum": ["strong", "supportive", "neutral", "no_case"]},
        "rationale": {"type": "string", "maxLength": 800},
        "citations": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "reason_code": {"type": "string"},
        "reason_label": {"type": "string"},
        "modifiers": {
            "type": "object",
            "required": ["material_mismatch", "time_anchor", "doc_dependency"],
            "properties": {
                "material_mismatch": {"type": "boolean"},
                "time_anchor": {"type": "boolean"},
                "doc_dependency": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}


_VALIDATOR = Draft7Validator(VALIDATION_DECISION_SCHEMA)

_KNOWN_REASON_MISMATCH = {"C4_TWO_MATCH_ONE_DIFF", "C5_ALL_DIFF"}


def _normalized_bureaus(finding: Mapping[str, Any] | None) -> set[str]:
    if finding is None:
        return set()
    bureaus = finding.get("bureaus")
    if isinstance(bureaus, Mapping):
        keys: Iterable[Any] = bureaus.keys()
    else:
        keys = bureaus if isinstance(bureaus, Sequence) and not isinstance(bureaus, (str, bytes)) else []
    normalized: set[str] = set()
    for item in keys:
        if item is None:
            continue
        try:
            text = str(item).strip().lower()
        except Exception:
            continue
        if text:
            normalized.add(text)
    return normalized


def _citations_with_known_bureau(citations: Sequence[Any], known_bureaus: set[str]) -> bool:
    for entry in citations:
        if not isinstance(entry, str):
            continue
        prefix, sep, _ = entry.partition(":")
        if not sep:
            continue
        bureau = prefix.strip().lower()
        if bureau in known_bureaus:
            return True
    return False


def _extract_documents(finding: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(finding, Mapping):
        return []
    documents = finding.get("documents")
    if isinstance(documents, Sequence) and not isinstance(documents, (str, bytes)):
        result: list[str] = []
        for entry in documents:
            if entry is None:
                continue
            try:
                text = str(entry).strip()
            except Exception:
                continue
            if text:
                result.append(text)
        return result
    if isinstance(documents, str):
        text = documents.strip()
        return [text] if text else []
    return []


def _normalized_values(finding: Mapping[str, Any] | None) -> set[str]:
    values: set[str] = set()
    if not isinstance(finding, Mapping):
        return values
    candidates = finding.get("bureau_values")
    if not isinstance(candidates, Mapping):
        candidates = finding.get("bureaus")
    if isinstance(candidates, Mapping):
        for record in candidates.values():
            if not isinstance(record, Mapping):
                continue
            value = record.get("normalized")
            if value is None:
                continue
            try:
                text = str(value).strip().lower()
            except Exception:
                continue
            if text:
                values.add(text)
    return values


def validate_llm_decision(
    obj: Mapping[str, Any] | None, finding: Mapping[str, Any] | None
) -> tuple[bool, list[str]]:
    """Validate an LLM decision object against business rules."""

    if not isinstance(obj, Mapping):
        return False, ["response_not_mapping"]

    errors = [error.message for error in _VALIDATOR.iter_errors(obj)]
    if errors:
        return False, errors

    reason_code = obj.get("reason_code")
    rationale = obj.get("rationale")
    if isinstance(reason_code, str) and reason_code:
        if not (isinstance(rationale, str) and reason_code in rationale):
            errors.append("rationale_missing_reason_code")

    citations = obj.get("citations")
    known_bureaus = _normalized_bureaus(finding)
    if isinstance(citations, Sequence) and citations:
        if known_bureaus and not _citations_with_known_bureau(citations, known_bureaus):
            errors.append("citations_missing_known_bureau")
    else:
        errors.append("citations_empty")

    modifiers = obj.get("modifiers")
    decision = obj.get("decision")
    documents = _extract_documents(finding)
    if (
        decision == "strong"
        and isinstance(modifiers, Mapping)
        and modifiers.get("doc_dependency")
        and not documents
    ):
        errors.append("doc_dependency_without_documents")

    if (
        isinstance(reason_code, str)
        and reason_code in _KNOWN_REASON_MISMATCH
        and isinstance(modifiers, Mapping)
        and not modifiers.get("material_mismatch")
    ):
        normalized = _normalized_values(finding)
        if len(normalized) >= 2:
            errors.append("material_mismatch_required")

    return (not errors), errors


__all__ = [
    "VALIDATION_DECISION_SCHEMA",
    "validate_llm_decision",
]
