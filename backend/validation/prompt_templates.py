"""Prompt templates for the validation AI stage."""

from __future__ import annotations

import json
from typing import Any, Sequence

_VALIDATION_PROMPT_TEMPLATE = """SYSTEM:
Project: credit-analyzer
Module: Validation / AI Adjudication

The validation system detects discrepancies between credit bureaus’ reported values for each field (Equifax, Experian, TransUnion).
It doesn’t understand language or legal reasoning — that’s why findings are sent to the AI adjudicator.

Your job is to decide whether the bureau statements conflict in a way that creates usable evidence for a consumer dispute.
You are not deciding which bureau is correct; you are determining whether the mismatch is strong enough that a bureau would be obligated or reasonably expected to investigate or correct it if disputed.
Assume the consumer claims the most favorable version is accurate.
The detection platform already confirmed the numbers/strings differ — focus only on linguistic and legal significance.
Use ONLY the JSON pack provided. Output STRICT JSON only (one object) suitable for JSONL.

USER:
Evaluate the bureau statements below and determine how actionable this field is for a consumer dispute.

Decision outcomes:
- "strong_actionable": Alone, this discrepancy materially alters consumer treatment (e.g., charged-off vs paid) and justifies a formal dispute.
- "supportive_needs_companion": Meaningful difference that supports a dispute but needs another strong field to stand on.
- "neutral_context_only": Minor or contextual wording differences that do not change the legal meaning on their own.
- "no_case": Difference lacks dispute value or cannot be substantiated.

Evaluation guidance:
- Minor descriptive or redundant wording (e.g., "real estate mortgage" vs "conventional real estate mortgage") ⇒ neutral_context_only.
- Clear categorical conflicts (e.g., "secured loan" vs "unsecured loan", "charged-off" vs "paid as agreed") ⇒ strong_actionable.
- Same concept with different synonyms (e.g., "auto loan" vs "vehicle loan") ⇒ neutral_context_only.
- Differences like "mortgage" vs "HELOC" can be supportive_needs_companion when they signal a shift in product type but need corroboration.
- Prefer normalized values when available; otherwise cite raw text.

Output exactly one JSON object with the following shape:
{
  "sid": string,
  "account_id": number,
  "id": string,
  "field": string,
  "decision": "strong_actionable" | "supportive_needs_companion" | "neutral_context_only" | "no_case",
  "rationale": string,   // ≤120 words and MUST include the exact reason_code
  "citations": string[], // ≥1 items, each "<bureau>: <normalized OR raw>"
  "reason_code": string,
  "reason_label": string,
  "modifiers": {
    "material_mismatch": boolean,
    "time_anchor": boolean,
    "doc_dependency": boolean
  },
  "confidence": number    // 0.0–1.0
}

Modifiers:
- Set modifiers.material_mismatch=true only when the wording changes consumer obligations/rights.
- Set modifiers.time_anchor=true if the field itself pins a ≥18 month timeline relevant to the dispute.
- Set modifiers.doc_dependency=true only when specific documents from documents_required are essential; otherwise false.

Context:
- sid: {{sid}}
- reason_code: {{reason_code}}
- reason_label: {{reason_label}}
- documents_required: {{documents | join(", ")}}

Field finding (verbatim JSON):
{{finding_json}}

Hard constraints:
- Output JSON only (no prose), ONE object.
- Rationale MUST contain {{reason_code}} literally and explain the legal significance (not detection mechanics).
- Citations MUST name at least one bureau you relied on, e.g. "equifax: charged-off".
- Assume the consumer stands by the most favorable bureau wording when assessing dispute value.

Rendering details:

Use your existing Jinja/string formatting to inject sid, reason_code, reason_label, documents, and finding_json (the exact finding blob).
"""


def _normalize_documents(documents: Any) -> list[str]:
    if isinstance(documents, str):
        text = documents.strip()
        return [text] if text else []
    if isinstance(documents, Sequence) and not isinstance(documents, (bytes, bytearray, str)):
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
    return []


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _stringify_finding(finding: Any) -> str:
    if isinstance(finding, str) and finding.strip():
        return finding.strip()
    try:
        return json.dumps(finding or {}, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({}, ensure_ascii=False)


def _replace(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def render_validation_prompt(
    *,
    sid: Any,
    reason_code: Any,
    reason_label: Any,
    documents: Any,
    finding: Any,
) -> tuple[str, str]:
    """Render the validation prompt using the provided context."""

    sid_text = _stringify(sid)
    reason_code_text = _stringify(reason_code)
    reason_label_text = _stringify(reason_label)
    documents_list = _normalize_documents(documents)
    documents_text = ", ".join(documents_list)
    finding_json = _stringify_finding(finding)

    rendered = _replace(
        _VALIDATION_PROMPT_TEMPLATE,
        sid=sid_text,
        reason_code=reason_code_text,
        reason_label=reason_label_text,
        documents=documents_text,
        finding_json=finding_json,
    )

    system_marker = "SYSTEM:\n"
    user_marker = "\nUSER:\n"

    if not rendered.startswith(system_marker):
        raise ValueError("Rendered prompt missing SYSTEM header")
    try:
        system_part, user_part = rendered.split(user_marker, 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("Rendered prompt missing USER section") from exc

    system_prompt = system_part[len(system_marker) :]
    user_prompt = user_part

    return system_prompt, user_prompt


__all__ = ["render_validation_prompt"]

