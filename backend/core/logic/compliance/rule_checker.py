from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from backend.core.logic.compliance.rules_loader import load_rules, load_state_rules
from backend.core.logic.utils.pii import redact_pii
from backend.core.models.letter import LetterContext


class RuleViolation(TypedDict):
    rule_id: str
    severity: Literal["critical", "warning"]
    span: tuple[int, int] | None
    message: str


def check_letter(
    text: str, state: str | None, context: LetterContext | dict[str, Any]
) -> tuple[str, list[RuleViolation]]:
    """
    Returns (possibly_fixed_text, violations)
    - Load systemic rules
    - Scan for block_patterns; for each match:
        * If fix_template exists → replace
        * Else → record violation
    - Mask PII according to RULE_PII_LIMIT
    - Append state-specific clauses if applicable
    - Return modified text + list of violations
    """
    rules = load_rules()
    state_rules = load_state_rules()

    modified_text = text
    violations: list[RuleViolation] = []

    # PII masking first
    for rule in rules:
        if rule.get("id") != "RULE_PII_LIMIT":
            continue
        for pattern in rule.get("block_patterns", []):
            regex = re.compile(pattern)
            matches = list(regex.finditer(modified_text))
            for m in matches:
                violations.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "warning"),
                        "span": m.span(),
                        "message": rule.get("description", ""),
                    }
                )
    modified_text = redact_pii(modified_text)

    # Apply other systemic rules
    for rule in rules:
        if rule.get("id") == "RULE_PII_LIMIT":
            continue
        for pattern in rule.get("block_patterns", []):
            regex = re.compile(pattern, flags=re.IGNORECASE)
            matches = list(regex.finditer(modified_text))
            for m in matches:
                violations.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "warning"),
                        "span": m.span(),
                        "message": rule.get("description", ""),
                    }
                )
            if matches and rule.get("fix_template"):
                modified_text = regex.sub(rule["fix_template"], modified_text)

    # Append state-specific clauses and handle prohibitions
    if state:
        state_key = state.upper()
        state_data = state_rules.get(state_key, {})
        clauses_added: list[str] = []

        if state_data.get("prohibit_service"):
            violations.append(
                {
                    "rule_id": "STATE_PROHIBITED",
                    "severity": "critical",
                    "span": None,
                    "message": f"Service is not available in {state_key}",
                }
            )

        # Insert state-specific clauses before the closing signature
        debt_type = context.get("debt_type")
        clause_texts: list[str] = []
        clauses_config = state_data.get("clauses", {})
        if debt_type and isinstance(clauses_config, dict):
            clause_info = clauses_config.get(debt_type)
            if clause_info:
                reference = clause_info.get("reference", "")
                text_clause = clause_info.get("text", "")
                sentence = (
                    f"Additionally, pursuant to {reference}, {text_clause}".rstrip()
                )
                if not sentence.endswith("."):
                    sentence += "."
                clause_texts.append(sentence)
                clauses_added.append(clause_info.get("label", text_clause))

        if clause_texts:
            lines = modified_text.splitlines()
            if len(lines) >= 2:
                signature_line = lines[-1]
                closing_line = lines[-2]
                body = "\n".join(lines[:-2]).rstrip()
                if body:
                    body += "\n"
                body += "\n".join(clause_texts)
                modified_text = body + "\n" + closing_line + "\n" + signature_line
            else:
                if not modified_text.endswith("\n"):
                    modified_text += "\n"
                modified_text += "\n".join(clause_texts)

        # Append required disclosures after the signature
        disclosures = state_data.get("disclosures") or []
        if disclosures:
            if not modified_text.endswith("\n"):
                modified_text += "\n"
            modified_text += "\n".join(disclosures)
            clauses_added.extend(disclosures)

        # Log clause additions to session if session_id provided
        session_id = context.get("session_id")
        if session_id and clauses_added:
            try:
                from backend.api.session_manager import update_session

                update_session(
                    session_id,
                    state_compliance={
                        "state": state_key,
                        "clauses_added": clauses_added,
                    },
                )
            except Exception:
                pass

    return modified_text, violations
