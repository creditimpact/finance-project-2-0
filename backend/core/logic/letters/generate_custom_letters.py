from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pdfkit
from jinja2 import Environment, FileSystemLoader

from backend.analytics.analytics_tracker import (
    log_letter_without_strategy,
    log_policy_override_reason,
)
from backend.telemetry.metrics import emit_counter
from backend.api import config as api_config
from backend.api.config import get_app_config
from backend.api.session_manager import get_session
from backend.assets.paths import templates_path
from backend.audit.audit import AuditLevel, AuditLogger, emit_event
from backend.core.letters import validators
from backend.core.letters.router import select_template
from backend.core.letters.sanitizer import sanitize_rendered_html
from backend.core.logic.compliance.rules_loader import get_neutral_phrase
from backend.core.logic.guardrails import generate_letter_with_guardrails
from backend.core.logic.guardrails.summary_validator import (
    validate_structured_summaries,
)
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.logic.utils.pdf_ops import gather_supporting_docs
from backend.core.models.account import Account
from backend.core.models.bureau import BureauPayload
from backend.core.models.client import ClientInfo
from backend.core.services.ai_client import AIClient

from .exceptions import StrategyContextMissing
from .utils import ensure_strategy_context

env = Environment(loader=FileSystemLoader(templates_path("")))


def _pdf_config(wkhtmltopdf_path: str | None):
    path = wkhtmltopdf_path or get_app_config().wkhtmltopdf_path
    return pdfkit.configuration(wkhtmltopdf=path)


def call_gpt_for_custom_letter(
    client_name: str,
    recipient_name: str,
    account_name: str,
    account_number: str,
    docs_text: str,
    structured_summary: Mapping[str, Any],
    classification_record: ClassificationRecord | None,
    state: str,
    session_id: str,
    audit: AuditLogger | None,
    ai_client: AIClient,
) -> str:
    docs_line = f"Supporting documents summary:\n{docs_text}" if docs_text else ""

    classification = (
        classification_record.classification if classification_record else {}
    )
    debt_type = structured_summary.get("debt_type")
    dispute_reason = classification.get("category")
    if not (debt_type and dispute_reason):
        log_letter_without_strategy()
        if not api_config.ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY:
            return "strategy_context_required"
        emit_event("strategy_applied", {"strategy_applied": False})
        debt_type = debt_type or "unknown"
        dispute_reason = dispute_reason or "unknown"
    neutral_phrase, neutral_reason = get_neutral_phrase(
        dispute_reason, structured_summary
    )
    prompt = f"""
Neutral legal phrase for this dispute type:
"{neutral_phrase or ''}"

Here is what the client explained about this account (structured summary):
{json.dumps(structured_summary, indent=2)}
Classification: {json.dumps(classification)}
Client name: {client_name}
Recipient: {recipient_name}
State: {state}
Account: {account_name} {account_number}
{docs_line}
Please draft a compliant letter body that blends the neutral legal phrase with the client's explanation. Do not copy either source verbatim.
"""
    if audit:
        audit.log_account(
            structured_summary.get("account_id"),
            {
                "stage": "custom_letter",
                "classification": classification,
                "neutral_phrase": neutral_phrase,
                "neutral_phrase_reason": neutral_reason,
                "structured_summary": structured_summary,
            },
        )
    body, _, _ = generate_letter_with_guardrails(
        prompt,
        state,
        {
            "debt_type": debt_type,
            "dispute_reason": dispute_reason,
        },
        session_id,
        "custom",
        ai_client=ai_client,
    )
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "custom_letter_prompt",
            {
                "account_id": structured_summary.get("account_id"),
                "prompt": prompt,
            },
        )
        audit.log_step(
            "custom_letter_response",
            {
                "account_id": structured_summary.get("account_id"),
                "response": body,
            },
        )
    return body


def generate_custom_letter(
    account: Account | dict[str, Any],
    client_info: ClientInfo | dict[str, Any],
    output_path: Path,
    audit: AuditLogger | None,
    *,
    ai_client: AIClient,
    classification_map: Mapping[str, ClassificationRecord] | None = None,
    run_date: str | None = None,
    wkhtmltopdf_path: str | None = None,
) -> None:
    try:
        ensure_strategy_context([account], api_config.STAGE4_POLICY_ENFORCEMENT)
    except StrategyContextMissing as exc:  # pragma: no cover - enforcement
        emit_event(
            "strategy_context_missing",
            {
                "account_id": exc.args[0] if exc.args else None,
                "letter_type": "custom",
            },
        )
        raise

    client_name = client_info.get("legal_name") or client_info.get("name", "Client")
    date_str = run_date or datetime.now().strftime("%B %d, %Y")
    recipient = account.get("name", "")
    select_template("custom_letter", {"recipient": recipient}, phase="candidate")
    acc_name = account.get("name", "")
    acc_number = account.get("account_number", "")
    session_id = client_info.get("session_id", "")
    state = client_info.get("state", "")

    session = get_session(session_id) or {}
    structured_summaries = session.get("structured_summaries", {})
    structured_summaries = validate_structured_summaries(structured_summaries)
    structured_summary = structured_summaries.get(account.get("account_id"), {})

    classification_record = (
        classification_map.get(account.get("account_id"))
        if classification_map
        else None
    )
    classification = (
        classification_record.classification if classification_record else {}
    )
    action_before = classification.get("action_tag", "")
    action_after = account.get("action_tag", action_before)
    strategy_applied = bool(
        structured_summary.get("debt_type") and classification.get("category")
    )
    if action_after == "goodwill" and classification.get("category") == "collection":
        emit_event(
            "goodwill_policy_override",
            {
                "policy_override_reason": "collection_no_goodwill",
                "account_id": account.get("account_id"),
            },
        )
        log_letter_without_strategy()
        log_policy_override_reason("collection_no_goodwill")
        if audit:
            audit.log_account(
                account.get("account_id"),
                {
                    "stage": "strategy_rule_enforcement",
                    "policy_override": True,
                    "policy_override_reason": "collection_no_goodwill",
                },
            )
        emit_event(
            "strategy_applied",
            {
                "account_id": account.get("account_id"),
                "strategy_applied": strategy_applied,
                "action_tag_before": action_before,
                "action_tag_after": action_after,
                "override_reason": "collection_no_goodwill",
                "policy_override_reason": "collection_no_goodwill",
            },
        )
        return

    docs_text, doc_names, _ = gather_supporting_docs(session_id)
    if docs_text and audit and audit.level is AuditLevel.VERBOSE:
        print(f"[INFO] Including supplemental docs for custom letter to {recipient}.")

    body_paragraph = call_gpt_for_custom_letter(
        client_name,
        recipient,
        acc_name,
        acc_number,
        docs_text,
        structured_summary,
        classification_record,
        state,
        session_id,
        audit,
        ai_client,
    )

    forbidden_actions = [str(a).lower() for a in account.get("forbidden_actions", [])]
    if "goodwill" in forbidden_actions and "goodwill" in body_paragraph.lower():
        body_paragraph = re.sub(r"(?i)good\s*-?will", "", body_paragraph)
        account["policy_override_reason"] = "custom_prompt_policy_conflict"
        log_policy_override_reason("custom_prompt_policy_conflict")
        emit_event(
            "strategy_applied",
            {
                "account_id": account.get("account_id"),
                "strategy_applied": strategy_applied,
                "action_tag_before": action_before,
                "action_tag_after": action_after,
                "override_reason": "custom_prompt_policy_conflict",
            },
        )
    else:
        emit_event(
            "strategy_applied",
            {
                "account_id": account.get("account_id"),
                "strategy_applied": strategy_applied,
                "action_tag_before": action_before,
                "action_tag_after": action_after,
                "override_reason": account.get("policy_override_reason", ""),
            },
        )

    greeting = f"Dear {recipient}" if recipient else "To whom it may concern"

    context = {
        "date": date_str,
        "client_name": client_name,
        "client_street": client_info.get("street", ""),
        "client_city": client_info.get("city", ""),
        "client_state": client_info.get("state", ""),
        "client_zip": client_info.get("zip", ""),
        "recipient_name": recipient,
        "greeting_line": greeting,
        "body_paragraph": body_paragraph,
        "supporting_docs": doc_names,
    }
    decision = select_template(
        "custom_letter", {"recipient": recipient}, phase="finalize"
    )
    if not decision.template_path:
        raise ValueError("router did not supply template_path")
    missing = validators.validate_substance(decision.template_path, context)
    if missing:
        for field in missing:
            emit_counter(f"validation.failed.{decision.template_path}.{field}")
        return
    tmpl = env.get_template(decision.template_path)
    html = tmpl.render(**context)
    emit_counter(f"letter_template_selected.{decision.template_path}")
    html, _ = sanitize_rendered_html(html, decision.template_path, context)
    safe_recipient = (recipient or "Recipient").replace("/", "_").replace("\\", "_")
    filename = f"Custom Letter - {safe_recipient}.pdf"
    full_path = output_path / filename
    options = {"quiet": ""}
    pdfkit.from_string(
        html,
        str(full_path),
        configuration=_pdf_config(wkhtmltopdf_path),
        options=options,
    )
    print(f"[INFO] Custom letter generated: {full_path}")

    response_path = output_path / f"{safe_recipient}_custom_gpt_response.txt"
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(body_paragraph)

    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "custom_letter_generated",
            {
                "account_id": account.get("account_id"),
                "output_pdf": str(full_path),
                "response": body_paragraph,
            },
        )


def generate_custom_letters(
    client_info: ClientInfo | dict[str, Any],
    bureau_data: BureauPayload | Mapping[str, Any],
    output_path: Path,
    audit: AuditLogger | None,
    *,
    ai_client: AIClient,
    classification_map: Mapping[str, ClassificationRecord] | None = None,
    run_date: str | None = None,
    log_messages: list[str] | None = None,
    wkhtmltopdf_path: str | None = None,
) -> None:
    if log_messages is None:
        log_messages = []
    for bureau, content in bureau_data.items():
        for acc in content.get("all_accounts", []):
            action = str(
                acc.get("action_tag") or acc.get("recommended_action") or ""
            ).lower()
            if acc.get("letter_type") == "custom" or action == "custom_letter":
                generate_custom_letter(
                    acc,
                    client_info,
                    output_path,
                    audit,
                    ai_client=ai_client,
                    classification_map=classification_map,
                    run_date=run_date,
                    wkhtmltopdf_path=wkhtmltopdf_path,
                )
            else:
                log_messages.append(
                    f"[{bureau}] No custom letter for '{acc.get('name')}' - not marked for custom correspondence"
                )
