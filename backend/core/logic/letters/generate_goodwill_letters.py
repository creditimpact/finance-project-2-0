"""Backward-compatible orchestrator for goodwill letter generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import backend.core.logic.letters.goodwill_preparation as goodwill_preparation
import backend.core.logic.letters.goodwill_prompting as goodwill_prompting
import backend.core.logic.letters.goodwill_rendering as goodwill_rendering
from backend.analytics.analytics_tracker import (
    log_letter_without_strategy,
    log_policy_override_reason,
)
from backend.api import config as api_config
from backend.api.session_manager import get_session
from backend.audit.audit import AuditLogger, emit_event
from backend.core.letters.router import select_template
from backend.core.logic.compliance.compliance_pipeline import run_compliance_pipeline
from backend.core.logic.guardrails.summary_validator import (
    validate_structured_summaries,
)
from backend.core.logic.rendering import pdf_renderer
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.pdf_ops import gather_supporting_docs
from backend.core.models import BureauPayload, ClientInfo
from backend.core.services.ai_client import AIClient

from .exceptions import StrategyContextMissing
from .utils import ensure_strategy_context, populate_required_fields

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _apply_strategy_fields(
    bureau_data: Mapping[str, Any], strategy_accounts: list[dict[str, Any]]
) -> None:
    """Merge strategist metadata into ``bureau_data`` accounts."""

    def _norm(name: str) -> str:
        return normalize_creditor_name(name or "")

    def _last4(num: str | None) -> str:
        digits = "".join(c for c in str(num or "") if c.isdigit())
        return digits[-4:]

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for acc in strategy_accounts:
        key = (_norm(acc.get("name", "")), _last4(acc.get("account_number")))
        index[key] = acc

    for payload in bureau_data.values():
        for section in payload.values():
            if isinstance(section, list):
                for acc in section:
                    key = (
                        _norm(acc.get("name", "")),
                        _last4(acc.get("account_number")),
                    )
                    strat = index.get(key)
                    before = acc.get("action_tag")
                    applied = False
                    override_reason = ""
                    if strat:
                        applied = True
                        override_reason = strat.get("policy_override_reason", "")
                        for field in [
                            "action_tag",
                            "priority",
                            "needs_evidence",
                            "legal_notes",
                            "flags",
                        ]:
                            if strat.get(field) is not None and not acc.get(field):
                                acc[field] = strat[field]
                    populate_required_fields(acc, strat)
                    emit_event(
                        "strategy_applied",
                        {
                            "account_id": acc.get("account_id"),
                            "strategy_applied": applied,
                            "action_tag_before": before,
                            "action_tag_after": acc.get("action_tag"),
                            "override_reason": override_reason,
                        },
                    )


# ---------------------------------------------------------------------------
# Orchestrator functions
# ---------------------------------------------------------------------------


def generate_goodwill_letter_with_ai(
    creditor: str,
    accounts: list[Account],
    client: ClientInfo,
    output_path: Path,
    run_date: str | None = None,
    audit: AuditLogger | None = None,
    *,
    ai_client: AIClient,
    classification_map: Mapping[str, ClassificationRecord] | None = None,
    strategy: Mapping[str, Any] | None = None,
) -> None:
    """Generate a single goodwill letter for ``creditor``."""

    if not strategy:
        emit_event(
            "goodwill_policy_override",
            {"policy_override_reason": "collection_no_goodwill"},
        )
        log_letter_without_strategy()
        log_policy_override_reason("collection_no_goodwill")
        return

    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    account_dicts = [
        a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in accounts
    ]

    for acc in account_dicts:
        if acc.get("action_tag") == "collection":
            emit_event(
                "goodwill_policy_override",
                {
                    "policy_override_reason": "collection_no_goodwill",
                    "account_id": acc.get("account_id"),
                },
            )
            log_policy_override_reason("collection_no_goodwill")
            return
    select_template("goodwill", {"creditor": creditor}, phase="candidate")
    client_info = client.to_dict()
    session_id = client_info.get("session_id")
    session = get_session(session_id or "") or {}
    structured_summaries = session.get("structured_summaries", {})
    structured_summaries = validate_structured_summaries(structured_summaries)

    account_summaries = goodwill_preparation.prepare_account_summaries(
        account_dicts,
        structured_summaries,
        classification_map,
        client_info.get("state"),
        session_id,
        audit=audit,
    )

    gpt_data, _ = goodwill_prompting.generate_goodwill_letter_draft(
        client_info.get("legal_name") or client_info.get("name", "Your Name"),
        creditor,
        account_summaries,
        tone=client_info.get("tone", "neutral"),
        session_id=session_id,
        ai_client=ai_client,
        audit=audit,
    )

    _, doc_names, _ = gather_supporting_docs(session_id or "")

    decision = select_template("goodwill", {"creditor": creditor}, phase="finalize")
    if not decision.template_path:
        raise ValueError("router did not supply template_path")
    goodwill_rendering.render_goodwill_letter(
        creditor,
        gpt_data,
        client_info,
        output_path,
        run_date,
        doc_names=doc_names,
        ai_client=ai_client,
        audit=audit,
        compliance_fn=run_compliance_pipeline,
        pdf_fn=pdf_renderer.render_html_to_pdf,
        template_path=decision.template_path,
    )


def generate_goodwill_letters(
    client: ClientInfo,
    bureau_map: Mapping[str, BureauPayload | dict[str, Any]],
    output_path: Path,
    audit: AuditLogger | None,
    run_date: str | None = None,
    *,
    ai_client: AIClient,
    identity_theft: bool = False,
    classification_map: Mapping[str, ClassificationRecord] | None = None,
    strategy: Mapping[str, Any] | None = None,
) -> None:
    """Generate goodwill letters for all eligible creditors in ``bureau_data``.

    Parameters
    ----------
    identity_theft:
        When ``True`` the function returns immediately without generating any
        letters.  This mirrors the higher level orchestration logic and acts as
        a defensive guard should callers invoke this helper directly.
    """

    if identity_theft:
        return

    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    client_info = client.to_dict()
    bureau_data = {
        k: (v.to_dict() if isinstance(v, BureauPayload) else dict(v))
        for k, v in bureau_map.items()
    }

    if strategy:
        _apply_strategy_fields(bureau_data, strategy.get("accounts", []))

    all_accounts: list[dict] = []
    for payload in bureau_data.values():
        for section in payload.values():
            if isinstance(section, list):
                all_accounts.extend(
                    a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in section
                )
    try:
        ensure_strategy_context(
            all_accounts,
            api_config.STAGE4_POLICY_ENFORCEMENT,
        )
    except StrategyContextMissing as exc:  # pragma: no cover - enforcement
        emit_event(
            "strategy_context_missing",
            {
                "account_id": exc.args[0] if exc.args else None,
                "letter_type": "goodwill",
            },
        )
        raise

    goodwill_accounts = goodwill_preparation.select_goodwill_candidates(
        client_info, bureau_data, strategy=strategy
    )
    for creditor, accounts in goodwill_accounts.items():
        account_dicts = [
            a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in accounts
        ]
        generate_goodwill_letter_with_ai(
            creditor,
            account_dicts,
            client,
            output_path,
            run_date,
            audit,
            ai_client=ai_client,
            classification_map=classification_map,
            strategy=strategy,
        )


__all__ = [
    "generate_goodwill_letter_with_ai",
    "generate_goodwill_letters",
]
