from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pdfkit
from jinja2 import Environment, FileSystemLoader

from backend.telemetry.metrics import emit_counter
from backend.api.config import get_app_config
from backend.assets.paths import templates_path
from backend.audit.audit import AuditLogger, emit_event
from backend.core.letters import validators
from backend.core.letters.client_context import format_safe_client_context
from backend.core.letters.router import select_template
from backend.core.letters.sanitizer import sanitize_rendered_html
from backend.core.logic.letters.exceptions import StrategyContextMissing
from backend.core.logic.letters.utils import ensure_strategy_context, populate_required_fields
from backend.core.logic.utils.note_handling import get_client_address_lines
from backend.core.models.account import Account
from backend.core.models.client import ClientInfo


env = Environment(loader=FileSystemLoader(templates_path("")))


def _pdf_config(wkhtmltopdf_path: str | None):
    path = wkhtmltopdf_path or get_app_config().wkhtmltopdf_path
    return pdfkit.configuration(wkhtmltopdf=path)


def generate_debt_validation_letter(
    account: Account | Mapping[str, Any],
    client_info: ClientInfo | Mapping[str, Any],
    output_path: Path,
    audit: AuditLogger | None,
    *,
    run_date: str | None = None,
    wkhtmltopdf_path: str | None = None,
) -> None:
    try:
        ensure_strategy_context([account], True)
    except StrategyContextMissing as exc:  # pragma: no cover - enforcement
        emit_event(
            "strategy_context_missing",
            {
                "account_id": exc.args[0] if exc.args else None,
                "letter_type": "debt_validation",
            },
        )
        raise

    acc = account if isinstance(account, Mapping) else account.to_dict()
    client = client_info if isinstance(client_info, Mapping) else client_info.to_dict()

    populate_required_fields(acc)

    select_template(
        "debt_validation",
        {"collector_name": acc.get("collector_name", "")},
        phase="candidate",
    )

    context = {
        "client": {
            "full_name": client.get("legal_name") or client.get("name", "Client"),
            "address_line": ", ".join(get_client_address_lines(client)),
        },
        "today": run_date or datetime.now().strftime("%B %d, %Y"),
        "collector_name": acc.get("collector_name", ""),
        "account_number_masked": acc.get("account_number_masked", ""),
        "bureau": acc.get("bureau", ""),
        "legal_safe_summary": acc.get("legal_safe_summary", ""),
        "days_since_first_contact": acc.get("days_since_first_contact", ""),
        "client_context_sentence": format_safe_client_context(
            "debt_validation", "", {}, []
        ),
    }

    decision = select_template("debt_validation", context, phase="finalize")
    if not decision.template_path:
        raise ValueError("router did not supply template_path")
    if decision.missing_fields:
        for field in decision.missing_fields:
            emit_counter(f"validation.failed.{decision.template_path}.{field}")
        return

    missing = validators.validate_substance(decision.template_path, context)
    if missing:
        for field in missing:
            emit_counter(f"validation.failed.{decision.template_path}.{field}")
        return

    tmpl = env.get_template(decision.template_path)
    html = tmpl.render(**context)
    emit_counter(f"letter_template_selected.{decision.template_path}")
    html, _ = sanitize_rendered_html(html, decision.template_path, context)

    safe_collector = (
        (context["collector_name"] or "Collector").replace("/", "_").replace("\\", "_")
    )
    filename = f"Debt Validation Letter - {safe_collector}.pdf"
    full_path = output_path / filename
    options = {"quiet": ""}
    pdfkit.from_string(
        html,
        str(full_path),
        configuration=_pdf_config(wkhtmltopdf_path),
        options=options,
    )
    print(f"[INFO] Debt validation letter generated: {full_path}")


def generate_debt_validation_letters(
    client_info: ClientInfo | Mapping[str, Any],
    bureau_data: Mapping[str, Any],
    output_path: Path,
    audit: AuditLogger | None,
    *,
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
            if (
                acc.get("letter_type") == "debt_validation"
                or action == "debt_validation"
            ):
                generate_debt_validation_letter(
                    acc,
                    client_info,
                    output_path,
                    audit,
                    run_date=run_date,
                    wkhtmltopdf_path=wkhtmltopdf_path,
                )
            else:
                log_messages.append(
                    f"[{bureau}] No debt validation letter for '{acc.get('name')}' - not marked for debt validation"
                )
