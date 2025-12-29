"""Rendering helpers for goodwill letters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping

from backend.assets.paths import data_path
from backend.audit.audit import AuditLevel, AuditLogger
from backend.core.logic.compliance.compliance_pipeline import (
    run_compliance_pipeline as default_compliance,
)
from backend.core.logic.rendering.pdf_renderer import (
    ensure_template_env,
    render_html_to_pdf as default_pdf_renderer,
)
from backend.core.logic.utils.file_paths import safe_filename
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.note_handling import get_client_address_lines
from backend.core.models.client import ClientInfo
from backend.core.services.ai_client import AIClient
from backend.telemetry.metrics import emit_counter
from backend.api.config import env_bool
from backend.core.letters.client_context import format_safe_client_context
from backend.core.letters import validators
from backend.core.letters.sanitizer import sanitize_rendered_html


def load_creditor_address_map() -> Mapping[str, str]:
    """Load a mapping of normalized creditor names to addresses."""
    try:
        with open(data_path("creditor_addresses.json"), encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return {
                    normalize_creditor_name(entry["name"]): entry["address"]
                    for entry in raw
                    if "name" in entry and "address" in entry
                }
            if isinstance(raw, dict):
                return {normalize_creditor_name(k): v for k, v in raw.items()}
            print("[WARN] Unknown address file format.")
            return {}
    except Exception as e:  # pragma: no cover - file IO issues
        print(f"[ERROR] Failed to load creditor addresses: {e}")
        return {}


def render_goodwill_letter(
    creditor: str,
    gpt_data: Mapping[str, Any],
    client_info: ClientInfo | Mapping[str, Any],
    output_path: Path,
    run_date: str | None = None,
    *,
    doc_names: List[str] | None = None,
    ai_client: AIClient,
    audit: AuditLogger | None = None,
    compliance_fn=default_compliance,
    pdf_fn=default_pdf_renderer,
    template_path: str,
) -> None:
    """Render a goodwill letter using ``gpt_data`` and save a PDF and JSON."""

    if not template_path:
        emit_counter("rendering.missing_template_path")
        raise ValueError("template_path is required")

    client_name = client_info.get("legal_name") or client_info.get("name", "Your Name")
    if not client_info.get("legal_name"):
        print(
            "[WARN] Warning: legal_name not found in client_info. Using fallback name."
        )

    date_str = run_date or datetime.now().strftime("%B %d, %Y")
    address_map = load_creditor_address_map()
    creditor_key = normalize_creditor_name(creditor)
    creditor_address = address_map.get(creditor_key)
    if not creditor_address:
        print(f"[WARN] No address found for: {creditor}")
        creditor_address = "Address not provided - please enter manually"

    session_id = client_info.get("session_id") or ""

    context = {
        "date": date_str,
        "client_name": client_name,
        "client_address_lines": get_client_address_lines(client_info),
        "client": type("C", (), {
            "full_name": client_name,
            "address_line": ", ".join(get_client_address_lines(client_info)),
        })(),
        "creditor": creditor,
        "creditor_address": creditor_address,
        "accounts": gpt_data.get("accounts", []),
        "intro_paragraph": gpt_data.get("intro_paragraph", ""),
        "hardship_paragraph": gpt_data.get("hardship_paragraph", ""),
        "recovery_paragraph": gpt_data.get("recovery_paragraph", ""),
        "closing_paragraph": gpt_data.get("closing_paragraph", ""),
        "supporting_docs": doc_names or [],
    }
    if env_bool("SAFE_CLIENT_SENTENCE_ENABLED", False):
        sentence = format_safe_client_context("goodwill", "", {}, [])
        if sentence:
            context["client_context_sentence"] = sentence
    missing = validators.validate_substance(template_path, context)
    if missing:
        for field in missing:
            emit_counter(f"validation.failed.{template_path}.{field}")
        return

    env = ensure_template_env()
    template = env.get_template(template_path)
    html = template.render(**context)
    emit_counter(f"letter_template_selected.{template_path}")
    if doc_names:
        html += "".join(doc_names)
    compliance_fn(
        html,
        client_info.get("state"),
        session_id,
        "goodwill",
        ai_client=ai_client,
    )

    html, _ = sanitize_rendered_html(html, template_path, context)

    safe_name = safe_filename(creditor)
    pdf_path = output_path / f"Goodwill Request - {safe_name}.pdf"
    pdf_fn(html, str(pdf_path), template_name=template_path)

    with open(output_path / f"{safe_name}_gpt_response.json", "w") as f:
        json.dump(gpt_data, f, indent=2)

    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "goodwill_letter_generated",
            {"creditor": creditor, "output_pdf": str(pdf_path), "response": gpt_data},
        )


__all__ = ["render_goodwill_letter", "load_creditor_address_map"]
