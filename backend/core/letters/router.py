from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Literal, Tuple

from backend.analytics.analytics_tracker import (
    check_canary_guardrails,
    log_canary_decision,
)
from backend.telemetry.metrics import emit_counter
from jinja2 import Environment, FileSystemLoader

from . import validators

logger = logging.getLogger(__name__)
TEMPLATES_DIRS = [
    Path(__file__).resolve().parents[2] / "assets" / "templates",
    Path(__file__).resolve().parents[3] / "letters" / "templates",
]

# Build a single Jinja environment and eagerly load templates so that each
# routing request avoids hitting the filesystem. Jinja's environment and cache
# are thread-safe as long as templates aren't mutated after load.
ENV = Environment(
    loader=FileSystemLoader([str(p) for p in TEMPLATES_DIRS]),
    trim_blocks=True,
    lstrip_blocks=True,
)
for base in TEMPLATES_DIRS:
    if base.exists():
        for tpl in base.glob("*.html"):
            try:  # pragma: no cover - preload best effort
                ENV.get_template(tpl.name)
            except Exception:  # pragma: no cover - preload best effort
                logger.exception("template preload failed for %s", tpl)


@dataclass
class TemplateDecision:
    template_path: str | None
    required_fields: List[str]
    missing_fields: List[str]
    router_mode: str


_ROUTER_CACHE: Dict[Tuple[str, str, str], TemplateDecision] = {}
_ROUTER_CACHE_LOCK = Lock()


def _enabled(phase: Literal["candidate", "final", "finalize"] = "candidate") -> bool:
    """Return True if the canary router should handle the request.

    Rollback: set environment variable ``ROUTER_CANARY_PERCENT=0`` to disable.
    Set ``FINALIZE_ROUTER_PHASED=0`` to bypass finalize routing, validators, and
    sanitizers.
    """

    ceiling = float(os.getenv("ROUTER_RENDER_MS_P95_CEILING", "250"))
    sanitizer_limit = float(os.getenv("ROUTER_SANITIZER_RATE_CAP", "1.0"))
    ai_cap = float(os.getenv("ROUTER_AI_DAILY_BUDGET", "100000"))
    if check_canary_guardrails(ceiling, sanitizer_limit, ai_cap):
        return False

    if "ROUTER_CANARY_PERCENT" not in os.environ and os.getenv(
        "LETTERS_ROUTER_PHASED", "",
    ).lower() in {"1", "true", "yes"}:
        base_percent = 100
    else:
        try:
            base_percent = int(os.getenv("ROUTER_CANARY_PERCENT", "0"))
        except ValueError:
            base_percent = 0
    base_percent = max(0, min(100, base_percent))
    if base_percent <= 0:
        return False
    if base_percent < 100 and random.randint(1, 100) > base_percent:
        return False

    if phase == "finalize":
        try:
            percent = int(os.getenv("FINALIZE_ROUTER_PHASED", "100"))
        except ValueError:
            percent = 100
        percent = max(0, min(100, percent))
        if percent <= 0:
            return False
        if percent < 100:
            return random.randint(1, 100) <= percent

    return True


def select_template(
    action_tag: str,
    ctx: dict,
    phase: Literal["candidate", "final", "finalize"],
    session_id: str | None = None,
) -> TemplateDecision:
    """Return the template selection for ``action_tag``.

    When ``LETTERS_ROUTER_PHASED`` is not set the router simply mirrors the
    previous hard-coded template choices so behavior remains unchanged.
    """

    tag = (action_tag or "").lower()
    session_id = session_id or ctx.get("session_id") or ""
    cache_key = (session_id, tag, phase)
    if session_id:
        with _ROUTER_CACHE_LOCK:
            if cache_key in _ROUTER_CACHE:
                return _ROUTER_CACHE[cache_key]

    routes = {
        "dispute": ("dispute_letter_template.html", ["bureau"]),
        "goodwill": ("goodwill_letter_template.html", ["creditor"]),
        "custom_letter": ("general_letter_template.html", ["recipient"]),
        "instruction": (
            "instruction_template.html",
            ["client_name", "date", "accounts_summary", "per_account_actions"],
        ),
        "fraud_dispute": (
            "fraud_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "is_identity_theft",
            ],
        ),
        "debt_validation": (
            "debt_validation_letter_template.html",
            [
                "collector_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "days_since_first_contact",
            ],
        ),
        "pay_for_delete": (
            "pay_for_delete_letter_template.html",
            [
                "collector_name",
                "account_number_masked",
                "legal_safe_summary",
                "offer_terms",
            ],
        ),
        "mov": (
            "mov_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "legal_safe_summary",
                "cra_last_result",
                "days_since_cra_result",
            ],
        ),
        "personal_info_correction": (
            "personal_info_correction_letter_template.html",
            [
                "client_name",
                "client_address_lines",
                "date_of_birth",
                "ssn_last4",
                "legal_safe_summary",
            ],
        ),
        "cease_and_desist": (
            "cease_and_desist_letter_template.html",
            ["collector_name", "account_number_masked", "legal_safe_summary"],
        ),
        "direct_dispute": (
            "direct_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "legal_safe_summary",
                "furnisher_address",
            ],
        ),
        "bureau_dispute": (
            "bureau_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
            ],
        ),
        "inquiry_dispute": (
            "inquiry_dispute_letter_template.html",
            [
                "inquiry_creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "inquiry_date",
            ],
        ),
        "medical_dispute": (
            "medical_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "amount",
                "medical_status",
            ],
        ),
        "paydown_first": (
            "instruction_template.html",
            ["client_name", "date", "accounts_summary", "per_account_actions"],
        ),
    }

    def _cache_and_return(decision: TemplateDecision) -> TemplateDecision:
        if session_id:
            with _ROUTER_CACHE_LOCK:
                _ROUTER_CACHE[cache_key] = decision
        return decision

    if tag in {"ignore", "paydown_first", "duplicate"}:
        emit_counter(f"router.skipped.{tag}")
    if tag == "ignore":
        return _cache_and_return(
            TemplateDecision(
                template_path=None,
                required_fields=[],
                missing_fields=[],
                router_mode="skip",
            )
        )

    if tag == "duplicate":
        if phase == "candidate":
            emit_counter("router.candidate_selected")
            emit_counter("router.candidate_selected.duplicate")
        elif phase in {"final", "finalize"}:
            emit_counter("router.finalized")
            emit_counter("router.finalized.duplicate")
        return _cache_and_return(
            TemplateDecision(
                template_path=None,
                required_fields=[],
                missing_fields=[],
                router_mode="memo",
            )
        )

    template_path, required = routes.get(tag, (None, []))

    if not _enabled(phase):
        log_canary_decision("legacy", template_path or "unknown")
        return _cache_and_return(
            TemplateDecision(
                template_path=template_path,
                required_fields=required,
                missing_fields=[],
                router_mode="bypass",
            )
        )

    if template_path is None:
        msg = f"Unknown action_tag '{action_tag}' for session '{session_id}'"
        emit_counter("router.candidate_errors")
        logger.error(msg)
        raise ValueError(msg)

    if tag == "bureau_dispute" and phase == "finalize":
        bureau = (ctx.get("bureau") or "").replace(" ", "").lower()
        if bureau:
            specific = f"{bureau}_bureau_dispute_letter_template.html"
            if any((base / specific).exists() for base in TEMPLATES_DIRS):
                template_path = specific
            else:
                template_path = "bureau_dispute_letter_template.html"

    if not any((base / template_path).exists() for base in TEMPLATES_DIRS):
        msg = f"Unknown template_name '{template_path}' for session '{session_id}'"
        emit_counter("router.candidate_errors")
        logger.error(msg)
        raise ValueError(msg)

    log_canary_decision("canary", template_path or "unknown")

    missing_fields: List[str] = []
    if template_path and phase == "finalize":
        missing_fields = set(
            validators.validate_required_fields(
                template_path, ctx, required, validators.CHECKLIST
            )
        )
        missing_fields.update(validators.validate_substance(template_path, ctx))
        missing_fields = sorted(missing_fields)
        if not missing_fields:
            try:
                # Rendering is performed to ensure template syntax errors are surfaced
                # early. The environment is shared across threads and caches compiled
                # templates, minimizing per-letter I/O.
                ENV.get_template(template_path).render(**ctx)
            except Exception:  # pragma: no cover - render guard
                logger.exception("template render failed for %s", template_path)
                emit_counter("router.render_error")
                return _cache_and_return(
                    TemplateDecision(
                        template_path=None,
                        required_fields=required,
                        missing_fields=missing_fields,
                        router_mode="error",
                    )
                )
    elif template_path and phase == "candidate":
        if tag == "instruction":
            missing_fields = []
        else:
            missing_fields = validators.validate_required_fields(
                template_path, ctx, required, validators.CHECKLIST
            )
            missing_fields = sorted(set(missing_fields))

    if template_path:
        template_name = os.path.basename(template_path)
        # Emit both legacy and tag-specific router metrics for transition
        if phase == "candidate":
            emit_counter("router.candidate_selected")  # deprecated
            if tag:
                emit_counter(f"router.candidate_selected.{tag}")
                emit_counter(f"router.candidate_selected.{tag}.{template_name}")
        elif phase in {"final", "finalize"}:
            emit_counter("router.finalized")  # deprecated
            if tag:
                emit_counter(f"router.finalized.{tag}")
                emit_counter(f"router.finalized.{tag}.{template_name}")

        if missing_fields:
            for field in missing_fields:
                emit_counter(f"router.missing_fields.{tag}.{template_name}.{field}")
                if phase == "finalize":
                    emit_counter(
                        f"router.missing_fields.finalize.{tag}.{field}"
                    )

    return _cache_and_return(
        TemplateDecision(
            template_path=template_path,
            required_fields=required,
            missing_fields=missing_fields,
            router_mode="auto_route",
        )
    )


def route_accounts(
    items: Iterable[tuple[str, dict, str | None]],
    *,
    phase: Literal["candidate", "final", "finalize"] = "candidate",
    max_workers: int | None = None,
) -> list[TemplateDecision]:
    """Route multiple accounts in parallel.

    Parameters
    ----------
    items:
        Iterable of ``(action_tag, ctx, session_id)`` tuples for each account.
    phase:
        Routing phase passed through to :func:`select_template`.
    max_workers:
        Optional limit for the thread pool size.

    Returns
    -------
    list[TemplateDecision]
        Decisions in the same order as ``items``.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(select_template, tag, ctx, phase, session_id)
            for tag, ctx, session_id in items
        ]
        return [f.result() for f in futures]


__all__ = ["TemplateDecision", "select_template", "route_accounts"]
