"""Helpers to populate missing fields prior to template validation."""

from __future__ import annotations

from typing import Any, Mapping

from backend.audit.audit import emit_event
from backend.telemetry.metrics import emit_counter
from backend.core.logic.letters.utils import populate_required_fields
from fields.populate_account_number_masked import populate_account_number_masked
from fields.populate_address import populate_address
from fields.populate_amount import populate_amount
from fields.populate_creditor_name import populate_creditor_name
from fields.populate_days_since_cra_result import populate_days_since_cra_result
from fields.populate_dob import populate_dob
from fields.populate_inquiry_creditor_name import populate_inquiry_creditor_name
from fields.populate_inquiry_date import populate_inquiry_date
from fields.populate_medical_status import populate_medical_status
from fields.populate_name import populate_name
from fields.populate_ssn_masked import populate_ssn_masked

_FILLER_CACHE: dict[tuple[str, str], Any] = {}


def clear_filler_cache() -> None:
    """Clear cached filler outputs."""
    _FILLER_CACHE.clear()

CRITICAL_FIELDS = {
    "name",
    "address",
    "date_of_birth",
    "ssn_masked",
    "creditor_name",
    "account_number_masked",
    "inquiry_creditor_name",
    "inquiry_date",
}

OPTIONAL_FIELDS = {
    "days_since_cra_result",
    "amount",
    "medical_status",
}


def _mark_missing(ctx: dict, tag: str, field: str, reason: str = "missing") -> None:
    emit_event(
        "fields.populate_errors",
        {"tag": tag, "field": field, "reason": reason},
    )
    emit_counter("fields.populate_errors")
    emit_counter(f"fields.populate_errors.tag.{tag}")
    emit_counter(f"fields.populate_errors.field.{field}")
    emit_counter(f"fields.populate_errors.reason.{reason}")
    missing = ctx.setdefault("missing_fields", [])
    if field not in missing:
        missing.append(field)
    if field in CRITICAL_FIELDS:
        critical = ctx.setdefault("critical_missing_fields", [])
        if field not in critical:
            critical.append(field)
        ctx["defer_action_tag"] = True


def apply_field_fillers(
    ctx: dict,
    *,
    strategy: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    corrections: Mapping[str, Any] | None = None,
) -> None:
    """Populate ``ctx`` using available field fillers.

    Parameters
    ----------
    ctx:
        Context dictionary mutated in-place.
    strategy:
        Optional per-account strategy data used by ``populate_required_fields``.
    profile:
        Optional client profile supplying PII fields.
    corrections:
        Optional corrections overriding profile values.
    """

    tri_merge = ctx.get("tri_merge") or {}
    inquiry = ctx.get("inquiry_evidence") or ctx.get("inquiry") or {}
    medical = ctx.get("medical_evidence") or ctx.get("medical") or {}
    outcome = ctx.get("cra_outcome") or ctx.get("outcome") or {}
    profile = profile or ctx.get("profile") or ctx.get("client") or {}
    corrections = corrections or ctx.get("corrections") or {}
    tag = str(ctx.get("action_tag") or "").lower()
    account_id = ctx.get("account_id")

    initial_missing = {
        f for f in CRITICAL_FIELDS | OPTIONAL_FIELDS if not ctx.get(f)
    }
    if account_id is not None:
        key = str(account_id)
        for field in CRITICAL_FIELDS | OPTIONAL_FIELDS:
            if field not in ctx:
                cached = _FILLER_CACHE.get((key, field))
                if cached is not None:
                    ctx[field] = cached

    # Client/profile fields -----------------------------------------------------
    if ctx.get("name") is None:
        populate_name(ctx, profile, corrections)
    if ctx.get("address") is None:
        populate_address(ctx, profile, corrections)
    if ctx.get("date_of_birth") is None:
        populate_dob(ctx, profile, corrections)
    if ctx.get("ssn_masked") is None:
        populate_ssn_masked(ctx, profile, corrections)

    # Evidence-driven account fields -------------------------------------------
    if tri_merge.get("name") and ctx.get("name") is None:
        ctx["name"] = tri_merge["name"]
    if ctx.get("creditor_name") is None:
        populate_creditor_name(ctx, tri_merge)
    if ctx.get("account_number_masked") is None:
        populate_account_number_masked(ctx, tri_merge)
    if ctx.get("days_since_cra_result") is None:
        populate_days_since_cra_result(ctx, outcome, now=ctx.get("now"))
    if ctx.get("inquiry_creditor_name") is None:
        populate_inquiry_creditor_name(ctx, inquiry)
    if ctx.get("inquiry_date") is None:
        populate_inquiry_date(ctx, inquiry)
    if ctx.get("amount") is None:
        populate_amount(ctx, medical)
    if ctx.get("medical_status") is None:
        populate_medical_status(ctx, medical)

    # Strategy provided fields -------------------------------------------------
    populate_required_fields(ctx, strategy)

    if account_id is not None:
        key = str(account_id)
        for field in CRITICAL_FIELDS | OPTIONAL_FIELDS:
            if ctx.get(field) is not None:
                _FILLER_CACHE[(key, field)] = ctx[field]

    # Missing field handling -----------------------------------------------------
    for field in CRITICAL_FIELDS | OPTIONAL_FIELDS:
        if field in initial_missing and ctx.get(field):
            emit_counter("fields.populated_total")
            emit_counter(f"fields.populated_total.tag.{tag}")
            emit_counter(f"fields.populated_total.field.{field}")
        if not ctx.get(field):
            _mark_missing(ctx, tag, field)


__all__ = ["apply_field_fillers", "clear_filler_cache"]
