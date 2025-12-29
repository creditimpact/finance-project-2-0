"""High-level orchestration for analyzing credit reports.

This module wires together parsing utilities, prompt generation/AI calls,
and a suite of post-processing helpers. Historically all of this logic lived
in a single file which made the responsibilities difficult to test and
reason about. The functionality has been split into dedicated modules:

- :mod:`backend.core.logic.report_analysis.report_parsing`
- :mod:`backend.core.logic.report_analysis.report_prompting`
- :mod:`backend.core.logic.report_analysis.report_postprocessing`
"""

from __future__ import annotations

import copy as _copy
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from rapidfuzz import fuzz

from backend.config import (
    CASESTORE_PARSER_LOG_PARITY,
    CASESTORE_REDACT_BEFORE_STORE,
    DETERMINISTIC_EXTRACTORS_ENABLED,
    ENABLE_CASESTORE_WRITE,
    PARSER_AUDIT_ENABLED,
    TEXT_NORMALIZE_ENABLED,
)
from backend.core.case_store.api import (
    MAX_RETRIES,
    create_session_case,
    load_session_case,
    save_session_case,
    upsert_account_fields,
)
from backend.core.case_store.errors import CaseStoreError
from backend.core.case_store.redaction import redact_account_fields
from backend.core.logic.report_analysis.block_exporter import load_account_blocks
from backend.core.logic.report_analysis.candidate_logger import (
    CandidateTokenLogger,
    StageATraceLogger,
)
from backend.core.logic.report_analysis.problem_detection import (
    evaluate_account_problem,
)
from backend.core.logic.utils.inquiries import extract_inquiries
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.norm import normalize_heading
from backend.core.logic.utils.text_parsing import (
    enforce_collection_status,
    extract_account_blocks,
    extract_account_headings,
    extract_late_history_blocks,
)
from backend.core.telemetry import metrics
from backend.core.telemetry.parser_metrics import emit_parser_audit

from .text_normalization import NormalizationStats, normalize_page
from .text_provider import load_cached_text


# Lazy wrapper to avoid importing PyMuPDF at module import time in environments
# without the binary. Tests can monkeypatch this name directly.
def char_count(s):
    from .pdf_io import char_count as _impl

    return _impl(s)


from .report_parsing import (  # noqa: E402,F401 - kept for test monkeypatching
    attach_bureau_meta_tables,
    build_block_fuzzy,
    extract_account_numbers,
    extract_creditor_remarks,
    extract_payment_statuses,
    extract_three_column_fields,
)
from .report_postprocessing import (  # noqa: E402
    _assign_issue_types,
    _cleanup_unverified_late_text,
    _inject_missing_late_accounts,
    _merge_parser_inquiries,
    _reconcile_account_headings,
    _sanitize_late_counts,
    enrich_account_metadata,
    validate_analysis_sanity,
)
from .report_prompting import (  # noqa: E402
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    PIPELINE_VERSION,
)

logger = logging.getLogger(__name__)
# Minimum advisor comment length after finalize; used for idempotent fills and QA
MIN_ADVISOR_COMMENT_LEN = 60

# Minimum advisor comment length after finalize; used for idempotent fills and QA
MIN_ADVISOR_COMMENT_LEN = 60


def _emit_metric(name: str, **tags: Any) -> None:
    """Best-effort metric emitter for lightweight counters."""
    logger.info("metric %s %s", name, tags)


def _track_parse_pass(session_id: str | None) -> None:
    """Increment per-session parse counter and warn on multiple passes."""
    if not session_id:
        _emit_metric("stage1.parse_passes", session_id=session_id)
        return

    passes = 1
    if ENABLE_CASESTORE_WRITE:
        for _ in range(MAX_RETRIES):
            try:
                case = load_session_case(session_id)
            except CaseStoreError as err:  # pragma: no cover - best effort
                logger.warning(
                    "casestore_session_error session=%s error=%s", session_id, err
                )
                break

            current = case.report_meta.raw_source.get("parse_passes", 0)
            passes = current + 1
            case.report_meta.raw_source["parse_passes"] = passes
            original_version = case.version
            case.version = original_version + 1

            try:
                stored = load_session_case(session_id)
            except CaseStoreError as err:  # pragma: no cover - best effort
                logger.warning(
                    "casestore_session_error session=%s error=%s", session_id, err
                )
                break
            if stored.version != original_version:
                continue

            try:
                save_session_case(case)
            except CaseStoreError as err:  # pragma: no cover - best effort
                logger.warning(
                    "casestore_session_error session=%s error=%s", session_id, err
                )
            break

    _emit_metric("stage1.parse_passes", session_id=session_id)
    logger.info("stage1.parse_pass session=%s pass=%d", session_id, passes)
    if passes > 1:
        logger.warning(
            "stage1.parse_multiple_passes session=%s passes=%d", session_id, passes
        )
        _emit_metric("stage1.parse_multiple_passes", session_id=session_id)


# Fallback patterns for unlabeled SmartCredit text blocks
BUREAU_RE = re.compile(r"^(TransUnion|Experian|Equifax)\b", re.IGNORECASE)
NEG_STATUS_RE = re.compile(
    r"\b(?:"
    r"late\s*(?:30|60|90|120|150\+?)\s*days"
    r"|past\s*due"
    r"|delinquent"
    r"|collection(?:/charge\s*off)?"
    r"|charge[- ]?off"
    r"|repossession"
    r"|foreclosure"
    r")\b",
    re.IGNORECASE,
)
MASKED_ACCT_RE = re.compile(
    r"^(?:"
    r"[A-Z]{0,4}\d{2,}[Xx\*]{2,}\d*"  # e.g. M20191************
    r"|[Xx\*]{2,}\d{2,}"  # **5678 or XX1234
    r"|XX\d{2,}"
    r"|\d{2,}\*{2,}\d*"  # 349992**********
    r"|\d{4,}\*+"
    r")$"
)


def normalize_for_regex(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", s or "")
    # Replace zero-width, BOM, NBSP to spaces; drop soft hyphen
    s = (
        s.replace("\ufeff", " ")
        .replace("\u200b", " ")
        .replace("\u200c", " ")
        .replace("\u200d", " ")
        .replace("\u00A0", " ")
        .replace("\u00AD", "")
    )
    s = re.sub(r"[ \t\f\v]+", " ", s)
    return s


def norm_bureau(name: str | None) -> str:
    low = (name or "").strip().lower().replace(" ", "")
    if low.startswith("transunion") or low == "tu":
        return "transunion"
    if low.startswith("experian") or low == "ex":
        return "experian"
    if low.startswith("equifax") or low == "eq":
        return "equifax"
    return "unknown"


# Stoplist of headings/labels to avoid false-positive negatives
STOP_HEADINGS: set[str] = {
    normalize_creditor_name("total accounts"),
    normalize_creditor_name("account history"),
    normalize_creditor_name("two-year payment history"),
    normalize_creditor_name("ok 30 60 90 120 150 pp rf"),
    normalize_creditor_name("summary"),
    normalize_creditor_name("payments"),
    normalize_creditor_name("limit"),
}


def add_status_hit(
    hits: dict,
    creditor: str | None,
    bureau: str | None,
    acct_mask: str | None,
    label: str,
    *,
    evidence: str | None = None,
) -> None:
    """Add a diagnostic-negative label into an additive structure.

    Structure: hits[creditor][bureau][acct] = {"labels": set(), "evidence": [..]}
    Unknown creditor/bureau/acct are kept as '__unknown__' so they can be
    promoted later.
    """

    cred = (creditor or "").strip() or "__unknown__"
    bure = norm_bureau(bureau)
    acct = (acct_mask or "").strip() or "__unknown__"
    info = (
        hits.setdefault(cred, {})
        .setdefault(bure, {})
        .setdefault(acct, {"labels": set(), "evidence": [], "mask": None})
    )
    info["labels"].add(label)
    if evidence:
        info["evidence"].append(evidence)
    if acct_mask and not info.get("mask"):
        info["mask"] = acct_mask


def safe_join_or_passthrough(
    result: dict,
    accounts_by_norm: dict[str, list[dict]],
    hits: dict,
) -> None:
    """Join fallback hits to existing accounts or synthesize accounts if missing.

    - Unknown creditors ('__unknown__') or unresolved names map to 'Unknown Creditor'.
    - Bureau keys are normalized via norm_bureau().
    - Injects synthetic accounts with source_stage='parser_aggregated' when needed.
    """

    for cred, bureaus in (hits or {}).items():
        # Skip stoplisted pseudo-headings
        if normalize_creditor_name(cred) in STOP_HEADINGS:
            continue
        cred_norm = (
            normalize_creditor_name(cred)
            if cred != "__unknown__"
            else "unknown creditor"
        )
        if cred_norm == "":
            cred_norm = "unknown creditor"
        targets = accounts_by_norm.get(cred_norm)
        if not targets:
            # Create a synthetic account entry
            acc = {
                "name": cred_norm.title(),
                "normalized_name": cred_norm,
                "source_stage": "parser_aggregated",
                "payment_statuses": {},  # will store lowercase bureau -> joined label string
                "flags": ["Late Payments"],
                "fbk_hits": {},  # additive map: bureau -> acct -> {labels,set; evidence}
            }
            if cred != cred_norm:
                acc["alias_resolved_to"] = cred_norm
            result.setdefault("all_accounts", []).append(acc)
            accounts_by_norm.setdefault(cred_norm, []).append(acc)
            targets = [acc]
            logger.info("FBK: synthesized account for unresolved key=%r", cred)
        # Merge hits into all target accounts
        for acc in targets or []:
            ps = acc.setdefault("payment_statuses", {})
            fbk = acc.setdefault("fbk_hits", {})
            for bureau, acct_map in (bureaus or {}).items():
                bnorm = norm_bureau(bureau)
                # Keep additive structure under fbk_hits
                node = fbk.setdefault(bnorm, {})
                # Aggregate labels into a readable value for UI
                # Aggregate labels into a readable value
                labels: set[str] = set()
                for acct_id, payload in acct_map.items():
                    # Merge payload into fbk node
                    cur = node.setdefault(
                        acct_id or "__unknown__",
                        {"labels": set(), "evidence": [], "mask": None},
                    )
                    for L in payload.get("labels", []) or []:
                        cur["labels"].add(L)
                        labels.add(L)
                    if payload.get("evidence"):
                        cur["evidence"].extend(payload.get("evidence") or [])
                    if payload.get("mask") and not cur.get("mask"):
                        cur["mask"] = payload.get("mask")
                if labels:
                    # Keep existing if present; otherwise set
                    val = "; ".join(sorted(labels))
                    ps.setdefault(bnorm, val)
            # Derive aggregate payment_status string
            if ps:
                acc["payment_status"] = "; ".join(sorted(set(ps.values())))
            # Tag negatives
            if ps and "Late Payments" not in (acc.get("flags") or []):
                acc.setdefault("flags", []).append("Late Payments")
            # Promote mask -> account_number_last4 if available
            if not acc.get("account_number_last4"):
                try:
                    for bkey, bmap in fbk.items():
                        for payload in bmap.values():
                            m = payload.get("mask")
                            if isinstance(m, str):
                                digits = re.sub(r"\D", "", m)
                                if len(digits) >= 4:
                                    acc["account_number_last4"] = digits[-4:]
                                    logger.info(
                                        "DBG last4_set name=%s bureau=%s mask=%s last4=%s",
                                        acc.get("normalized_name"),
                                        bkey,
                                        m,
                                        acc.get("account_number_last4"),
                                    )
                                    raise StopIteration
                except StopIteration:
                    pass


def finalize_problem_accounts(result: dict, hits: dict | None = None) -> None:
    """Ensure negatives are represented as problem accounts, even when unknown.

    Any account that has 'payment_statuses' with negative labels or was
    synthesized from hits will be promoted to problem_accounts.
    """

    # Promote based on current accounts
    promoted = 0

    def _derive_primary(labels: set[str], acc: dict) -> str:
        L = {s.lower() for s in labels}
        if any(
            x in L
            for x in {"collection", "collection/chargeoff", "charge off", "chargeoff"}
        ):
            return "collection"
        # Candidate past-due from labels
        if any(
            any(
                y in lab
                for y in [
                    "past due",
                    "late 30",
                    "late 60",
                    "late 90",
                    "late 120",
                    "late 150",
                ]
            )
            for lab in L
        ):
            # Gate by real evidence: late history or positive past-due amount
            if _has_late_history(acc) or _has_positive_past_due(acc):
                return "past_due"
        if any(x in L for x in {"repossession"}):
            return "repossessed"
        if any(x in L for x in {"foreclosure"}):
            return "foreclosure"
        if any(x in L for x in {"bankruptcy"}):
            return "bankruptcy"
        # Derive from payment_statuses text if labels insufficient, with gating
        ps_map = acc.get("payment_statuses") or {}
        if ps_map:
            joined = " ".join(str(v or "").lower() for v in ps_map.values())
            past_due_hit = bool(
                re.search(r"\bpast\s*due\b", joined)
                or re.search(r"late\s*(30|60|90|120|150\+?)", joined)
            )
            if past_due_hit and (_has_late_history(acc) or _has_positive_past_due(acc)):
                return "past_due"
        return "late_payment"

    def _has_late_history(acc: dict) -> bool:
        lm = acc.get("late_payments") or {}
        try:
            if isinstance(lm, dict):
                for v in lm.values():
                    if isinstance(v, dict):
                        for c in v.values():
                            if int(c) > 0:
                                return True
                    else:
                        if int(v) > 0:
                            return True
        except Exception:
            return False
        return False

    def _has_positive_past_due(acc: dict) -> bool:
        # account-level value
        try:
            val = acc.get("past_due_amount")
            if (
                val is not None
                and float(str(val).replace(",", "").replace("$", "")) > 0
            ):
                return True
        except Exception:
            pass
        # bureau details
        try:
            for fields in (acc.get("bureau_details") or {}).values():
                v = (fields or {}).get("past_due_amount")
                if v is None:
                    continue
                if float(str(v).replace(",", "").replace("$", "")) > 0:
                    return True
        except Exception:
            pass
        return False

    for acc in result.get("all_accounts", []) or []:
        ps_map = acc.get("payment_statuses") or {}
        joined = " ".join(str(v or "").lower() for v in ps_map.values())
        neg_hit = any(NEG_STATUS_RE.search(str(v or "")) for v in ps_map.values())
        # If only 'past due' matches but no late history and amount is zero, downgrade
        only_past_due = bool(re.search(r"\bpast\s*due\b", joined)) and not (
            re.search(r"collection|charge\s*off|repossession|foreclosure", joined)
        )
        if (
            neg_hit
            and only_past_due
            and (not _has_late_history(acc))
            and (not _has_positive_past_due(acc))
        ):
            logger.info(
                "DBG past_due_downgrade name=%s reason=no_history_and_amount_zero",
                acc.get("normalized_name") or acc.get("name"),
            )
            neg_hit = False
        if neg_hit:
            acc["_detector_is_problem"] = True
            # Try to derive primary_issue from additive hits if present
            additive = acc.get("fbk_hits") or {}
            labels: set[str] = set()
            for bure_map in additive.values():
                for payload in bure_map.values():
                    labels |= set(payload.get("labels", []))
            acc["primary_issue"] = _derive_primary(labels, acc)
            logger.info(
                "DBG primary_map name=%s labels=%s ps=%s => primary=%s",
                acc.get("normalized_name"),
                sorted(list(labels)),
                ps_map,
                acc.get("primary_issue"),
            )
            promoted += 1
    if promoted:
        logger.info(
            "FBK: promoted problem accounts from payment_statuses: %d", promoted
        )
    # Fallback: for hits with unknown creditor, synthesize accounts now
    if hits:
        accounts_by_norm: dict[str, list[dict]] = {}
        for a in result.get("all_accounts", []) or []:
            accounts_by_norm.setdefault(a.get("normalized_name") or "", []).append(a)
        safe_join_or_passthrough(result, accounts_by_norm, hits)
    # Build final problem_accounts list
    result["problem_accounts"] = [
        a
        for a in result.get("all_accounts", [])
        if a.get("_detector_is_problem")
        or any(
            NEG_STATUS_RE.search(str(v or ""))
            for v in (a.get("payment_statuses") or {}).values()
        )
    ]
    # Advisor comments and basic recommendations
    for a in result["problem_accounts"]:
        pi = (a.get("primary_issue") or "").lower()
        # Problem reasons from additive hits or statuses
        if not a.get("problem_reasons"):
            reasons: list[str] = []
            fbk = a.get("fbk_hits") or {}
            for bure, acctmap in fbk.items():
                labs = set()
                for payload in acctmap.values():
                    labs |= set(payload.get("labels", []))
                if labs:
                    reasons.append(f"{bure}: {', '.join(sorted(labs))}")
            if not reasons:
                ps = a.get("payment_statuses") or {}
                for bure, val in ps.items():
                    reasons.append(f"{bure}: {val}")
            if reasons:
                a["problem_reasons"] = reasons
        if (
            not a.get("advisor_comment")
            or len(str(a.get("advisor_comment"))) < MIN_ADVISOR_COMMENT_LEN
        ):
            if pi in ("collection", "charge_off"):
                a["advisor_comment"] = (
                    "Account appears in collection/charge-off. Consider debt validation and goodwill/dispute steps."
                )
            elif pi in ("past_due", "late_payment"):
                a["advisor_comment"] = (
                    "Late payment markers detected. Consider on-time streak plan and a goodwill letter."
                )
        recs = a.setdefault("recommendations", [])
        if not recs:
            if pi in ("collection", "charge_off"):
                recs.extend(["Send debt validation", "Prepare goodwill/dispute letter"])
            else:
                recs.extend(["Set up auto-pay", "Send goodwill letter"])
        logger.info(
            "DBG advisor_fill name=%s primary=%s comment_len=%d recs=%d",
            a.get("normalized_name"),
            a.get("primary_issue"),
            len(str(a.get("advisor_comment", ""))),
            len(a.get("recommendations", []) or []),
        )
        # QA guard: log WARN if missing critical fields
        ps_map = a.get("payment_statuses") or {}
        if any(NEG_STATUS_RE.search(str(v or "")) for v in ps_map.values()):
            if (
                (a.get("primary_issue") or "").lower() == "unknown"
                or not a.get("advisor_comment")
                or not (a.get("recommendations") or [])
            ):
                logger.warning(
                    "WARN qa_missing_fields name=%s primary=%s ps=%s",
                    a.get("normalized_name"),
                    a.get("primary_issue"),
                    ps_map,
                )
    logger.info("FBK: problem_accounts=%d", len(result.get("problem_accounts", [])))


def _parity_counts(
    original: dict[str, Any], stored: dict[str, Any]
) -> tuple[int, int, int, int]:
    """Return counts of added/changed/masked/missing keys between ``original`` and ``stored``."""

    added = len(set(stored) - set(original))
    missing = len(set(original) - set(stored))
    changed = 0
    masked = 0
    redacted = redact_account_fields(original)
    for key in set(original) & set(stored):
        if stored.get(key) != original.get(key):
            changed += 1
            if stored.get(key) == redacted.get(key):
                masked += 1
    return added, changed, masked, missing


def _split_account_buckets(accounts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split accounts into negative and open issue buckets.

    The heuristics consider any charge-off/collection/closed indicators from
    merged fields such as ``payment_status`` and ``remarks``. Accounts that are
    currently open (e.g. "Open", "Current", "Pays as agreed") are classified
    under ``open_accounts_with_issues`` while all others default to
    ``negative_accounts``.
    """

    negatives: list[dict] = []
    open_issues: list[dict] = []
    for acc in accounts or []:
        if not acc.get("issue_types") and not acc.get("high_utilization"):
            continue

        parts = [
            acc.get("status"),
            acc.get("account_status"),
            acc.get("payment_status"),
            acc.get("remarks"),
        ]
        parts.extend((acc.get("payment_statuses") or {}).values())
        parts.extend((acc.get("status_texts") or {}).values())
        for fields in (acc.get("bureau_details") or {}).values():
            val = fields.get("account_status")
            if val:
                parts.append(val)
        status_text = " ".join(str(p) for p in parts if p).lower()

        evidence = {
            "status_text": status_text,
            "closed_date": acc.get("closed_date"),
            "past_due_amount": acc.get("past_due_amount"),
            "late_payments": bool(acc.get("late_payments")),
            "high_utilization": acc.get("high_utilization"),
        }

        negative_re = r"charge\s*off|charged\s*off|chargeoff|collection|derog|repossess"
        has_negative = bool(re.search(negative_re, status_text))
        if not has_negative and acc.get("closed_date"):
            has_negative = bool(
                re.search(r"derog|delinquent|charge|collection|repossess", status_text)
            )

        if has_negative and acc.get("primary_issue") == "late_payment":
            # Do not overwrite core primary_issue after finalize; compute a UI-only suggestion
            if "collection" in status_text:
                new_primary = "collection"
            elif re.search(r"charge\s*off|chargeoff", status_text):
                new_primary = "charge_off"
            elif "repossess" in status_text:
                new_primary = "repossessed"
            else:
                new_primary = "derogatory"
            # Suggest into UI field only; keep core field intact
            acc.setdefault("ui_primary_issue", new_primary)
            acc.setdefault("issue_types", [])
            if new_primary not in acc["issue_types"]:
                acc["issue_types"].insert(0, new_primary)

        if has_negative:
            bucket = "negative"
            negatives.append(acc)
        else:
            has_open = bool(
                re.search(r"\bopen\b|current|pays\s+as\s+agreed", status_text)
            ) and not acc.get("closed_date")
            has_issue = (
                (
                    isinstance(acc.get("past_due_amount"), (int, float))
                    and acc.get("past_due_amount") > 0
                )
                or bool(acc.get("late_payments"))
                or bool(acc.get("high_utilization"))
            )

            if has_open and has_issue:
                bucket = "open_issues"
                open_issues.append(acc)
            else:
                bucket = "negative"
                negatives.append(acc)

        logger.debug(
            "bucket_decision %s",
            json.dumps(
                {"name": acc.get("name"), "bucket": bucket, "evidence": evidence}
            ),
        )

    return negatives, open_issues


def _attach_parser_signals(
    accounts: list[dict] | None,
    payment_statuses_by_heading: dict[str, dict[str, str]],
    remarks_by_heading: dict[str, str],
    payment_status_raw_by_heading: dict[str, str],
) -> None:
    """Populate parser-derived fields for aggregated accounts."""

    for acc in accounts or []:
        if acc.get("source_stage") != "parser_aggregated":
            continue
        norm = acc.get("normalized_name") or normalize_creditor_name(
            acc.get("name", "")
        )
        acc["normalized_name"] = norm
        bureau_map = payment_statuses_by_heading.get(norm, {})
        acc["payment_statuses"] = bureau_map
        if bureau_map:
            acc["payment_status"] = "; ".join(sorted(set(bureau_map.values())))
        else:
            raw = payment_status_raw_by_heading.get(norm, "")
            acc["payment_status_raw"] = raw
            if raw:
                if re.search(r"\bcharge[-\s]?off\b", raw, re.I):
                    acc["payment_status"] = "charge_off"
                elif re.search(r"\bcollection(s)?\b", raw, re.I):
                    acc["payment_status"] = "collection"
        acc["remarks"] = remarks_by_heading.get(norm, "")


# ---------------------------------------------------------------------------
# Join helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(name: str, choices: set[str]) -> str | None:
    """Return best fuzzy match for *name* within *choices* when >= 0.9."""

    best_score = 0.0
    best: str | None = None
    for cand in choices:
        score = fuzz.WRatio(name, cand) / 100.0
        if score > best_score:
            best_score = score
            best = cand
    if best_score >= 0.8:
        return best
    return None


def _normalize_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *mapping* with creditor names normalized."""

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        norm = normalize_creditor_name(key)
        existing = normalized.get(norm)
        if existing and isinstance(existing, dict) and isinstance(value, dict):
            existing.update(value)
        else:
            normalized[norm] = value
    return normalized


def _join_heading_map(
    accounts: Mapping[str, list[dict]],
    existing_norms: set[str],
    mapping: dict[str, Any],
    field_name: str | None,
    heading_map: Mapping[str, str],
    *,
    is_bureau_map: bool = False,
    aggregate_field: str | None = None,
) -> None:
    """Join a heading-keyed *mapping* onto *accounts* in-place.

    When *field_name* is ``None`` the mapping keys are reconciled (alias/fuzzy)
    but no fields are attached to accounts.  ``is_bureau_map`` controls whether
    values are ``{bureau: value}`` mappings that should be merged into the
    account's ``bureaus`` list.  For payment status maps an additional
    ``aggregate_field`` (e.g. ``"payment_status"``) may be specified to store a
    combined string value.
    """

    size = len(mapping or {})
    logger.info(
        "DBG join_source_sizes map=%s size=%d targets=%d",
        field_name or "<none>",
        size,
        len(accounts or {}),
    )
    if size == 0:
        logger.debug(
            "CASEBUILDER: empty_three_column_maps map=%s", field_name or "<none>"
        )
        return

    for key, value in list(mapping.items()):
        norm = normalize_creditor_name(key)
        raw = heading_map.get(norm, key)
        if norm != key:
            mapping.pop(key)
            if norm in mapping and field_name:
                if (
                    is_bureau_map
                    and isinstance(mapping[norm], dict)
                    and isinstance(value, dict)
                ):
                    mapping[norm].update(value)
                else:
                    mapping[norm] = value
            else:
                mapping[norm] = value
            value = mapping[norm]
        targets = accounts.get(norm)
        method: str | None = None
        if targets is None:
            match = _fuzzy_match(norm, existing_norms)
            if match:
                mapping.pop(norm)
                if match in mapping and field_name:
                    # merge dictionaries if both present
                    if (
                        is_bureau_map
                        and isinstance(mapping[match], dict)
                        and isinstance(value, dict)
                    ):
                        mapping[match].update(value)
                    else:
                        mapping[match] = value
                else:
                    mapping[match] = value
                targets = accounts.get(match)
                norm = match
                value = mapping[match]
                method = "fuzzy"
            else:
                details = {
                    "raw_key": raw,
                    "normalized": norm,
                    "target_present": False,
                    "map": field_name or "",
                }
                logger.debug(
                    "heading_join_miss %s",
                    json.dumps(details, sort_keys=True),
                )
                # Avoid noisy unresolved logs when mapping is empty or no targets exist
                if size > 0 and len(accounts or {}) > 0:
                    logger.debug(
                        "heading_join_unresolved %s",
                        json.dumps(details, sort_keys=True),
                    )
                continue
        elif raw.upper() != norm.upper():
            method = "alias"

        if field_name is None:
            if method:
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {"raw_key": raw, "normalized_target": norm, "method": method},
                        sort_keys=True,
                    ),
                )
            continue

        for acc in targets or []:
            if is_bureau_map and isinstance(value, dict):
                acc.setdefault(field_name, {})
                acc[field_name].update(value)
                if aggregate_field:
                    acc[aggregate_field] = "; ".join(
                        sorted(set(acc[field_name].values()))
                    )
                acc.setdefault("bureaus", [])
                for bureau, val in value.items():
                    info = None
                    for b in acc["bureaus"]:
                        if isinstance(b, dict) and b.get("bureau") == bureau:
                            info = b
                            break
                    if info is None:
                        info = {"bureau": bureau}
                        acc["bureaus"].append(info)
                    if field_name == "payment_statuses":
                        if not info.get("payment_status"):
                            info["payment_status"] = val
                    else:
                        if not info.get(field_name):
                            info[field_name] = val
            else:
                acc[field_name] = value

        if method:
            logger.info(
                "heading_join_linked %s",
                json.dumps(
                    {"raw_key": raw, "normalized_target": norm, "method": method},
                    sort_keys=True,
                ),
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_credit_report(
    pdf_path,
    output_json_path,
    client_info,
    *,
    request_id: str,
    session_id: str,
    ai_client: Any | None = None,
    **kwargs,
):
    """Analyze ``pdf_path`` and write structured analysis to ``output_json_path``."""
    # Parsing is fully deterministic; Stage A adjudication (AI) is separate.

    # Swallow orchestration kwargs we don't need here. Keep backward-compatible
    # tolerance for callers that pass AI-related flags or clients.
    # Note: If callers used keyword-only args not in our signature, they'll land in kwargs.
    run_ai = kwargs.pop("run_ai", None)
    _ = kwargs.pop("ai_client", None)  # in case it arrived via kwargs
    if kwargs:
        logging.getLogger(__name__).warning(
            "analyze_credit_report ignored extra kwargs: %s", list(kwargs.keys())
        )
    # Optional debug visibility for tolerated args
    if ai_client is not None:
        logger.debug(
            "ai_client passed to analyze_report.analyze_credit_report but unused; ignoring"
        )
    if run_ai is not None:
        logger.debug(
            "run_ai passed to analyze_report.analyze_credit_report but unused; ignoring"
        )

    pages_total = 0
    pages_with_text = 0
    pages_empty_text = 0
    extract_text_ms = 0
    pages_ocr = 0
    ocr_latency_ms_total = 0
    ocr_errors = 0
    norm_stats = NormalizationStats()
    fields_written: int | None = None
    errors: str | None = None
    extract_sections_ms: int | None = None
    extract_accounts_ms: int | None = None
    extract_report_meta_ms: int | None = None
    extract_summary_ms: int | None = None
    extractor_accounts_total: dict[str, int] | None = None
    _finalized_snapshot: list[dict[str, Any]] | None = None

    sid = session_id or request_id
    cached = load_cached_text(sid)
    if not cached:
        raise ValueError("no_cached_text_for_session")
    pages_text = list(cached["pages"])
    text = cached.get("full_text") or cached.get("full", "")
    meta = cached.get("meta", {})
    extract_text_ms = int(meta.get("extract_text_ms", 0))
    pages_ocr = int(meta.get("pages_ocr", 0))
    ocr_latency_ms_total = int(meta.get("ocr_latency_ms_total", 0))
    ocr_errors = int(meta.get("ocr_errors", 0))
    pages_total = len(pages_text)
    counts = [char_count(t) for t in pages_text]
    pages_with_text = sum(1 for c in counts if c > 0)
    pages_empty_text = pages_total - pages_with_text
    if TEXT_NORMALIZE_ENABLED:
        normalized_pages: list[str] = []
        for t in pages_text:
            normed, s = normalize_page(t)
            normalized_pages.append(normed)
            norm_stats.dates_converted += s.dates_converted
            norm_stats.amounts_converted += s.amounts_converted
            norm_stats.bidi_stripped += s.bidi_stripped
            norm_stats.space_reduced_chars += s.space_reduced_chars
        pages_text = normalized_pages
        text = "\n".join(pages_text) or text

    def _emit_audit() -> None:
        if PARSER_AUDIT_ENABLED:
            emit_parser_audit(
                session_id=session_id or "",
                pages_total=pages_total,
                pages_with_text=pages_with_text,
                pages_empty_text=pages_empty_text,
                extract_text_ms=extract_text_ms,
                fields_written=fields_written,
                errors=errors,
                parser_pdf_pages_ocr=pages_ocr,
                parser_ocr_latency_ms_total=ocr_latency_ms_total,
                parser_ocr_errors=ocr_errors,
                normalize_dates_converted=norm_stats.dates_converted,
                normalize_amounts_converted=norm_stats.amounts_converted,
                normalize_bidi_stripped=norm_stats.bidi_stripped,
                normalize_space_reduced_chars=norm_stats.space_reduced_chars,
                extract_sections_ms=extract_sections_ms,
                extract_accounts_ms=extract_accounts_ms,
                extract_report_meta_ms=extract_report_meta_ms,
                extract_summary_ms=extract_summary_ms,
                extractor_field_coverage_total=fields_written,
                extractor_accounts_total=extractor_accounts_total,
            )

    if not text.strip():
        raise ValueError("[ERROR] No text extracted from PDF")

    headings = extract_account_headings(text)
    heading_map = {normalize_creditor_name(norm): raw for norm, raw in headings}

    def detected_late_phrases(txt: str) -> bool:
        # Use regex hook centralized in patterns; keeps alignment and easier refinement.
        try:
            from .patterns import LATE_HEADERS  # local import to avoid cycles
        except Exception:
            return bool(re.search(r"late|past due", txt, re.I))
        return bool(LATE_HEADERS.search(txt))

    raw_goal = client_info.get("goal", "").strip().lower()
    if raw_goal in ["", "not specified", "improve credit", "repair credit"]:
        strategic_context = (
            "Improve credit score significantly within the next 3-6 months using strategies such as authorized users, "
            "credit building tools, and removal of negative items."
        )
    else:
        strategic_context = client_info.get("goal", "Not specified")

    is_identity_theft = client_info.get("is_identity_theft", False)
    doc_fingerprint = hashlib.sha256(
        f"{text}|{ANALYSIS_PROMPT_VERSION}|{ANALYSIS_SCHEMA_VERSION}|{PIPELINE_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()

    if ENABLE_CASESTORE_WRITE and session_id:
        try:
            try:
                case = load_session_case(session_id)
            except CaseStoreError:
                meta: dict[str, Any] = {
                    "raw_source": {
                        "vendor": "SmartCredit",
                        "version": None,
                        "doc_fingerprint": doc_fingerprint,
                    }
                }
                report_date = client_info.get("report_date")
                if report_date:
                    meta["credit_report_date"] = report_date
                case = create_session_case(session_id, meta=meta)
                logger.debug(
                    "CASEBUILDER: create_session_case(session_id=%s, accounts_count=%d)",
                    session_id,
                    len(case.accounts),
                )
            else:
                case.report_meta.raw_source.update(
                    {
                        "vendor": "SmartCredit",
                        "version": None,
                        "doc_fingerprint": doc_fingerprint,
                    }
                )
                report_date = client_info.get("report_date")
                if report_date:
                    case.report_meta.credit_report_date = report_date
            save_session_case(case)
            try:
                from backend.config import CASESTORE_DIR

                path = Path(CASESTORE_DIR) / f"{session_id}.json"
                size = path.stat().st_size if path.exists() else 0
            except Exception:
                path = None
                size = 0
            logger.debug(
                "CASEBUILDER: save_session_case(session_id=%s, path=%s, size=%d)",
                session_id,
                path,
                size,
            )
        except CaseStoreError as err:  # pragma: no cover - best effort
            logger.warning(
                "casestore_session_error session=%s error=%s",
                session_id,
                err,
            )
    _track_parse_pass(session_id)
    extractor_accounts_total = {}
    if DETERMINISTIC_EXTRACTORS_ENABLED and ENABLE_CASESTORE_WRITE and session_id:
        from .extractors import (
            accounts as acc_mod,
            report_meta as rm_mod,
            sections as sec_mod,
            summary as sum_mod,
        )

        _start = time.perf_counter()
        sec = sec_mod.detect(pages_text)
        extract_sections_ms = int((time.perf_counter() - _start) * 1000)

        bureaus = sec.get("bureaus", {})
        _start = time.perf_counter()
        for bureau, lines in bureaus.items():
            res = acc_mod.extract(lines, session_id=session_id, bureau=bureau)
            extractor_accounts_total[bureau] = len(res)
            fields_written = (fields_written or 0) + sum(len(r["fields"]) for r in res)
        extract_accounts_ms = int((time.perf_counter() - _start) * 1000)

        _start = time.perf_counter()
        meta_fields = rm_mod.extract(sec.get("report_meta", []), session_id=session_id)
        extract_report_meta_ms = int((time.perf_counter() - _start) * 1000)
        fields_written = (fields_written or 0) + len(meta_fields)

        _start = time.perf_counter()
        summary_fields = sum_mod.extract(sec.get("summary", []), session_id=session_id)
        extract_summary_ms = int((time.perf_counter() - _start) * 1000)
        fields_written = (fields_written or 0) + len(summary_fields)

    # Initialize deterministic result container; AI parsing path removed.
    result = {
        "negative_accounts": [],
        "open_accounts_with_issues": [],
        "positive_accounts": [],
        "high_utilization_accounts": [],
        "all_accounts": [],
        "inquiries": [],
        "needs_human_review": False,
        "missing_bureaus": [],
    }

    result["prompt_version"] = ANALYSIS_PROMPT_VERSION
    result["schema_version"] = ANALYSIS_SCHEMA_VERSION

    _reconcile_account_headings(result, heading_map)

    parsed_inquiries = extract_inquiries(text)
    inquiry_raw_map = {
        normalize_heading(i["creditor_name"]): i["creditor_name"]
        for i in parsed_inquiries
    }
    if parsed_inquiries:
        print(f"[INFO] Parser found {len(parsed_inquiries)} inquiries in text.")
    else:
        print("[WARN] Parser did not find any inquiries in the report text.")

    # Always use parser-derived inquiries in deterministic pipeline
    result["inquiries"] = parsed_inquiries

    payment_status_map: dict[str, dict[str, str]] = {}
    _payment_status_raw_map: dict[str, str] = {}
    remarks_map: dict[str, dict[str, str]] = {}
    status_text_map: dict[str, dict[str, str]] = {}
    account_number_map: dict[str, dict[str, str]] = {}
    try:
        account_names = {acc.get("name", "") for acc in result.get("all_accounts", [])}
        history_all, raw_map, grid_all = extract_late_history_blocks(
            text, return_raw_map=True
        )
        _sanitize_late_counts(history_all)
        history_all = _normalize_keys(history_all)
        raw_map = _normalize_keys(raw_map)
        grid_all = _normalize_keys(grid_all)
        history, _, grid_map = extract_late_history_blocks(
            text, account_names, return_raw_map=True
        )
        _sanitize_late_counts(history)
        history = _normalize_keys(history)
        grid_map = _normalize_keys(grid_map)
        (
            col_payment_map,
            col_remarks_map,
            status_text_map,
            col_payment_raw,
            _col_remarks_raw,
            _col_status_raw,
            detail_map,
        ) = extract_three_column_fields(pdf_path, session_id=session_id)
        three_col_header_missing = not (
            col_payment_map or col_remarks_map or status_text_map or detail_map
        )
        payment_status_map, _payment_status_raw_map = extract_payment_statuses(text)
        payment_status_map = _normalize_keys(payment_status_map)
        # Normalize bureau keys to lowercase for consistency
        for k in list(payment_status_map.keys()):
            bmap = payment_status_map.get(k) or {}
            payment_status_map[k] = {norm_bureau(b): v for b, v in bmap.items()}
        _payment_status_raw_map = _normalize_keys(_payment_status_raw_map)
        for name, vals in col_payment_map.items():
            norm = normalize_creditor_name(name)
            payment_status_map.setdefault(norm, {}).update(vals)
        for name, raw in col_payment_raw.items():
            norm = normalize_creditor_name(name)
            _payment_status_raw_map.setdefault(norm, raw)
        # Fallback for unlabeled per-bureau negative statuses under bureau headings
        try:
            full_text = text
            logger.info("FBK: scanning fallback on text len=%d", len(full_text))
            text_norm = normalize_for_regex(full_text)
            blocks = extract_account_blocks(text_norm)
            fallback_statuses: dict[str, dict[str, str]] = {}
            fallback_hits_additive: dict = {}
            for block in blocks:
                if not block:
                    continue
                acc_name = normalize_creditor_name(block[0].strip())
                current_bureau: str | None = None
                in_values = False
                for line in block[1:]:
                    clean = line.strip()
                    if (
                        "Two-Year Payment History" in clean
                        or "Days Late - 7 Year History" in clean
                    ):
                        in_values = False
                        continue
                    m = BUREAU_RE.match(clean)
                    if m:
                        current_bureau = m.group(1)
                        in_values = True
                        continue
                    if not in_values or not current_bureau:
                        continue
                    hit = NEG_STATUS_RE.search(clean)
                    if hit:
                        val = hit.group(0).lower()
                        fbureau = norm_bureau(current_bureau)
                        fallback_statuses.setdefault(acc_name, {})[fbureau] = val
                        mask_val = clean if MASKED_ACCT_RE.search(clean) else None
                        add_status_hit(
                            fallback_hits_additive,
                            acc_name,
                            fbureau,
                            mask_val,
                            val,
                            evidence=clean,
                        )
            tot_accounts = len(fallback_statuses)
            tot_pairs = sum(len(v) for v in fallback_statuses.values())
            logger.info(
                "FBK: negatives found accounts=%d bureaus=%d", tot_accounts, tot_pairs
            )
            for acc, bureaus in list(fallback_statuses.items())[:5]:
                logger.info("FBK: negatives for %s -> %s", acc, bureaus)
            # Doc-level scan independent of blocks
            try:
                fallback_statuses_doc: dict[str, dict[str, str]] = {}
                fallback_hits_doc: dict = {}
                recent_mask: str | None = None
                current_bureau2: str | None = None
                current_heading: str | None = None
                heading_re = re.compile(
                    r"^(?!transunion$|experian$|equifax$)(?!.*(days\s+late|payment\s+history|year\s+history))(?=[A-Za-z0-9])[A-Za-z0-9/&\-',\. ]{3,60}$",
                    re.I,
                )
                lines = text_norm.splitlines()
                n = len(lines)
                # Precompute context flags
                is_bureau = [bool(BUREAU_RE.match((ln or "").strip())) for ln in lines]
                is_mask = [
                    bool(MASKED_ACCT_RE.search((ln or "").strip())) for ln in lines
                ]
                for idx, raw in enumerate(lines):
                    line = (raw or "").strip()
                    if not line:
                        continue
                    # Track heading candidates loosely
                    if heading_re.match(line):
                        current_heading = normalize_creditor_name(line)
                        # Mask capture directly on heading line
                        if MASKED_ACCT_RE.search(line):
                            recent_mask = line
                            logger.info(
                                "DBG last4_candidate name=%s line_idx=%d text=%r",
                                current_heading,
                                idx,
                                line,
                            )
                    # Track bureau
                    m_b = BUREAU_RE.match(line)
                    if m_b:
                        current_bureau2 = m_b.group(1)
                        if MASKED_ACCT_RE.search(line):
                            recent_mask = line
                            logger.info(
                                "DBG last4_candidate name=%s line_idx=%d text=%r",
                                current_heading or "__unknown__",
                                idx,
                                line,
                            )
                        continue
                    # Track masked account numbers opportunistically
                    if MASKED_ACCT_RE.search(line):
                        recent_mask = line
                    # Negative pattern with context
                    mneg = NEG_STATUS_RE.search(line)
                    if mneg:
                        # Only accept if within ±7 lines window we have a bureau header AND a masked account
                        start = idx - 7 if idx - 7 >= 0 else 0
                        end = idx + 8 if idx + 8 <= n else n
                        window_bureau = any(is_bureau[j] for j in range(start, end))
                        window_mask = any(is_mask[j] for j in range(start, end))
                        if not (window_bureau and window_mask):
                            continue
                        # Back-attach to nearest heading above if missing
                        if not current_heading:
                            for j in range(idx - 1, max(idx - 10, -1), -1):
                                prev = (lines[j] or "").strip()
                                if heading_re.match(prev):
                                    current_heading = normalize_creditor_name(prev)
                                    break
                        key = current_heading or "__unknown__"
                        # Stoplist filter
                        if normalize_creditor_name(key) in STOP_HEADINGS:
                            continue
                        bureau = norm_bureau(current_bureau2) or "__unknown__"
                        val = mneg.group(0).lower()
                        fallback_statuses_doc.setdefault(key, {}).setdefault(
                            bureau, val
                        )
                        add_status_hit(
                            fallback_hits_doc,
                            key,
                            bureau,
                            recent_mask,
                            val,
                            evidence=line,
                        )
                if fallback_statuses_doc:
                    logger.info(
                        "FBK: doc-scan negatives accounts=%d",
                        len(fallback_statuses_doc),
                    )
                    for acc, vals in list(fallback_statuses_doc.items())[:5]:
                        logger.info("FBK: doc-scan negatives for %s -> %s", acc, vals)
                # Merge doc-level findings
                for acc, vals in fallback_statuses_doc.items():
                    for bureau, val in vals.items():
                        payment_status_map.setdefault(acc, {}).setdefault(bureau, val)
                # Keep additive hits for passthrough
                if fallback_hits_doc:
                    for cred, bure_map in fallback_hits_doc.items():
                        for bure, acct_map in bure_map.items():
                            for acct, payload in acct_map.items():
                                for lab in payload.get("labels", []) or []:
                                    add_status_hit(
                                        fallback_statuses_doc,  # dummy sink not used
                                        cred,
                                        bure,
                                        acct,
                                        lab,
                                    )

            except Exception:
                pass
            # merge fallback only where missing
            for acc, vals in fallback_statuses.items():
                for bureau, val in vals.items():
                    payment_status_map.setdefault(acc, {}).setdefault(bureau, val)
            # Sample dump post-merge
            try:
                sample = list(payment_status_map.items())[:5]
                logger.info("FBK: payment_status_map sample (post-merge): %r", sample)
            except Exception:
                pass
        except Exception:
            pass
        remarks_map = extract_creditor_remarks(text)
        remarks_map = _normalize_keys(remarks_map)
        for name, vals in col_remarks_map.items():
            norm = normalize_creditor_name(name)
            remarks_map.setdefault(norm, {}).update(vals)
        account_number_map = extract_account_numbers(text)
        account_number_map = _normalize_keys(account_number_map)
        for acc_name, bureaus in detail_map.items():
            acc_norm = normalize_creditor_name(acc_name)
            for bureau, fields in bureaus.items():
                num = fields.get("account_number")
                if num:
                    account_number_map.setdefault(acc_norm, {})[bureau] = str(num)
        # Fallback masked account numbers when label is missing
        try:
            fallback_numbers: dict[str, dict[str, str]] = {}
            blocks_nums = (
                blocks if "blocks" in locals() else extract_account_blocks(text)
            )
            for block in blocks_nums:
                if not block:
                    continue
                acc_name = normalize_creditor_name(block[0].strip())
                current_bureau: str | None = None
                in_values = False
                within_idx = 0
                for line in block[1:]:
                    clean = line.strip()
                    if (
                        "Two-Year Payment History" in clean
                        or "Days Late - 7 Year History" in clean
                    ):
                        in_values = False
                        continue
                    m = BUREAU_RE.match(clean)
                    if m:
                        current_bureau = m.group(1).title()
                        in_values = True
                        within_idx = 0
                        continue
                    if not in_values or not current_bureau:
                        continue
                    within_idx += 1
                    if within_idx <= 3 and MASKED_ACCT_RE.match(clean):
                        # only if not already present
                        existing = account_number_map.get(acc_name, {}).get(
                            current_bureau
                        )
                        if not existing and current_bureau not in fallback_numbers.get(
                            acc_name, {}
                        ):
                            fallback_numbers.setdefault(acc_name, {})[
                                current_bureau
                            ] = clean
                            logger.info(
                                "FBK: masked account set for %s/%s: %s",
                                acc_name,
                                current_bureau,
                                clean,
                            )
                        # don't break; subsequent bureaus may appear later
            for acc, vals in fallback_numbers.items():
                for bureau, val in vals.items():
                    account_number_map.setdefault(acc, {}).setdefault(bureau, val)
        except Exception:
            pass
        fallback_used = three_col_header_missing and bool(
            payment_status_map or remarks_map or account_number_map
        )
        if three_col_header_missing:
            metrics.increment("casebuilder.columns.header_missing")
            if fallback_used:
                metrics.increment("casebuilder.columns.fallback_used")
            logger.debug(
                "CASEBUILDER: three_column_header_missing fallback_used=%s persisted=True",
                fallback_used,
            )
        status_text_map = _normalize_keys(status_text_map)
        detail_map = _normalize_keys(detail_map)

        if history:
            print(f"[INFO] Found {len(history)} late payment block(s):")
            for creditor, bureaus in history.items():
                print(
                    f"[INFO] Detected late payments for: '{creditor.title()}' -> {bureaus}"
                )
        else:
            print("[ERROR] No late payment history blocks detected.")

        accounts_by_norm: dict[str, list[dict]] = {}
        sections = [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]
        for section in sections:
            for acc in result.get(section, []):
                raw_name = acc.get("name", "")
                norm = normalize_creditor_name(raw_name)
                if raw_name and norm != raw_name.lower().strip():
                    print(f"[~] Normalized account heading '{raw_name}' -> '{norm}'")
                acc["normalized_name"] = norm
                accounts_by_norm.setdefault(norm, []).append(acc)
        existing_norms = set(accounts_by_norm.keys())
        # Pre-create synthesized accounts for fallback hits before join to reduce unresolved logs
        try:
            fhits = locals().get("fallback_hits_additive", {})
            if fhits:
                logger.info(
                    "DBG snapshot_before_finalize accounts=%d",
                    len(result.get("all_accounts", [])),
                )
                safe_join_or_passthrough(result, accounts_by_norm, fhits)
                # refresh existing norms after synthesis
                existing_norms = set(accounts_by_norm.keys())
        except Exception:
            pass

        _join_heading_map(
            accounts_by_norm, existing_norms, history, "late_payments", raw_map
        )
        _join_heading_map(
            accounts_by_norm, existing_norms, grid_map, "grid_history_raw", raw_map
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            payment_status_map,
            "payment_statuses",
            heading_map,
            is_bureau_map=True,
            aggregate_field="payment_status",
        )
        try:
            sample_accounts = []
            for k, lst in list(accounts_by_norm.items())[:5]:
                if lst:
                    sample_accounts.append((k, lst[0].get("payment_statuses")))
            logger.info(
                "FBK: accounts payment_statuses sample (post-join): %r", sample_accounts
            )
        except Exception:
            pass

        # Ensure problem classifier can see negative statuses from merged map
        try:
            affected = 0
            for acc in result.get("all_accounts", []):
                ps_map = acc.get("payment_statuses") or {}
                if any(NEG_STATUS_RE.search(str(v or "")) for v in ps_map.values()):
                    flags = acc.setdefault("flags", [])
                    if not any(str(f).lower() == "late payments" for f in flags):
                        flags.append("Late Payments")
                        affected += 1
            logger.info(
                "FBK: accounts flagged by NEG_STATUS_RE in payment_statuses: %d",
                affected,
            )
        except Exception:
            pass

        # NEW: Passthrough unresolved/unknown fallback hits and finalize problems
        try:
            # Use fallback_hits_additive if present from local scope
            fhits = locals().get("fallback_hits_additive", {})
            safe_join_or_passthrough(result, accounts_by_norm, fhits)
            # One-time bureau keys sanity
            for acc in (result.get("all_accounts") or [])[:5]:
                ps_keys = sorted(list((acc.get("payment_statuses") or {}).keys()))
                fbk_keys = sorted(list((acc.get("fbk_hits") or {}).keys()))
                bd_keys = sorted(list((acc.get("bureau_details") or {}).keys()))
                logger.info(
                    "DBG bureau_keys_ok name=%s keys_ps=%s keys_fbk=%s keys_bd=%s",
                    acc.get("normalized_name"),
                    ps_keys,
                    fbk_keys,
                    bd_keys,
                )
            finalize_problem_accounts(result, fhits)
            # Inject derived account number fields (last4/display/source) prior to SSOT freeze
            try:
                from backend.core.logic.report_analysis.report_postprocessing import (
                    enrich_problem_accounts_with_numbers,
                )

                if isinstance(result.get("problem_accounts"), list):
                    result["problem_accounts"] = enrich_problem_accounts_with_numbers(
                        result.get("problem_accounts") or []
                    )
            except Exception:
                pass
            # Snapshot and freeze immediately after finalize to prevent mutation later in this pipeline
            try:
                _finalized_snapshot = _copy.deepcopy(
                    result.get("problem_accounts") or []
                )
                result["problem_accounts"] = tuple(
                    MappingProxyType(_copy.deepcopy(a)) if isinstance(a, dict) else a
                    for a in _finalized_snapshot
                )
                prims = [
                    (a.get("normalized_name"), a.get("primary_issue"))
                    for a in list(_finalized_snapshot)[:3]
                ]
                logger.info(
                    "DBG snapshot_after_finalize source=SSOT problem_accounts=%d sample_primary=%s",
                    len(result.get("problem_accounts") or []),
                    prims,
                )
            except Exception:
                _finalized_snapshot = None
        except Exception:
            pass
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            _payment_status_raw_map,
            "payment_status_raw",
            heading_map,
        )
        # Summaries before merges to reduce unresolved warnings
        try:
            total = len(account_number_map or {})
            matched = sum(
                1
                for k in (account_number_map or {})
                if normalize_creditor_name(k) in accounts_by_norm
            )
            unresolved = total - matched
            logger.info(
                "DBG join_result map=account_number matched=%d unresolved=%d",
                matched,
                unresolved,
            )
        except Exception:
            pass
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            remarks_map,
            "remarks",
            heading_map,
            is_bureau_map=True,
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            status_text_map,
            "account_status",
            heading_map,
            is_bureau_map=True,
        )
        _join_heading_map(
            accounts_by_norm,
            existing_norms,
            account_number_map,
            None,
            heading_map,
        )
        try:
            total = len(detail_map or {})
            matched = sum(
                1
                for k in (detail_map or {})
                if normalize_creditor_name(k) in accounts_by_norm
            )
            unresolved = total - matched
            logger.info(
                "DBG join_result map=bureau_details matched=%d unresolved=%d",
                matched,
                unresolved,
            )
        except Exception:
            pass
        # Reconcile global maps for later bookkeeping
        _join_heading_map(accounts_by_norm, existing_norms, history_all, None, raw_map)
        _join_heading_map(accounts_by_norm, existing_norms, grid_all, None, raw_map)

        def _apply_late_flags(acc_list, section_name):
            for acc in acc_list or []:
                norm = acc.get("normalized_name")
                if norm in history and any(
                    v >= 1 for vals in history[norm].values() for v in vals.values()
                ):
                    acc.setdefault("flags", []).append("Late Payments")
                    if section_name not in [
                        "negative_accounts",
                        "open_accounts_with_issues",
                    ]:
                        acc["goodwill_candidate"] = True
                    status_text = (
                        str(acc.get("status") or acc.get("account_status") or "")
                        .strip()
                        .lower()
                    )
                    if status_text == "closed":
                        acc["goodwill_on_closed"] = True

        for section in sections:
            _apply_late_flags(result.get(section, []), section)

        for raw_norm, bureaus in history_all.items():
            linked = raw_norm in history
            if linked:
                print(
                    f"[INFO] Linked late payment block '{raw_map.get(raw_norm, raw_norm)}' to account '{raw_norm.title()}'"
                )
            else:
                snippet = raw_map.get(raw_norm, raw_norm)
                print(f"[WARN] Unlinked late-payment block detected near: '{snippet}'")

        # Remove any late_payment fields that were not verified by parser
        verified_names = set(history.keys())

        def strip_unverified(acc_list):
            for acc in acc_list:
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                acc["normalized_name"] = norm
                if "late_payments" in acc and norm not in verified_names:
                    acc.pop("late_payments", None)

        for sec in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            strip_unverified(result.get(sec, []))

        _cleanup_unverified_late_text(result, verified_names)

        _inject_missing_late_accounts(result, history_all, raw_map, grid_all)

        _attach_parser_signals(
            result.get("all_accounts"),
            payment_status_map,
            remarks_map,
            _payment_status_raw_map,
        )

        _merge_parser_inquiries(result, parsed_inquiries, inquiry_raw_map)

        def _merge_account_numbers(acc_list, field_map):
            for acc in acc_list or []:
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                raw_name = acc.get("name", "")
                acc["normalized_name"] = norm
                values_map = field_map.get(norm)
                if not values_map:
                    logger.debug(
                        "heading_join_unresolved %s",
                        json.dumps(
                            {
                                "raw_key": raw_name,
                                "normalized": norm,
                                "target_present": False,
                                "map": "account_number",
                            },
                            sort_keys=True,
                        ),
                    )
                    continue
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {
                            "raw_key": raw_name,
                            "normalized_target": norm,
                            "method": "canonical",
                        },
                        sort_keys=True,
                    ),
                )
                acc.setdefault("bureaus", [])
                raw_unique: set[str] = set()
                digit_unique: set[str] = set()
                for bureau, raw in values_map.items():
                    info = None
                    for b in acc["bureaus"]:
                        if isinstance(b, dict) and b.get("bureau") == bureau:
                            info = b
                            break
                    if info is None:
                        info = {"bureau": bureau}
                        acc["bureaus"].append(info)
                    if not info.get("account_number_raw"):
                        info["account_number_raw"] = raw
                    digits = re.sub(r"\D", "", raw)
                    if digits and not info.get("account_number"):
                        info["account_number"] = digits
                    if raw:
                        raw_unique.add(raw)
                    if digits:
                        digit_unique.add(digits)
                if not acc.get("account_number_raw") and len(raw_unique) == 1:
                    acc["account_number_raw"] = next(iter(raw_unique))
                if not acc.get("account_number") and len(digit_unique) == 1:
                    acc["account_number"] = next(iter(digit_unique))

        def _merge_bureau_details(acc_list, detail_map):
            for acc in acc_list or []:
                norm = acc.get("normalized_name") or normalize_creditor_name(
                    acc.get("name", "")
                )
                raw_name = acc.get("name", "")
                acc["normalized_name"] = norm
                values_map = detail_map.get(norm)
                if not values_map:
                    logger.debug(
                        "heading_join_unresolved %s",
                        json.dumps(
                            {
                                "raw_key": raw_name,
                                "normalized": norm,
                                "target_present": False,
                                "map": "bureau_details",
                            },
                            sort_keys=True,
                        ),
                    )
                    continue
                logger.info(
                    "heading_join_linked %s",
                    json.dumps(
                        {
                            "raw_key": raw_name,
                            "normalized_target": norm,
                            "method": "canonical",
                        },
                        sort_keys=True,
                    ),
                )
                bd = acc.setdefault("bureau_details", {})
                for bureau, fields in values_map.items():
                    bkey = norm_bureau(bureau)
                    bd.setdefault(bkey, {}).update(fields)

        for sec in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            _merge_account_numbers(result.get(sec, []), account_number_map)
            _merge_bureau_details(result.get(sec, []), detail_map)

        for section in [
            "all_accounts",
            "negative_accounts",
            "open_accounts_with_issues",
            "positive_accounts",
            "high_utilization_accounts",
        ]:
            for acc in result.get(section, []):
                enforce_collection_status(acc)

        candidate_logger = CandidateTokenLogger()
        trace_logger = StageATraceLogger(request_id)
        for acc in result.get("all_accounts", []):
            candidate_logger.collect(acc)
            verdict = evaluate_account_problem(acc)
            if acc.get("_detector_is_problem"):
                trace_logger.append(
                    {
                        "normalized_name": acc.get("normalized_name"),
                        "account_id": acc.get("account_number_last4")
                        or acc.get("account_fingerprint"),
                        "decision_source": verdict.get("decision_source"),
                        "primary_issue": verdict.get("primary_issue"),
                        "confidence": verdict.get("confidence"),
                        "tier": verdict.get("tier"),
                        "reasons": verdict.get("problem_reasons", []),
                        "ai_latency_ms": verdict.get("debug", {}).get(
                            "ai_latency_ms", 0
                        ),
                        "ai_tokens_in": verdict.get("debug", {}).get("ai_tokens_in", 0),
                        "ai_tokens_out": verdict.get("debug", {}).get(
                            "ai_tokens_out", 0
                        ),
                        "ai_error": verdict.get("debug", {}).get("ai_error"),
                    }
                )

        # candidate logger only; do not re-derive problem_accounts here (finalize handles it)
        candidate_logger.save(Path("client_output") / request_id)

        if os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1":
            for acc in result.get("all_accounts", []):
                acc["primary_issue"] = "unknown"
                acc["issue_types"] = []
            result["negative_accounts"] = []
            result["open_accounts_with_issues"] = []
            result["positive_accounts"] = []
            result["high_utilization_accounts"] = []
        else:
            for acc in result.get("all_accounts", []):
                _assign_issue_types(acc)

            _all = result.get("all_accounts") or []
            _all_copy = _copy.deepcopy(_all)
            negatives, open_issues = _split_account_buckets(_all_copy)
            result["negative_accounts"] = negatives
            result["open_accounts_with_issues"] = open_issues

        # In deterministic mode, inquiries come solely from parser

    except Exception as e:
        print(f"[WARN] Late history parsing failed: {e}")

    # Load previously exported blocks and build fuzzy index without re-exporting
    fbk_blocks = load_account_blocks(sid)
    logger.info("ANZ: loaded %d pre-exported blocks for sid=%s", len(fbk_blocks), sid)
    result["fbk_blocks"] = fbk_blocks
    result["blocks_by_account_fuzzy"] = (
        build_block_fuzzy(fbk_blocks) if fbk_blocks else {}
    )

    # --- BEGIN: persist identifiers on analysis result ---
    try:
        result["session_id"] = session_id or request_id
        result["request_id"] = result.get("request_id") or request_id
    except Exception:
        logger.exception("persist_identifiers_failed")
    # --- END: persist identifiers on analysis result ---

    # --- BEGIN: ANZ diagnostics ---
    logger.warning(
        "ANZ: pre-save fbk=%d fuzzy=%d sid=%s req=%s",
        len(result.get("fbk_blocks") or []),
        len((result.get("blocks_by_account_fuzzy") or {}).keys()),
        result.get("session_id"),
        result.get("request_id"),
    )
    # --- END: ANZ diagnostics ---

    issues = validate_analysis_sanity(result)
    try:
        # Warn only if late terms present but no relevant problem accounts
        relevant = {
            "past_due",
            "late_payment",
            "collection",
            "charge_off",
            "collection/chargeoff",
            "repossessed",
            "foreclosure",
            "bankruptcy",
        }
        has_relevant = any(
            (acc.get("primary_issue") or "").lower() in relevant
            for acc in result.get("problem_accounts", [])
        )
        late_terms = detected_late_phrases(text)
        if late_terms and not has_relevant:
            msg = "WARN Late payment terms found in text but no accounts marked with issues."
            issues.append(msg)
            print(msg)
    except Exception:
        pass

    for section in [
        "all_accounts",
        "negative_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "high_utilization_accounts",
    ]:
        result[section] = [
            enrich_account_metadata(acc) for acc in result.get(section, [])
        ]

    if ENABLE_CASESTORE_WRITE and session_id:
        try:
            case = load_session_case(session_id)
            case.report_meta.personal_information.name = client_info.get("name")
            case.report_meta.inquiries = result.get("inquiries") or []
            case.report_meta.public_information = result.get("public_information") or []
            if not DETERMINISTIC_EXTRACTORS_ENABLED:
                case.summary.total_accounts = len(result.get("all_accounts") or [])
            save_session_case(case)
            try:
                from backend.config import CASESTORE_DIR

                path = Path(CASESTORE_DIR) / f"{session_id}.json"
                size = path.stat().st_size if path.exists() else 0
            except Exception:
                path = None
                size = 0
            logger.debug(
                "CASEBUILDER: save_session_case(session_id=%s, path=%s, size=%d)",
                session_id,
                path,
                size,
            )
        except CaseStoreError as err:  # pragma: no cover - best effort
            logger.warning(
                "casestore_session_error session=%s error=%s",
                session_id,
                err,
            )

        def _count_fields(mapping: Mapping[str, Any]) -> int:
            return sum(
                1
                for v in mapping.values()
                if v not in (None, "") and not isinstance(v, (dict, list, tuple, set))
            )

        fields_written = 0
        for idx, acc in enumerate(result.get("all_accounts", []), start=1):
            bureau_details = acc.get("bureau_details") or {}
            fingerprint = acc.get("account_fingerprint") or f"acc{idx}"
            for bureau, fields in bureau_details.items():
                account_id = f"{fingerprint}_{bureau}"
                try:
                    upsert_account_fields(
                        session_id=session_id,
                        account_id=account_id,
                        bureau=bureau,
                        fields=fields,
                    )
                    fields_written += _count_fields(fields)
                    if CASESTORE_PARSER_LOG_PARITY:
                        stored = (
                            redact_account_fields(fields)
                            if CASESTORE_REDACT_BEFORE_STORE
                            else fields
                        )
                        added, changed, masked, missing = _parity_counts(fields, stored)
                        logger.info(
                            "casestore_parity: session=%s account=%s bureau=%s added=%d changed=%d masked=%d missing=%d",
                            session_id,
                            account_id,
                            bureau,
                            added,
                            changed,
                            masked,
                            missing,
                        )
                except CaseStoreError as err:  # pragma: no cover - best effort
                    logger.warning(
                        "casestore_upsert_error session=%s account=%s bureau=%s error=%s",
                        session_id,
                        account_id,
                        bureau,
                        err,
                    )

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from backend.core.utils.json_utils import _json_safe as __json_safe
    except Exception:

        def __json_safe(x):
            return x

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(__json_safe(result), f, indent=4, ensure_ascii=False)

    _emit_audit()
    # Compare finalized snapshot with emit-time problem_accounts to detect overwrites
    try:
        if _finalized_snapshot is not None:
            _final_now = [dict(a) for a in (result.get("problem_accounts") or [])]
            for before, after in zip(_finalized_snapshot, _final_now):
                if before.get("normalized_name") != after.get("normalized_name"):
                    continue
                if before.get("primary_issue") != after.get("primary_issue"):
                    logger.error(
                        "BUG overwrite_primary_after_finalize name=%s before=%s after=%s",
                        after.get("normalized_name"),
                        before.get("primary_issue"),
                        after.get("primary_issue"),
                    )
                b_adv = before.get("advisor_comment") or ""
                a_adv = after.get("advisor_comment") or ""
                if b_adv != a_adv:
                    logger.error(
                        "BUG overwrite_advisor_after_finalize name=%s before_len=%d after_len=%d",
                        after.get("normalized_name"),
                        len(b_adv),
                        len(a_adv),
                    )
    except Exception as e:
        logger.exception("BUG finalize_integrity_check_failed: %s", e)

    # Ensure per-bureau meta tables with the 25-field set exist on each account
    try:
        attach_bureau_meta_tables(result)
    except Exception:
        # Non-fatal; continue without blocking
        logger.exception("attach_bureau_meta_tables_failed")

    return result


def dev_parse_text_dump(dump_path: str, expect_bureau: str | None = None):
    """Developer hook: parse an extracted text dump and print diagnostics.

    This utility does not alter pipeline behavior; it's a quick way to test
    fallback regexes directly on a text dump.
    """
    import json as _json
    import re as _re
    import unicodedata as _ud

    full_text = Path(dump_path).read_text(encoding="utf-8", errors="ignore")
    print(f"FBK: raw len={len(full_text)}")

    # Quick raw scan for NEG_STATUS_RE
    raw_hits = list(NEG_STATUS_RE.finditer(full_text))
    print("FBK: NEG_STATUS_RE hits (raw):", len(raw_hits))
    for h in raw_hits[:5]:
        s = max(0, h.start() - 30)
        e = min(len(full_text), h.end() + 30)
        print("FBK: HIT(raw):", full_text[s:e].replace("\n", "\\n"))

    # Unicode normalization probe
    n = _ud.normalize("NFKC", full_text)
    print("FBK: NFKC changed?", len(n) != len(full_text))
    print("FBK: contains NBSP?", "\u00A0" in full_text)
    print("FBK: contains soft hyphen?", "\u00AD" in full_text)

    # Diagnostic relaxed pattern
    DIAG_NEG_STATUS_RE = _re.compile(
        r"(?i)(?:"
        r"late[\s\W_]*(?:30|60|90|120|150)[\s\W_]*days?"
        r"|(?:30|60|90|120|150)[\s\W_]*days?[\s\W_]*late"
        r"|late[\s\W_]*(?:30|60|90|120|150)"
        r"|past[\s\W_]*due(?:[\s\W_]*\d+[\s\W_]*days?)?"
        r"|collections?"
        r"|charge[\s\W_]*-?[\s\W_]*off|charged[\s\W_]*-?[\s\W_]*off|chargeoff"
        r"|repossession|foreclosure|bankruptcy"
        r"|delinquen\w+|derogator\w+|settled[\s\W_]*for[\s\W_]*less"
        r")"
    )

    text_norm = normalize_for_regex(full_text)
    diag_hits = list(DIAG_NEG_STATUS_RE.finditer(text_norm))
    print("FBK: DIAG_NEG_STATUS_RE hits (norm):", len(diag_hits))
    for h in diag_hits[:5]:
        s = max(0, h.start() - 30)
        e = min(len(text_norm), h.end() + 30)
        print("FBK: HIT(norm):", text_norm[s:e].replace("\n", "\\n"))

    # Use the existing block extractor on normalized text
    try:
        blocks = extract_account_blocks(text_norm)
    except Exception:
        blocks = []
    print(f"FBK-DEV: blocks found={len(blocks)}")

    if any("PALISADES" in (b[0] if b else "") for b in blocks):
        print("FBK-DEV: PALISADES heading present in blocks")
    else:
        print("FBK-DEV: PALISADES heading NOT found in blocks")

    from collections import defaultdict

    payment_status_map: dict[str, dict[str, str]] = defaultdict(dict)
    bureaus_seen = 0
    for block in blocks:
        if not block:
            continue
        acc = normalize_creditor_name(block[0].strip())
        current_bureau: str | None = None
        in_values = False
        line_idx_in_bureau = 0
        for raw in block[1:]:
            clean = raw.strip()
            if (
                "Two-Year Payment History" in clean
                or "Days Late - 7 Year History" in clean
            ):
                in_values = False
                continue
            m = BUREAU_RE.match(clean)
            if m:
                current_bureau = m.group(1).title()
                in_values = True
                line_idx_in_bureau = 0
                bureaus_seen += 1
                continue
            if not in_values or not current_bureau:
                continue
            line_idx_in_bureau += 1
            hit = DIAG_NEG_STATUS_RE.search(clean) or NEG_STATUS_RE.search(clean)
            if hit and current_bureau not in payment_status_map.get(acc, {}):
                payment_status_map.setdefault(acc, {})[current_bureau] = hit.group(0)
                if expect_bureau and current_bureau == expect_bureau:
                    print(f"FBK-DEV: {acc} under {expect_bureau} -> {hit.group(0)}")

    sample = list(payment_status_map.items())[:5]
    print("FBK-DEV: payment_status_map sample:", _json.dumps(sample, indent=2))

    def _is_neg(s: str) -> bool:
        return bool(DIAG_NEG_STATUS_RE.search(s or "") or NEG_STATUS_RE.search(s or ""))

    problem = []
    for acc, bureaus in payment_status_map.items():
        if any(_is_neg(v) for v in bureaus.values()):
            problem.append({"creditor": acc, "payment_statuses": bureaus})

    print(f"FBK-DEV: bureaus_seen={bureaus_seen} problem_accounts={len(problem)}")
    for row in problem[:5]:
        print("FBK-DEV: example:", _json.dumps(row, indent=2))
