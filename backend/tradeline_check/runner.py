"""Tradeline check runner: orchestrates per-account checks and writes outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from backend.validation.pipeline import AccountContext
from backend.tradeline_check.config import TradlineCheckConfig
from backend.tradeline_check.schema import bureau_output_template, SUPPORTED_BUREAUS
from backend.tradeline_check.writer import write_bureau_findings

log = logging.getLogger(__name__)


def _isoformat_timestamp(now: datetime | None = None) -> str:
    """Return ISO 8601 UTC timestamp string."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_bureaus_list(bureaus_path: Path) -> set[str]:
    """Load list of bureaus present in bureaus.json.

    Parameters
    ----------
    bureaus_path
        Path to cases/accounts/<id>/bureaus.json

    Returns
    -------
    set[str]
        Set of bureau names found in the file (e.g., {"equifax", "experian"})
    """
    if not bureaus_path.exists():
        return set()

    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as exc:
        log.warning(
            "TRADELINE_CHECK_BUREAUS_LOAD_FAILED path=%s error=%s",
            bureaus_path,
            exc,
            exc_info=True,
        )
        return set()

    if not isinstance(data, Mapping):
        return set()

    # bureaus.json is keyed by bureau name
    bureaus_found = set()
    for key in data.keys():
        if isinstance(key, str):
            key_lower = key.lower().strip()
            if key_lower in SUPPORTED_BUREAUS:
                bureaus_found.add(key_lower)

    return bureaus_found


def _load_bureaus_data(bureaus_path: Path) -> Mapping[str, object]:
    """Load raw bureaus.json content as a mapping.

    Returns an empty mapping on error. This is used to extract the per-bureau
    object strictly by bureau name (no merging across bureaus).
    """
    if not bureaus_path.exists():
        return {}

    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as exc:
        log.warning(
            "TRADELINE_CHECK_BUREAUS_DATA_LOAD_FAILED path=%s error=%s",
            bureaus_path,
            exc,
            exc_info=True,
        )
        return {}

    if not isinstance(data, Mapping):
        return {}

    return data


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Return True if value is considered missing per presence-only rules.

    Missing when:
    - value is None
    - value is a string that is empty after trim
    - value is a string that matches any configured placeholder token (case-insensitive)
    """
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return True
        if s.lower() in placeholders:
            return True
        return False
    # Non-string values are considered present by default for presence-only gate
    return False


def _compute_coverage(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Compute coverage map (missing core and branch fields) per bureau.

    Coverage is presence-only and non-blocking: it reports which fields are missing
    for deeper analysis, without affecting eligibility or status.

    Parameters
    ----------
    bureau_obj
        The per-bureau object from bureaus.json (e.g., bureaus_data["equifax"])
    bureaus_data
        Full bureaus.json mapping (for accessing shared history blocks by bureau key)
    bureau
        Bureau name (lowercase, e.g., "equifax")
    placeholders
        Set of placeholder tokens (lowercase, e.g., {"--", "n/a", "unknown"})

    Returns
    -------
    dict
        Coverage block with missing_core_fields and missing_branch_fields per Q
    """

    def _is_field_missing(field_name: str) -> bool:
        """Check if a specific field is missing in bureau_obj."""
        return _is_missing(bureau_obj.get(field_name), placeholders)

    def _is_history_present(history_key: str) -> bool:
        """Check if a history block (e.g., 'two_year_payment_history') is present for this bureau."""
        # History blocks are keyed by bureau name in bureaus.json root
        hist_block = bureaus_data.get(history_key)
        if not isinstance(hist_block, Mapping):
            return False
        # Check if this bureau has an entry in the history block (non-empty)
        hist_entry = hist_block.get(bureau)
        if hist_entry is None or hist_entry == "" or (isinstance(hist_entry, str) and hist_entry.strip() == ""):
            return False
        return True

    missing_core: dict[str, list[str]] = {
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "Q4": [],
        "Q5": [],
    }
    missing_branch: dict[str, list[str]] = {
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "Q4": [],
        "Q5": [],
    }

    # ── Q1: Account State ──────────────────────────────────────────────
    q1_core = ["account_status", "account_rating", "payment_status"]
    q1_branch = ["dispute_status", "creditor_remarks", "date_reported", "date_of_last_activity"]

    for field in q1_core:
        if _is_field_missing(field):
            missing_core["Q1"].append(field)

    for field in q1_branch:
        if _is_field_missing(field):
            missing_branch["Q1"].append(field)

    # ── Q2: Active vs Historical ──────────────────────────────────────
    q2_core = ["date_of_last_activity", "last_payment", "date_reported", "closed_date"]
    q2_branch = [
        "payment_amount",
        "past_due_amount",
        "balance_owed",
    ]
    # History blocks (bureau-scoped)
    q2_branch_history = ["two_year_payment_history", "seven_year_history"]

    for field in q2_core:
        if _is_field_missing(field):
            missing_core["Q2"].append(field)

    for field in q2_branch:
        if _is_field_missing(field):
            missing_branch["Q2"].append(field)

    for hist_key in q2_branch_history:
        if not _is_history_present(hist_key):
            missing_branch["Q2"].append(hist_key)

    # ── Q3: Timeline Coherence ────────────────────────────────────────
    q3_core = ["date_opened", "date_reported", "date_of_last_activity", "last_payment", "closed_date"]
    q3_branch = []
    # History blocks (bureau-scoped)
    q3_branch_history = ["two_year_payment_history", "seven_year_history"]

    for field in q3_core:
        if _is_field_missing(field):
            missing_core["Q3"].append(field)

    for field in q3_branch:
        if _is_field_missing(field):
            missing_branch["Q3"].append(field)

    for hist_key in q3_branch_history:
        if not _is_history_present(hist_key):
            missing_branch["Q3"].append(hist_key)

    # ── Q4: Account Type Integrity ────────────────────────────────────
    q4_core = ["account_type", "creditor_type", "term_length", "payment_frequency"]
    q4_branch = ["credit_limit", "high_balance", "payment_amount", "original_creditor"]

    for field in q4_core:
        if _is_field_missing(field):
            missing_core["Q4"].append(field)

    for field in q4_branch:
        if _is_field_missing(field):
            missing_branch["Q4"].append(field)

    # ── Q5: Ownership & Responsibility ─────────────────────────────────
    q5_core = ["account_description"]
    q5_branch: list[str] = []  # no branch fields in Q5 v1

    for field in q5_core:
        if _is_field_missing(field):
            missing_core["Q5"].append(field)

    for field in q5_branch:
        if _is_field_missing(field):
            missing_branch["Q5"].append(field)

    return {
        "version": "coverage_v1",
        "placeholders": sorted(placeholders),
        "missing_core_fields": missing_core,
        "missing_branch_fields": missing_branch,
    }


def run_for_account(
    acc_ctx: AccountContext,
    *,
    cfg: TradlineCheckConfig | None = None,
) -> dict:
    """Run tradeline_check for a single account.

    Outputs are written per-bureau under:
      cases/accounts/<account_dir>/tradeline_check/<bureau>.json

    Parameters
    ----------
    acc_ctx
        Account context with paths and identifiers
    cfg
        Optional TradlineCheckConfig; if None, loaded from environment

    Returns
    -------
    dict
        Summary result with keys: wrote_files, bureaus_checked, status, errors
    """
    if cfg is None:
        cfg = TradlineCheckConfig.from_env()

    if not cfg.enabled:
        return {
            "status": "disabled",
            "wrote_files": 0,
            "bureaus_checked": 0,
            "errors": 0,
        }

    account_key = acc_ctx.account_key
    account_dir = acc_ctx.account_dir
    bureaus_path = acc_ctx.bureaus_path

    log.info(
        "TRADELINE_CHECK_START account_key=%s dir=%s",
        account_key,
        account_dir,
    )

    result = {
        "status": "ok",
        "wrote_files": 0,
        "bureaus_checked": 0,
        "errors": 0,
    }

    try:
        # Discover which bureaus are present in the account
        bureaus_present = _load_bureaus_list(bureaus_path)
        result["bureaus_checked"] = len(bureaus_present)

        if not bureaus_present:
            log.info(
                "TRADELINE_CHECK_NO_BUREAUS account_key=%s path=%s",
                account_key,
                bureaus_path,
            )
            return result

        # Create tradeline_check output directory
        tradeline_output_dir = account_dir / "tradeline_check"

        timestamp = _isoformat_timestamp()

        # Load raw bureaus data once for per-bureau isolation reads
        bureaus_data = _load_bureaus_data(bureaus_path)
        placeholders = set(cfg.placeholder_tokens or set())

        # Write one JSON per bureau found
        for bureau in sorted(bureaus_present):
            try:
                # Create minimal payload for this bureau
                payload = bureau_output_template(
                    account_key=account_key,
                    bureau=bureau,
                    generated_at=timestamp,
                )

                # ── Q6 Presence-only Gate (bureau-isolated) ─────────────
                bureau_obj = (
                    bureaus_data.get(bureau) if isinstance(bureaus_data, Mapping) else None
                )
                if not isinstance(bureau_obj, Mapping):
                    bureau_obj = {}

                # Fields groups
                status_fields = ("account_status", "account_rating", "payment_status")
                activity_fields = ("date_of_last_activity", "last_payment")
                secondary_dates = (
                    "date_reported",
                    "date_of_last_activity",
                    "last_payment",
                    "closed_date",
                )

                # Presence checks
                status_present = any(not _is_missing(bureau_obj.get(f), placeholders) for f in status_fields)
                activity_present = any(not _is_missing(bureau_obj.get(f), placeholders) for f in activity_fields)
                opened_present = not _is_missing(bureau_obj.get("date_opened"), placeholders)
                secondary_present = any(not _is_missing(bureau_obj.get(f), placeholders) for f in secondary_dates)
                type_present = any(
                    not _is_missing(bureau_obj.get(f), placeholders)
                    for f in ("account_type", "creditor_type")
                )
                desc_present = not _is_missing(bureau_obj.get("account_description"), placeholders)

                eligible = {
                    "Q1": bool(status_present),
                    "Q2": bool(status_present and activity_present),
                    "Q3": bool(opened_present and secondary_present),
                    "Q4": bool(type_present),
                    "Q5": bool(desc_present),
                }

                missing_fields: dict[str, list[str]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": [], "Q5": []}

                # Q1 missing: include all three only if none present
                if not status_present:
                    missing_fields["Q1"] = list(status_fields)

                # Q2 missing: reflect missing group(s)
                if not status_present:
                    missing_fields["Q2"].extend(status_fields)
                if not activity_present:
                    missing_fields["Q2"].extend(activity_fields)

                # Q3 missing: require date_opened and at least one secondary
                if not opened_present:
                    missing_fields["Q3"].append("date_opened")
                if not secondary_present:
                    missing_fields["Q3"].extend(list(secondary_dates))

                # Q4 missing: include both only if none present
                if not type_present:
                    missing_fields["Q4"] = ["account_type", "creditor_type"]

                # Q5 missing: account_description
                if not desc_present:
                    missing_fields["Q5"] = ["account_description"]

                payload["gate"] = {
                    "version": "q6_presence_v1",
                    "eligible": eligible,
                    "missing_fields": missing_fields,
                    "placeholders": sorted(placeholders),
                }

                payload["root_checks"] = {
                    "Q1": {"status": "not_implemented_yet"},
                    "Q2": {"status": "not_implemented_yet"},
                    "Q3": {"status": "not_implemented_yet"},
                    "Q4": {"status": "not_implemented_yet"},
                    "Q5": {"status": "skipped_missing_data"},
                }

                # ── Coverage Map (non-blocking, capability awareness) ───
                coverage = _compute_coverage(bureau_obj, bureaus_data, bureau, placeholders)
                payload["coverage"] = coverage

                # ── Q1 Account State Declaration (non-blocking) ─────────
                try:
                    from backend.tradeline_check.q1_account_state import evaluate_q1
                    payload["root_checks"]["Q1"] = evaluate_q1(bureau_obj, placeholders)
                except Exception as q1_exc:
                    log.warning(
                        "TRADELINE_CHECK_Q1_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        q1_exc,
                        exc_info=True,
                    )

                # ── Q2 Activity expectation vs evidence (non-blocking) ──
                try:
                    from backend.tradeline_check.q2_activity import evaluate_q2
                    payload["root_checks"]["Q2"] = evaluate_q2(
                        bureau_obj,
                        payload["root_checks"].get("Q1", {}),
                        placeholders,
                    )
                except Exception as q2_exc:
                    log.warning(
                        "TRADELINE_CHECK_Q2_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        q2_exc,
                        exc_info=True,
                    )

                # ── Q3 Timeline coherence (non-blocking) ──────────────
                try:
                    from backend.tradeline_check.q3_timeline import evaluate_q3
                    payload["root_checks"]["Q3"] = evaluate_q3(
                        bureau_obj,
                        placeholders,
                    )
                except Exception as q3_exc:
                    log.warning(
                        "TRADELINE_CHECK_Q3_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        q3_exc,
                        exc_info=True,
                    )

                # ── Q4 Account Type Integrity (non-blocking) ─────────
                try:
                    from backend.tradeline_check.q4_type_integrity import evaluate_q4
                    payload["root_checks"]["Q4"] = evaluate_q4(
                        bureau_obj,
                        placeholders,
                    )
                except Exception as q4_exc:
                    log.warning(
                        "TRADELINE_CHECK_Q4_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        q4_exc,
                        exc_info=True,
                    )

                # ── Q5 Ownership & Responsibility (non-blocking) ─────────
                try:
                    from backend.tradeline_check.q5_ownership import evaluate_q5
                    payload["root_checks"]["Q5"] = evaluate_q5(
                        bureau_obj,
                        placeholders,
                    )
                except Exception as q5_exc:
                    log.warning(
                        "TRADELINE_CHECK_Q5_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        q5_exc,
                        exc_info=True,
                    )

                # Optional strict mode: block if any of Q1/Q3/Q4 ineligible
                if cfg.gate_strict:
                    failed = [q for q in ("Q1", "Q3", "Q4") if not eligible.get(q, False)]
                    if failed:
                        payload["status"] = "blocked"
                        payload["blocked_questions"] = failed

                # Atomically write the bureau output
                output_path = write_bureau_findings(
                    tradeline_output_dir,
                    account_key,
                    bureau,
                    payload,
                )

                log.debug(
                    "TRADELINE_CHECK_BUREAU_WRITTEN account_key=%s bureau=%s path=%s",
                    account_key,
                    bureau,
                    output_path,
                )

                result["wrote_files"] += 1

            except Exception as exc:
                log.error(
                    "TRADELINE_CHECK_BUREAU_FAILED account_key=%s bureau=%s error=%s",
                    account_key,
                    bureau,
                    exc,
                    exc_info=True,
                )
                result["errors"] += 1

    except Exception as exc:
        log.error(
            "TRADELINE_CHECK_FAILED account_key=%s error=%s",
            account_key,
            exc,
            exc_info=True,
        )
        result["status"] = "error"
        result["errors"] += 1
        return result

    log.info(
        "TRADELINE_CHECK_DONE account_key=%s wrote_files=%d errors=%d",
        account_key,
        result["wrote_files"],
        result["errors"],
    )

    return result
