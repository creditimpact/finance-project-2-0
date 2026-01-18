"""Tradeline check runner: orchestrates per-account checks and writes outputs."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from backend.validation.pipeline import AccountContext
from backend.tradeline_check.config import TradlineCheckConfig
from backend.tradeline_check.schema import bureau_output_template, SUPPORTED_BUREAUS
from backend.tradeline_check.date_convention import load_date_convention
from backend.tradeline_check.writer import write_bureau_findings
from backend.tradeline_check.branch_registry import BRANCH_REGISTRY, is_branch_eligible, invoke_branch_by_path

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
        # Prefer monthly TSV v2 for 2Y history, fall back to legacy list/string
        if history_key == "two_year_payment_history":
            monthly_block = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
            if isinstance(monthly_block, Mapping):
                monthly_entries = monthly_block.get(bureau)
                if isinstance(monthly_entries, list) and any(
                    isinstance(entry, Mapping) and str(entry.get("month", "")).strip()
                    for entry in monthly_entries
                ):
                    return True

        # History blocks are keyed by bureau name in bureaus.json root
        hist_block = bureaus_data.get(history_key)
        if not isinstance(hist_block, Mapping):
            return False

        # two_year_payment_history legacy fallback: non-empty list or string
        if history_key == "two_year_payment_history":
            hist_entry = hist_block.get(bureau)
            if isinstance(hist_entry, list):
                return len(hist_entry) > 0
            if isinstance(hist_entry, str):
                return bool(hist_entry.strip())
            return False

        # Default behavior for other history blocks (unchanged)
        hist_entry = hist_block.get(bureau)
        if hist_entry is None or hist_entry == "" or (isinstance(hist_entry, str) and hist_entry.strip() == ""):
            return False
        return True

    missing_core: dict[str, list[str]] = {
        "Q1": [],
    }
    missing_branch: dict[str, list[str]] = {
        "Q1": [],
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

    return {
        "version": "coverage_v1",
        "placeholders": sorted(placeholders),
        "missing_core_fields": missing_core,
        "missing_branch_fields": missing_branch,
    }


def _prune_root_checks_for_public(root_checks: Mapping[str, object] | None) -> dict:
    """Return a reduced root_checks view for public payloads."""

    if not isinstance(root_checks, Mapping):
        return {}

    public_root = {}

    q1 = root_checks.get("Q1")
    if isinstance(q1, Mapping):
        q1_public = {}
        for key in ("declared_state", "status", "explanation"):
            if key in q1:
                q1_public[key] = q1.get(key)
        public_root["Q1"] = q1_public

    return public_root


def _prune_routing_for_public(routing: Mapping[str, object] | None) -> dict:
    """Return a reduced routing view for public payloads."""

    if not isinstance(routing, Mapping):
        return {}

    public_routing = {}

    r1 = routing.get("R1")
    if isinstance(r1, Mapping):
        r1_public = {}
        for key in ("version", "state_id", "state_num"):
            if key in r1:
                r1_public[key] = r1.get(key)
        public_routing["R1"] = r1_public

    return public_routing


def project_public_payload(payload: Mapping[str, object]) -> dict:
    """Produce a filtered copy suitable for public output without mutating input."""

    public_payload = copy.deepcopy(payload) if isinstance(payload, Mapping) else payload
    if not isinstance(public_payload, dict):
        return public_payload

    # date_convention: collapse to string convention when available; otherwise omit
    convention_value = None
    if isinstance(payload, Mapping):
        date_conv_block = payload.get("date_convention")
        if isinstance(date_conv_block, Mapping):
            raw_conv = date_conv_block.get("convention")
            if isinstance(raw_conv, str) and raw_conv.strip():
                convention_value = raw_conv.strip()

    public_payload.pop("date_convention", None)
    if convention_value:
        public_payload["date_convention"] = convention_value

    # coverage: remove entirely
    public_payload.pop("coverage", None)

    # root_checks: reduce to allowed fields
    public_payload["root_checks"] = _prune_root_checks_for_public(payload.get("root_checks") if isinstance(payload, Mapping) else None)

    # routing: reduce to allowed fields
    public_payload["routing"] = _prune_routing_for_public(payload.get("routing") if isinstance(payload, Mapping) else None)

    return public_payload


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

        # Load date convention once per account (non-blocking)
        try:
            date_convention_block = load_date_convention(account_dir)
        except Exception as dc_exc:  # pragma: no cover - defensive
            log.warning(
                "TRADELINE_CHECK_DATE_CONVENTION_LOAD_FAILED account_key=%s error=%s",
                account_key,
                dc_exc,
                exc_info=True,
            )
            date_convention_block = {
                "version": "date_convention_v1",
                "scope": "unknown",
                "convention": "unknown",
                "month_language": "unknown",
                "confidence": 0.0,
                "evidence_counts": {},
                "detector_version": "unknown",
                "source": {
                    "file_abs": None,
                    "file_rel": "traces/date_convention.json",
                    "created_at": None,
                },
            }

        # Write one JSON per bureau found
        for bureau in sorted(bureaus_present):
            try:
                # Create minimal payload for this bureau
                payload = bureau_output_template(
                    account_key=account_key,
                    bureau=bureau,
                    generated_at=timestamp,
                )

                payload["date_convention"] = date_convention_block

                # Get bureau object for evaluations
                bureau_obj = (
                    bureaus_data.get(bureau) if isinstance(bureaus_data, Mapping) else None
                )
                if not isinstance(bureau_obj, Mapping):
                    bureau_obj = {}

                payload["root_checks"] = {
                    "Q1": {"status": "not_implemented_yet"},
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

                # ── F0.A01: Time Ceiling Integrity (record-level, non-blocking) ──
                try:
                    from backend.tradeline_check.f0_a01_time_ceiling_integrity import evaluate_f0_a01

                    record_integrity = payload.setdefault("record_integrity", {})
                    family_block = record_integrity.setdefault("F0", {})
                    family_block["A01"] = evaluate_f0_a01(
                        payload,
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                except Exception as f0_a01_exc:
                    log.warning(
                        "TRADELINE_CHECK_F0_A01_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        f0_a01_exc,
                        exc_info=True,
                    )

                # ── F0.A02: Opening Date Lower Bound Integrity (record-level, non-blocking) ──
                try:
                    from backend.tradeline_check.f0_a02_opening_date_lower_bound import evaluate_f0_a02

                    record_integrity = payload.setdefault("record_integrity", {})
                    family_block = record_integrity.setdefault("F0", {})
                    family_block["A02"] = evaluate_f0_a02(
                        payload,
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                except Exception as f0_a02_exc:
                    log.warning(
                        "TRADELINE_CHECK_F0_A02_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        f0_a02_exc,
                        exc_info=True,
                    )

                # ── F0.A03: Monthly Ceiling Integrity (record-level, non-blocking) ──
                try:
                    from backend.tradeline_check.f0_a03_monthly_ceiling_integrity import evaluate_f0_a03

                    record_integrity = payload.setdefault("record_integrity", {})
                    family_block = record_integrity.setdefault("F0", {})
                    family_block["A03"] = evaluate_f0_a03(
                        payload,
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                except Exception as f0_a03_exc:
                    log.warning(
                        "TRADELINE_CHECK_F0_A03_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        f0_a03_exc,
                        exc_info=True,
                    )

                # ── F0.A04: Monthly Floor Integrity (record-level, non-blocking) ──
                try:
                    from backend.tradeline_check.f0_a04_monthly_floor_integrity import evaluate_f0_a04

                    record_integrity = payload.setdefault("record_integrity", {})
                    family_block = record_integrity.setdefault("F0", {})
                    family_block["A04"] = evaluate_f0_a04(
                        payload,
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                except Exception as f0_a04_exc:
                    log.warning(
                        "TRADELINE_CHECK_F0_A04_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        f0_a04_exc,
                        exc_info=True,
                    )

                # ── R1 Router (4-state classifier, non-blocking) ────
                try:
                    from backend.tradeline_check.r1_router import evaluate_r1
                    payload.setdefault("routing", {})
                    payload["routing"]["R1"] = evaluate_r1(payload.get("root_checks", {}))
                except Exception as r1_exc:
                    log.warning(
                        "TRADELINE_CHECK_R1_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        r1_exc,
                        exc_info=True,
                    )

                # ── Branch Families Visibility (always-on, non-blocking) ───
                try:
                    from backend.tradeline_check.branch_visibility import build_branches_block

                    payload["branches"] = build_branches_block(payload)
                except Exception as branches_exc:
                    log.warning(
                        "TRADELINE_CHECK_BRANCH_VISIBILITY_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        branches_exc,
                        exc_info=True,
                    )

                # ── Branch Results Container (always-on, non-blocking) ─────
                try:
                    from backend.tradeline_check.branch_results import ensure_branch_results_container

                    ensure_branch_results_container(payload)
                except Exception as branch_results_exc:
                    log.warning(
                        "TRADELINE_CHECK_BRANCH_RESULTS_INIT_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        branch_results_exc,
                        exc_info=True,
                    )

                # ── FX: Always-Run Behavioral Branches (ungated) ─────────────
                # FX branches run unconditionally for every bureau. They do NOT:
                #   - read or depend on R1 state
                #   - affect routing or root_checks
                #   - modify payload status or gate
                # FX provides behavioral consistency signals complementary to Q1/R1.

                # ── FX.B01: Last Payment Monotonicity (ungated, always-run) ────
                try:
                    from backend.tradeline_check.fx_b01_last_payment_monotonicity import evaluate_fx_b01
                    fx_b01_result = evaluate_fx_b01(
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                    if isinstance(payload.get("branch_results"), Mapping):
                        if isinstance(payload["branch_results"].get("results"), dict):
                            payload["branch_results"]["results"]["FX.B01"] = fx_b01_result
                except Exception as fx_b01_exc:
                    log.warning(
                        "TRADELINE_CHECK_FX_B01_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        fx_b01_exc,
                        exc_info=True,
                    )

                # ── FX.B02: Seven-Year vs Two-Year Late Consistency (ungated, always-run) ────
                try:
                    from backend.tradeline_check.fx_b02_seven_year_vs_two_year_consistency import (
                        evaluate_fx_b02,
                    )

                    fx_b02_result = evaluate_fx_b02(
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                    if isinstance(payload.get("branch_results"), Mapping):
                        if isinstance(payload["branch_results"].get("results"), dict):
                            payload["branch_results"]["results"]["FX.B02"] = fx_b02_result
                except Exception as fx_b02_exc:
                    log.warning(
                        "TRADELINE_CHECK_FX_B02_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        fx_b02_exc,
                        exc_info=True,
                    )

                # ── FX.B03: Last Payment vs Monthly Coverage (ungated, always-run) ────
                try:
                    from backend.tradeline_check.fx_b03_last_payment_vs_monthly_coverage import evaluate_fx_b03
                    fx_b03_result = evaluate_fx_b03(
                        bureau_obj,
                        bureaus_data,
                        bureau,
                        placeholders,
                    )
                    if isinstance(payload.get("branch_results"), Mapping):
                        if isinstance(payload["branch_results"].get("results"), dict):
                            payload["branch_results"]["results"]["FX.B03"] = fx_b03_result
                except Exception as fx_b03_exc:
                    log.warning(
                        "TRADELINE_CHECK_FX_B03_EVAL_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        fx_b03_exc,
                        exc_info=True,
                    )

                # ── F1–F6: Conditional Family Branches (eligibility-gated) ────
                # These branches are only invoked when eligible for the current R1.state_num.
                # If a branch is not eligible, it is not invoked and does not appear in results.
                try:
                    r1_state_num = payload.get("routing", {}).get("R1", {}).get("state_num")
                    
                    # Iterate registry for F1–F6 branches (excluding F0/FX)
                    for branch_entry in BRANCH_REGISTRY:
                        branch_id = branch_entry.get("branch_id")
                        family_id = branch_entry.get("family_id")
                        
                        # Check eligibility before invocation
                        if not is_branch_eligible(branch_entry, r1_state_num):
                            # Branch is ineligible; skip invocation and result storage
                            log.debug(
                                "TRADELINE_CHECK_BRANCH_SKIPPED_INELIGIBLE account_key=%s bureau=%s branch=%s r1_state_num=%s",
                                account_key,
                                bureau,
                                branch_id,
                                r1_state_num,
                            )
                            continue
                        
                        # Branch is eligible; invoke it and store result
                        try:
                            evaluator_path = branch_entry.get("evaluator_path")
                            evaluator_args = branch_entry.get("evaluator_args", [])
                            
                            # Build argument dict for dynamic invocation
                            args_dict = {
                                "payload": payload,
                                "bureau_obj": bureau_obj,
                                "bureaus_data": bureaus_data,
                                "bureau": bureau,
                                "placeholders": placeholders,
                            }
                            
                            # Dynamically invoke branch evaluator
                            branch_result = invoke_branch_by_path(evaluator_path, args_dict, evaluator_args)
                            
                            # Store result only if eligible and successfully invoked
                            if isinstance(payload.get("branch_results"), Mapping):
                                if isinstance(payload["branch_results"].get("results"), dict):
                                    payload["branch_results"]["results"][branch_id] = branch_result
                            
                            log.debug(
                                "TRADELINE_CHECK_BRANCH_INVOKED account_key=%s bureau=%s branch=%s",
                                account_key,
                                bureau,
                                branch_id,
                            )
                        except Exception as branch_exc:
                            log.warning(
                                "TRADELINE_CHECK_BRANCH_EVAL_FAILED account_key=%s bureau=%s branch=%s error=%s",
                                account_key,
                                bureau,
                                branch_id,
                                branch_exc,
                                exc_info=True,
                            )
                except Exception as registry_exc:
                    log.warning(
                        "TRADELINE_CHECK_REGISTRY_DISPATCH_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        registry_exc,
                        exc_info=True,
                    )

                # ── Update branches visibility lists (non-blocking) ────────
                try:
                    from backend.tradeline_check.branch_visibility import update_branches_visibility
                    update_branches_visibility(payload)
                except Exception as vis_exc:
                    log.warning(
                        "TRADELINE_CHECK_BRANCHES_VISIBILITY_UPDATE_FAILED account_key=%s bureau=%s error=%s",
                        account_key,
                        bureau,
                        vis_exc,
                        exc_info=True,
                    )

                # Atomically write the bureau output (public projection)
                public_payload = project_public_payload(payload)
                output_path = write_bureau_findings(
                    tradeline_output_dir,
                    account_key,
                    bureau,
                    public_payload,
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
