from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping

import backend.config as config
from backend.core.ai.adjudicator_client import call_adjudicator
from backend.core.ai.models import AIAdjudicateRequest
from backend.core.case_store.api import append_artifact, get_account_case, list_accounts
from backend.core.case_store.errors import CaseStoreError
from backend.core.case_store.redaction import redact_for_ai
from backend.core.case_store import telemetry
from backend.core.telemetry import metrics
from backend.core.logic.report_analysis.candidate_logger import log_stageA_candidates
from backend.core.taxonomy.problem_taxonomy import compare_tiers, normalize_decision

logger = logging.getLogger(__name__)

# Field groups used for constructing problem reasons.  These are a subset of
# the fields fetched for Stage A; ``STAGEA_REQUIRED_FIELDS`` enumerates the
# complete minimal field set we request from Case Store.
EVIDENCE_FIELDS_NUMERIC = (
    "past_due_amount",
    "balance_owed",
    "credit_limit",
)
EVIDENCE_FIELDS_STATUS = ("payment_status", "account_status")
EVIDENCE_FIELDS_HISTORY = ("two_year_payment_history", "days_late_7y")

# Explicit list of all fields Stage A fetches from Case Store.  Keeping this
# centralized avoids drifting field usage between the detector and orchestrator.
STAGEA_REQUIRED_FIELDS = [
    "balance_owed",
    "payment_status",
    "account_status",
    "credit_limit",
    "past_due_amount",
    "account_rating",
    "account_description",
    "creditor_remarks",
    "original_creditor",
    "account_type",
    "creditor_type",
    "dispute_status",
    "two_year_payment_history",
    "days_late_7y",
]

NEUTRAL_TIER = "none"


def neutral_stageA_decision(debug: dict | None = None) -> dict:
    return {
        "primary_issue": "unknown",
        "issue_types": [],
        "problem_reasons": [],
        "decision_source": "rules",
        "confidence": 0.0,
        "tier": NEUTRAL_TIER,
        "debug": debug or {},
    }


def adopt_or_fallback(ai_resp: dict | None, min_conf: float) -> dict:
    """Return an AI decision if it meets quality gates, otherwise neutral."""
    if (
        ai_resp
        and ai_resp.get("primary_issue") not in {"none", "unknown", None}
        and ai_resp.get("confidence", 0.0) >= min_conf
    ):
        return {
            "primary_issue": ai_resp.get("primary_issue", "unknown"),
            "issue_types": ai_resp.get("issue_types", []),
            "problem_reasons": ai_resp.get("problem_reasons", []),
            "decision_source": "ai",
            "confidence": float(ai_resp.get("confidence", 0.0)),
            "tier": ai_resp.get("tier", NEUTRAL_TIER),
            "debug": {"fields_used": ai_resp.get("fields_used", [])},
        }
    return neutral_stageA_decision()


def evaluate_with_optional_ai(
    session_id: str,
    account_id: str,
    case_fields: dict,
    doc_fingerprint: str,
    account_fingerprint: str,
) -> tuple[dict, bool, float | None, str | None, float | None]:
    """Attempt AI adjudication and return decision with telemetry info.

    Returns a tuple ``(decision, ai_called, ai_latency_ms, fallback_reason,
    ai_confidence)`` where ``ai_called`` indicates whether the adjudicator was
    invoked and ``ai_latency_ms`` is the duration of that call when available.
    ``fallback_reason`` is populated when AI was attempted but not adopted.
    """

    if not config.ENABLE_AI_ADJUDICATOR:
        return neutral_stageA_decision(debug={"source": "rules_v1"}), False, None, None, None

    ai_fields = redact_for_ai({"fields": case_fields})["fields"]
    req = AIAdjudicateRequest(
        doc_fingerprint=doc_fingerprint or "",
        account_fingerprint=account_fingerprint or "",
        hierarchy_version=config.AI_HIERARCHY_VERSION,
        fields=ai_fields,
    )

    meta: dict[str, Any] = {}
    prev_emit = telemetry.get_emitter()

    def _capture(event: str, fields: Mapping[str, Any]) -> None:
        if event == "stageA_ai_call":
            meta.update(fields)
        if prev_emit:
            try:
                prev_emit(event, fields)
            except Exception:
                pass

    telemetry.set_emitter(_capture)
    try:
        resp = call_adjudicator(None, req)
    finally:
        telemetry.set_emitter(prev_emit)

    ai_latency = meta.get("duration_ms")
    status = meta.get("status")
    ai_conf = meta.get("confidence")

    resp_dict = None
    if resp:
        resp_dict = {
            "primary_issue": resp.primary_issue,
            "tier": resp.tier,
            "confidence": float(resp.confidence),
            "problem_reasons": resp.problem_reasons,
            "fields_used": resp.fields_used,
        }
    decision = adopt_or_fallback(resp_dict, config.AI_MIN_CONFIDENCE)

    fallback_reason: str | None = None
    if decision.get("decision_source") != "ai":
        if resp_dict:
            fallback_reason = "low_confidence"
            ai_conf = resp_dict.get("confidence")
        else:
            mapping = {
                "TimeoutException": "timeout",
                "HTTPStatusError": "http_error",
                "HTTPError": "http_error",
                "JSONDecodeError": "invalid_json",
                "ValidationError": "schema_reject",
            }
            fallback_reason = mapping.get(status, "http_error")

    return decision, True, ai_latency, fallback_reason, ai_conf if ai_conf is not None else None


def _format_amount(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _extract_late_counts(history) -> dict:
    codes: List[str] = []
    if history is None:
        return {}
    if isinstance(history, str):
        codes = [c.strip() for c in history.split(",")]
    elif isinstance(history, list):
        codes = [str(c).strip() for c in history]
    buckets = {"30": 0, "60": 0, "90": 0, "120": 0}
    for c in codes:
        if c in buckets:
            buckets[c] += 1
        elif c.endswith("D") and c[:-1] in buckets:
            buckets[c[:-1]] += 1
        elif c.endswith("+") and c[:-1] in buckets:
            buckets[c[:-1]] += 1
    return {k: v for k, v in buckets.items() if v > 0}


def build_problem_reasons(fields: dict) -> List[str]:
    reasons: List[str] = []
    if fields.get("past_due_amount", 0):
        reasons.append(f"past_due_amount: {_format_amount(fields['past_due_amount'])}")
    for fname in EVIDENCE_FIELDS_STATUS:
        if fields.get(fname):
            reasons.append(f"status_present: {fname}")
    for hname in EVIDENCE_FIELDS_HISTORY:
        counts = _extract_late_counts(fields.get(hname))
        if counts:
            bits = [f"{v}×{k}" for k, v in counts.items()]
            reasons.append(f"late: {','.join(bits)}")
    return reasons


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    reasons = build_problem_reasons(acct)
    signals: List[Any] = []
    if acct.get("past_due_amount", 0):
        signals.append("past_due_amount")
    for fname in EVIDENCE_FIELDS_STATUS:
        if acct.get(fname):
            signals.append(f"status_present:{fname}")
    for hname in EVIDENCE_FIELDS_HISTORY:
        counts = _extract_late_counts(acct.get(hname))
        if counts:
            signals.append({hname: counts})
    decision = neutral_stageA_decision(debug={"signals": signals})
    decision["problem_reasons"] = reasons
    acct.update({k: v for k, v in decision.items() if k != "debug"})
    acct["debug"] = decision["debug"]
    acct["_detector_is_problem"] = bool(reasons)
    return decision


def run_stage_a(
    session_id: str,
    legacy_accounts: List[Mapping[str, Any]] | None = None,
) -> None:
    try:
        account_ids = list_accounts(session_id)  # type: ignore[operator]
    except Exception:
        logger.warning("stageA_list_accounts_failed session=%s", session_id)
        return

    if not account_ids:
        raise CaseStoreError("no_account_cases", "Stage-A requires cases; got 0")

    for acc_id in account_ids:
        with telemetry.timed(
            "stageA_casestore_eval",
            session_id=session_id,
            account_id=acc_id,
            used_source="casestore",
        ):
            try:
                case = get_account_case(session_id, acc_id)  # type: ignore[operator]
            except Exception:
                logger.warning(
                    "stageA_missing_account session=%s account=%s", session_id, acc_id
                )
                telemetry.emit(
                    "stageA_missing_account",
                    session_id=session_id,
                    account_id=acc_id,
                )
                continue

            by_bureau = getattr(case.fields, "by_bureau", {}) or {}
            decisions: List[dict] = []
            for bureau, bureau_fields in by_bureau.items():
                metrics.increment(
                    "stageA.run.count", tags={"bureau": bureau}
                )
                try:
                    telemetry.emit(
                        "stageA.bureau_evaluations",
                        session_id=session_id,
                        account_id=acc_id,
                        bureau=bureau,
                    )
                except Exception:
                    pass

                rules_input = {
                    k: bureau_fields.get(k) for k in STAGEA_REQUIRED_FIELDS
                }

                if config.ENABLE_CANDIDATE_TOKEN_LOGGER:
                    try:
                        log_stageA_candidates(
                            session_id,
                            acc_id,
                            bureau,
                            "pre",
                            dict(rules_input),
                            decision={},
                            meta={"source": "stageA"},
                        )
                    except Exception:
                        logger.debug(
                            "candidate_tokens_log_failed session=%s account=%s phase=pre",
                            session_id,
                            acc_id,
                            exc_info=True,
                        )

                t0 = time.perf_counter()
                rules_verdict = evaluate_account_problem(dict(rules_input))
                if config.ENABLE_AI_ADJUDICATOR:
                    (
                        ai_verdict,
                        ai_called,
                        ai_latency_ms,
                        fallback_reason,
                        ai_confidence,
                    ) = evaluate_with_optional_ai(
                        session_id,
                        acc_id,
                        dict(rules_input),
                        str(bureau_fields.get("doc_fingerprint") or ""),
                        str(bureau_fields.get("account_fingerprint") or ""),
                    )
                else:
                    (
                        ai_verdict,
                        ai_called,
                        ai_latency_ms,
                        fallback_reason,
                        ai_confidence,
                    ) = (rules_verdict, False, 0, None, None)

                verdict = (
                    ai_verdict
                    if ai_called and ai_verdict.get("decision_source") == "ai"
                    else rules_verdict
                )
                verdict = normalize_decision(verdict)
                total_latency = (time.perf_counter() - t0) * 1000.0
                try:
                    telemetry.emit(
                        "stageA_eval",
                        session_id=session_id,
                        account_id=acc_id,
                        bureau=bureau,
                        decision_source=verdict.get("decision_source"),
                        primary_issue=verdict.get("primary_issue"),
                        tier=verdict.get("tier"),
                        confidence=float(verdict.get("confidence", 0.0)),
                        latency_ms=round(total_latency, 3),
                        ai_latency_ms=ai_latency_ms if ai_called else None,
                    )
                    if ai_called and verdict.get("decision_source") != "ai":
                        telemetry.emit(
                            "stageA_fallback",
                            session_id=session_id,
                            account_id=acc_id,
                            bureau=bureau,
                            reason=fallback_reason,
                            ai_confidence=ai_confidence,
                            latency_ms=ai_latency_ms,
                        )
                except Exception:
                    pass

                payload = {
                    **verdict,
                    "decision_source": "rules+ai" if ai_called else "rules",
                    "bureau": bureau,
                    "debug": {
                        "stage": "StageA",
                        "bureau": bureau,
                        "ai_called": ai_called,
                    },
                }

                if config.ENABLE_CANDIDATE_TOKEN_LOGGER:
                    try:
                        log_stageA_candidates(
                            session_id,
                            acc_id,
                            bureau,
                            "post",
                            dict(rules_input),
                            verdict,
                            meta={"source": "stageA"},
                        )
                    except Exception:
                        logger.debug(
                            "candidate_tokens_log_failed session=%s account=%s phase=post",
                            session_id,
                            acc_id,
                            exc_info=True,
                        )

                try:
                    append_artifact(  # type: ignore[operator]
                        session_id,
                        acc_id,
                        f"stageA_detection.{bureau}",
                        payload,
                        attach_provenance={
                            "module": "problem_detection",
                            "algo": "rules_v1",
                        },
                    )
                    try:
                        telemetry.emit(
                            "stageA.namespaced_artifact_written",
                            session_id=session_id,
                            account_id=acc_id,
                            bureau=bureau,
                        )
                    except Exception:
                        pass
                except Exception:
                    logger.warning(
                        "stageA_append_failed session=%s account=%s", session_id, acc_id
                    )
                    telemetry.emit(
                        "stageA_append_failed",
                        session_id=session_id,
                        account_id=acc_id,
                    )
                    continue

                decisions.append(payload)
                try:
                    logger.debug(
                        "stageA.per_bureau",
                        extra={
                            "session_id": session_id,
                            "account_id": acc_id,
                            "bureau": bureau,
                            "primary_issue": payload.get("primary_issue"),
                            "tier": payload.get("tier"),
                        },
                    )
                except Exception:
                    pass

            if not decisions:
                continue

            winner = decisions[0]
            for dec in decisions[1:]:
                tier = compare_tiers(dec.get("tier", "none"), winner.get("tier", "none"))
                if tier != winner.get("tier", "none"):
                    winner = dec
                    continue
                if dec.get("tier") == winner.get("tier") and float(
                    dec.get("confidence", 0.0)
                ) > float(winner.get("confidence", 0.0)):
                    winner = dec

            winner_payload = dict(winner)
            winner_bureau = winner_payload.get("bureau")
            winner_payload.pop("bureau", None)

            try:
                append_artifact(  # type: ignore[operator]
                    session_id,
                    acc_id,
                    "stageA_detection",
                    winner_payload,
                    attach_provenance={
                        "module": "problem_detection",
                        "algo": "rules_v1",
                    },
                )
                try:
                    telemetry.emit(
                        "stageA.legacy_winner_written",
                        session_id=session_id,
                        account_id=acc_id,
                        bureau=winner_bureau,
                    )
                except Exception:
                    pass
            except Exception:
                logger.warning(
                    "stageA_append_failed session=%s account=%s", session_id, acc_id
                )
                telemetry.emit(
                    "stageA_append_failed",
                    session_id=session_id,
                    account_id=acc_id,
                )
                continue
