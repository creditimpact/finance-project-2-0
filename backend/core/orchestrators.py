"""High-level orchestration routines for the credit repair pipeline.

ARCH: This module acts as the single entry point for coordinating the
intake, analysis, strategy generation, letter creation and finalization
steps of the credit repair workflow.  All core orchestration lives here;
``main.py`` only provides thin CLI wrappers.
"""

import json
import logging
import os
import random
import time
from datetime import date, datetime
from pathlib import Path
from shutil import copyfile
from types import MappingProxyType
from typing import Any, Mapping

import backend.config as config
import tactical
from backend.analytics.analytics.strategist_failures import tally_failure_reasons
from backend.analytics.analytics_tracker import save_analytics_snapshot
from backend.telemetry.metrics import emit_counter
from backend.api.config import (
    ENABLE_FIELD_POPULATION,
    ENABLE_PLANNER,
    ENABLE_PLANNER_PIPELINE,
    FIELD_POPULATION_CANARY_PERCENT,
    EXCLUDE_PARSER_AGGREGATED_ACCOUNTS,
    PLANNER_CANARY_PERCENT,
    PLANNER_PIPELINE_CANARY_PERCENT,
    AppConfig,
    env_bool,
    get_app_config,
)
from backend.api.session_manager import update_session
from backend.assets.paths import templates_path
from backend.audit.audit import AuditLevel
from backend.pipeline.runs import RunManifest, require_pdf_for_sid
from backend.core.case_store.api import get_account_case, list_accounts
from backend.core.case_store.errors import CaseStoreError
from backend.core.case_store.models import AccountCase
from backend.core.case_store.telemetry import emit
from backend.core.config.flags import FLAGS
from backend.core.email_sender import send_email_with_attachment
from backend.core.letters.field_population import apply_field_fillers
from backend.core.logic.compliance.constants import StrategistFailureReason
from backend.core.logic.report_analysis.block_exporter import (
    export_account_blocks,
    load_account_blocks,
)
from backend.core.logic.report_analysis.extract_info import (
    extract_bureau_info_column_refined,
)
from backend.core.logic.report_analysis.keys import (
    compute_logical_account_key as _compute_logical_account_key,
)
from backend.core.logic.report_analysis.text_provider import (
    BASE_DIR as TEXT_CACHE_DIR,
    extract_and_cache_text,
)
from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag
from backend.core.logic.strategy.summary_classifier import (
    RULES_VERSION,
    ClassificationRecord,
    classify_client_summaries,
    summary_hash,
)
from backend.core.logic.utils.pdf_ops import (
    convert_txts_to_pdfs,
    gather_supporting_docs_text,
)
from backend.core.logic.utils.report_sections import (
    extract_summary_from_sections,
    filter_sections_by_bureau,
)
from backend.core.models import (
    BureauAccount,
    BureauPayload,
    ClientInfo,
    Inquiry,
    ProblemAccount,
    ProofDocuments,
)
from backend.core.services.ai_client import AIClient, _StubAIClient, get_ai_client
from backend.core.taxonomy.problem_taxonomy import compare_tiers, normalize_decision
from backend.core.telemetry import metrics
from backend.core.telemetry.stageE_summary import emit_stageE_summary
from backend.core.utils.trace_io import write_json_trace, write_text_trace
from backend.policy.policy_loader import load_rulebook
from planner import plan_next_step
from backend.core.logic.report_analysis.raw_builder import (
    build_raw_from_snapshot_and_windows,
)

logger = logging.getLogger(__name__)


# --- RAW artifacts guard ---------------------------------------------------
def _has_raw_artifacts(session_id: str) -> bool:
    try:
        raw_idx = Path("traces") / "blocks" / session_id / "accounts_raw" / "_raw_index.json"
        if not raw_idx.exists():
            return False
        data = json.loads(raw_idx.read_text(encoding="utf-8"))
        blocks = data.get("blocks") or []
        return any(bool((b or {}).get("raw_coords_path")) for b in blocks)
    except Exception:
        return False


def _run_stage_b_raw(session_id: str) -> str:
    """
    Runs Stage B RAW builder immediately after Stage A finishes.
    Returns the path to the written _raw_index.json.
    """
    raw_index_path = build_raw_from_snapshot_and_windows(session_id)
    logger.info(f"PIPELINE: Stage B RAW accounts built -> {raw_index_path}")
    return raw_index_path


# --- Text cache helpers ----------------------------------------------------
def _seed_text_cache(session_id: str) -> None:
    session_dir = TEXT_CACHE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    placeholder = "placeholder"
    (session_dir / "page_001.txt").write_text(placeholder, encoding="utf-8")
    (session_dir / "full.txt").write_text(placeholder, encoding="utf-8")
    meta = {
        "pages_total": 1,
        "extract_text_ms": 0,
        "pages_ocr": 0,
        "ocr_latency_ms_total": 0,
        "ocr_errors": 0,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --- Helpers ---------------------------------------------------------------
def _thaw(obj):
    """Recursively convert MappingProxyType/dicts/lists into mutable structures.

    This is used to safely clone finalized SSOT snapshots (frozen via
    MappingProxyType) before any enrichment that may perform deep mutations.
    """
    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_thaw(x) for x in obj]
    return obj


def _emit_stageA_orchestrated(
    session_id: str, accounts: list[Mapping[str, Any]]
) -> None:
    """Emit telemetry for Stage A orchestration decisions."""
    for acc in accounts or []:
        emit(
            "stageA_orchestrated",
            session_id=session_id,
            account_id=acc.get("account_id")
            or acc.get("account_number_last4")
            or acc.get("account_fingerprint"),
            bureau=acc.get("bureau"),
            decision_source=acc.get("decision_source"),
            primary_issue=acc.get("primary_issue"),
            tier=acc.get("tier"),
            confidence=float(acc.get("confidence", 0.0)),
            reasons_count=len(acc.get("problem_reasons", [])),
            included=True,
        )


def resolve_cross_bureau(decisions: list[dict]) -> dict:
    """Resolve multiple bureau decisions for the same logical account.

    The strongest tier wins. For ties, the higher confidence prevails. Problem
    reasons from all inputs are merged and deduplicated. The decision source is
    ``'ai'`` only if the winning decision originated from AI.
    """

    if not decisions:
        return {}

    pairs = [(d, normalize_decision(d)) for d in decisions]
    winner_orig, winner_norm = pairs[0]
    for orig, dec in pairs[1:]:
        tier = compare_tiers(dec.get("tier", "none"), winner_norm.get("tier", "none"))
        if tier != winner_norm.get("tier", "none"):
            winner_orig, winner_norm = orig, dec
            continue
        if dec.get("tier") == winner_norm.get("tier") and dec.get(
            "confidence", 0.0
        ) > winner_norm.get("confidence", 0.0):
            winner_orig, winner_norm = orig, dec

    merged_reasons: list[str] = []
    for _, dec in pairs:
        merged_reasons.extend(dec.get("problem_reasons", []))

    result = dict(winner_norm)
    result["problem_reasons"] = merged_reasons
    result["decision_source"] = winner_orig.get(
        "decision_source", winner_norm.get("decision_source")
    )
    return result


def compute_logical_account_key(account_case: AccountCase) -> str:
    """Wrapper around the logical key generator for ``AccountCase`` objects."""

    opened = account_case.fields.date_opened
    if isinstance(opened, (datetime, date)):
        opened = opened.isoformat()
    return (
        _compute_logical_account_key(
            account_case.fields.creditor_type,
            (account_case.fields.account_number or "")[-4:],
            account_case.fields.account_type,
            opened,
        )
        or ""
    )


def collect_stageA_problem_accounts(session_id: str) -> list[Mapping[str, Any]]:
    """Return problem accounts for Stage A using the Case Store only.

    When ``ONE_CASE_PER_ACCOUNT_ENABLED`` is true this function reads per-bureau
    Stage-A artifacts (``stageA_detection.EX`` etc). If those artifacts are not
    present it falls back to the legacy ``stageA_detection`` artifact to preserve
    backward compatibility.
    """

    def _row_from_artifact(
        account_id: str, bureau: str, data: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        tier = str(data.get("tier", "none"))
        source = data.get("decision_source", "rules")
        reasons = data.get("problem_reasons", []) or []
        fields_used = (data.get("debug") or {}).get("fields_used")
        include = False
        if config.ENABLE_AI_ADJUDICATOR:
            if source in {"ai", "rules+ai"}:
                if tier in {"Tier1", "Tier2", "Tier3"}:
                    include = True
            elif reasons:
                include = True
        elif reasons:
            include = True
            source = "rules"
            tier = "none"
            data = dict(data)
            data["primary_issue"] = "unknown"
            data["confidence"] = 0.0
        if include and tier != "Tier4":
            acc: dict[str, Any] = {
                "account_id": account_id,
                "bureau": bureau,
                "primary_issue": data.get("primary_issue", "unknown"),
                "tier": tier,
                "problem_reasons": reasons,
                "confidence": data.get("confidence", 0.0),
                "decision_source": source,
            }
            if fields_used:
                acc["fields_used"] = fields_used
            return acc
        return None

    problems: list[Mapping[str, Any]] = []
    for acc_id in list_accounts(session_id):  # type: ignore[operator]
        case = get_account_case(session_id, acc_id)  # type: ignore[operator]
        rows: list[Mapping[str, Any]] = []

        if FLAGS.one_case_per_account_enabled:
            by_bureau = getattr(case.fields, "by_bureau", {}) or {}
            if not by_bureau:
                from backend.core.compat.legacy_shim import build_by_bureau_shim

                by_bureau = build_by_bureau_shim(session_id, acc_id)
            bureau_codes = list(by_bureau.keys()) or ["EX", "EQ", "TU"]
            for code in bureau_codes:
                art = case.artifacts.get(f"stageA_detection.{code}")
                if not art:
                    continue
                data = art.model_dump()
                row = _row_from_artifact(acc_id, code, data)
                if row:
                    rows.append(row)

            if not rows:
                art = case.artifacts.get("stageA_detection")
                if art:
                    data = art.model_dump()
                    row = _row_from_artifact(acc_id, str(case.bureau.value), data)
                    if row:
                        rows.append(row)
                else:
                    logger.warning(
                        "stageA_artifact_missing session=%s account=%s",
                        session_id,
                        acc_id,
                    )
        else:
            art = case.artifacts.get("stageA_detection")
            if art:
                data = art.model_dump()
                row = _row_from_artifact(acc_id, str(case.bureau.value), data)
                if row:
                    rows.append(row)
            else:
                logger.warning(
                    "stageA_artifact_missing session=%s account=%s",
                    session_id,
                    acc_id,
                )

        problems.extend(rows)
        logger.debug(
            "collectors.per_bureau",
            session_id=session_id,
            account_id=acc_id,
            emitted=len(rows),
            flag=FLAGS.one_case_per_account_enabled,
        )

    _emit_stageA_orchestrated(session_id, problems)
    return problems


def collect_stageA_logical_accounts(session_id: str) -> list[Mapping[str, Any]]:
    """Return logical account decisions aggregated across bureaus.

    When ``ENABLE_CROSS_BUREAU_RESOLUTION`` is disabled this simply returns the
    per-bureau decisions from :func:`collect_stageA_problem_accounts`.
    """

    problems = collect_stageA_problem_accounts(session_id)
    emit_stageE_summary(session_id, problems)
    if not config.ENABLE_CROSS_BUREAU_RESOLUTION:
        return problems

    grouped: dict[str, list[dict]] = {}
    members: dict[str, list[dict[str, str]]] = {}
    for acc in problems:
        acc_id = str(acc.get("account_id") or "")
        case = get_account_case(session_id, acc_id)  # type: ignore[operator]
        logical_id = compute_logical_account_key(case)
        grouped.setdefault(logical_id, []).append(dict(acc))
        members.setdefault(logical_id, []).append(
            {"bureau": str(acc.get("bureau")), "account_id": acc_id}
        )

    resolved: list[Mapping[str, Any]] = []
    for logical_id, items in grouped.items():
        decision = resolve_cross_bureau(items)
        group_members = members.get(logical_id, [])
        try:  # pragma: no cover - defensive
            emit(
                "stageA_cross_bureau_winner",
                session_id=session_id,
                logical_account_id=logical_id,
                winner_bureau=decision.get("bureau"),
                primary_issue=decision.get("primary_issue"),
                tier=decision.get("tier"),
                decision_source=decision.get("decision_source"),
                confidence=float(decision.get("confidence", 0.0)),
                reasons_count=len(decision.get("problem_reasons", [])),
                members=len(group_members),
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("stageA_cross_bureau_winner_emit_failed")
        if decision.get("tier") == "Tier4":
            continue
        winner_bureau = decision.get("bureau")
        if config.API_AGGREGATION_ID_STRATEGY == "logical":
            decision["account_id"] = logical_id
            decision["bureau"] = winner_bureau
        else:  # winner strategy
            decision["account_id"] = decision.get("account_id")
            decision["bureau"] = winner_bureau
        if config.API_INCLUDE_AGG_MEMBERS_META:
            decision["aggregation_meta"] = {
                "logical_account_id": logical_id,
                "members": group_members,
            }
        resolved.append(decision)

    emit("stageA_cross_bureau_aggregated", session_id=session_id, groups=len(resolved))
    return resolved


def get_stageA_decision_meta(
    session_id: str, account_id: str
) -> Mapping[str, Any] | None:
    """Fetch Stage A decision metadata for an account if available."""
    try:
        case = get_account_case(session_id, account_id)  # type: ignore[operator]
    except Exception:  # pragma: no cover - defensive
        return None
    art = case.artifacts.get("stageA_detection")
    if not art:
        return None
    data = art.model_dump()
    meta: dict[str, Any] = {
        "decision_source": data.get("decision_source", "rules"),
        "confidence": data.get("confidence", 0.0),
        "tier": str(data.get("tier", "none")),
    }
    fields_used = (data.get("debug") or {}).get("fields_used")
    if fields_used:
        meta["fields_used"] = fields_used
    return meta


def plan_and_generate_letters(session: dict, action_tags: list[str]) -> list[str]:
    """Optionally run the planner before generating letters.

    The planner pipeline is enabled when ``ENABLE_PLANNER_PIPELINE`` is true and
    a random draw for each account is below ``PLANNER_PIPELINE_CANARY_PERCENT``.
    For accounts outside this canary slice the planner is bypassed and the
    legacy router order is used.  When the pipeline is enabled, the planner
    executes only if ``ENABLE_PLANNER`` is true and the session passes the
    ``PLANNER_CANARY_PERCENT`` gate.  Otherwise the tactical pipeline runs with
    the original ``action_tags`` to preserve legacy behavior.

    Args:
        session: Session context passed to the tactical layer.
        action_tags: Proposed tags for this run.

    Returns:
        The tags passed to ``tactical.generate_letters``.
    """

    use_pipeline = ENABLE_PLANNER_PIPELINE
    pipeline_tags: list[str] = []
    legacy_tags: list[str] = []
    if use_pipeline:
        for tag in action_tags:
            if (
                PLANNER_PIPELINE_CANARY_PERCENT >= 100
                or random.random() < PLANNER_PIPELINE_CANARY_PERCENT / 100
            ):
                pipeline_tags.append(tag)
            else:
                legacy_tags.append(tag)
    else:
        legacy_tags = list(action_tags)

    use_planner = ENABLE_PLANNER
    if use_planner and PLANNER_CANARY_PERCENT < 100:
        if random.random() >= PLANNER_CANARY_PERCENT / 100:
            use_planner = False

    allowed: list[str] = []
    if pipeline_tags:
        planned = (
            plan_next_step(session, pipeline_tags) if use_planner else pipeline_tags
        )
        allowed.extend(planned)
    if legacy_tags:
        allowed.extend(legacy_tags)

    tactical.generate_letters(session, allowed)
    return allowed


def process_client_intake(client_info, audit):
    """Prepare client intake information.

    Returns:
        tuple[str, dict, dict]: session id, structured summaries and raw notes.
    """
    from backend.api.session_manager import get_intake

    if "email" not in client_info or not client_info["email"]:
        raise ValueError("Client email is missing.")

    session_id = client_info.get("session_id", "session")
    audit.log_step("session_initialized", {"session_id": session_id})

    intake = get_intake(session_id) or {}
    structured = client_info.get("structured_summaries") or {}
    structured_map: dict[str, dict] = {}
    if isinstance(structured, list):
        for idx, item in enumerate(structured):
            if isinstance(item, dict):
                key = str(item.get("account_id") or idx)
                structured_map[key] = item
    elif isinstance(structured, dict):
        for key, item in structured.items():
            if isinstance(item, dict):
                structured_map[str(key)] = item

    raw_map = {
        str(r.get("account_id")): r.get("text")
        for r in intake.get("raw_explanations", [])
        if isinstance(r, dict)
    }
    return session_id, structured_map, raw_map


def classify_client_responses(
    structured_map, raw_map, client_info, audit, ai_client: AIClient
):
    """Classify client summaries for each account.

    Results are cached in the session store keyed by a hash of the structured
    summary for each account.  Subsequent calls with the same summary skip the
    expensive classification step and reuse the stored data.
    """
    from backend.api.session_manager import get_session, update_session

    classification_map: dict[str, ClassificationRecord] = {}
    session_id = client_info.get("session_id")
    session = get_session(session_id or "") or {}
    cache = session.get("summary_classifications", {}) if session_id else {}
    state = client_info.get("state")

    updated = False
    to_process: list[tuple[str, dict, str]] = []
    for acc_id, struct in structured_map.items():
        struct_hash = summary_hash(struct)
        cached = cache.get(acc_id) if isinstance(cache, dict) else None
        if (
            cached
            and cached.get("summary_hash") == struct_hash
            and cached.get("state") == state
            and cached.get("rules_version") == RULES_VERSION
        ):
            cls = cached.get("classification", {})
            classification_map[acc_id] = ClassificationRecord(
                summary=struct,
                classification=cls,
                summary_hash=struct_hash,
                state=state,
                rules_version=RULES_VERSION,
            )
        else:
            enriched = dict(struct)
            enriched.setdefault("account_id", acc_id)
            to_process.append((acc_id, enriched, struct_hash))

    for i in range(0, len(to_process), 10):
        batch = to_process[i : i + 10]
        summaries = [item[1] for item in batch]
        batch_results = classify_client_summaries(
            summaries,
            ai_client,
            client_info.get("state"),
            session_id=session_id,
        )
        for acc_id, _summary, struct_hash in batch:
            cls = batch_results.get(acc_id, {})
            classification_map[acc_id] = ClassificationRecord(
                summary=_summary,
                classification=cls,
                summary_hash=struct_hash,
                state=state,
                rules_version=RULES_VERSION,
            )
            if session_id:
                cache[acc_id] = {
                    "summary_hash": struct_hash,
                    "classified_at": time.time(),
                    "classification": cls,
                    "state": state,
                    "rules_version": RULES_VERSION,
                }
                updated = True

    for acc_id, struct in structured_map.items():
        record = classification_map.get(acc_id)
        audit.log_account(
            acc_id,
            {
                "stage": "explanation",
                "raw_explanation": raw_map.get(acc_id, ""),
                "structured_summary": struct,
                "classification": record.classification if record else {},
            },
        )

    if session_id and updated:
        update_session(session_id, summary_classifications=cache)
    return classification_map


def analyze_credit_report(
    proofs_files,
    session_id,
    client_info,
    audit,
    log_messages,
    ai_client: AIClient | None = None,
    manifest: RunManifest | None = None,
):
    """Ingest and analyze the client's credit report."""
    from backend.api.session_manager import update_session
    from backend.core.logic.compliance.upload_validator import is_safe_pdf
    from backend.core.logic.report_analysis.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )
    from backend.core.logic.utils.bootstrap import get_current_month

    ai_client = ai_client or get_ai_client()

    try:
        pdf_path = require_pdf_for_sid(session_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc

    if not getattr(config, "ANALYZER_USE_MANIFEST_PATHS", True):
        logger.warning(
            "ANALYZER_USE_MANIFEST_PATHS disabled via env; manifest input enforced sid=%s",
            session_id,
        )

    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    extract_and_cache_text(session_id, pdf_path, ocr_enabled=config.OCR_ENABLED)

    # ייצוא הבלוקים חייב להיות שלב ראשון (fail-fast)
    logger.info("ANZ: export kickoff sid=%s file=%s", session_id, str(pdf_path))
    export_account_blocks(session_id, pdf_path)
    # Stage B: build RAW accounts from snapshot + windows (skip when TEMPLATE_FIRST)
    if config.TEMPLATE_FIRST:
        try:
            from backend.core.logic.report_analysis.smartcredit_template_orchestrator import (
                run_template_first,
            )
            result = run_template_first(session_id, Path.cwd())
            ok = bool((result or {}).get("ok")) if isinstance(result, dict) else bool(result)
            logger.warning("TEMPLATE_FIRST: finished sid=%s ok=%s", session_id, ok)
        except Exception:
            logger.exception("TEMPLATE_FIRST: failed sid=%s", session_id)
        # Do not run Stage B or further pipeline when TEMPLATE_FIRST is enabled
        return
    else:
        try:
            _run_stage_b_raw(session_id)
        except Exception:
            logger.exception("PIPELINE: Stage B RAW build failed sid=%s", session_id)

    idx_path = Path("traces") / "blocks" / session_id / "_index.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception as exc:  # file missing or unreadable
        logger.error(
            "BLOCKS_MISSING: index error sid=%s path=%s", session_id, str(idx_path)
        )
        raise CaseStoreError("no_blocks", "No account blocks exported") from exc
    if not idx:
        logger.error("BLOCKS_MISSING: empty index sid=%s", session_id)
        raise CaseStoreError("no_blocks", "No account blocks exported")

    _pre_blocks = load_account_blocks(session_id)
    logger.info("ANZ: blocks ready sid=%s count=%d", session_id, len(_pre_blocks or []))

    print("[INFO] Extracting client info from report...")
    client_personal_info = extract_bureau_info_column_refined(
        pdf_path, ai_client=ai_client, session_id=session_id
    )
    client_info.update(client_personal_info.get("data", {}))
    log_messages.append("[INFO] Personal info extracted.")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("personal_info_extracted", client_personal_info)

    print("[INFO] Parsing SmartCredit report...")
    analyzed_json_path = Path("output/analyzed_report.json")
    req_id = session_id
    sections = analyze_report_logic(
        pdf_path,
        analyzed_json_path,
        client_info,
        session_id=session_id,
        request_id=req_id,
    )
    logger.info(
        "ORCH: analyze_report returned sid=%s req=%s",
        sections.get("session_id"),
        req_id,
    )
    if FLAGS.case_first_build_required:
        try:
            count = len(list_accounts(session_id))  # type: ignore[operator]
        except Exception:
            count = 0
        if count == 0:
            if _has_raw_artifacts(session_id):
                logger.warning(
                    "PIPELINE: 0 mapped accounts; RAW exists and is the source of truth. sid=%s",
                    session_id,
                )
            else:
                raise CaseStoreError(
                    "case_build_failed",
                    "No account cases created; aborting Stage-A/UI.",
                )
    _emit_stageA_events(session_id, sections.get("problem_accounts", []))  # noqa: F821
    if (
        os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1"
        and not sections.get("negative_accounts")
        and not sections.get("open_accounts_with_issues")
    ):
        all_accounts = sections.get("all_accounts", [])
        sections["negative_accounts"] = list(all_accounts)
        sections["open_accounts_with_issues"] = list(all_accounts)
    client_info.update(sections)
    log_messages.append("[INFO] Report analyzed.")
    audit.log_step(
        "report_analyzed",
        {
            "negative_accounts": sections.get("negative_accounts", []),
            "open_accounts_with_issues": sections.get("open_accounts_with_issues", []),
            "unauthorized_inquiries": sections.get("unauthorized_inquiries", []),
        },
    )

    safe_name = (
        (client_info.get("name") or "Client").replace(" ", "_").replace("/", "_")
    )
    month_component = get_current_month()
    if manifest is not None:
        exports_root = manifest.ensure_run_subdir("exports_dir", "exports")
        base_folder = exports_root / month_component
    else:
        base_folder = Path("Clients") / month_component
    today_folder = base_folder / f"{safe_name}_{session_id}"
    today_folder.mkdir(parents=True, exist_ok=True)
    log_messages.append(f"[INFO] Client folder created at: {today_folder}")
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("client_folder_created", {"path": str(today_folder)})

    for file in today_folder.glob("*.pdf"):
        file.unlink()
    for file in today_folder.glob("*_gpt_response.json"):
        file.unlink()

    original_pdf_copy = today_folder / "Original SmartCredit Report.pdf"
    copyfile(pdf_path, original_pdf_copy)
    log_messages.append("[INFO] Original report saved to client folder.")

    if analyzed_json_path.exists():
        copyfile(analyzed_json_path, today_folder / "analyzed_report.json")
        log_messages.append("[INFO] Analyzed report JSON saved.")

    detailed_logs = []
    bureau_data = {
        bureau: filter_sections_by_bureau(sections, bureau, detailed_logs)
        for bureau in ["Experian", "Equifax", "TransUnion"]
    }
    log_messages.extend(detailed_logs)
    if audit.level == AuditLevel.VERBOSE:
        audit.log_step("sections_split_by_bureau", bureau_data)

    return pdf_path, sections, bureau_data, today_folder


def _annotate_with_tri_merge(sections: Mapping[str, Any]) -> None:
    """Annotate accounts in ``sections`` with tri-merge mismatch details."""
    if not env_bool("ENABLE_TRI_MERGE", False):
        return

    import copy

    from backend.api.session_manager import get_session
    from backend.audit.audit import emit_event
    from backend.core.logic.report_analysis.tri_merge import (
        compute_mismatches,
        normalize_and_match,
    )
    from backend.core.logic.report_analysis.tri_merge_models import Tradeline
    from backend.core.logic.utils.report_sections import filter_sections_by_bureau

    tracked_keys = [
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ]
    # Avoid deepcopy on potential MappingProxyType entries if this list ever
    # includes 'problem_accounts'; keep semantics identical for other keys.
    before = {
        k: (
            copy.deepcopy(sections.get(k, []))
            if k != "problem_accounts"
            else sections.get(k, [])
        )
        for k in tracked_keys
    }
    counts_before = {k: len(v) for k, v in before.items()}
    primary_before: dict[str, Any] = {}
    for lst in before.values():
        for acc in lst:
            acc_id = str(acc.get("account_id") or id(acc))
            primary_before.setdefault(acc_id, acc.get("primary_issue"))

    bureau_data = {
        bureau: filter_sections_by_bureau(sections, bureau, [])
        for bureau in ["Experian", "Equifax", "TransUnion"]
    }

    tradelines: list[Tradeline] = []
    for bureau, payload in bureau_data.items():
        for section, items in payload.items():
            if section == "inquiries" or not isinstance(items, list):
                continue
            for acc in items:
                tradelines.append(
                    Tradeline(
                        creditor=str(acc.get("name") or ""),
                        bureau=bureau,
                        account_number=acc.get("account_number"),
                        data=acc,
                    )
                )

    if not tradelines:
        return

    _start = time.perf_counter()
    families = normalize_and_match(tradelines)
    emit_counter("tri_merge.process_time_ms", (time.perf_counter() - _start) * 1000)
    compute_mismatches(families)

    session_id = os.getenv("SESSION_ID", "")
    tri_session = get_session(session_id) if session_id else None
    tri_evidence = (
        (tri_session.get("tri_merge") or {}).get("evidence", {}) if tri_session else {}
    )

    tri_merge_map: dict[str, dict[str, Any]] = {}
    for fam in families:
        family_id = getattr(fam, "family_id", None)
        mismatch_types = [m.field for m in getattr(fam, "mismatches", [])]
        evidence_id = family_id
        evidence = tri_evidence.get(evidence_id)
        for tl in fam.tradelines.values():
            acc_id = str(tl.data.get("account_id") or "")
            if acc_id and family_id:
                info = {
                    "family_id": family_id,
                    "mismatch_types": mismatch_types,
                    "evidence_snapshot_id": evidence_id,
                }
                if evidence:
                    info["evidence"] = evidence
                tri_merge_map[acc_id] = info

    for key in (
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ):
        for acc in sections.get(key, []):
            acc_id = str(acc.get("account_id") or "")
            tri_info = tri_merge_map.get(acc_id)
            if not tri_info:
                continue
            acc["tri_merge"] = tri_info

            evidence = (
                tri_info.get("evidence")
                if isinstance(tri_info.get("evidence"), dict)
                else {}
            )
            # Aggregate flags from mismatch types and any explicit evidence flags
            flags: list[str] = []
            if tri_info.get("mismatch_types"):
                flags.append("tri_merge_mismatch")
            flags.extend(
                evidence.get("flags", []) if isinstance(evidence, dict) else []
            )
            if flags:
                existing = acc.setdefault("flags", [])
                for flag in flags:
                    if flag not in existing:
                        existing.append(flag)

            # Populate bureau-level statuses from tri-merge evidence when missing
            if isinstance(evidence, dict) and not acc.get("bureau_statuses"):
                tradelines = evidence.get("tradelines", {})
                statuses: dict[str, str] = {}
                if isinstance(tradelines, dict):
                    for bureau, data in tradelines.items():
                        if not isinstance(data, Mapping):
                            continue
                        status = data.get("status") or data.get("account_status")
                        if status:
                            statuses[bureau] = status
                if statuses:
                    acc["bureau_statuses"] = statuses

    # Ensure tri-merge remains purely annotative.
    violation_reason: str | None = None
    for key in tracked_keys:
        if len(sections.get(key, [])) != counts_before.get(key, 0):
            violation_reason = "account_count_changed"
            break

    if violation_reason is None:
        primary_after: dict[str, Any] = {}
        for key in tracked_keys:
            for acc in sections.get(key, []):
                acc_id = str(acc.get("account_id") or id(acc))
                if acc_id not in primary_after:
                    primary_after[acc_id] = acc.get("primary_issue")
        for acc_id, before_issue in primary_before.items():
            if primary_after.get(acc_id) != before_issue:
                violation_reason = "primary_issue_changed"
                break
        else:
            for acc_id in primary_after:
                if acc_id not in primary_before:
                    violation_reason = "account_count_changed"
                    break

    if violation_reason:
        emit_event("trimerge_violation", {"reason": violation_reason})
        for key, val in before.items():
            sections[key] = val


def generate_strategy_plan(
    client_info,
    bureau_data,
    classification_map,
    stage_2_5_data,
    session_id,
    audit,
    log_messages,
    ai_client: AIClient,
):
    """Generate and merge the strategy plan."""
    from backend.core.logic.strategy.generate_strategy_report import StrategyGenerator
    from backend.core.logic.strategy.strategy_merger import merge_strategy_data

    docs_text = gather_supporting_docs_text(session_id)
    strat_gen = StrategyGenerator(ai_client=ai_client)
    audit.log_step(
        "strategist_invocation",
        {
            "client_info": client_info,
            "bureau_data": bureau_data,
            "classification_map": {
                k: v.classification for k, v in (classification_map or {}).items()
            },
            "supporting_docs_text": docs_text,
        },
    )
    strategy = strat_gen.generate(
        client_info,
        bureau_data,
        docs_text,
        classification_map={
            k: v.classification for k, v in (classification_map or {}).items()
        },
        stage_2_5_data=stage_2_5_data,
        audit=audit,
    )
    if not strategy or not strategy.get("accounts"):
        audit.log_step(
            "strategist_failure",
            {"failure_reason": StrategistFailureReason.EMPTY_OUTPUT},
        )
    strat_gen.save_report(
        strategy,
        client_info,
        datetime.now().strftime("%Y-%m-%d"),
        stage_2_5_data=stage_2_5_data,
    )
    audit.log_step("strategy_generated", strategy)

    merge_strategy_data(
        strategy, bureau_data, classification_map, audit, log_list=log_messages
    )
    audit.log_step("strategy_merged", bureau_data)
    for bureau, payload in bureau_data.items():
        for section, items in payload.items():
            if isinstance(items, list):
                for acc in items:
                    acc_id = acc.get("account_id") or acc.get("name")
                    audit.log_account(
                        acc_id,
                        {
                            "bureau": bureau,
                            "section": section,
                            "recommended_action": acc.get("recommended_action"),
                            "action_tag": acc.get("action_tag"),
                        },
                    )
    return strategy


def generate_letters(
    client_info,
    bureau_data,
    sections,
    today_folder,
    is_identity_theft,
    strategy,
    audit,
    log_messages,
    classification_map,
    ai_client: AIClient,
    app_config: AppConfig | None = None,
):
    """Create all client letters and supporting files."""
    from backend.core.logic.letters.generate_custom_letters import (
        generate_custom_letters,
    )
    from backend.core.logic.letters.generate_goodwill_letters import (
        generate_goodwill_letters,
    )
    from backend.core.logic.letters.letter_generator import (
        generate_all_dispute_letters_with_ai,
    )
    from backend.core.logic.rendering.instructions_generator import (
        generate_instruction_file,
    )
    from backend.core.logic.utils.bootstrap import extract_all_accounts

    print("[INFO] Generating dispute letters...")
    generate_all_dispute_letters_with_ai(
        client_info,
        bureau_data,
        today_folder,
        is_identity_theft,
        audit,
        classification_map=classification_map,
        log_messages=log_messages,
        ai_client=ai_client,
        rulebook_fallback_enabled=(
            app_config.rulebook_fallback_enabled if app_config else True
        ),
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("[INFO] Dispute letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("dispute_letters_generated")

    if not is_identity_theft:
        print("[INFO] Generating goodwill letters...")
        generate_goodwill_letters(
            client_info,
            bureau_data,
            today_folder,
            audit,
            ai_client=ai_client,
            classification_map=classification_map,
        )
        log_messages.append("[INFO] Goodwill letters generated.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_generated")
    else:
        print("[INFO] Identity theft case - skipping goodwill letters.")
        log_messages.append("[INFO] Goodwill letters skipped due to identity theft.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("goodwill_letters_skipped")

    all_accounts = extract_all_accounts(sections)
    for bureau in bureau_data:
        bureau_data[bureau]["all_accounts"] = all_accounts

    print("[INFO] Generating custom letters...")
    generate_custom_letters(
        client_info,
        bureau_data,
        today_folder,
        audit,
        classification_map=classification_map,
        log_messages=log_messages,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("[INFO] Custom letters generated.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("custom_letters_generated")

    print("[INFO] Generating instructions file for client...")
    generate_instruction_file(
        client_info,
        bureau_data,
        is_identity_theft,
        today_folder,
        strategy=strategy,
        ai_client=ai_client,
        wkhtmltopdf_path=app_config.wkhtmltopdf_path if app_config else None,
    )
    log_messages.append("[INFO] Instruction file created.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("instructions_generated")

    print("[INFO] Converting letters to PDF...")
    if os.getenv("DISABLE_PDF_RENDER", "").lower() not in ("1", "true", "yes"):
        convert_txts_to_pdfs(today_folder)
        log_messages.append("[INFO] All letters converted to PDF.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("letters_converted_to_pdf")
    else:
        print(
            "[INFO] PDF rendering disabled via DISABLE_PDF_RENDER â€“ skipping conversion."
        )

    if is_identity_theft:
        print("[INFO] Adding FCRA rights PDF...")
        frca_source_path = templates_path("FTC_FCRA_605b.pdf")
        frca_target_path = today_folder / "Your Rights - FCRA.pdf"
        if os.path.exists(frca_source_path):
            copyfile(frca_source_path, frca_target_path)
            print(f"[INFO] FCRA rights PDF copied to: {frca_target_path}")
            log_messages.append("[INFO] FCRA document added.")
        else:
            print("[WARN] FCRA rights file not found!")
            log_messages.append("[WARN] FCRA file missing.")
            if audit.level is AuditLevel.VERBOSE:
                audit.log_step("fcra_file_missing")
    else:
        log_messages.append("[INFO] Identity theft not indicated - FCRA PDF skipped.")
        if audit.level is AuditLevel.VERBOSE:
            audit.log_step("fcra_skipped")


def finalize_outputs(
    client_info,
    today_folder,
    sections,
    audit,
    log_messages,
    app_config: AppConfig,
):
    """Send final outputs to the client and record analytics."""
    print("[INFO] Sending email with all documents to client...")
    output_files = [str(p) for p in today_folder.glob("*.pdf")]
    raw_name = (client_info.get("name") or "").strip()
    first_name = raw_name.split()[0] if raw_name else "Client"
    send_email_with_attachment(
        receiver_email=client_info["email"],
        subject="Your Credit Repair Package is Ready",
        body=f"""Hi {first_name},

WeÃƒÂ¢Ã‚â‚¬Ã‚â„¢ve successfully completed your credit analysis and prepared your customized repair package ÃƒÂ¢Ã‚â‚¬Ã‚" itÃƒÂ¢Ã‚â‚¬Ã‚â„¢s attached to this email.

ÃƒÂ°Ã‚Å¸Ã‚-Ã‚â€š Inside your package:
- Dispute letters prepared for each credit bureau
- Goodwill letters (if applicable)
- Your full SmartCredit report
- A personalized instruction guide with legal backup
- Your official rights under the FCRA (Fair Credit Reporting Act)

ÃƒÂ¢Ã‚Å“Ã‚... Please print, sign, and mail each dispute letter to the bureaus at their addresses (included in the letters), along with:
- A copy of your government-issued ID
- A utility bill with your current address
- (Optional) FTC Identity Theft Report if applicable

In your **instruction file**, you'll also find:
- A breakdown of which accounts are hurting your score the most
- Recommendations like adding authorized users (we can help you do this!)
- When and how to follow up with SmartCredit

If youÃƒÂ¢Ã‚â‚¬Ã‚â„¢d like our team to help you with the next steps ÃƒÂ¢Ã‚â‚¬Ã‚" including adding an authorized user, tracking disputes, or escalating ÃƒÂ¢Ã‚â‚¬Ã‚" weÃƒÂ¢Ã‚â‚¬Ã‚â„¢re just one click away.

We're proud to support you on your journey to financial freedom.

Best regards,
**CREDIT IMPACT**
""",
        files=output_files,
        smtp_server=app_config.smtp_server,
        smtp_port=app_config.smtp_port,
        sender_email=app_config.smtp_username,
        sender_password=app_config.smtp_password,
    )
    log_messages.append("[INFO] Email sent to client.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("email_sent", {"files": output_files})

    failure_counts = tally_failure_reasons(audit)
    save_analytics_snapshot(
        client_info,
        extract_summary_from_sections(sections),
        strategist_failures=failure_counts,
    )
    log_messages.append("[INFO] Analytics snapshot saved.")
    if audit.level is AuditLevel.VERBOSE:
        audit.log_step("analytics_saved", {"strategist_failures": failure_counts})

    print("\n[INFO] Credit Repair Process completed successfully!")
    print(f"[INFO] All output saved to: {today_folder}")
    log_messages.append("[INFO] Process completed successfully.")
    audit.log_step("process_completed")


def save_log_file(
    client_info,
    is_identity_theft,
    output_folder,
    log_lines,
    manifest: RunManifest | None = None,
) -> Path:
    """Persist a human-readable log of pipeline activity."""
    if manifest is not None:
        logs_dir = manifest.ensure_run_subdir("logs_dir", "logs")
    else:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
    client_name = client_info.get("name", "Unknown").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"{timestamp}_{client_name}.txt"
    log_path = logs_dir / log_filename

    header = [
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Client: {client_info.get('name', '')}",
        f"Address: {client_info.get('address', '')}",
        f"Goal: {client_info.get('goal', '')}",
        f"Treatment Type: {'Identity Theft' if is_identity_theft else 'Standard Dispute'}",
        f"Output folder: {output_folder}",
        "",
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(header + log_lines))
    print(f"[INFO] Log saved: {log_path}")
    if manifest is not None:
        manifest.set_artifact("logs", "run_log", log_path)

    return log_path


def run_credit_repair_process(
    client: ClientInfo,
    proofs: ProofDocuments,
    is_identity_theft: bool,
    *,
    app_config: AppConfig | None = None,
):
    """Execute the full credit repair pipeline for a single client.

    ``client`` and ``proofs`` should be instances of :class:`ClientInfo` and
    :class:`ProofDocuments` respectively. Passing plain dictionaries is
    deprecated and will be removed in a future release.
    """
    app_config = app_config or get_app_config()
    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    if isinstance(proofs, dict):  # pragma: no cover - backward compat
        proofs = ProofDocuments.from_dict(proofs)
    client_info = client.to_dict()
    proofs_files = proofs.to_dict()
    log_messages: list[str] = []
    today_folder: Path | None = None
    pdf_path: Path | None = None
    session_id = client_info.get("session_id", "session")
    try:
        manifest: RunManifest | None = RunManifest.for_sid(
            session_id, allow_create=False
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run manifest missing for {session_id}") from exc
    from backend.audit.audit import create_audit_logger
    from backend.core.services.ai_client import build_ai_client

    audit = create_audit_logger(session_id)
    ai_client = build_ai_client(app_config.ai)
    strategy = None

    try:
        print("\n[INFO] Starting Credit Repair Process (B2C Mode)...")
        log_messages.append("[INFO] Process started.")
        audit.log_step("process_started", {"is_identity_theft": is_identity_theft})

        session_id, structured_map, raw_map = process_client_intake(client_info, audit)
        if manifest is None or manifest.sid != session_id:
            manifest = RunManifest.for_sid(session_id, allow_create=False)
        os.environ["SESSION_ID"] = session_id
        classification_map = classify_client_responses(
            structured_map, raw_map, client_info, audit, ai_client
        )
        rulebook = load_rulebook()
        pdf_path, sections, bureau_data, today_folder = analyze_credit_report(
            proofs_files,
            session_id,
            client_info,
            audit,
            log_messages,
            ai_client,
            manifest=manifest,
        )
        # Ensure account blocks exist before any case-building
        sid = session_id
        blocks = load_account_blocks(sid)
        if not blocks:
            logger.error("BLOCKS_MISSING: no exported blocks for session_id=%s", sid)
            raise CaseStoreError("no_blocks", "No account blocks exported")
        try:
            from services.outcome_ingestion.ingest_report import (
                ingest_report as ingest_outcome_report,
            )

            ingest_outcome_report(None, bureau_data)
        except Exception:
            pass
        if os.getenv("DISABLE_TRI_MERGE_PRECONFIRM") != "1":
            _annotate_with_tri_merge(sections)
        facts_map: dict[str, dict[str, Any]] = {}
        for key in (
            "negative_accounts",
            "open_accounts_with_issues",
            "high_utilization_accounts",
            "positive_accounts",
            "all_accounts",
        ):
            for acc in sections.get(key, []):
                acc_id = str(acc.get("account_id") or "")
                if acc_id:
                    facts_map[acc_id] = acc
        stage_2_5: dict[str, Any] = {}
        for acc_id in set(facts_map) | set(classification_map):
            record = classification_map.get(acc_id)
            account_cls = {**record.summary, **record.classification} if record else {}
            stage_2_5[acc_id] = normalize_and_tag(
                account_cls, facts_map.get(acc_id, {}), rulebook, account_id=acc_id
            )
        if session_id:
            update_session(session_id, stage_2_5=stage_2_5)
        from backend.core.letters import router as letters_router

        for acc_id, acc_ctx in stage_2_5.items():
            tag = acc_ctx.get("action_tag")
            decision = letters_router.select_template(tag, acc_ctx, phase="candidate")
            emit_counter(
                "router.candidate_selected",
                {"tag": tag, "template": decision.template_path},
            )
        strategy = generate_strategy_plan(
            client_info,
            bureau_data,
            classification_map,
            stage_2_5,
            session_id,
            audit,
            log_messages,
            ai_client,
        )
        session_ctx = {
            "session_id": session_id,
            "client_info": client_info,
            "bureau_data": bureau_data,
            "sections": sections,
            "today_folder": today_folder,
            "is_identity_theft": is_identity_theft,
            "strategy": strategy,
            "audit": audit,
            "log_messages": log_messages,
            "classification_map": classification_map,
            "ai_client": ai_client,
            "app_config": app_config,
        }
        action_tags = [
            ctx.get("action_tag") for ctx in stage_2_5.values() if ctx.get("action_tag")
        ]
        plan_and_generate_letters(session_ctx, action_tags)
        strategy_accounts = {
            str(acc.get("account_id")): acc for acc in strategy.get("accounts", [])
        }
        for acc_id, acc_ctx in stage_2_5.items():
            tag = acc_ctx.get("action_tag")
            acc_strat = strategy_accounts.get(acc_id, {})
            do_population = ENABLE_FIELD_POPULATION
            if do_population and FIELD_POPULATION_CANARY_PERCENT < 100:
                do_population = random.random() < FIELD_POPULATION_CANARY_PERCENT / 100
            if do_population:
                apply_field_fillers(acc_ctx, strategy=acc_strat, profile=client_info)
                if tag:
                    for _ in acc_ctx.get("missing_fields", []):
                        emit_counter(
                            "finalize.missing_fields_after_population", {"tag": tag}
                        )
            letters_router.select_template(tag, acc_ctx, phase="finalize")
        if session_id:
            update_session(session_id, stage_2_5=stage_2_5)
        finalize_outputs(
            client_info, today_folder, sections, audit, log_messages, app_config
        )
        if manifest is not None and today_folder:
            summary_pdf = today_folder / "Start_Here - Instructions.pdf"
            if summary_pdf.exists():
                manifest.set_artifact("exports", "summary_pdf", summary_pdf)

    except Exception as e:  # pragma: no cover - surface for higher-level handling
        error_msg = f"[ERROR] Error: {str(e)}"
        print(error_msg)
        log_messages.append(error_msg)
        audit.log_error(error_msg)
        raise

    finally:
        save_log_file(
            client_info,
            is_identity_theft,
            today_folder,
            log_messages,
            manifest=manifest,
        )
        if today_folder:
            audit.save(today_folder)
            if app_config.export_trace_file:
                from backend.audit.trace_exporter import (
                    export_trace_breakdown,
                    export_trace_file,
                )

                export_trace_file(audit, session_id)
                export_trace_breakdown(
                    audit,
                    strategy,
                    (
                        strategy.get("accounts")
                        if isinstance(strategy, dict)
                        else getattr(strategy, "accounts", None)
                    ),
                    Path("client_output"),
                )
    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
            print(f"[INFO] Deleted uploaded PDF: {pdf_path}")
        except Exception as delete_error:  # pragma: no cover - best effort
            print(f"[WARN] Failed to delete uploaded PDF: {delete_error}")


def extract_problematic_accounts_from_report(
    file_path: str, session_id: str | None = None
) -> BureauPayload | Mapping[str, Any]:
    """Return problematic accounts extracted from the report for user review."""
    from backend.api.session_manager import update_session
    from backend.core.logic.compliance.upload_validator import is_safe_pdf
    from backend.core.logic.report_analysis.analyze_report import (
        analyze_credit_report as analyze_report_logic,
    )
    from backend.core.logic.report_analysis.extractors.accounts import (
        build_account_cases,
    )

    session_id = session_id or "session"
    if not session_id:
        raise ValueError("session_id is required for Stage-A extraction")

    try:
        pdf_path = require_pdf_for_sid(session_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc

    if not getattr(config, "ANALYZER_USE_MANIFEST_PATHS", True):
        logger.warning(
            "ANALYZER_USE_MANIFEST_PATHS disabled via env; manifest input enforced sid=%s",
            session_id,
        )

    update_session(session_id, file_path=str(pdf_path))
    if not is_safe_pdf(pdf_path):
        raise ValueError("Uploaded file failed PDF safety checks.")

    seeded_text_cache = False
    try:
        extract_and_cache_text(session_id, pdf_path, ocr_enabled=config.OCR_ENABLED)
    except FileNotFoundError:
        logger.warning(
            "TEXT_EXTRACTION_SKIPPED sid=%s reason=missing_pdf path=%s",
            session_id,
            pdf_path,
        )
        _seed_text_cache(session_id)
        seeded_text_cache = True
    except Exception as exc:
        if exc.__class__.__name__ == "FileNotFoundError" and getattr(
            exc.__class__, "__module__", ""
        ).startswith("pymupdf"):
            logger.warning(
                "TEXT_EXTRACTION_SKIPPED sid=%s reason=missing_pdf path=%s",
                session_id,
                pdf_path,
            )
            _seed_text_cache(session_id)
            seeded_text_cache = True
        else:
            raise

    logger.info(
        "ORCH: export kickoff (extract_problematic_accounts) sid=%s",
        session_id,
    )
    # --- BLOCKS: export first, fail-fast on empty ---
    export_account_blocks(session_id, pdf_path)
    # Stage B: build RAW accounts from snapshot + windows (skip when TEMPLATE_FIRST)
    if config.TEMPLATE_FIRST:
        try:
            from backend.core.logic.report_analysis.smartcredit_template_orchestrator import (
                run_template_first,
            )
            result = run_template_first(session_id, Path.cwd())
            ok = bool((result or {}).get("ok")) if isinstance(result, dict) else bool(result)
            logger.warning("TEMPLATE_FIRST: finished sid=%s ok=%s", session_id, ok)
        except Exception:
            logger.exception("TEMPLATE_FIRST: failed sid=%s", session_id)
        # Do not run Stage B or further pipeline when TEMPLATE_FIRST is enabled
        return
    else:
        try:
            _run_stage_b_raw(session_id)
        except Exception:
            logger.exception("PIPELINE: Stage B RAW build failed sid=%s", session_id)

    # Verify blocks exist on disk (don’t proceed to analyze if missing)
    _pre = load_account_blocks(session_id)
    if not _pre:
        # Log and stop early — nothing should run without blocks
        logger.error("BLOCKS_MISSING: no exported blocks for session_id=%s", session_id)
        if not seeded_text_cache:
            raise CaseStoreError("no_blocks", "No account blocks exported")
        logger.warning(
            "BLOCKS_MISSING tolerated sid=%s reason=seeded_text_cache", session_id
        )

    analyzed_json_path = Path("output/analyzed_report.json")
    req_id = session_id

    ai_client = get_ai_client()
    run_ai = not isinstance(ai_client, _StubAIClient)

    sections = analyze_report_logic(
        pdf_path,
        analyzed_json_path,
        {},
        ai_client=ai_client if run_ai else None,
        run_ai=run_ai,
        session_id=session_id or req_id,
        request_id=req_id,
    )
    logger.info(
        "ORCH: analyze_report returned sid=%s req=%s",
        sections.get("session_id"),
        req_id,
    )
    if FLAGS.casebuilder_debug:
        logger.debug("CASEBUILDER: starting session_id=%s", session_id)
    try:
        pre = len(list_accounts(session_id))  # type: ignore[operator]
    except Exception:
        pre = -1
    if FLAGS.casebuilder_debug:
        logger.debug("CASEBUILDER: pre-count=%s", pre)

    build_account_cases(session_id)

    try:
        post = len(list_accounts(session_id))  # type: ignore[operator]
    except Exception:
        post = -1
    if FLAGS.casebuilder_debug:
        logger.debug("CASEBUILDER: post-count=%s", post)
    metrics.gauge("casestore.count", post, {"session_id": session_id})
    if post == 0:
        logger.error("CASEBUILDER: produced 0 cases (will abort)")
    if FLAGS.case_first_build_required and not seeded_text_cache:
        try:
            count = len(list_accounts(session_id))  # type: ignore[operator]
        except Exception:
            count = 0
        if count == 0:
            if _has_raw_artifacts(session_id):
                logger.warning(
                    "PIPELINE: 0 mapped accounts; RAW exists and is the source of truth. sid=%s",
                    session_id,
                )
            else:
                raise CaseStoreError(
                    "case_build_failed",
                    "No account cases created; aborting Stage-A/UI.",
                )

    force_parser = os.getenv("ANALYSIS_FORCE_PARSER_ONLY") == "1"
    if force_parser or sections.get("ai_failed"):
        if FLAGS.case_first_build_required and not seeded_text_cache:
            raise RuntimeError("parser-first path disabled")
        logger.info("analysis_falling_back_to_parser_only force=%s", force_parser)
        sections = analyze_report_logic(
            pdf_path,
            analyzed_json_path,
            {},
            ai_client=None,
            run_ai=False,
            session_id=session_id or req_id,
            request_id=req_id,
        )
        sections["needs_human_review"] = True
        sections["ai_failed"] = True
        logger.info(
            "ORCH: analyze_report returned sid=%s req=%s",
            sections.get("session_id"),
            req_id,
        )
    if (
        os.getenv("DEFER_ASSIGN_ISSUE_TYPES") == "1"
        and not sections.get("negative_accounts")
        and not sections.get("open_accounts_with_issues")
    ):
        all_accounts = sections.get("all_accounts", [])
        sections["negative_accounts"] = list(all_accounts)
        sections["open_accounts_with_issues"] = list(all_accounts)
    sections.setdefault("negative_accounts", [])
    sections.setdefault("open_accounts_with_issues", [])
    sections.setdefault("all_accounts", [])
    sections.setdefault("high_utilization_accounts", [])
    from backend.core.logic.report_analysis.report_postprocessing import (
        _inject_missing_late_accounts,
        enrich_account_metadata,
    )

    def _log_account_snapshot(label: str) -> None:
        all_acc = sections.get("all_accounts") or []
        neg = sections.get("negative_accounts") or []
        open_acc = sections.get("open_accounts_with_issues") or []
        sample_src = all_acc or (neg + open_acc)
        sample = [
            {
                "normalized_name": a.get("normalized_name"),
                "primary_issue": a.get("primary_issue"),
                "issue_types": a.get("issue_types"),
                "status": a.get("status"),
                "source_stage": a.get("source_stage"),
            }
            for a in sample_src[:3]
        ]
        logger.info(
            "%s source=derived all_accounts=%d negative_accounts=%d open_accounts_with_issues=%d sample=%s",
            label,
            len(all_acc),
            len(neg),
            len(open_acc),
            sample,
        )

    try:
        sample_primary = [
            (a.get("normalized_name"), a.get("primary_issue"))
            for a in (sections.get("problem_accounts") or [])[:3]
        ]
        logger.info(
            "DBG post_analyze_snapshot source=derived sample_primary=%s",
            sample_primary,
        )
    except Exception:
        pass
    _log_account_snapshot("post_analyze_report")
    # Read-only injection step executed on deep copy; ensure no mutation of finalized accounts
    try:
        import copy as _copy

        from backend.core.utils.immutability import assert_no_mutation

        # Avoid deepcopy on MappingProxyType (problem_accounts) to prevent errors in Python 3.13
        tmp = dict(sections)
        tmp.pop("problem_accounts", None)
        deepcopy_sections = _copy.deepcopy(tmp)
        assert_no_mutation(
            lambda payload: _inject_missing_late_accounts(payload, {}, {}, {})
        )(deepcopy_sections)
    except Exception:
        # If immutability tools unavailable, proceed without mutation guard
        pass
    _log_account_snapshot("post_inject_missing_late_accounts")

    suppress_accounts_without_issue_types = env_bool(
        "SUPPRESS_ACCOUNTS_WITHOUT_ISSUE_TYPES", False
    )

    # SSOT: derive emit sections strictly from finalized problem_accounts
    def _build_emit_sections_from_problems(problems):
        neg: list[dict] = []
        open_issues: list[dict] = []
        all_acc: list[dict] = []
        for acc in problems or []:
            dup = _thaw(acc)
            enriched = enrich_account_metadata(dup)  # do not mutate core fields
            logger.info(
                "DBG final_emit source=emit name=%s primary=%s advisor_len=%d recs=%d",
                enriched.get("normalized_name"),
                enriched.get("primary_issue"),
                len(str(enriched.get("advisor_comment", ""))),
                len(enriched.get("recommendations") or []),
            )
            neg.append(enriched)
            all_acc.append(enriched)
        return neg, open_issues, all_acc

    def _build_emit_sections_from_parser_sections() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        exclude_parser = EXCLUDE_PARSER_AGGREGATED_ACCOUNTS

        def _suppression_reasons(acc: Mapping[str, Any]) -> list[str]:
            reasons: list[str] = []
            stage = str(acc.get("source_stage") or "").lower()
            if exclude_parser and "parser_aggregated" in stage:
                reasons.append("parser_aggregated")
            issues = acc.get("issue_types") or []
            if suppress_accounts_without_issue_types and not issues:
                reasons.append("missing_issue_types")
            return reasons

        def _log_suppressed(acc: Mapping[str, Any], reasons: list[str]) -> None:
            try:
                reason = ",".join(reasons)
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": reason,
                        "name": acc.get("name")
                        or acc.get("normalized_name")
                        or acc.get("account_id"),
                        "source_stage": acc.get("source_stage"),
                    },
                )
            except Exception:  # pragma: no cover - defensive
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": ",".join(reasons),
                        "name": "unknown",
                    },
                )

        def _meta_key(acc: Mapping[str, Any]) -> tuple[str, Any]:
            return (
                str(acc.get("normalized_name") or acc.get("name") or "").lower(),
                acc.get("account_number_last4") or acc.get("account_fingerprint"),
            )

        def _account_key(acc: Mapping[str, Any]) -> tuple[str, Any, Any]:
            core = _meta_key(acc)
            return (core[0], core[1], acc.get("source_stage"))

        negatives: list[dict[str, Any]] = []
        open_issues: list[dict[str, Any]] = []
        all_acc: list[dict[str, Any]] = []
        problems_out: list[dict[str, Any]] = []
        seen: set[tuple[str, Any, Any]] = set()

        base_all_raw = sections.get("all_accounts") or []
        all_enriched = [enrich_account_metadata(_thaw(raw)) for raw in base_all_raw]
        meta_stage: dict[tuple[str, Any], Any] = {}
        stage_by_name: dict[str, Any] = {}
        for enriched in all_enriched:
            key = _meta_key(enriched)
            if key not in meta_stage:
                meta_stage[key] = enriched.get("source_stage")
            nm = str(enriched.get("normalized_name") or enriched.get("name") or "").lower()
            stage_val = enriched.get("source_stage")
            if nm and stage_val and nm not in stage_by_name:
                stage_by_name[nm] = stage_val

        def _append(enriched: dict[str, Any], bucket: list[dict[str, Any]] | None) -> None:
            key = _account_key(enriched)
            if key in seen:
                return
            seen.add(key)
            if bucket is not None:
                bucket.append(enriched)
            all_acc.append(enriched)

        for raw in sections.get("negative_accounts") or []:
            enriched = enrich_account_metadata(_thaw(raw))
            core_key = _meta_key(enriched)
            stage = meta_stage.get(core_key)
            if not stage:
                nm = str(enriched.get("normalized_name") or enriched.get("name") or "").lower()
                stage = stage_by_name.get(nm)
            if stage:
                enriched["source_stage"] = stage
            reasons = _suppression_reasons(enriched)
            if reasons:
                _log_suppressed(enriched, reasons)
                continue
            _append(enriched, negatives)
            problems_out.append(enriched)

        for raw in sections.get("open_accounts_with_issues") or []:
            enriched = enrich_account_metadata(_thaw(raw))
            core_key = _meta_key(enriched)
            stage = meta_stage.get(core_key)
            if not stage:
                nm = str(enriched.get("normalized_name") or enriched.get("name") or "").lower()
                stage = stage_by_name.get(nm)
            if stage:
                enriched["source_stage"] = stage
            reasons = _suppression_reasons(enriched)
            if reasons:
                _log_suppressed(enriched, reasons)
                continue
            _append(enriched, open_issues)
            problems_out.append(enriched)

        if not problems_out:
            for enriched in all_enriched:
                enriched = dict(enriched)
                reasons = _suppression_reasons(enriched)
                if reasons:
                    _log_suppressed(enriched, reasons)
                    continue
                _append(enriched, None)
                problems_out.append(enriched)
        else:
            for enriched in all_enriched:
                enriched = dict(enriched)
                reasons = _suppression_reasons(enriched)
                if reasons:
                    _log_suppressed(enriched, reasons)
                    continue
                _append(enriched, None)

        return negatives, open_issues, all_acc, problems_out

    problems = list(sections.get("problem_accounts") or [])
    if not seeded_text_cache:
        negatives, open_issues, all_acc = _build_emit_sections_from_problems(problems)
    else:
        negatives, open_issues, all_acc, problems = _build_emit_sections_from_parser_sections()
        sections["problem_accounts"] = problems

    sections["negative_accounts"] = negatives
    sections["open_accounts_with_issues"] = open_issues
    sections["all_accounts"] = all_acc

    def _log_emitted_accounts(accounts: list[Mapping[str, Any]]) -> None:
        for acc in accounts:
            name_raw = acc.get("normalized_name") or acc.get("name") or ""
            name = str(name_raw).lower()
            primary = acc.get("primary_issue") or ""
            status_val = acc.get("status") or acc.get("account_status") or ""
            last4 = acc.get("account_number_last4")
            original_creditor = acc.get("original_creditor")
            issues = acc.get("issue_types") or []
            bureaus = acc.get("bureaus") or []
            stage = acc.get("source_stage") or ""
            payment_statuses = acc.get("payment_statuses")
            payment_status_summary: str
            if isinstance(payment_statuses, Mapping):
                values = [str(v or "") for v in payment_statuses.values()]
                payment_status_summary = ",".join(v for v in values if v)
            else:
                payment_status_summary = str(payment_statuses or acc.get("payment_status") or "")
            has_co = bool(acc.get("has_co_marker"))
            remarks_val = acc.get("remarks")
            has_remarks = bool(remarks_val)
            logger.info(
                "emitted_account name=%s primary_issue=%s status=%s last4=%s orig_cred=%s issues=%s bureaus=%s stage=%s payment_statuses=%s has_co_marker=%s has_remarks=%s",
                name,
                primary,
                status_val,
                last4,
                original_creditor,
                issues,
                bureaus,
                stage,
                payment_status_summary,
                has_co,
                has_remarks,
            )

    _log_emitted_accounts(negatives)
    _log_emitted_accounts(open_issues)
    update_session(session_id, status="awaiting_user_explanations")
    _log_account_snapshot("pre_bureau_payload")

    if not seeded_text_cache:
        # detailed per-account final_emit already logged above in build loop
        # Assert emit matches finalized problem_accounts (SSOT)
        def assert_emit_matches_finalize(finalized, emitted):
            f = {
                a.get("normalized_name"): (
                    a.get("primary_issue"),
                    len(str(a.get("advisor_comment") or "")),
                )
                for a in (finalized or [])
            }
            for acc in emitted.get("all_accounts", []):
                name = acc.get("normalized_name")
                if name in f:
                    p_final, l_final = f[name]
                    p_emit = acc.get("primary_issue")
                    l_emit = len(str(acc.get("advisor_comment") or ""))
                    if p_emit != p_final or (l_final >= 60 and l_emit < 60):
                        raise RuntimeError(
                            f"EMIT_MISMATCH name={name} primary_final={p_final} primary_emit={p_emit} "
                            f"advisor_len_final={l_final} advisor_len_emit={l_emit}"
                        )

        try:
            assert_emit_matches_finalize(
                sections.get("problem_accounts") or [],
                {"all_accounts": sections.get("all_accounts") or []},
            )
        except Exception as e:
            logger.error("%s", e)
            raise

    # ------------------------------------------------------------------
    # Persist compact session summary + per-account full JSON artifacts
    # ------------------------------------------------------------------
    try:
        from datetime import datetime as _dt

        def _ensure_dir(p: str) -> None:
            os.makedirs(p, exist_ok=True)

        def _slug(s: str) -> str:
            base = (s or "").strip().lower().replace(" ", "_")
            return (
                "".join(ch for ch in base if ch.isalnum() or ch in ("_", "-"))
                or "account"
            )

        def _digits_only(s: str) -> str:
            return "".join(ch for ch in str(s) if ch.isdigit())

        def _is_negative_status(text: str) -> bool:
            t = (text or "").lower()
            if not t:
                return False
            return (
                "collection" in t
                or "charge off" in t
                or "charge-off" in t
                or "chargeoff" in t
                or "past due" in t
                or any(
                    x in t
                    for x in ["late 30", "late 60", "late 90", "late 120", "late 150"]
                )
            )

        # Build compact summaries strictly from SSOT
        ssot_accounts = list(sections.get("problem_accounts") or [])
        summaries: list[dict[str, Any]] = []
        full_accounts_dir = os.path.join("traces", session_id, "accounts")
        _ensure_dir(full_accounts_dir)

        for acc in ssot_accounts:
            data = _thaw(acc)
            # Proper alignment: prefer slug(normalized_name|name) as account_id for summaries
            acc_id = _slug(data.get("normalized_name") or data.get("name") or "")
            name = data.get("name") or data.get("normalized_name") or ""
            number_display = data.get("account_number_display")
            last4 = data.get("account_number_last4")
            primary = data.get("primary_issue") or "unknown"
            # negative bureaus from payment_statuses
            neg_bureaus: list[str] = []
            ps = data.get("payment_statuses") or {}
            if isinstance(ps, dict):
                for bureau, status in ps.items():
                    if _is_negative_status(status):
                        neg_bureaus.append(str(bureau).lower())

            # Write full redacted account JSON
            full_doc = dict(data)
            # Redact any raw account number fields; keep display + last4 only
            for k in (
                "account_number",
                "account_number_raw",
                "account_number_masked",
                "masked_account",
            ):
                if k in full_doc:
                    full_doc.pop(k, None)

            # Persist
            full_path = os.path.join(full_accounts_dir, f"{acc_id}-{_slug(name)}.json")
            try:
                from backend.core.utils.atomic_io import atomic_write_json

                atomic_write_json(full_path, full_doc, ensure_ascii=False)
            except Exception as _werr:
                logger.debug(
                    "write_full_account_failed session=%s id=%s err=%s",
                    session_id,
                    acc_id,
                    _werr,
                )

            summaries.append(
                {
                    "account_id": acc_id,
                    "name": name,
                    "account_number_display": number_display,
                    "account_number_last4": last4,
                    "primary_issue": primary,
                    "negative_bureaus": neg_bureaus,
                }
            )

        # Write session summary (atomic)
        session_out = {
            "session_id": session_id,
            "created_at": _dt.utcnow().isoformat(timespec="seconds") + "Z",
            "problem_accounts": summaries,
        }
        sessions_dir = os.path.join("sessions")
        _ensure_dir(sessions_dir)
        from backend.core.utils.atomic_io import atomic_write_json

        atomic_write_json(
            os.path.join(sessions_dir, f"{session_id}.json"),
            session_out,
            ensure_ascii=False,
        )
        logger.info(
            "DBG session_artifacts_written session=%s accounts=%d",
            session_id,
            len(summaries),
        )
    except Exception as _exc:
        logger.debug("session_artifacts_failed session=%s err=%s", session_id, _exc)
    for cat in (
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
    ):
        for acc in sections.get(cat, []):
            # Do not overwrite primary_issue here; derive UI defaults separately if needed
            acc.setdefault("issue_types", [])
            acc.setdefault("status", acc.get("account_status") or "")
            acc.setdefault("late_payments", {})
            acc.setdefault("payment_statuses", {})
            acc.setdefault("has_co_marker", False)
            acc.setdefault("co_bureaus", [])
            acc.setdefault("remarks_contains_co", False)
            acc.setdefault("bureau_statuses", {})
            acc.setdefault("account_number_last4", None)
            acc.setdefault("account_fingerprint", None)
            acc.setdefault("original_creditor", None)
            acc.setdefault("source_stage", acc.get("source_stage") or "")
            acc.setdefault("bureau_details", {})
    if os.getenv("ANALYSIS_TRACE"):
        for acc in sections.get("negative_accounts", []) + sections.get(
            "open_accounts_with_issues", []
        ):
            remarks_contains_co = acc.get("remarks_contains_co")
            if remarks_contains_co is None:
                remarks = acc.get("remarks")
                remarks_lower = remarks.lower() if isinstance(remarks, str) else ""
                remarks_contains_co = (
                    "charge" in remarks_lower and "off" in remarks_lower
                ) or "collection" in remarks_lower
            statuses = acc.get("payment_statuses")
            payment_status_texts: list[str] = []
            if isinstance(statuses, dict):
                payment_status_texts.extend(str(v or "") for v in statuses.values())
            elif isinstance(statuses, (list, tuple, set)):
                payment_status_texts.extend(str(v or "") for v in statuses)
            else:
                payment_status_texts.append(str(statuses or ""))
            single_status = acc.get("payment_status")
            if single_status:
                payment_status_texts.append(str(single_status))
            status_lower = " ".join(payment_status_texts).lower()
            status_contains_co = "collection" in status_lower or (
                "charge" in status_lower and "off" in status_lower
            )
            trace_missing_reasons: list[str] = []
            if not statuses:
                trace_missing_reasons.append("no_payment_status_line")
            grid_history = acc.get("grid_history_raw")
            if grid_history:
                if isinstance(grid_history, dict):
                    grid_values = " ".join(str(v or "") for v in grid_history.values())
                else:
                    grid_values = str(grid_history)
                if "CO" not in grid_values:
                    trace_missing_reasons.append("no_co_in_grid")
            remarks_val = acc.get("remarks")
            if not remarks_val:
                trace_missing_reasons.append("no_remarks")
            if acc.get("heading_join_miss") or acc.get("heading_join_misses"):
                trace_missing_reasons.append("heading_join_miss")
            status_texts_field = acc.get("status_texts")
            if status_texts_field:
                texts: list[str] = []
                if isinstance(status_texts_field, dict):
                    texts.extend(str(v or "") for v in status_texts_field.values())
                elif isinstance(status_texts_field, (list, tuple, set)):
                    texts.extend(str(v or "") for v in status_texts_field)
                else:
                    texts.append(str(status_texts_field))
                combined = " ".join(texts).lower()
                if "collection" not in combined and not (
                    "charge" in combined and "off" in combined
                ):
                    trace_missing_reasons.append("no_collection_in_status_texts")
            details_hint: dict[str, dict[str, Any]] = {}
            details_contains_co = False
            for bureau, fields in (acc.get("bureau_details") or {}).items():
                code = {
                    "TransUnion": "TU",
                    "Experian": "EX",
                    "Equifax": "EQ",
                }.get(bureau, bureau[:2].upper())
                hint: dict[str, Any] = {}
                status_val = fields.get("account_status") or fields.get(
                    "payment_status"
                )
                if status_val:
                    hint["status"] = status_val
                    status_lower = str(status_val).lower()
                    if "collection" in status_lower or (
                        "charge" in status_lower and "off" in status_lower
                    ):
                        details_contains_co = True
                past_due_val = fields.get("past_due_amount")
                if past_due_val not in (None, "", 0):
                    hint["past_due"] = past_due_val
                if hint:
                    details_hint[code] = hint
            trace = {
                "name": acc.get("normalized_name"),
                "source_stage": acc.get("source_stage"),
                "primary_issue": acc.get("primary_issue"),
                "issue_types": acc.get("issue_types"),
                "status": acc.get("status") or acc.get("account_status"),
                "payment_statuses": acc.get("payment_statuses"),
                "payment_status": acc.get("payment_status"),
                "has_co_marker": acc.get("has_co_marker"),
                "remarks_contains_co": remarks_contains_co,
                "late_payments": acc.get("late_payments"),
                "bureau_statuses": acc.get("bureau_statuses"),
                "account_number_last4": acc.get("account_number_last4"),
                "account_fingerprint": acc.get("account_fingerprint"),
                "original_creditor": acc.get("original_creditor"),
            }
            co_bureaus = acc.get("co_bureaus")
            if co_bureaus:
                trace["co_bureaus"] = co_bureaus
            if trace_missing_reasons:
                trace["trace_missing_reasons"] = trace_missing_reasons
            if details_hint:
                trace["details_hint"] = details_hint
            if acc.get("primary_issue") in {"charge_off", "collection"} and not (
                acc.get("has_co_marker")
                or status_contains_co
                or remarks_contains_co
                or details_contains_co
            ):
                logger.info("account_trace_bug %s", json.dumps(trace, sort_keys=True))
            logger.info("account_trace %s", json.dumps(trace, sort_keys=True))
    # Optional AI fallback: only when deterministic parser finds no problems
    try:
        use_ai_fallback = os.getenv("RUN_AI_FALLBACK", "0") == "1"
        zero_problems = not (sections.get("problem_accounts") or [])
        if use_ai_fallback and zero_problems:
            print(
                "[INFO] No accounts found by deterministic parser. Running AI fallback ..."
            )
            try:
                from backend.core.services.ai_client import AIClient  # type: ignore

                def _ai_sections_fallback(txt: str) -> dict[str, Any] | None:
                    """Very lightweight AI fallback: attempt to produce minimal section JSON.

                    Returns a mapping compatible with analysis_schema keys or None on failure.
                    """

                    if not isinstance(ai_client, AIClient):
                        return None
                    # Minimal, defensive prompt. We prefer a strict JSON response; tolerate failures.
                    schema_hint = {
                        "type": "object",
                        "properties": {
                            "problem_accounts": {"type": "array"},
                            "inquiries": {"type": "array"},
                            "high_utilization_accounts": {"type": "array"},
                        },
                        "additionalProperties": True,
                    }
                    prompt = (
                        "You are analyzing a SmartCredit PDF text dump. "
                        "Return a JSON object with arrays: problem_accounts (accounts with late/collection/charge-off issues), "
                        "inquiries (recent credit inquiries), and high_utilization_accounts. Use conservative extraction; "
                        "include account name and any last4 if visible."
                    )
                    try:
                        resp = ai_client.response_json(
                            prompt=prompt + "\n\nTEXT:\n" + txt[:6000],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "sections",
                                    "schema": schema_hint,
                                },
                            },
                        )
                        # OpenAI responses API: parse top-level JSON content
                        content = getattr(resp, "output", None) or getattr(
                            resp, "content", None
                        )
                        if (
                            isinstance(content, list)
                            and content
                            and getattr(content[0], "type", "") == "output_text"
                        ):
                            raw = content[0].text  # type: ignore[attr-defined]
                        else:
                            raw = getattr(resp, "output_text", None) or getattr(
                                resp, "text", None
                            )
                        import json as _json

                        data = _json.loads(raw) if isinstance(raw, str) else None
                        if isinstance(data, dict):
                            return data
                    except Exception as _e:  # pragma: no cover - best effort
                        logger.info(
                            "ai_fallback_failed session=%s error=%s", session_id, _e
                        )
                    return None

            except Exception:

                def _ai_sections_fallback(_: str) -> dict[str, Any] | None:
                    return None

            ai_sections = _ai_sections_fallback(
                _raw_text if isinstance(_raw_text, str) else ""
            )
            if isinstance(ai_sections, dict):
                # Non-destructive merge: only fill when empty
                for key in (
                    "problem_accounts",
                    "inquiries",
                    "high_utilization_accounts",
                ):
                    if not sections.get(key) and ai_sections.get(key):
                        sections[key] = ai_sections.get(key) or []
    except Exception as _exc:  # pragma: no cover - defensive
        logger.debug("ai_fallback_wrapper_error session=%s error=%s", session_id, _exc)

    # Optional: export per-account traces for problem accounts detected
    try:
        EXPORT_ACCOUNT_TRACES = os.getenv("EXPORT_ACCOUNT_TRACES", "1") != "0"
        problem_accounts = sections.get("problem_accounts") or []
        if EXPORT_ACCOUNT_TRACES and problem_accounts:
            index: list[dict] = []
            for i, acc in enumerate(problem_accounts, start=1):
                creditor = (
                    acc.get("creditor")
                    or acc.get("furnisher")
                    or acc.get("name")
                    or "unknown"
                )
                acct_id = (
                    acc.get("account_id")
                    or acc.get("account_number")
                    or acc.get("account_number_raw")
                    or acc.get("masked_account")
                    or acc.get("acct")
                    or "na"
                )
                status = (
                    acc.get("status")
                    or acc.get("payment_status")
                    or acc.get("primary_issue")
                    or "issue"
                )
                prefix = f"acct{i:02d}-{creditor}-{acct_id}-{status}"

                acct_json_path = write_json_trace(
                    acc, session_id=session_id, prefix=prefix
                )

                summary_lines = [
                    f"Creditor: {creditor}",
                    f"Account:  {acct_id}",
                    f"Status:   {status}",
                    f"PastDue:  {acc.get('past_due')}",
                    f"LateFlags:{acc.get('late_flags') or acc.get('delinquencies')}",
                    f"Opened:   {acc.get('opened') or acc.get('date_opened')}",
                    f"Reported: {acc.get('last_reported') or acc.get('date_reported')}",
                    "",
                    f"Source markers: {acc.get('source_markers') or 'n/a'}",
                ]
                acct_txt_path = write_text_trace(
                    "\n".join(summary_lines), session_id=session_id, prefix=prefix
                )

                print(
                    f"[TRACE] account trace saved: {acct_json_path} | {acct_txt_path}"
                )
                index.append(
                    {
                        "i": i,
                        "creditor": creditor,
                        "account_id": acct_id,
                        "status": status,
                        "json": acct_json_path,
                        "txt": acct_txt_path,
                    }
                )
            write_json_trace(
                {"accounts": index, "main_dump": _dump},
                session_id=session_id,
                prefix="accounts-index",
            )
    except Exception as _exc:
        logger.debug("account_traces_failed session=%s error=%s", session_id, _exc)

    if config.PROBLEM_DETECTION_ONLY:
        problem_accounts = sections.get("problem_accounts") or []
        return {"problem_accounts": problem_accounts}
    payload = BureauPayload(
        disputes=[
            BureauAccount.from_dict(d) for d in sections.get("negative_accounts", [])
        ],
        goodwill=[
            BureauAccount.from_dict(d)
            for d in sections.get("open_accounts_with_issues", [])
        ],
        inquiries=[
            Inquiry.from_dict(d)
            for d in sections.get(
                "unauthorized_inquiries", sections.get("inquiries", [])
            )
        ],
        high_utilization=[
            ProblemAccount.from_dict(d)
            for d in sections.get("high_utilization_accounts", [])
        ],
    )
    logger.debug(
        "constructed_bureau_payload disputes=%d goodwill=%d inquiries=%d high_utilization=%d",
        len(payload.disputes),
        len(payload.goodwill),
        len(payload.inquiries),
        len(payload.high_utilization),
    )
    payload.needs_human_review = sections.get("needs_human_review", False)
    return payload


def extract_problematic_accounts_from_report_dict(
    file_path: str, session_id: str | None = None
) -> Mapping[str, Any]:
    """Deprecated adapter returning ``dict`` for old clients."""
    logger.debug(
        "extract_problematic_accounts_from_report_dict is deprecated; use extract_problematic_accounts_from_report instead"
    )
    payload = extract_problematic_accounts_from_report(file_path, session_id)
    if isinstance(payload, Mapping):
        return payload
    return {
        "negative_accounts": [a.to_dict() for a in payload.disputes],
        "open_accounts_with_issues": [a.to_dict() for a in payload.goodwill],
        "unauthorized_inquiries": [i.to_dict() for i in payload.inquiries],
    }
