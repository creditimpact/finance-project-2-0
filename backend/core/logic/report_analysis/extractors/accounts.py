"""Per-bureau account block parser."""

from __future__ import annotations

import json
import logging
import re
from hashlib import sha1
from typing import Dict, List, Tuple

from backend.core.case_store.api import (
    get_account_case,
    get_or_create_logical_account_id,
    upsert_account_fields,
)
from backend.core.case_store import storage
from backend.core.config.flags import FLAGS
from backend.core.logic.report_analysis.keys import compute_logical_account_key
from backend.core.metrics import emit_metric
from backend.core.metrics.field_coverage import (
    EXPECTED_FIELDS,
    _is_filled,
    emit_account_field_coverage,
    emit_session_field_coverage_summary,
)
from backend.core.telemetry import metrics

from .tokens import (
    ACCOUNT_FIELD_MAP,
    ACCOUNT_RE,
    normalize_issuer,
    parse_amount,
    parse_date,
    parse_date_any,
)

logger = logging.getLogger(__name__)


def extract_last4(account_line: str) -> str:
    """Extract the last 4 digits from ``account_line``.

    Returns an empty string if fewer than four digits are present.
    """
    if not account_line:
        return ""
    digits = re.findall(r"\d", account_line)
    if len(digits) < 4:
        return ""
    return "".join(digits[-4:])


_BUREAU_CODES = {"TransUnion": "TU", "Experian": "EX", "Equifax": "EQ"}

_mode_emitted: set[str] = set()
_logical_ids: Dict[Tuple[str, str], str] = {}
_dedup_disabled_emitted: set[str] = set()
_used_ids: Dict[str, set[str]] = {}


def _dbg(msg: str, *args: object) -> None:
    if FLAGS.casebuilder_debug:
        logger.debug("CASEBUILDER: " + msg, *args)


def _bureau_code(name: str) -> str:
    return _BUREAU_CODES.get(name, name[:2].upper())


def _redact_key(k: str) -> str:
    if not k:
        return ""
    return ("..." + k[-6:]) if len(k) > 6 else k


def _detect_columns(bureau: str) -> Dict[str, bool]:
    b = bureau.lower()
    return {
        "tu": b.startswith("trans"),
        "xp": b.startswith("exp"),
        "eq": b.startswith("equ"),
    }


def _digest_first_account_line(block_lines: list[str]) -> str:
    """Return the first line with an account number or first non-empty line."""
    for ln in block_lines:
        if "account" in ln.lower() and "#" in ln:
            return ln.strip()
    for ln in block_lines:
        if ln.strip():
            return ln.strip()
    return ""


def _split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    prev_non_empty = ""
    for line in lines:
        if ACCOUNT_RE.search(line):
            if current:
                blocks.append(current)
            current = []
            if prev_non_empty:
                current.append(f"__ISSUER_HEADING__: {prev_non_empty.strip()}")
            current.append(line)
        else:
            if line.strip() == "" and current:
                blocks.append(current)
                current = []
            elif current:
                current.append(line)
        if line.strip():
            prev_non_empty = line
    if current:
        blocks.append(current)
    return blocks


def _parse_block(block: List[str]) -> Tuple[str, Dict[str, object], str]:
    account_idx = 0
    if not ACCOUNT_RE.search(block[account_idx]) and len(block) > 1:
        account_idx = 1
    account_line = block[account_idx]
    last4 = extract_last4(account_line)
    account_id = last4 if last4 else f"synthetic-{hash(' '.join(block)) & 0xffff:x}"
    fields: Dict[str, object] = {}
    for idx, line in enumerate(block):
        if idx == account_idx:
            continue
        if line.startswith("__ISSUER_HEADING__:"):
            value = line.split(":", 1)[1].strip()
            fields["issuer"] = normalize_issuer(value)
            _dbg('issuer_captured issuer="%s"', fields["issuer"])
            continue
        if ":" not in line:
            continue
        label, value = [p.strip() for p in line.split(":", 1)]
        label_lc = label.lower()
        key = ACCOUNT_FIELD_MAP.get(label_lc)
        if not key:
            extra = fields.setdefault("extra_fields", {})
            extra[label_lc] = value.strip()
            logger.debug("CASEBUILDER: extra_field label=%r value=%r", label, value)
            continue
        if "amount" in key or key in {
            "high_balance",
            "balance_owed",
            "past_due_amount",
            "credit_limit",
            "payment_amount",
        }:
            fields[key] = parse_amount(value)
        elif key == "date_opened":
            fields[key] = parse_date_any(value) or value.strip()
            if fields.get("date_opened") and fields["date_opened"] != value.strip():
                logger.debug(
                    "CASEBUILDER: date_opened_parsed raw=%r iso=%r",
                    value,
                    fields["date_opened"],
                )
        elif key.endswith("date") or key in {
            "last_verified",
            "last_payment",
            "date_of_last_activity",
        }:
            fields[key] = parse_date(value) or value.strip()
        else:
            fields[key] = value.strip()
    return account_id, fields, account_line


def extract(
    lines: List[str], *, session_id: str, bureau: str
) -> List[Dict[str, object]]:
    """Extract accounts from ``lines`` and write to Case Store."""

    blocks = _split_blocks(lines)
    results: List[Dict[str, object]] = []
    input_blocks = 0
    upserted = 0
    dropped = {"missing_logical_key": 0, "write_error": 0}
    surrogate_key_used = 0

    if session_id not in _mode_emitted:
        emit_metric(
            "stage1.per_account_mode.enabled",
            1.0 if FLAGS.one_case_per_account_enabled else 0.0,
            session_id=session_id,
        )
        _mode_emitted.add(session_id)
    if not FLAGS.one_case_per_account_enabled and session_id not in _dedup_disabled_emitted:
        metrics.increment(
            "casebuilder.dedup.disabled_session", tags={"session_id": session_id}
        )
        _dedup_disabled_emitted.add(session_id)

    for block_index, block in enumerate(blocks):
        input_blocks += 1
        metrics.increment("casebuilder.input_blocks", tags={"session_id": session_id})
        account_id, fields, account_line = _parse_block(block)
        if not FLAGS.one_case_per_account_enabled:
            used = _used_ids.setdefault(session_id, set())
            base_id = account_id
            idx = 1
            while account_id in used:
                account_id = f"{base_id}_{idx}"
                idx += 1
            used.add(account_id)
        raw_block_lines = [
            ln.split(":", 1)[1].strip() if ln.startswith("__ISSUER_HEADING__:") else ln
            for ln in block
        ]
        raw_block = "\n".join(raw_block_lines)
        logger.debug(
            "CASEBUILDER: raw_block_attached lines=%d", len(raw_block_lines)
        )
        fields["raw_block"] = raw_block
        issuer = (
            fields.get("creditor_type") or fields.get("account_type") or ""
        ).strip()
        _dbg(
            "block_detected bureau=%s index=%d issuer=%s",
            bureau,
            block_index,
            issuer,
        )
        last4 = extract_last4(account_line)
        logger.debug(
            "CASEBUILDER: last4_extracted line=%r last4=%r", account_line, last4
        )
        expected = EXPECTED_FIELDS.get(bureau, [])
        filled_count = sum(1 for f in expected if _is_filled(fields.get(f)))
        fields_present_count = filled_count

        lk = compute_logical_account_key(
            fields.get("issuer") or fields.get("creditor_type"),
            last4 or None,
            fields.get("account_type"),
            fields.get("date_opened"),
        )
        logger.debug(
            "CASEBUILDER: logical_key issuer=%r creditor_type=%r last4=%r opened=%r lk=%r",
            fields.get("issuer"),
            fields.get("creditor_type"),
            last4,
            fields.get("date_opened"),
            lk,
        )
        issuer_for_surrogate = (
            fields.get("issuer") or fields.get("creditor_type") or ""
        ).strip()
        first_account_line = _digest_first_account_line(block)
        surrogate_components = (
            f"{issuer_for_surrogate}|{block_index}|{first_account_line}"
        )
        if not lk:
            lk = (
                "surrogate_"
                + sha1(surrogate_components.encode("utf-8")).hexdigest()[:16]
            )
            logger.debug(
                "CASEBUILDER: surrogate_key_generated issuer=%r block_index=%r first_line=%r lk=%r",
                issuer_for_surrogate,
                block_index,
                first_account_line,
                lk,
            )
            surrogate_key_used += 1
            metrics.increment(
                "casebuilder.surrogate_key_used",
                tags={"session_id": session_id},
            )

        min_fields_threshold = getattr(FLAGS, "CASEBUILDER_MIN_FIELDS", 0)
        if min_fields_threshold > 0 and filled_count < min_fields_threshold:
            fields["_weak_fields"] = True
            logger.debug(
                "CASEBUILDER: weak_fields threshold=%d present=%d",
                min_fields_threshold,
                filled_count,
            )
            metrics.increment(
                "casebuilder.tag.weak_fields",
                tags={"session_id": session_id},
            )
        columns_state = _detect_columns(bureau)
        persisted = False
        try:
            if FLAGS.one_case_per_account_enabled:
                logical_key = lk
                account_id = get_or_create_logical_account_id(session_id, logical_key)
                previous = _logical_ids.get((session_id, logical_key))
                if previous and previous != account_id:
                    emit_metric(
                        "stage1.logical_index.collisions",
                        1.0,
                        session_id=session_id,
                        logical_key=logical_key,
                        ids=f"{previous},{account_id}",
                    )
                    logger.warning(
                        "logical_index_collision %s",
                        {
                            "session_id": session_id,
                            "logical_key": logical_key,
                            "ids": [previous, account_id],
                        },
                )
                _logical_ids[(session_id, logical_key)] = account_id
                upsert_account_fields(
                    session_id=session_id,
                    account_id=account_id,
                    bureau=bureau,
                    fields={"by_bureau": {_bureau_code(bureau): fields}},
                )
            else:
                upsert_account_fields(
                    session_id=session_id,
                    account_id=account_id,
                    bureau=bureau,
                    fields=fields,
                )
            upserted += 1
            metrics.increment("casebuilder.upserted", tags={"session_id": session_id})
            persisted = True
        except Exception as e:  # pragma: no cover - diagnostic path
            dropped["write_error"] += 1
            logger.exception(
                "CASEBUILDER: write_error issuer=%r last4=%r err=%s",
                issuer,
                last4,
                e,
            )
            metrics.increment(
                "casebuilder.dropped",
                tags={"reason": "write_error", "session_id": session_id},
            )
        if FLAGS.casebuilder_debug:
            session_case_path = f"{storage.CASESTORE_DIR}/{session_id}.json"
            ledger = {
                "heading": fields.get("issuer") or fields.get("creditor_type") or "",
                "key_built": bool(lk),
                "lk": _redact_key(lk),
                "fields_present_count": fields_present_count,
                "columns_detected": {
                    "tu": bool(columns_state.get("tu")),
                    "xp": bool(columns_state.get("xp")),
                    "eq": bool(columns_state.get("eq")),
                },
                "weak_fields": bool(fields.get("_weak_fields")),
                "persisted": bool(persisted),
                "filename": session_case_path,
                "block_index": int(block_index),
                "session_id": session_id or "",
            }
            logger.debug(
                "CASEBUILDER: ledger %s",
                json.dumps(ledger, ensure_ascii=False, sort_keys=True),
            )
        if not persisted:
            continue

        emit_metric(
            "stage1.by_bureau.present",
            1.0,
            session_id=session_id,
            account_id=account_id,
            bureau=_bureau_code(bureau),
        )
        emit_account_field_coverage(
            session_id=session_id,
            account_id=account_id,
            bureau=bureau,
            fields=fields,
        )
        if FLAGS.normalized_overlay_enabled:
            try:
                from backend.core.normalize.apply import (
                    build_normalized,
                    emit_mapping_coverage_metrics,
                    load_registry,
                )

                reg = load_registry()
                case = get_account_case(session_id, account_id)
                by_bureau = case.fields.model_dump().get("by_bureau", {})
                overlay = build_normalized(by_bureau, reg)
                upsert_account_fields(
                    session_id=session_id,
                    account_id=account_id,
                    bureau=None,
                    fields={"normalized": overlay},
                )
                emit_mapping_coverage_metrics(session_id, account_id, by_bureau, reg)
            except Exception:
                logger.exception("normalized_overlay_failed")
        results.append({
            "account_id": account_id,
            "fields": fields,
            "raw_block": raw_block,
        })
    emit_session_field_coverage_summary(session_id=session_id)
    logger.info(
        "CASEBUILDER: session_summary blocks_detected=%d cases_written=%d session_id=%s dropped=%s",
        input_blocks,
        upserted,
        session_id,
        dropped,
    )
    metrics.gauge(
        "casebuilder.input_blocks.total",
        input_blocks,
        {"session_id": session_id},
    )
    metrics.gauge(
        "casebuilder.upserted.total",
        upserted,
        {"session_id": session_id},
    )
    metrics.gauge(
        "casebuilder.surrogate_key_used.total",
        surrogate_key_used,
        {"session_id": session_id},
    )
    for reason, count in dropped.items():
        metrics.gauge(
            "casebuilder.dropped.total",
            count,
            {"reason": reason, "session_id": session_id},
        )
    return results


def build_account_cases(session_id: str) -> None:
    """Build AccountCase records for ``session_id`` if needed."""

    # The extractor writes directly to Case Store during analysis, so no
    # additional work is required here. The function exists to provide an
    # explicit orchestration hook and remains idempotent.
    return None
