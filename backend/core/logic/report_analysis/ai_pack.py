from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import date, datetime
from typing import Iterable, Mapping

from backend.core.ai.paths import ensure_merge_paths, pair_pack_path
from backend.core.logic.report_analysis.ai_packs import (
    build_duplicate_audit_payload,
    build_duplicate_debt_messages,
)

from . import config as merge_config


logger = logging.getLogger(__name__)
_candidate_logger = logging.getLogger("ai_packs")

WANTED_CONTEXT_KEYS: list[str] = [
    "Account #",
    "High Balance:",
    "Last Verified:",
    "Date of Last Activity:",
    "Date Reported:",
    "Date Opened:",
    "Balance Owed:",
    "Closed Date:",
    "Account Rating:",
    "Account Description:",
    "Dispute Status:",
    "Creditor Type:",
    "Account Status:",
    "Payment Status:",
    "Creditor Remarks:",
    "Payment Amount:",
    "Last Payment:",
    "Past Due Amount:",
    "Account Type:",
    "Credit Limit:",
]

DEFAULT_MAX_LINES = 20
MAX_CONTEXT_LINE_LENGTH = 240


def _coerce_text(entry: object) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        value = entry.get("text")
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    if entry is None:
        return ""
    return str(entry)


def _normalize_line(text: str) -> str:
    norm = text or ""
    norm = norm.replace("\u2013", "-").replace("\u2014", "-")
    norm = re.sub(r"\s+", " ", norm).strip()
    if len(norm) > MAX_CONTEXT_LINE_LENGTH:
        norm = norm[: MAX_CONTEXT_LINE_LENGTH - 3].rstrip() + "..."
    return norm


def _is_only_dashes(text: str) -> bool:
    if not text:
        return True
    return re.sub(r"[-\s]", "", text) == ""


def _should_include_value(text: str, wanted_keys: Iterable[str]) -> bool:
    for key in wanted_keys:
        if key and key in text:
            return True
    return False


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        number = int(str(value))
    except Exception:
        return default
    return number if number > 0 else default


def extract_context_raw(
    raw_lines: list[dict] | list[object],
    wanted_keys: list[str] | None,
    max_lines: int,
) -> list[str]:
    keys = wanted_keys or WANTED_CONTEXT_KEYS
    limit = max_lines if max_lines and max_lines > 0 else DEFAULT_MAX_LINES
    if limit <= 0:
        return []

    original_texts = [_coerce_text(entry) for entry in raw_lines or []]
    normalized = [_normalize_line(text) for text in original_texts]

    interesting_indices: list[int] = []
    for idx, line in enumerate(normalized):
        if not line or _is_only_dashes(line):
            continue
        if _should_include_value(line, keys):
            interesting_indices.append(idx)

    if not interesting_indices:
        return []

    header_index: int | None = None
    first_idx = interesting_indices[0]
    for idx in range(first_idx):
        candidate = normalized[idx]
        if candidate and not _is_only_dashes(candidate):
            header_index = idx
            break

    ordered_indices: list[int] = []
    if header_index is not None:
        ordered_indices.append(header_index)
    for idx in interesting_indices:
        if idx not in ordered_indices:
            ordered_indices.append(idx)

    context: list[str] = []
    seen: set[str] = set()
    for idx in ordered_indices:
        if len(context) >= limit:
            break
        line = normalized[idx]
        if not line or _is_only_dashes(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        context.append(line)

    return context[:limit]


def _load_raw_lines(path: Path) -> list[object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"raw_lines payload must be a list: {path}")


def _extract_account_number(raw_lines: Iterable[str]) -> str | None:
    pattern = re.compile(r"Account #\s*(.*)", re.IGNORECASE)
    for line in raw_lines:
        match = pattern.search(line)
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part.strip(" -:") for part in re.split(r"--", tail)]
        for part in parts:
            if part and not _is_only_dashes(part):
                return part
    return None


def _normalize_highlights(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("highlights payload must be a mapping if provided")


def _extract_field_values(lines: Iterable[str], field: str) -> list[str]:
    pattern = re.compile(rf"{re.escape(field)}\s*:?(.*)$", re.IGNORECASE)
    values: list[str] = []
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part.strip(" -") for part in re.split(r"--", tail)]
        for part in parts:
            if not part or part in {"--", "-"}:
                continue
            cleaned = part.strip()
            if cleaned:
                values.append(cleaned)
    return values


_AMOUNT_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def _parse_amount(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().strip("$")
    if not cleaned or cleaned.lower() in {"--", "n/a", "na"}:
        return None
    cleaned = cleaned.replace("$", "").replace(",", "")
    match = _AMOUNT_RE.search(cleaned)
    if not match:
        return None
    number = match.group(1)
    if not number:
        return None
    try:
        return float(number)
    except ValueError:
        return None


def _amounts_close(left: float, right: float, tol_abs: float, tol_ratio: float) -> bool:
    tol_abs = max(float(tol_abs), 0.0)
    tol_ratio = max(float(tol_ratio), 0.0)
    base = min(abs(left), abs(right))
    allowed = max(tol_abs, base * tol_ratio)
    return abs(left - right) <= allowed


def _first_amount(values: Iterable[str]) -> float | None:
    for raw in values:
        amount = _parse_amount(raw)
        if amount is not None:
            return amount
    return None


def _strip_ordinal_suffix(text: str) -> str:
    return re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


_DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%Y.%m.%d",
    "%b %d, %Y",
    "%d %b %Y",
    "%b %Y",
    "%Y",
)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    cleaned = value.strip().strip("-–—")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"--", "n/a", "na"}:
        return None
    normalized = _strip_ordinal_suffix(cleaned.replace("–", "-").replace("—", "-"))
    normalized = re.sub(r"\s+", " ", normalized)
    for fmt in _DATE_FORMATS:
        candidate = normalized
        if fmt in {"%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y"}:
            candidate = normalized.replace("-", "/")
        elif fmt in {"%m-%d-%Y", "%m-%d-%y", "%d-%m-%Y", "%d-%m-%y"}:
            candidate = normalized.replace("/", "-")
        elif fmt in {"%d.%m.%Y", "%Y.%m.%d"}:
            candidate = normalized.replace("-", ".").replace("/", ".")
        try:
            if fmt == "%Y-%m-%d":
                return date.fromisoformat(candidate)
            dt = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        if fmt == "%b %Y":
            return date(dt.year, dt.month, 1)
        if fmt == "%Y":
            return date(dt.year, 1, 1)
        return dt.date()
    numeric = re.sub(r"[^0-9]", "", normalized)
    if len(numeric) == 8:
        try:
            dt = datetime.strptime(numeric, "%Y%m%d")
            return dt.date()
        except ValueError:
            pass
    return None


def _first_date(values: Iterable[str]) -> date | None:
    for raw in values:
        parsed = _parse_date(raw)
        if parsed is not None:
            return parsed
    return None


_COLLECTION_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_collection_text(value: str) -> str:
    lowered = value.lower().replace("&", " and ")
    cleaned = _COLLECTION_NORMALIZE_RE.sub(" ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_COLLECTION_EXACT_TERMS = {
    "collection",
    "collections",
    "collection agency",
    "collection agencies",
    "collection services",
}

_COLLECTION_KEYWORDS = (
    "collection agency",
    "collection agencies",
    "collection dept",
    "collection department",
    "collection division",
    "collection office",
    "collection bureau",
    "collection company",
    "collections dept",
    "collections department",
    "collections division",
    "collections office",
    "collections bureau",
    "collections company",
    "other collection agencies",
    "debt collector",
    "debt collectors",
    "debt collection",
    "debt collections",
    "debt collection agency",
    "debt collection company",
    "debt buyer",
)


def _is_collection_agency(values: Iterable[str]) -> bool:
    for raw in values:
        normalized = _normalize_collection_text(raw)
        if not normalized:
            continue
        if normalized in _COLLECTION_EXACT_TERMS:
            return True
        for keyword in _COLLECTION_KEYWORDS:
            if keyword in normalized:
                return True
    return False


_ORIGINAL_KEYWORDS = {
    "bank",
    "financial",
    "finance",
    "lender",
    "mortgage",
    "credit union",
    "servicing",
    "services",
    "loan",
    "credit card",
}


def _is_original(values: Iterable[str]) -> bool:
    for raw in values:
        lowered = raw.lower()
        for token in _ORIGINAL_KEYWORDS:
            if token in lowered:
                return True
    return False


def _resolve_tolerances() -> tuple[float, float, int]:
    from backend.core.logic.report_analysis.account_merge import get_merge_cfg

    try:
        cfg = get_merge_cfg()
        tolerances = cfg.tolerances if isinstance(cfg.tolerances, Mapping) else {}
    except Exception:
        tolerances = {}
    try:
        tol_abs = float(tolerances.get("AMOUNT_TOL_ABS", 50.0))
    except (TypeError, ValueError):
        tol_abs = 50.0
    try:
        tol_ratio = float(tolerances.get("AMOUNT_TOL_RATIO", 0.01))
    except (TypeError, ValueError):
        tol_ratio = 0.01
    try:
        months_tol = int(tolerances.get("CA_DATE_MONTH_TOL", 6))
    except (TypeError, ValueError):
        months_tol = 6
    return tol_abs, tol_ratio, max(months_tol, 0)


def _months_apart(left: date, right: date) -> int:
    return abs((left.year - right.year) * 12 + (left.month - right.month))


def build_context_flags(
    highlights: Mapping[str, object] | None,
    lines_a: Iterable[str],
    lines_b: Iterable[str],
) -> dict[str, bool]:
    tol_abs, tol_ratio, ca_month_tol = _resolve_tolerances()

    highlights_map = dict(highlights or {})
    matched_fields = {}
    raw_matched = highlights_map.get("matched_fields")
    if isinstance(raw_matched, Mapping):
        matched_fields = {str(key): bool(value) for key, value in raw_matched.items()}

    creditor_type_a = list(_extract_field_values(lines_a, "Creditor Type"))
    creditor_type_b = list(_extract_field_values(lines_b, "Creditor Type"))
    account_type_a = list(_extract_field_values(lines_a, "Account Type"))
    account_type_b = list(_extract_field_values(lines_b, "Account Type"))
    account_description_a = list(_extract_field_values(lines_a, "Account Description"))
    account_description_b = list(_extract_field_values(lines_b, "Account Description"))
    creditor_description_a = list(_extract_field_values(lines_a, "Creditor Description"))
    creditor_description_b = list(_extract_field_values(lines_b, "Creditor Description"))
    creditor_remarks_a = list(_extract_field_values(lines_a, "Creditor Remarks"))
    creditor_remarks_b = list(_extract_field_values(lines_b, "Creditor Remarks"))

    descriptors_a = (
        creditor_type_a
        + account_type_a
        + account_description_a
        + creditor_description_a
        + creditor_remarks_a
    )
    descriptors_b = (
        creditor_type_b
        + account_type_b
        + account_description_b
        + creditor_description_b
        + creditor_remarks_b
    )

    is_collection_a = _is_collection_agency(descriptors_a)
    is_collection_b = _is_collection_agency(descriptors_b)

    is_original_a = (not is_collection_a) and _is_original(
        creditor_type_a + account_type_a + account_description_a
    )
    is_original_b = (not is_collection_b) and _is_original(
        creditor_type_b + account_type_b + account_description_b
    )

    balance_a = _first_amount(_extract_field_values(lines_a, "Balance Owed"))
    balance_b = _first_amount(_extract_field_values(lines_b, "Balance Owed"))
    past_due_a = _first_amount(_extract_field_values(lines_a, "Past Due Amount"))
    past_due_b = _first_amount(_extract_field_values(lines_b, "Past Due Amount"))

    amounts_equal = False
    if matched_fields.get("balance_owed") or matched_fields.get("past_due_amount"):
        amounts_equal = True
    else:
        amount_pairs = [
            (balance_a, balance_b),
            (past_due_a, past_due_b),
        ]
        for left, right in amount_pairs:
            if left is None or right is None:
                continue
            if left <= 0 or right <= 0:
                continue
            if _amounts_close(left, right, tol_abs, tol_ratio):
                amounts_equal = True
                break

    dla_a = _first_date(_extract_field_values(lines_a, "Date of Last Activity"))
    dla_b = _first_date(_extract_field_values(lines_b, "Date of Last Activity"))
    opened_a = _first_date(_extract_field_values(lines_a, "Date Opened"))
    opened_b = _first_date(_extract_field_values(lines_b, "Date Opened"))
    closed_a = _first_date(_extract_field_values(lines_a, "Closed Date"))
    closed_b = _first_date(_extract_field_values(lines_b, "Closed Date"))
    reported_a = _first_date(_extract_field_values(lines_a, "Date Reported"))
    reported_b = _first_date(_extract_field_values(lines_b, "Date Reported"))

    date_pairs: list[tuple[date, date]] = []
    for left, right in (
        (opened_a, opened_b),
        (opened_a, dla_b),
        (dla_a, dla_b),
        (dla_a, opened_b),
    ):
        if left is None or right is None:
            continue
        date_pairs.append((left, right))

    dates_plausible = any(right >= left for left, right in date_pairs)

    if not dates_plausible and is_collection_a and is_collection_b:
        ca_after_closed_pairs = [
            (closed_a, opened_b),
            (closed_a, reported_b),
            (closed_b, opened_a),
            (closed_b, reported_a),
        ]
        for left, right in ca_after_closed_pairs:
            if left is None or right is None:
                continue
            if right >= left:
                dates_plausible = True
                break

        if not dates_plausible and ca_month_tol > 0:
            ca_tolerance_pairs = [
                (opened_a, opened_b),
                (opened_a, reported_b),
                (reported_a, opened_b),
                (reported_a, reported_b),
                (closed_a, opened_b),
                (closed_a, reported_b),
                (closed_b, opened_a),
                (closed_b, reported_a),
            ]
            for left, right in ca_tolerance_pairs:
                if left is None or right is None:
                    continue
                if left.year != right.year:
                    continue
                if _months_apart(left, right) <= ca_month_tol:
                    dates_plausible = True
                    break

    return {
        "is_collection_agency_a": bool(is_collection_a),
        "is_collection_agency_b": bool(is_collection_b),
        "is_original_creditor_a": bool(is_original_a),
        "is_original_creditor_b": bool(is_original_b),
        "amounts_equal_within_tol": bool(amounts_equal),
        "dates_plausible_chain": bool(dates_plausible),
    }


def _build_pack_payload(
    sid: str,
    first_idx: int,
    second_idx: int,
    first_context: list[str],
    second_context: list[str],
    first_account_number: str | None,
    second_account_number: str | None,
    highlights: Mapping[str, object] | None,
    max_lines: int,
) -> dict:
    return {
        "sid": sid,
        "pair": {"a": first_idx, "b": second_idx},
        "highlights": _normalize_highlights(highlights),
        "context": {"a": list(first_context), "b": list(second_context)},
        "ids": {
            "account_number_a": first_account_number or "--",
            "account_number_b": second_account_number or "--",
        },
        "limits": {"max_lines_per_side": max_lines},
    }


def _validate_pack_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("pack payload must be a mapping")

    sid_value = payload.get("sid")
    if not isinstance(sid_value, str) or not sid_value:
        raise ValueError("pack payload requires a non-empty sid string")

    pair_payload = payload.get("pair")
    if not isinstance(pair_payload, Mapping):
        raise ValueError("pack payload requires pair metadata")

    pair_a = pair_payload.get("a")
    pair_b = pair_payload.get("b")
    if not isinstance(pair_a, int) or not isinstance(pair_b, int):
        raise ValueError("pair metadata must include integer indices")

    context_payload = payload.get("context")
    if not isinstance(context_payload, Mapping):
        raise ValueError("pack payload requires context metadata")

    for side in ("a", "b"):
        lines = context_payload.get(side)
        if not isinstance(lines, list):
            raise ValueError("context lines must be a list")
        if any(not isinstance(line, str) for line in lines):
            raise ValueError("context lines must be strings")

    ids_payload = payload.get("ids")
    if not isinstance(ids_payload, Mapping):
        raise ValueError("pack payload requires id metadata")

    limits_payload = payload.get("limits")
    if not isinstance(limits_payload, Mapping):
        raise ValueError("pack payload requires limits metadata")


def build_ai_pack_for_pair(
    sid: str,
    runs_root: str | os.PathLike[str],
    a_idx: int,
    b_idx: int,
    highlights: Mapping[str, object] | None = None,
) -> dict:
    sid_str = str(sid)
    runs_root_path = Path(runs_root)

    try:
        account_a = int(a_idx)
        account_b = int(b_idx)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Account indices must be integers") from exc

    merge_paths = ensure_merge_paths(runs_root_path, sid_str, create=True)
    pack_path = pair_pack_path(merge_paths, *sorted((account_a, account_b)))

    payload = build_duplicate_audit_payload(runs_root_path, sid_str, account_a, account_b)
    messages = build_duplicate_debt_messages(payload)

    merge_logic = None
    gate_cfg = None
    try:  # pragma: no cover - defensive import
        from backend.core.logic.report_analysis import account_merge as merge_logic  # type: ignore
    except Exception:
        merge_logic = None

    if merge_logic is not None:
        try:
            gate_cfg = merge_logic.get_merge_cfg()
        except Exception:  # pragma: no cover - defensive
            gate_cfg = None

    require_original_creditor = bool(
        getattr(gate_cfg, "require_original_creditor_for_ai", False)
    )

    gate_allowed = True
    gate_reason = ""
    if require_original_creditor and merge_logic is not None:
        bureaus_a = payload.get("a", {}).get("bureaus") if isinstance(payload.get("a"), Mapping) else {}
        bureaus_b = payload.get("b", {}).get("bureaus") if isinstance(payload.get("b"), Mapping) else {}
        gate_allowed, gate_reason = merge_logic._ai_pack_gate_allows(  # type: ignore[attr-defined]
            gate_cfg,
            bureaus_a if isinstance(bureaus_a, Mapping) else {},
            bureaus_b if isinstance(bureaus_b, Mapping) else {},
        )
        if not gate_reason:
            gate_reason = "missing_original_creditor"

    if require_original_creditor and not gate_allowed:
        reason_suffix = f" reason={gate_reason}" if gate_reason else ""
        skip_message = f"PACK_SKIPPED {account_a}-{account_b}{reason_suffix}"
        _candidate_logger.info(skip_message)
        logger.info(
            "MERGE_DUP_PACK_GATE sid=%s i=%s j=%s reason=%s stage=builder",
            sid_str,
            account_a,
            account_b,
            gate_reason,
        )
        if pack_path.exists():
            try:
                pack_path.unlink()
            except OSError:  # pragma: no cover - defensive
                logger.warning(
                    "MERGE_DUP_PACK_REMOVE_FAILED sid=%s pair=(%s,%s) path=%s",
                    sid_str,
                    account_a,
                    account_b,
                    pack_path,
                    exc_info=True,
                )
        return {
            "sid": sid_str,
            "pair": {"a": account_a, "b": account_b},
            "skipped": True,
            "reason": gate_reason,
        }

    points_mode_active = False
    score_total: float | int | None = None
    if isinstance(highlights, Mapping):
        if merge_logic is not None:
            points_mode_active = merge_logic.detect_points_mode_from_payload(highlights)
            score_total = merge_logic.coerce_score_value(
                highlights.get("total"), points_mode=points_mode_active
            )
        else:
            try:
                from backend.core.logic.report_analysis.account_merge import (
                    coerce_score_value,
                    detect_points_mode_from_payload,
                )

                points_mode_active = detect_points_mode_from_payload(highlights)
                score_total = coerce_score_value(
                    highlights.get("total"), points_mode=points_mode_active
                )
            except Exception:  # pragma: no cover - defensive fallback
                points_mode_active = False
                score_total = None

    pack_record: dict[str, object] = {
        "sid": sid_str,
        "pair": {"a": account_a, "b": account_b},
        "messages": messages,
        "schema": "duplicate_debt_audit:v1",
        "sources": {
            "a": ["bureaus.json", "meta.json", "tags.json"],
            "b": ["bureaus.json", "meta.json", "tags.json"],
        },
    }
    if score_total is not None:
        pack_record["score_total"] = score_total
        pack_record["points_mode"] = points_mode_active

    serialized = json.dumps(pack_record, ensure_ascii=False, sort_keys=True)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(serialized + "\n", encoding="utf-8")

    return pack_record

