"""AI adjudication pack builder for merge V2 flows."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Union

from backend.core.io.tags import read_tags
from backend.core.merge.acctnum import normalize_level
from backend.core.logic.normalize import last4 as normalize_last4
from backend.core.logic.normalize import normalize_acctnum
from backend.core.logic.report_analysis.constants import BUREAUS
from backend.core.logic.report_analysis.keys import normalize_issuer

from . import config as merge_config


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _account_merge_module():
    from backend.core.logic.report_analysis import account_merge

    return account_merge


DUPLICATE_DEBT_SYSTEM = (
    "You are an adjudicator for credit-report disputes.\n"
    "Decide whether two accounts represent the same underlying debt shown twice (a duplicate).\n"
    "Focus strictly on:\n"
    "- Original creditor (OC) linkage and naming chains.\n"
    "- Whether one account identifies the other account’s creditor as its OC.\n"
    "- Whether names indicate a creditor → collection/servicer relationship.\n"
    "- Only use the provided JSON (bureaus per account, account display name, primary_issue).\n"
    "- If a bureau’s normalized original_creditor matches the other account’s normalized name, treat as duplicate unless explicit evidence disproves it.\n"
    "\n"
    "Output JSON only: {\"decision\": \"...\", \"reason\": \"...\", \"flags\": {\"duplicate\": true|false}}\n"
    "Valid decisions: \"duplicate\" | \"not_duplicate\".\n"
    "Make the reason short and cite which bureau/field(s) established the OC link or disproved it.\n"
    "If data is insufficient, lean \"not_duplicate\" and say why."
)


def _load_json(path: Path, default: object) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    except OSError:
        logger.warning("MERGE_DUP_AUDIT_FILE_READ_FAILED path=%s", path, exc_info=True)
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("MERGE_DUP_AUDIT_JSON_DECODE_FAILED path=%s", path, exc_info=True)
        return default


def _extract_account_name(meta_payload: Mapping[str, object] | None) -> str:
    if not isinstance(meta_payload, Mapping):
        return ""

    for key in ("account_name", "label", "heading", "heading_guess", "name"):
        value = meta_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_primary_issue(tags_payload: object) -> str | None:
    if isinstance(tags_payload, Mapping):
        candidate = tags_payload.get("primary_issue")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    if isinstance(tags_payload, Sequence):
        for entry in tags_payload:
            if not isinstance(entry, Mapping):
                continue
            candidate = entry.get("primary_issue")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if str(entry.get("kind")).lower() == "issue":
                issue_value = entry.get("type")
                if isinstance(issue_value, str) and issue_value.strip():
                    return issue_value.strip()

    return None


_DUPLICATE_NAME_PREFIX_RE = re.compile(r"^[0-9]+[\s\-:./]*")
_DUPLICATE_NAME_SANITIZE_RE = re.compile(r"[^A-Z0-9]+")


def _normalize_duplicate_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = _DUPLICATE_NAME_PREFIX_RE.sub("", text)
    text = text.upper()
    text = _DUPLICATE_NAME_SANITIZE_RE.sub(" ", text)
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized or None


def _augment_bureaus_with_normalized_creditor(
    bureaus_payload: object,
) -> object:
    if not isinstance(bureaus_payload, Mapping):
        return bureaus_payload

    enriched: Dict[str, object] = {}
    for key, value in bureaus_payload.items():
        if key in BUREAUS and isinstance(value, Mapping):
            entry = dict(value)
            normalized_oc = _normalize_duplicate_name(entry.get("original_creditor"))
            if normalized_oc:
                entry["original_creditor_normalized"] = normalized_oc
            else:
                entry.pop("original_creditor_normalized", None)
            enriched[key] = entry
        else:
            enriched[key] = value
    return enriched


def build_duplicate_audit_payload(
    runs_root: Path | str,
    sid: str,
    account_a: int,
    account_b: int,
) -> Dict[str, object]:
    """Return minimal payload for duplicate debt adjudication."""

    base = Path(runs_root) / str(sid) / "cases" / "accounts"

    def load_account(idx: int) -> Dict[str, object]:
        account_dir = base / str(idx)
        bureaus_raw = _load_json(account_dir / "bureaus.json", {})
        meta_raw = _load_json(account_dir / "meta.json", {})
        tags_raw = _load_json(account_dir / "tags.json", [])

        name = _extract_account_name(meta_raw)
        name_normalized = _normalize_duplicate_name(name)
        primary_issue = _extract_primary_issue(tags_raw)
        bureaus = _augment_bureaus_with_normalized_creditor(bureaus_raw)

        return {
            "name": name,
            "name_normalized": name_normalized,
            "primary_issue": primary_issue,
            "bureaus": bureaus,
        }

    return {
        "sid": sid,
        "pair": {"a": account_a, "b": account_b},
        "a": load_account(account_a),
        "b": load_account(account_b),
        "contract": {
            "response_format": "json",
            "fields": ["decision", "reason", "flags"],
        },
    }


def build_duplicate_debt_messages(payload: Mapping[str, object]) -> List[Dict[str, str]]:
    user_payload = {
        "sid": payload.get("sid"),
        "pair": payload.get("pair"),
        "a": {
            "name": payload.get("a", {}).get("name"),
            "name_normalized": payload.get("a", {}).get("name_normalized"),
            "primary_issue": payload.get("a", {}).get("primary_issue"),
            "bureaus": payload.get("a", {}).get("bureaus"),
        },
        "b": {
            "name": payload.get("b", {}).get("name"),
            "name_normalized": payload.get("b", {}).get("name_normalized"),
            "primary_issue": payload.get("b", {}).get("primary_issue"),
            "bureaus": payload.get("b", {}).get("bureaus"),
        },
    }

    user_json = json.dumps(user_payload, ensure_ascii=False)

    return [
        {"role": "system", "content": DUPLICATE_DEBT_SYSTEM},
        {"role": "user", "content": user_json},
    ]

MAX_CONTEXT_LINE_LENGTH = 240


FIELD_ORDER: Sequence[str] = (
    "Account #",
    "Balance Owed",
    "Last Payment",
    "Past Due Amount",
    "High Balance",
    "Creditor Type",
    "Account Type",
    "Payment Amount",
    "Credit Limit",
    "Last Verified",
    "Date of Last Activity",
    "Date Reported",
    "Date Opened",
    "Closed Date",
)

REMARK_PREFIXES: Sequence[str] = ("Creditor Remarks", "Remarks")

SKIP_KEYWORDS: Sequence[str] = (
    "two-year payment history",
    "two year payment history",
    "days late - 7 year history",
    "days late-7 year history",
)

HEADER_BUREAU_LINE_RE = re.compile(
    r"(transunion|experian|equifax).*(transunion|experian|equifax)", re.IGNORECASE
)
PAGINATION_RE = re.compile(r"^page\s+\d+\s+of\s+\d+", re.IGNORECASE)
ACCOUNT_NUMBER_RE = re.compile(r"Account #\s*(.*)", re.IGNORECASE)

LENDER_DROP_TOKENS = {
    "LLC",
    "INC",
    "CORP",
    "NA",
    "N A",
    "BANKCARD",
    "CARD",
    "CARDS",
    "SERV",
    "SERVICE",
    "SERVICES",
    "SVCS",
    "CACS",
}


def _empty_ids_by_bureau() -> Dict[str, Dict[str, str | None]]:
    return {
        bureau: {"raw": None, "digits": None, "last4": None}
        for bureau in BUREAUS
    }


def _normalize_bureau_value(value: str) -> Dict[str, str | None]:
    cleaned = str(value or "").strip()
    raw_value = cleaned if cleaned and cleaned != "--" else None
    digits = re.sub(r"\D", "", cleaned)
    digits_value = digits or None
    last4_value = digits[-4:] if len(digits) >= 4 else None
    return {"raw": raw_value, "digits": digits_value, "last4": last4_value}


def _extract_account_numbers_by_bureau(lines: Iterable[str]) -> Dict[str, Dict[str, str | None]]:
    result = _empty_ids_by_bureau()
    for line in lines:
        match = ACCOUNT_NUMBER_RE.search(str(line) if line is not None else "")
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part for part in tail.split() if part]
        if not parts:
            continue
        for idx, bureau in enumerate(BUREAUS):
            if idx >= len(parts):
                continue
            result[bureau] = _normalize_bureau_value(parts[idx])
        break
    return result


@dataclass(frozen=True)
class _PairTag:
    source_idx: int
    kind: str
    payload: Mapping[str, object]


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


def _load_raw_lines(path: Path) -> List[object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"raw_lines payload must be a list: {path}")


def _normalize_lender_display(raw: str) -> str:
    issuer = normalize_issuer(raw or "")
    if not issuer:
        return ""
    tokens: List[str] = []
    for token in issuer.split():
        adjusted = "BANK" if token == "BK" else token
        if adjusted in LENDER_DROP_TOKENS:
            continue
        tokens.append(adjusted)
    if not tokens:
        tokens = issuer.split()
    normalized = " ".join(tokens)
    return normalized.strip()


def _line_matches_label(line: str, label: str) -> bool:
    normalized_line = line.lower()
    normalized_label = label.lower().rstrip(":")
    return normalized_line.startswith(normalized_label)


def _is_skip_line(line: str) -> bool:
    lowered = line.lower()
    if any(keyword in lowered for keyword in SKIP_KEYWORDS):
        return True
    if PAGINATION_RE.match(lowered):
        return True
    if HEADER_BUREAU_LINE_RE.search(line):
        return True
    return False


def _find_header_line(lines: Sequence[str]) -> str | None:
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        if _is_skip_line(line):
            continue
        if any(_line_matches_label(line, label) for label in FIELD_ORDER):
            continue
        if any(line.lower().startswith(prefix.lower()) for prefix in REMARK_PREFIXES):
            continue
        return line
    return None


def _collect_field_lines(lines: Sequence[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        if _is_skip_line(line):
            continue
        for label in FIELD_ORDER:
            if label in results:
                continue
            if _line_matches_label(line, label):
                results[label] = line
                break
    return results


def _collect_remarks(lines: Sequence[str]) -> List[str]:
    remarks: List[str] = []
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        for prefix in REMARK_PREFIXES:
            if line.lower().startswith(prefix.lower()):
                remarks.append(line)
                break
    return remarks


def _extract_account_number(lines: Iterable[str]) -> str | None:
    for line in lines:
        match = ACCOUNT_NUMBER_RE.search(line)
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part.strip(" -:") for part in re.split(r"--", tail)]
        for part in parts:
            cleaned = part.strip()
            if cleaned and not _is_only_dashes(cleaned):
                return cleaned
    return None


def _build_account_context(lines: Sequence[str], max_lines: int) -> List[str]:
    limit = max_lines if max_lines and max_lines > 0 else 0
    if limit <= 0:
        return []

    header = _find_header_line(lines)
    context: List[str] = []
    seen: set[str] = set()

    if header:
        context.append(header)
        seen.add(header)
        normalized = _normalize_lender_display(header)
        if normalized:
            normalized_line = f"Lender normalized: {normalized}"
            if normalized_line not in seen:
                context.append(normalized_line)
                seen.add(normalized_line)

    field_lines = _collect_field_lines(lines)
    for label in FIELD_ORDER:
        line = field_lines.get(label)
        if not line:
            continue
        if line in seen:
            continue
        context.append(line)
        seen.add(line)
        if len(context) >= limit:
            return context[:limit]

    remark_lines = _collect_remarks(lines)
    for line in remark_lines:
        if line in seen:
            continue
        context.append(line)
        seen.add(line)
        if len(context) >= limit:
            break

    return context[:limit]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _sum_parts(
    parts: Mapping[str, object] | None,
    fields: Iterable[str],
    *,
    as_float: bool,
) -> Union[int, float]:
    if not isinstance(parts, Mapping):
        return 0.0 if as_float else 0

    if as_float:
        total_float = 0.0
        for field in fields:
            try:
                value = parts.get(field, 0.0)
            except AttributeError:
                value = 0.0
            try:
                total_float += float(value or 0.0)
            except (TypeError, ValueError):
                continue
        return total_float

    total_int = 0
    for field in fields:
        try:
            total_int += int(parts.get(field, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total_int


def _select_primary_tag(entries: Sequence[_PairTag]) -> Mapping[str, object] | None:
    if not entries:
        return None
    best_entry = None
    best_score = None
    for entry in entries:
        payload = entry.payload
        points_mode_active = _account_merge_module().detect_points_mode_from_payload(payload)
        total = _account_merge_module().coerce_score_value(
            payload.get("total"), points_mode=points_mode_active
        )
        score = (total, 1 if entry.kind == "merge_pair" else 0)
        if best_score is None or score > best_score:
            best_entry = entry
            best_score = score
    return best_entry.payload if best_entry else None


def _build_highlights(tag_payload: Mapping[str, object]) -> Mapping[str, object]:
    aux = tag_payload.get("aux") if isinstance(tag_payload.get("aux"), Mapping) else {}
    raw_parts = tag_payload.get("parts") if isinstance(tag_payload.get("parts"), Mapping) else {}
    points_mode_active = _account_merge_module().detect_points_mode_from_payload(tag_payload)
    parts = _account_merge_module().normalize_parts_for_serialization(
        raw_parts, points_mode=points_mode_active
    )
    conflicts = (
        list(tag_payload.get("conflicts"))
        if isinstance(tag_payload.get("conflicts"), Sequence)
        else []
    )

    identity_fields, debt_fields = _account_merge_module().resolve_identity_debt_fields()
    identity_score = _sum_parts(
        parts,
        identity_fields,
        as_float=points_mode_active,
    )
    debt_score = _sum_parts(
        parts,
        debt_fields,
        as_float=points_mode_active,
    )

    total_value = _account_merge_module().coerce_score_value(
        tag_payload.get("total"), points_mode=points_mode_active
    )
    mid_candidate = tag_payload.get("mid")
    if mid_candidate is None:
        mid_candidate = tag_payload.get("mid_sum")
    mid_value = _account_merge_module().coerce_score_value(
        mid_candidate, points_mode=points_mode_active
    )

    return {
        "total": total_value,
        "strong": bool(tag_payload.get("strong")),
        "mid_sum": mid_value,
        "parts": dict(parts),
        "identity_score": identity_score,
        "debt_score": debt_score,
        "matched_fields": dict(aux.get("matched_fields", {})) if isinstance(aux, Mapping) else {},
        "conflicts": conflicts,
        "acctnum_level": normalize_level(aux.get("acctnum_level"))
        if isinstance(aux, Mapping)
        else "none",
        "points_mode": points_mode_active,
    }


def _tolerance_hint() -> Mapping[str, float | int]:
    cfg = _account_merge_module().get_merge_cfg()
    tolerances = cfg.tolerances if isinstance(cfg.tolerances, Mapping) else {}

    def _get_float(key: str, default: float) -> float:
        raw = tolerances.get(key)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _get_int(key: str, default: int) -> int:
        raw = tolerances.get(key)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    return {
        "amount_abs_usd": _get_float("AMOUNT_TOL_ABS", 50.0),
        "amount_ratio": _get_float("AMOUNT_TOL_RATIO", 0.01),
        "last_payment_day_tol": _get_int("LAST_PAYMENT_DAY_TOL", 5),
    }


def _load_account_payload(
    accounts_root: Path,
    account_idx: int,
    cache: MutableMapping[int, Mapping[str, object]],
    max_lines: int,
) -> Mapping[str, object]:
    if account_idx in cache:
        return cache[account_idx]

    raw_path = accounts_root / str(account_idx) / "raw_lines.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_lines.json not found for account {account_idx}")

    raw_lines = _load_raw_lines(raw_path)
    normalized_lines = [_normalize_line(_coerce_text(line)) for line in raw_lines]
    context = _build_account_context(normalized_lines, max_lines)
    account_number = _extract_account_number(normalized_lines)

    payload = {"context": context, "account_number": account_number, "lines": normalized_lines}
    cache[account_idx] = payload
    return payload


def _collect_pair_entries(accounts_root: Path) -> tuple[Dict[tuple[int, int], List[_PairTag]], Dict[int, set[int]]]:
    pair_entries: Dict[tuple[int, int], List[_PairTag]] = {}
    best_partners: Dict[int, set[int]] = {}

    allowed_kinds = {"merge_pair", "merge_best"}

    for entry in sorted(accounts_root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue

        tags_path = entry / "tags.json"
        tags = read_tags(tags_path)
        for tag in tags:
            if not isinstance(tag, Mapping):
                continue
            kind = str(tag.get("kind"))
            if kind not in allowed_kinds:
                continue
            decision = str(tag.get("decision", "")).lower()
            pack_allowed_flag = tag.get("pack_allowed")
            if pack_allowed_flag is not None and not bool(pack_allowed_flag):
                continue
            acct_level = _extract_acct_level(tag)
            if decision != "ai":
                is_hard_auto = decision == "auto" and acct_level == "exact_or_known_match"
                if not is_hard_auto:
                    continue
            partner = tag.get("with")
            try:
                partner_idx = int(partner)
            except (TypeError, ValueError):
                continue
            pair_key = tuple(sorted((idx, partner_idx)))
            pair_entries.setdefault(pair_key, []).append(
                _PairTag(source_idx=idx, kind=kind, payload=tag)
            )
            if kind == "merge_best":
                best_partners.setdefault(idx, set()).add(partner_idx)

    return pair_entries, best_partners


def _is_merge_best_pair(
    pair: tuple[int, int],
    best_partners: Mapping[int, set[int]],
) -> bool:
    a, b = pair
    best_a = best_partners.get(a, set())
    best_b = best_partners.get(b, set())
    return (b in best_a) or (a in best_b)


def _extract_acct_level(tag_payload: Mapping[str, object] | None) -> str:
    level = normalize_level(None)
    if not isinstance(tag_payload, Mapping):
        return level

    level = normalize_level(tag_payload.get("acctnum_level"))
    if level != "none":
        return level

    aux_payload = tag_payload.get("aux")
    if isinstance(aux_payload, Mapping):
        aux_level = normalize_level(aux_payload.get("acctnum_level"))
        if aux_level != "none":
            return aux_level

    return level


def build_merge_ai_packs(
    sid: str,
    runs_root: Path | str,
    *,
    only_merge_best: bool = True,
    max_lines_per_side: int = 20,  # Legacy signature retained; value unused for duplicate packs.
) -> List[Mapping[str, object]]:
    """Build duplicate-debt audit packs for merge adjudication."""

    sid_str = str(sid)
    runs_root_path = Path(runs_root)
    accounts_root = runs_root_path / sid_str / "cases" / "accounts"

    if not accounts_root.exists():
        raise FileNotFoundError(
            f"cases/accounts directory not found for sid={sid_str!r} under {runs_root_path}"
        )

    pair_entries, best_partners = _collect_pair_entries(accounts_root)
    packs: List[Mapping[str, object]] = []

    for pair in sorted(pair_entries.keys()):
        entries = pair_entries.get(pair, [])
        primary_tag = _select_primary_tag(entries)
        if primary_tag is None:
            logger.warning("MERGE_DUP_PACK_MISSING_TAG sid=%s pair=%s", sid_str, pair)
            continue

        a_idx, b_idx = pair

        acct_level = _extract_acct_level(primary_tag)
        is_merge_best = _is_merge_best_pair(pair, best_partners)
        hard_acct = acct_level == "exact_or_known_match"
        include_pair = (not only_merge_best) or is_merge_best or hard_acct

        if not include_pair:
            logger.info(
                "MERGE_DUP_PACK_SKIP %s",
                json.dumps(
                    {
                        "sid": sid_str,
                        "pair": {"a": a_idx, "b": b_idx},
                        "only_merge_best": bool(only_merge_best),
                        "is_merge_best": bool(is_merge_best),
                        "acctnum_level": acct_level,
                        "include": False,
                        "reason": "filtered_only_merge_best",
                    },
                    sort_keys=True,
                ),
            )
            continue

        pack_payload = build_duplicate_audit_payload(runs_root_path, sid_str, a_idx, b_idx)
        messages = build_duplicate_debt_messages(pack_payload)

        points_mode_active = _account_merge_module().detect_points_mode_from_payload(primary_tag)
        total_score = _account_merge_module().coerce_score_value(
            primary_tag.get("total"), points_mode=points_mode_active
        )

        pack_record: Dict[str, object] = {
            "sid": sid_str,
            "pair": {"a": a_idx, "b": b_idx},
            "messages": messages,
            "schema": "duplicate_debt_audit:v1",
            "sources": {
                "a": ["bureaus.json", "meta.json", "tags.json"],
                "b": ["bureaus.json", "meta.json", "tags.json"],
            },
            "score_total": total_score,
            "points_mode": points_mode_active,
        }

        packs.append(pack_record)

    return packs


__all__ = ["build_merge_ai_packs"]

