# ruff: noqa
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import backend.config as config
from backend.config import RAW_JOIN_TOKENS_WITH_SPACE
from backend.core.logic.report_analysis.account_packager_coords import (
    package_block_raw_coords,
    write_block_raw_coords,
)
from backend.core.logic.report_analysis.block_segmenter import segment_account_blocks
from backend.core.logic.report_analysis.column_reader import (
    build_debug_rows,
    detect_bureau_columns,
    extract_bureau_table,
)
from backend.core.logic.report_analysis.normalize_fields import (
    clean_value,
    is_dash_placeholder,
    is_effectively_blank,
)
from backend.core.logic.report_analysis.report_parsing import (
    build_block_fuzzy,
    detect_bureau_order,
)
from backend.core.logic.report_analysis.text_provider import load_cached_text
from backend.core.text.env_guard import ensure_env_and_paths
from backend.pipeline.runs import RunManifest, require_pdf_for_sid
from backend.core.text.text_provider import load_text_with_layout
from scripts.split_general_info_from_tsv import split_general_info

# Optional G1 infra (label/bureau detection)
try:  # pragma: no cover - optional import
    from ._bureau_parse_utils import (
        find_label_groups as G1_find_label_groups,
        is_bureau as G1_is_bureau,
    )

    _G1_AVAILABLE = True
except Exception:  # pragma: no cover - best effort
    _G1_AVAILABLE = False


_SPACE_RE = re.compile(r"\s+")


def join_tokens_with_space(tokens: list[str]) -> str:
    """
    מחבר טוקנים עם רווח בודד, תוך נרמול רווחים מיותרים ושמירה על פיסוק.
    """
    # חיבור בסיסי
    s = " ".join(t.strip() for t in tokens if t is not None)
    # נרמול רווחים לבן־אחד
    s = _SPACE_RE.sub(" ", s)
    return s.strip()


from scripts.split_accounts_from_tsv import split_accounts as split_accounts_from_tsv


def _log_join_sample(sid: str, raw_tokens: list[str], target_file: str) -> None:
    try:
        joined = join_tokens_with_space(raw_tokens)
        raw_preview = "".join(raw_tokens)[:120]
        joined_preview = joined[:120]
        logger.info(
            "RAW_JOIN: using space joiner=%s sid=%s file=%s",
            RAW_JOIN_TOKENS_WITH_SPACE,
            sid,
            target_file,
        )
        logger.info("RAW_JOIN sample before=%r after=%r", raw_preview, joined_preview)
    except Exception:  # pragma: no cover - best effort
        logger.exception(
            "RAW_JOIN: failed to log sample sid=%s file=%s", sid, target_file
        )


def _first_line_tokens(tsv_path: Path) -> list[str]:
    try:
        with tsv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            first = next(reader, None)
            if not first:
                return []
            page = first.get("page")
            line = first.get("line")
            tokens = [first.get("text", "")]
            for row in reader:
                if row.get("page") == page and row.get("line") == line:
                    tokens.append(row.get("text", ""))
                else:
                    break
            return tokens
    except Exception:  # pragma: no cover - best effort
        return []


def load_account_blocks(session_id: str) -> List[Dict[str, Any]]:
    """Load previously exported account blocks for ``session_id``.

    The blocks are expected under ``traces/blocks/<session_id>/_index.json``.
    The index must be a JSON array where each element is a mapping with
    exactly the keys ``{"i", "heading", "file"}``. If the directory or index
    file is missing, an entry is malformed, or a referenced block file cannot
    be read/parsed, the function fails softly and simply returns an empty
    list.

    Parameters
    ----------
    session_id:
        Identifier used for locating ``traces/blocks/<session_id>``.

    Returns
    -------
    list[dict]
        List of block dictionaries of the form ``{"heading": str,
        "lines": list[str]}``.
    """

    # Prefer manifest-provided blocks dir under runs/<SID>/traces/blocks
    try:
        m = RunManifest.for_sid(session_id, allow_create=False)
        base = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
    except FileNotFoundError:
        base = Path("traces") / "blocks" / session_id
    except Exception:
        base = Path("traces") / "blocks" / session_id
    index_path = base / "_index.json"
    if not index_path.exists():
        return []
    try:
        idx = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    blocks: List[Dict[str, Any]] = []
    expected_keys = {"i", "heading", "file"}
    for entry in idx or []:
        if not isinstance(entry, dict):
            continue
        if set(entry.keys()) != expected_keys:
            # ignore unexpected/legacy index rows
            continue
        f = entry.get("file")
        if not isinstance(f, str) or not f:
            continue
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "heading" in data and "lines" in data:
                blocks.append(data)
        except Exception:
            continue
    return blocks


logger = logging.getLogger(__name__)
try:
    ensure_env_and_paths()
except Exception:
    pass


ENRICH_ENABLED = os.getenv("BLOCK_ENRICH", "1") != "0"
BLOCK_DEBUG = os.getenv("BLOCK_DEBUG", "0") == "1"
USE_LAYOUT_TEXT = os.getenv("USE_LAYOUT_TEXT", "1") != "0"
RAW_TWO_STAGE = os.getenv("RAW_TWO_STAGE", "1") == "1"


FIELD_LABELS: dict[str, str] = {
    "account #": "account_number_display",
    "high balance": "high_balance",
    "last verified": "last_verified",
    "date of last activity": "date_of_last_activity",
    "date reported": "date_reported",
    "date opened": "date_opened",
    "balance owed": "balance_owed",
    "closed date": "closed_date",
    "account rating": "account_rating",
    "account description": "account_description",
    "dispute status": "dispute_status",
    "creditor type": "creditor_type",
    "original creditor 01": "original_creditor",
    "original creditor 02": "original_creditor",
    "orig. creditor": "original_creditor",
    "orig creditor": "original_creditor",
    "original creditor": "original_creditor",
    "account status": "account_status",
    "payment status": "payment_status",
    "creditor remarks": "creditor_remarks",
    "payment amount": "payment_amount",
    "last payment": "last_payment",
    "term length": "term_length",
    "past due amount": "past_due_amount",
    "account type": "account_type",
    "payment frequency": "payment_frequency",
    "credit limit": "credit_limit",
}

# Regex label mapping for robust SmartCredit variants (prefix match)
LABEL_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^ACCOUNT\s*#", re.I), "account_number_display"),
    (re.compile(r"^HIGH\s*BALANCE", re.I), "high_balance"),
    (re.compile(r"^LAST\s*VERIFIED", re.I), "last_verified"),
    (re.compile(r"^DATE\s+OF\s+LAST\s+ACTIVITY", re.I), "date_of_last_activity"),
    (re.compile(r"^DATE\s+REPORTED", re.I), "date_reported"),
    (re.compile(r"^DATE\s+OPENED", re.I), "date_opened"),
    (re.compile(r"^BALANCE\s+OWED", re.I), "balance_owed"),
    (re.compile(r"^CLOSED\s+DATE", re.I), "closed_date"),
    (re.compile(r"^ACCOUNT\s+RATING", re.I), "account_rating"),
    (re.compile(r"^ACCOUNT\s+DESCRIPTION", re.I), "account_description"),
    (re.compile(r"^DISPUTE\s+STATUS", re.I), "dispute_status"),
    (re.compile(r"^CREDITOR\s+TYPE", re.I), "creditor_type"),
    (
        re.compile(r"^ORIG(?:INAL)?\.?\s*CREDITOR(?:\s*\d{1,2})?\s*:?", re.I),
        "original_creditor",
    ),
    (re.compile(r"^ACCOUNT\s+STATUS", re.I), "account_status"),
    (re.compile(r"^PAYMENT\s+STATUS", re.I), "payment_status"),
    (re.compile(r"^CREDITOR\s+REMARKS", re.I), "creditor_remarks"),
    (re.compile(r"^PAYMENT\s+AMOUNT", re.I), "payment_amount"),
    (re.compile(r"^LAST\s+PAYMENT", re.I), "last_payment"),
    (re.compile(r"^TERM\s+LENGTH", re.I), "term_length"),
    (re.compile(r"^PAST\s+DUE\s+AMOUNT", re.I), "past_due_amount"),
    (re.compile(r"^ACCOUNT\s+TYPE", re.I), "account_type"),
    (re.compile(r"^PAYMENT\s+FREQUENCY", re.I), "payment_frequency"),
    (re.compile(r"^CREDIT\s+LIMIT", re.I), "credit_limit"),
    # Optional sections
    (re.compile(r"^TWO[- ]YEAR\s+PAYMENT\s+HISTORY", re.I), "two_year_payment_history"),
    (
        re.compile(r"^DAYS\s+LATE\s*[-–]\s*7\s*YEAR\s*HISTORY", re.I),
        "seven_year_days_late",
    ),
]


def _norm_hdr(s: str) -> str:
    s = (s or "").replace("\u00A0", " ").replace("®", " ").replace("™", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[^A-Za-z ]+", " ", s)
    return s.upper()


def _find_bureau_header_idx(lines: list[str]) -> tuple[int | None, list[str] | None]:
    """Return (index, order) for the first tri-bureau header occurrence.

    Supports headers split across two lines.
    """
    tokens = ["TRANSUNION", "EXPERIAN", "EQUIFAX"]
    for i in range(len(lines)):
        joined = _norm_hdr(lines[i])
        pairs = [joined]
        if i + 1 < len(lines):
            pairs.append(_norm_hdr(lines[i] + " " + lines[i + 1]))
        for text in pairs:
            if all(t in text for t in tokens):
                positions = {t: text.find(t) for t in tokens}
                order = [
                    k.lower() for k, _ in sorted(positions.items(), key=lambda x: x[1])
                ]
                return i, order
    return None, None


# Labels grouped per SmartCredit layout
LABELS_TOP = [
    "Account #",
    "High Balance",
    "Last Verified",
    "Date of Last Activity",
    "Date Reported",
    "Date Opened",
    "Balance Owed",
    "Closed Date",
    "Account Rating",
    "Account Description",
    "Dispute Status",
    "Creditor Type",
    "Original Creditor",
]

LABELS_BOTTOM = [
    "Account Status",
    "Payment Status",
    "Creditor Remarks",
    "Payment Amount",
    "Last Payment",
    "Term Length",
    "Past Due Amount",
    "Account Type",
    "Payment Frequency",
    "Credit Limit",
]


def _norm_token(s: str) -> str:
    s = (s or "").replace("\u00A0", " ").replace("®", " ").replace("™", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = s.rstrip(":")
    return s.upper()


def _is_noise_line(s: str) -> bool:
    t = (s or "").strip()
    return t in {"", "®", "--", "-"}


def _find_label_block(lines: list[str], labels: list[str]) -> int | None:
    """Find the index where a label block starts by matching the first label."""
    first = _norm_token(labels[0])
    for i, ln in enumerate(lines):
        if _norm_token(ln).startswith(first):
            return i
    return None


def _collect_values_sequence(
    lines: list[str], start_idx: int, stop_tokens: set[str], limit: int | None = None
) -> list[str]:
    vals: list[str] = []
    i = max(0, start_idx)
    while i < len(lines):
        tok = _norm_token(lines[i])
        if tok in stop_tokens:
            break
        if _is_noise_line(lines[i]):
            i += 1
            continue
        vals.append(lines[i].strip())
        if limit and len(vals) >= limit:
            break
        i += 1
    return vals


def _split_equal_parts(seq: list[str], parts: int) -> list[list[str]] | None:
    if parts <= 0:
        return None
    if len(seq) % parts != 0:
        return None
    step = len(seq) // parts
    return [seq[i * step : (i + 1) * step] for i in range(parts)]


def _zip_values_to_labels(raw_values: list[str], labels: list[str]) -> dict[str, str]:
    """Map values to labels with simple continuation joining and padding.

    - If extra values exist, join them to the last label's value.
    - Normalize tokens so dash placeholders stay as "--" and blanks stay empty.
    """
    cleaned = [clean_value(v) for v in raw_values]
    # Pad
    if len(cleaned) < len(labels):
        cleaned = cleaned + [""] * (len(labels) - len(cleaned))
    # Join surplus
    if len(cleaned) > len(labels):
        head = cleaned[: len(labels)]
        tail = cleaned[len(labels) :]
        joiner = " ".join(x for x in tail if x)
        if joiner:
            head[-1] = (head[-1] + " " + joiner).strip() if head[-1] else joiner
        cleaned = head
    out: dict[str, str] = {}
    for lab, val in zip(labels, cleaned):
        out[lab] = clean_value(val)
    return out


def _segments_by_bureau(
    block_lines: list[str], order: list[str]
) -> dict[str, dict[str, list[str]]] | None:
    """Return values per bureau for top/bottom sections when OCR yields stacked segments.

    Returns { 'top': {bureau: [..]}, 'bottom': {bureau: [..]} } or None if not matched.
    """
    if not order:
        return None
    # Normalize tokens and prepare stop set
    tokens = {"TRANSUNION", "EXPERIAN", "EQUIFAX"}
    stop_tokens = tokens.union({_norm_token(t) for t in LABELS_TOP + LABELS_BOTTOM})

    # Find the first occurrence of each bureau in order
    idxs: dict[str, int] = {}
    for i, ln in enumerate(block_lines):
        tok = _norm_token(ln)
        if tok in tokens and tok.lower() in order and tok.lower() not in idxs:
            idxs[tok.lower()] = i
            if len(idxs) == 3:
                break
    if len(idxs) < 2:  # need at least 2 to segment
        return None

    # Attempt to locate top/bottom label anchors
    top_anchor = _find_label_block(block_lines, LABELS_TOP)
    bot_anchor = _find_label_block(block_lines, LABELS_BOTTOM)

    result: dict[str, dict[str, list[str]]] = {"top": {}, "bottom": {}}

    # Collect TOP values if anchor exists and bureau tokens follow
    if top_anchor is not None:
        # For each bureau, collect up to len(LABELS_TOP) values after its token
        for b in order:
            pos = idxs.get(b)
            if pos is None:
                result["top"][b] = []
                continue
            # Stop at next bureau token or bottom anchor if present
            next_positions = [
                idxs.get(ob) for ob in order if idxs.get(ob) and idxs.get(ob) > pos
            ]
            bound = min(
                [p for p in next_positions if p is not None]
                + ([bot_anchor] if bot_anchor is not None else [])
                + [len(block_lines)]
            )
            vals = _collect_values_sequence(
                block_lines,
                pos + 1,
                tokens.union(
                    {
                        _norm_token("Two-Year Payment History"),
                        _norm_token("Days Late - 7 Year History"),
                    }
                ),
                limit=None,
            )
            # Trim to within bound window
            # Since _collect_values_sequence ignores stop token but not bound, slice manually
            vals = vals[: max(0, min(len(vals), bound - (pos + 1)))]
            # Keep only the first LABELS_TOP values; extra lines will be joined in zipper
            result["top"][b] = vals

    # Collect BOTTOM values sequence if anchor exists
    if bot_anchor is not None:
        # Gather all value lines after bot_anchor until next section (history) or end
        after = []
        for ln in block_lines[bot_anchor + 1 :]:
            t = _norm_token(ln)
            if t in {
                _norm_token("Two-Year Payment History"),
                _norm_token("Days Late - 7 Year History"),
            }:
                break
            if _is_noise_line(ln):
                continue
            # skip pure bureau tokens in this region; many OCRs omit them here
            if t in tokens:
                continue
            after.append(ln.strip())
        # If the total count fits 3x of labels, split equally
        parts = _split_equal_parts(after, 3)
        if parts:
            for idx, b in enumerate(order):
                result["bottom"][b] = parts[idx]
        else:
            # Best effort: assign sequentially by chunk of LABELS_BOTTOM
            for idx, b in enumerate(order):
                start = idx * len(LABELS_BOTTOM)
                result["bottom"][b] = after[start : start + len(LABELS_BOTTOM)]

    return result


# ------------------------------ G2: TOP-only parser ------------------------------
def _g2_make_label_pattern(label: str) -> re.Pattern[str]:
    esc = label.replace("-", "[-–]").replace("#", r"\s*#").replace(" ", r"\s+")
    return re.compile(rf"^{esc}\s*:?$", re.IGNORECASE)


_G2_TOP_PATS: list[tuple[re.Pattern[str], str]] = [
    (_g2_make_label_pattern(l), l) for l in LABELS_TOP
]


def _g2_is_bureau(s: str) -> str | None:
    t = _norm_hdr(s)
    if "TRANSUNION" in t:
        return "transunion"
    if "EXPERIAN" in t:
        return "experian"
    if "EQUIFAX" in t:
        return "equifax"
    return None


def _g2_find_top_group(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    """Return (start, end, labels) for the first TOP label group with >=3 labels."""
    n = len(lines)
    for i in range(n):
        j = i
        found: list[str] = []
        while j < n:
            m = None
            for pat, lab in _G2_TOP_PATS:
                if pat.match(lines[j].strip()):
                    m = lab
                    break
            if not m:
                break
            found.append(m)
            j += 1
        if len(found) >= 3:
            return i, j - 1, found
    return None, None, []


def _g2_detect_order(lines: list[str], start_idx: int) -> list[str]:
    order: list[str] = []
    for s in lines[start_idx:]:
        b = _g2_is_bureau(s)
        if b and b not in order:
            order.append(b)
            if len(order) == 3:
                break
    return order


def _g2_parse_top_only(
    lines: list[str],
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, int], int]:
    """Parse only the TOP label group into per-bureau fields.

    Returns (fields_by_bureau, counts_by_bureau, joined_fragments_count) or (None, {}, 0) if not matched.
    """
    start, end, labs = _g2_find_top_group(lines)
    if start is None or end is None or not labs:
        return None, {}, 0
    n_labels = len(labs)
    # Determine bureau order by scanning after the group
    order = _g2_detect_order(lines, end + 1)
    if not order:
        return None, {}, 0
    # Stop markers
    STOP_PAT = re.compile(
        r"^(Two\s*[- ]\s*Year\s+Payment\s+History|Days\s+Late|Account\s*#|Account\s+Status|Payment\s+Status)",
        re.IGNORECASE,
    )

    # Collect raw values per bureau
    i = end + 1
    fields: dict[str, dict[str, Any]] = {b: {} for b in order}
    counts: dict[str, int] = {b: 0 for b in order}
    joined_total = 0
    for b in order:
        vals: list[str] = []
        while i < len(lines):
            s = lines[i].strip()
            if _is_noise_line(s):
                i += 1
                continue
            # next bureau or stop
            if _g2_is_bureau(s) and _g2_is_bureau(s) != b:
                break
            if STOP_PAT.match(s):
                break
            # new top label group implies end
            if any(pat.match(s) for pat, _ in _G2_TOP_PATS):
                break
            vals.append(s)
            i += 1
        # Zip values to labels
        mapped = _zip_values_to_labels(vals, labs)
        # Count joined fragments
        if len(vals) > n_labels:
            joined_total += len(vals) - n_labels
        # Map to snake_case keys into fields[b]
        key_map = {
            "Account #": "account_number_display",
            "High Balance": "high_balance",
            "Last Verified": "last_verified",
            "Date of Last Activity": "date_of_last_activity",
            "Date Reported": "date_reported",
            "Date Opened": "date_opened",
            "Balance Owed": "balance_owed",
            "Closed Date": "closed_date",
            "Account Rating": "account_rating",
            "Account Description": "account_description",
            "Dispute Status": "dispute_status",
            "Creditor Type": "creditor_type",
        }
        for lab, val in mapped.items():
            fields[b][key_map[lab]] = val
        counts[b] = sum(1 for v in fields[b].values() if v)
        # move to next bureau token if current line was a bureau header consumed implicitly
        while (
            i < len(lines)
            and not _g2_is_bureau(lines[i])
            and not STOP_PAT.match(lines[i])
        ):
            # Skip residuals until next header; safety hatch
            break
        # if next line is bureau token for next bureau, it will be handled in next loop
    return fields, counts, joined_total


# ------------------------------ G3: BOTTOM-only parser ------------------------------
_G3_BOTTOM_PATS: list[tuple[re.Pattern[str], str]] = [
    (_g2_make_label_pattern(l), l) for l in LABELS_BOTTOM
]


def _g3_find_bottom_group(lines: list[str]) -> tuple[int | None, int | None, list[str]]:
    n = len(lines)
    for i in range(n):
        j = i
        found: list[str] = []
        while j < n:
            m = None
            for pat, lab in _G3_BOTTOM_PATS:
                if pat.match(lines[j].strip()):
                    m = lab
                    break
            if not m:
                break
            found.append(m)
            j += 1
        if len(found) >= 3:  # require at least a few labels for robustness
            return i, j - 1, found
    return None, None, []


def _g3_parse_bottom_only(
    lines: list[str],
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, int], int]:
    start, end, labs = _g3_find_bottom_group(lines)
    if start is None or end is None or not labs:
        return None, {}, 0
    n_labels = len(labs)
    order = _g2_detect_order(lines, end + 1)
    if not order:
        return None, {}, 0
    STOP_PAT = re.compile(
        r"^(Two\s*[- ]\s*Year\s+Payment\s+History|Days\s+Late|Account\s*#|Account\s+Description|Account\s+Rating|Creditor\s+Type)",
        re.IGNORECASE,
    )

    i = end + 1
    fields: dict[str, dict[str, Any]] = {b: {} for b in order}
    counts: dict[str, int] = {b: 0 for b in order}
    joined_total = 0
    for b in order:
        vals: list[str] = []
        while i < len(lines):
            s = lines[i].strip()
            if _is_noise_line(s):
                i += 1
                continue
            if _g2_is_bureau(s) and _g2_is_bureau(s) != b:
                break
            if STOP_PAT.match(s):
                break
            # If a new bottom group starts, stop
            if any(pat.match(s) for pat, _ in _G3_BOTTOM_PATS):
                break
            vals.append(s)
            i += 1
        mapped = _zip_values_to_labels(vals, labs)
        if len(vals) > n_labels:
            joined_total += len(vals) - n_labels
        key_map = {
            "Account Status": "account_status",
            "Payment Status": "payment_status",
            "Creditor Remarks": "creditor_remarks",
            "Payment Amount": "payment_amount",
            "Last Payment": "last_payment",
            "Term Length": "term_length",
            "Past Due Amount": "past_due_amount",
            "Account Type": "account_type",
            "Payment Frequency": "payment_frequency",
            "Credit Limit": "credit_limit",
        }
        for lab, val in mapped.items():
            fields[b][key_map[lab]] = val
        counts[b] = sum(1 for v in fields[b].values() if v)
    return fields, counts, joined_total


# ------------------------------ G4: presence, money cleaning, account tail ------------------------------
BUREAUS = ("transunion", "experian", "equifax")

_MONEY_KEYS = {
    "high_balance",
    "balance_owed",
    "credit_limit",
    "payment_amount",
    "past_due_amount",
}


def _g4_clean_money(val: str | None) -> str:
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if not s:
        return ""
    if is_dash_placeholder(s):
        return "--"
    # gentle remove of $ and commas
    s = s.replace("$", "").replace(",", "").strip()
    return s


def _g4_apply(
    fields: dict[str, dict[str, Any]] | None, meta: dict, lines: list[str]
) -> tuple[dict[str, dict[str, Any]], dict]:
    fields = fields or {}
    # Clean values and compute presence
    presence: dict[str, bool] = {}
    for b in BUREAUS:
        bureau_map = fields.get(b) or {}
        cleaned: dict[str, Any] = {}
        for k, v in bureau_map.items():
            raw = "" if v is None else v
            text = clean_value(raw)
            if k in _MONEY_KEYS:
                text = _g4_clean_money(text)
            cleaned[k] = text
        fields[b] = cleaned
        presence[b] = any(not is_effectively_blank(v) for v in cleaned.values())

    # Account number tail — if not already present, try to derive from fields
    if not meta.get("account_number_tail"):
        import re as _re

        tail_found = None
        for b in BUREAUS:
            acct = (fields.get(b) or {}).get("account_number_display") or (
                fields.get(b) or {}
            ).get("account_number")
            if isinstance(acct, str):
                m = _re.search(r"\*+(\d{2,4})\b", acct)
                if m:
                    tail_found = m.group(1)
                    break
        meta["account_number_tail"] = tail_found

    # Update presence
    meta.setdefault("bureau_presence", {})
    meta["bureau_presence"].update(presence)

    if BLOCK_DEBUG:
        try:
            logger.info(
                "ENRICH:G4 presence tu=%s ex=%s eq=%s tail=%s",
                presence.get("transunion", False),
                presence.get("experian", False),
                presence.get("equifax", False),
                meta.get("account_number_tail") or "",
            )
        except Exception:
            pass

    return fields, meta


# ------------------------------ G5: payment history and days-late meta (raw) ------------------------------
def _g5_enrich_meta(meta: dict, lines: list[str]) -> dict:
    try:
        # Two-Year Payment History tokens per bureau
        hist: dict[str, list[str]] = {}
        dl7: dict[str, dict[str, str]] = {}

        # Normalize token function and predicates
        def is_history_hdr(s: str) -> bool:
            t = _norm_token(s)
            return t.startswith(_norm_token("Two-Year Payment History"))

        def is_days_hdr(s: str) -> bool:
            t = _norm_token(s)
            return t.startswith(_norm_token("Days Late - 7 Year History"))

        tokens = {"TRANSUNION", "EXPERIAN", "EQUIFAX"}

        # Payment history
        try:
            start = next((i for i, ln in enumerate(lines) if is_history_hdr(ln)), None)
            if start is not None:
                i = start + 1
                current: str | None = None
                while i < len(lines):
                    t = _norm_token(lines[i])
                    if t in tokens:
                        current = t.lower()
                        hist.setdefault(current, [])
                        i += 1
                        continue
                    # stop if next section header
                    if is_days_hdr(lines[i]) or any(
                        pat.match(lines[i].strip())
                        for pat, _ in _G2_TOP_PATS + _G3_BOTTOM_PATS
                    ):
                        break
                    if current:
                        val = lines[i].strip()
                        if not _is_noise_line(val):
                            hist[current].append(val)
                    i += 1
        except Exception:
            pass

        # Days Late - 7 Year History
        try:
            start2 = next((i for i, ln in enumerate(lines) if is_days_hdr(ln)), None)
            if start2 is not None:
                i = start2 + 1
                current: str | None = None
                import re as _re

                while i < len(lines):
                    t = _norm_token(lines[i])
                    if t in tokens:
                        current = t.lower()
                        dl7.setdefault(current, {"30": "0", "60": "0", "90": "0"})
                        i += 1
                        continue
                    if any(
                        pat.match(lines[i].strip())
                        for pat, _ in _G2_TOP_PATS + _G3_BOTTOM_PATS
                    ) or is_history_hdr(lines[i]):
                        break
                    if current:
                        m30 = _re.search(r"30\s*:\s*(\d+)", lines[i])
                        m60 = _re.search(r"60\s*:\s*(\d+)", lines[i])
                        m90 = _re.search(r"90\s*:\s*(\d+)", lines[i])
                        if m30:
                            dl7[current]["30"] = m30.group(1)
                        if m60:
                            dl7[current]["60"] = m60.group(1)
                        if m90:
                            dl7[current]["90"] = m90.group(1)
                    i += 1
        except Exception:
            pass

        if hist:
            meta.setdefault("payment_history", {})
            meta["payment_history"].update(hist)
        if dl7:
            meta.setdefault("days_late_7y", {})
            meta["days_late_7y"].update(dl7)
    except Exception:
        # best-effort only
        pass
    return meta


# ---------------------------------------------------------------------------
# Canonicalization helpers for issuer headings
# ---------------------------------------------------------------------------

# Patterns apply to a basic-normalized form (lowercased, non-alnum -> single space)
CANON_MAP: dict[str, str] = {
    r"^bk of amer.*$": "Bank of America",
    r"^bankamerica.*$": "Bank of America",
    r"^bofa.*$": "Bank of America",
    r"^jpmcb( card)?$": "JPMorgan Chase",
    r"^chase.*$": "JPMorgan Chase",
    r"^wfbna( card)?$": "Wells Fargo",
    r"^us bk cacs$": "U.S. Bank",
    r"^syncb.*$": "Synchrony Bank",
    r"^cbna$": "Citibank",
    r"^amex$": "American Express",
    r"^nstar cooper$": "NSTAR/COOPER",
    r"^seterus inc$": "Seterus",
    r"^roundpoint$": "RoundPoint",
}

_CANON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.I), canon) for pat, canon in CANON_MAP.items()
]


def _basic_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    # Replace non-alphanumeric with spaces, collapse to single spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "account"


def normalize_heading(raw: str) -> dict:
    """Return canonical issuer fields for a raw heading.

    Returns a dict with keys: canonical, slug, variant.
    """
    variant = (raw or "").strip()
    base = _basic_normalize(variant)
    canonical = None
    for pat, canon in _CANON_PATTERNS:
        if pat.match(base):
            canonical = canon
            break
    if not canonical:
        canonical = base.title()
    slug = _slugify(canonical)
    logger.info(
        "BLOCK: canonical issuer heading=%r -> canonical=%r slug=%r",
        raw,
        canonical,
        slug,
    )
    return {"canonical": canonical, "slug": slug, "variant": variant}


def tail_digits_from_lines(lines: list[str]) -> str | None:
    """Attempt to extract last 2–4 digits of a masked account number.

    Looks for patterns like '****1234', 'XX1234', '1234****' on lines
    containing the word 'Account'/'Acct'. Returns the tail digits or None.
    """
    if not isinstance(lines, list):
        return None
    # Prefer lines that mention account number
    candidates = [
        ln
        for ln in lines
        if isinstance(ln, str) and re.search(r"\b(acc(?:ount)?|acct)\b", ln, re.I)
    ] or [ln for ln in lines if isinstance(ln, str)]
    for ln in candidates:
        s = str(ln)
        # masked then digits
        m = re.search(r"(?:[*xX]{2,}|\*{2,}|[xX]{2,})\s*(\d{2,4})\b", s)
        if m:
            return m.group(1)
        # digits then mask
        m = re.search(r"\b(\d{2,4})\s*(?:[*xX]{2,}|\*{2,}|[xX]{2,})", s)
        if m:
            return m.group(1)
        # fallback: trailing 2–4 digits at end of line
        m = re.search(r"(\d{2,4})\s*$", s)
        if m:
            return m.group(1)
    return None


def _split_vals(text: str, parts: int) -> list[str]:
    """Split ``text`` into ``parts`` values using column heuristics."""

    if not text:
        return [""] * parts

    stripped = text.rstrip()
    if (
        parts == 3
        and config.STAGEA_COLONLESS_TU_SPLIT
        and ":" not in text
        and stripped.endswith("-- --")
    ):
        marker = "-- --"
        cutoff = stripped.rfind(marker)
        head = stripped[:cutoff].strip()
        if head:
            return [head] + [""] * (parts - 1)

    vals = re.split(r"\s{2,}", text.strip())
    if len(vals) != parts:
        tokens = text.strip().split()
        if len(tokens) >= parts:
            vals = tokens[: parts - 1] + [" ".join(tokens[parts - 1 :])]
        else:
            vals = tokens + [""] * (parts - len(tokens))
    if len(vals) > parts:
        vals = vals[: parts - 1] + [" ".join(vals[parts - 1 :])]
    if len(vals) < parts:
        vals += [""] * (parts - len(vals))
    return [v.strip() for v in vals]


def enrich_block_v2(blk: dict) -> dict:
    """Enhanced enrichment for SmartCredit 3-bureau account blocks.

    Parses tri-bureau header order, then maps known labels into per-bureau fields.
    Keeps fields empty and meta consistent when header is missing.
    """

    heading = blk.get("heading", "")
    logger.warning("ENRICH: start heading=%r", heading)

    raw_lines = [str(line or "") for line in (blk.get("lines") or [])]
    lines = [(line or "").strip() for line in raw_lines]
    # G1 debug: label groups + bureaus seen
    if BLOCK_DEBUG and _G1_AVAILABLE:
        try:
            groups = G1_find_label_groups(raw_lines)
            bureaus_seen = sorted(
                {b for b in (G1_is_bureau(ln) for ln in raw_lines) if b}
            )
            logger.info(
                "ENRICH:G1 label_groups=%s bureaus_seen=%s",
                json.dumps(groups, ensure_ascii=False),
                ",".join(bureaus_seen),
            )
        except Exception:
            pass
    hdr_idx, order = _find_bureau_header_idx(lines)
    if not order:
        det = detect_bureau_order(lines)
        if det:
            order = det
            for i2, ln in enumerate(lines):
                low = ln.lower()
                if all(b in low for b in order):
                    hdr_idx = i2
                    break

    canon = normalize_heading(heading)
    tail = tail_digits_from_lines(lines)
    if not order:
        # No tri-header detected. Proceed to G2/G3 fallbacks below.
        order = []

    # Try stacked-segment parse first (bureau-wise values)
    segments = _segments_by_bureau(raw_lines, order)
    if segments:
        fields_segm: dict[str, dict[str, Any]] = {b: {} for b in order}
        # Map TOP
        for b in order:
            top_vals = segments.get("top", {}).get(b, [])
            if top_vals:
                mapped = _zip_values_to_labels(top_vals, LABELS_TOP)
                # Normalize label keys to snake_case set
                for lab, val in mapped.items():
                    key = {
                        "Account #": "account_number_display",
                        "High Balance": "high_balance",
                        "Last Verified": "last_verified",
                        "Date of Last Activity": "date_of_last_activity",
                        "Date Reported": "date_reported",
                        "Date Opened": "date_opened",
                        "Balance Owed": "balance_owed",
                        "Closed Date": "closed_date",
                        "Account Rating": "account_rating",
                        "Account Description": "account_description",
                        "Dispute Status": "dispute_status",
                        "Creditor Type": "creditor_type",
                    }[lab]
                    fields_segm[b][key] = val
        # Map BOTTOM
        for b in order:
            bot_vals = segments.get("bottom", {}).get(b, [])
            if bot_vals:
                mapped = _zip_values_to_labels(bot_vals, LABELS_BOTTOM)
                for lab, val in mapped.items():
                    key = {
                        "Account Status": "account_status",
                        "Payment Status": "payment_status",
                        "Creditor Remarks": "creditor_remarks",
                        "Payment Amount": "payment_amount",
                        "Last Payment": "last_payment",
                        "Term Length": "term_length",
                        "Past Due Amount": "past_due_amount",
                        "Account Type": "account_type",
                        "Payment Frequency": "payment_frequency",
                        "Credit Limit": "credit_limit",
                    }[lab]
                    fields_segm[b][key] = val

        # Fill presence/meta, apply G4 cleaning/derivations, return
        presence = {
            b: any(v not in (None, "") for v in fields_segm.get(b, {}).values())
            for b in ["transunion", "experian", "equifax"]
        }
        base_meta = dict(blk.get("meta") or {})
        meta = {
            "issuer_canonical": canon["canonical"],
            "issuer_slug": canon["slug"],
            "issuer_variant": canon["variant"],
            "bureau_presence": presence,
        }
        if tail:
            meta["account_number_tail"] = tail
        if BLOCK_DEBUG:
            meta.setdefault("debug", {})
            meta["debug"].update(
                {
                    "top_counts": {
                        b: len(segments.get("top", {}).get(b, [])) for b in order
                    },
                    "bottom_counts": {
                        b: len(segments.get("bottom", {}).get(b, [])) for b in order
                    },
                }
            )
        cleaned_fields, cleaned_meta = _g4_apply(
            fields_segm, {**base_meta, **meta}, raw_lines
        )
        cleaned_meta = _g5_enrich_meta(cleaned_meta, raw_lines)
        return {**blk, "fields": cleaned_fields, "meta": cleaned_meta}

    # G2: TOP-only parsing fallback (order-agnostic)
    fields_top, counts_top, joined_total = _g2_parse_top_only(raw_lines)
    if fields_top:
        # Also try BOTTOM and merge
        bottom_fields, bottom_counts, bottom_joined = _g3_parse_bottom_only(raw_lines)
        if bottom_fields:
            for b in ("transunion", "experian", "equifax"):
                src = fields_top.setdefault(b, {})
                upd = bottom_fields.get(b, {})
                for k, v in upd.items():
                    if k == "creditor_remarks" and src.get(k) and v:
                        src[k] = f"{src[k]} {v}".strip()
                    else:
                        if v:
                            src[k] = v
        presence = {
            b: any(v not in (None, "") for v in fields_top.get(b, {}).values())
            for b in ["transunion", "experian", "equifax"]
        }
        base_meta = dict(blk.get("meta") or {})
        meta = {
            "issuer_canonical": canon["canonical"],
            "issuer_slug": canon["slug"],
            "issuer_variant": canon["variant"],
            "bureau_presence": presence,
        }
        if tail:
            meta["account_number_tail"] = tail
        if BLOCK_DEBUG:
            logger.info(
                "ENRICH:G2 top_counts tu=%d ex=%d eq=%d joined=%d",
                counts_top.get("transunion", 0),
                counts_top.get("experian", 0),
                counts_top.get("equifax", 0),
                joined_total,
            )
            if bottom_fields:
                logger.info(
                    "ENRICH:G3 bottom_counts tu=%d ex=%d eq=%d merged=%s",
                    bottom_counts.get("transunion", 0),
                    bottom_counts.get("experian", 0),
                    bottom_counts.get("equifax", 0),
                    "True",
                )
        cleaned_fields2, cleaned_meta2 = _g4_apply(
            fields_top, {**base_meta, **meta}, raw_lines
        )
        cleaned_meta2 = _g5_enrich_meta(cleaned_meta2, raw_lines)
        return {**blk, "fields": cleaned_fields2, "meta": cleaned_meta2}

    # Full key set
    field_keys = [
        "account_number_display",
        "high_balance",
        "last_verified",
        "date_of_last_activity",
        "date_reported",
        "date_opened",
        "balance_owed",
        "closed_date",
        "account_rating",
        "account_description",
        "dispute_status",
        "creditor_type",
        "account_status",
        "payment_status",
        "creditor_remarks",
        "payment_amount",
        "last_payment",
        "term_length",
        "past_due_amount",
        "account_type",
        "payment_frequency",
        "credit_limit",
        "two_year_payment_history",
        "seven_year_days_late",
    ]
    fields: dict[str, dict[str, Any]] = {
        b: {k: None for k in field_keys} for b in ["transunion", "experian", "equifax"]
    }

    start_row = (hdr_idx or 0) + 1
    if hdr_idx is None:
        start_row = 0

    def _split_three(s: str) -> list[str]:
        parts = (
            re.split(r"\t+", s.strip()) if "\t" in s else re.split(r"\s{2,}", s.strip())
        )
        if len(parts) < 3:
            parts += [""] * (3 - len(parts))
        if len(parts) > 3:
            parts = parts[:2] + [" ".join(parts[2:])]
        return [p.strip() for p in parts]

    def _clean_cell(v: str) -> str:
        return clean_value(v)

    if hdr_idx is not None:
        logger.info(
            "ENRICH: bureau_header idx=%d order=%s",
            hdr_idx,
            "/".join([o[:2].upper() for o in order]),
        )

    i = start_row
    while i < len(lines):
        row = raw_lines[i].strip()
        if not row:
            i += 1
            continue
        matched = False
        for pat, key in LABEL_MAP:
            m = pat.match(row)
            if not m:
                continue
            matched = True
            rest = row[m.end() :].strip()
            rest = rest[1:].strip() if rest.startswith(":") else rest
            if key == "two_year_payment_history":
                block_lines = [row]
                j = i + 1
                while j < len(lines):
                    nxt = raw_lines[j].strip()
                    if not nxt:
                        break
                    if any(p.match(nxt) for p, _ in LABEL_MAP):
                        break
                    block_lines.append(nxt)
                    j += 1
                blob = "\n".join(block_lines)
                for b in order:
                    fields[b]["two_year_payment_history"] = blob
                i = j
                break
            if key == "seven_year_days_late":
                candidate_lines = [rest] if rest else []
                look = 3
                j = i + 1
                while j < len(lines) and look > 0:
                    candidate_lines.append(raw_lines[j].strip())
                    look -= 1
                    j += 1
                selected = None
                for s in candidate_lines:
                    if not s:
                        continue
                    if re.search(r"\t|\s{2,}", s):
                        selected = s
                        break
                selected = selected or (candidate_lines[0] if candidate_lines else "")
                cols = _split_three(selected)
                for idx2, b in enumerate(order):
                    col = cols[idx2] if idx2 < len(cols) else ""
                    m30 = re.search(r"30\s*:\s*(\d+)", col)
                    m60 = re.search(r"60\s*:\s*(\d+)", col)
                    m90 = re.search(r"90\s*:\s*(\d+)", col)
                    if m30 or m60 or m90:
                        fields[b]["seven_year_days_late"] = {
                            "30": int(m30.group(1)) if m30 else 0,
                            "60": int(m60.group(1)) if m60 else 0,
                            "90": int(m90.group(1)) if m90 else 0,
                        }
                i += 1
                break
            cols = _split_three(rest)
            for idx2, b in enumerate(order):
                v = _clean_cell(cols[idx2] if idx2 < len(cols) else "")
                fields[b][key] = v
            if BLOCK_DEBUG:
                logger.info(
                    "ENRICH: parsed %s -> tu=%r ex=%r eq=%r",
                    key,
                    fields[order[0]].get(key),
                    fields[order[1]].get(key),
                    fields[order[2]].get(key),
                )
            break
        if not matched:
            i += 1
            continue

    tu_count = sum(1 for v in fields["transunion"].values() if v not in (None, ""))
    ex_count = sum(1 for v in fields["experian"].values() if v not in (None, ""))
    eq_count = sum(1 for v in fields["equifax"].values() if v not in (None, ""))
    if BLOCK_DEBUG:
        logger.warning(
            "ENRICH: fields_done tu=%d ex=%d eq=%d", tu_count, ex_count, eq_count
        )
        logger.info(
            "BLOCK: enrichment_summary heading=%r tu_filled=%d ex_filled=%d eq_filled=%d",
            heading,
            tu_count,
            ex_count,
            eq_count,
        )

    presence = {
        "transunion": any(
            v not in (None, "") for v in fields.get("transunion", {}).values()
        ),
        "experian": any(
            v not in (None, "") for v in fields.get("experian", {}).values()
        ),
        "equifax": any(v not in (None, "") for v in fields.get("equifax", {}).values()),
    }
    if BLOCK_DEBUG:
        logger.info(
            "BLOCK: bureau_presence tu=%d ex=%d eq=%d tail=%s",
            1 if presence["transunion"] else 0,
            1 if presence["experian"] else 0,
            1 if presence["equifax"] else 0,
            tail or "",
        )
    base_meta = dict(blk.get("meta") or {})
    meta = {
        "issuer_canonical": canon["canonical"],
        "issuer_slug": canon["slug"],
        "issuer_variant": canon["variant"],
        "bureau_presence": presence,
    }
    if tail:
        meta["account_number_tail"] = tail
    if BLOCK_DEBUG:
        meta.setdefault("debug", {})
        meta["debug"].update(
            {
                "bureau_header_line": (hdr_idx + 1) if hdr_idx is not None else None,
                "head_sample": raw_lines[:8],
            }
        )
    cleaned_fields3, cleaned_meta3 = _g4_apply(fields, {**base_meta, **meta}, raw_lines)
    cleaned_meta3 = _g5_enrich_meta(cleaned_meta3, raw_lines)
    return {**blk, "fields": cleaned_fields3, "meta": cleaned_meta3}


def enrich_block(blk: dict) -> dict:
    """Add structured ``fields`` map parsed from ``blk['lines']``."""

    heading = blk.get("heading", "")
    logger.warning("ENRICH: start heading=%r", heading)

    # Normalize anomalies in potential bureau header lines
    lines = [(line or "").replace("®", "").strip() for line in (blk.get("lines") or [])]
    order = detect_bureau_order(lines)
    # Always compute canonical issuer + tail/meta, even if no bureau order found
    canon = normalize_heading(heading)
    tail = tail_digits_from_lines(lines)

    if not order:
        logger.warning(
            "ENRICH: no bureau columns; skipping enrichment heading=%r", heading
        )
        base_meta = dict(blk.get("meta") or {})
        meta = {
            "issuer_canonical": canon["canonical"],
            "issuer_slug": canon["slug"],
            "issuer_variant": canon["variant"],
            "bureau_presence": {
                "transunion": False,
                "experian": False,
                "equifax": False,
            },
        }
        if tail:
            meta["account_number_tail"] = tail
        meta = {**base_meta, **meta}
        return {**blk, "fields": {}, "meta": meta}

    # initialise fields map with empty strings
    field_keys = list(FIELD_LABELS.values())
    fields = {
        b: {k: "" for k in field_keys} for b in ["transunion", "experian", "equifax"]
    }

    in_section = False
    for line in blk.get("lines") or []:
        clean = line.strip()
        if not clean:
            continue
        if not in_section:
            norm = re.sub(r"[^a-z]+", " ", clean.lower())
            if all(b in norm for b in order):
                in_section = True
            continue

        norm_line = clean.lower()
        for label, key in FIELD_LABELS.items():
            if norm_line.startswith(label):
                rest = clean[len(label) :].strip()
                if rest.startswith(":"):
                    rest = rest[1:].strip()
                vals = _split_vals(rest, len(order))
                for idx, bureau in enumerate(order):
                    v = vals[idx] if idx < len(vals) else ""
                    fields[bureau][key] = clean_value(v)
                break

    tu_count = sum(1 for v in fields["transunion"].values() if not is_effectively_blank(v))
    ex_count = sum(1 for v in fields["experian"].values() if not is_effectively_blank(v))
    eq_count = sum(1 for v in fields["equifax"].values() if not is_effectively_blank(v))
    if BLOCK_DEBUG:
        logger.warning(
            "ENRICH: fields_done tu=%d ex=%d eq=%d", tu_count, ex_count, eq_count
        )
        logger.info(
            "BLOCK: enrichment_summary heading=%r tu_filled=%d ex_filled=%d eq_filled=%d",
            heading,
            tu_count,
            ex_count,
            eq_count,
        )

    # Meta: presence flags and identifiers
    presence = {
        "transunion": any(
            not is_effectively_blank(v) for v in fields.get("transunion", {}).values()
        ),
        "experian": any(
            not is_effectively_blank(v) for v in fields.get("experian", {}).values()
        ),
        "equifax": any(
            not is_effectively_blank(v) for v in fields.get("equifax", {}).values()
        ),
    }
    logger.info(
        "BLOCK: bureau_presence tu=%d ex=%d eq=%d tail=%s",
        1 if presence["transunion"] else 0,
        1 if presence["experian"] else 0,
        1 if presence["equifax"] else 0,
        tail or "",
    )
    base_meta = dict(blk.get("meta") or {})
    meta = {
        "issuer_canonical": canon["canonical"],
        "issuer_slug": canon["slug"],
        "issuer_variant": canon["variant"],
        "bureau_presence": presence,
    }
    if tail:
        meta["account_number_tail"] = tail
    meta = {**base_meta, **meta}

    return {**blk, "fields": fields, "meta": meta}


def _token_sort_key(tok: dict, page: int) -> tuple[float, float, float]:
    try:
        line = float(tok.get("line"))
    except Exception:
        line = float(tok.get("y0", 0.0))
    try:
        x0 = float(tok.get("x0", 0.0))
    except Exception:
        x0 = 0.0
    return page, line, x0


def _dump_full_tsv(layout: dict, out_path: Path) -> int:
    rows: List[Tuple[int, Any, float, float, float, float, str]] = []
    pages = list(layout.get("pages") or [])
    for idx, page in enumerate(pages, start=1):
        tokens = list(page.get("tokens") or [])
        for tok in tokens:
            rows.append(
                (
                    idx,
                    tok.get("line"),
                    float(tok.get("y0", 0.0)),
                    float(tok.get("y1", 0.0)),
                    float(tok.get("x0", 0.0)),
                    float(tok.get("x1", 0.0)),
                    (tok.get("text") or "").replace("\t", " "),
                )
            )
    rows.sort(
        key=lambda r: _token_sort_key({"line": r[1], "x0": r[4], "y0": r[2]}, r[0])
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("page\tline\ty0\ty1\tx0\tx1\ttext\n")
        for pg, ln, y0, y1, x0, x1, text in rows:
            ln_str = "" if ln is None else str(ln)
            fh.write(f"{pg}\t{ln_str}\t{y0}\t{y1}\t{x0}\t{x1}\t{text}\n")
    return len(rows)


def _build_accounts_table(
    session_id: str, accounts_dir: Path, layout: dict, layout_pages: list | None = None, block_windows: dict | None = None
) -> dict[str, str]:
    try:
        # Safety: ensure accounts_dir points to canonical runs/<sid>/traces/accounts_table
        try:
            resolved_dir = accounts_dir.resolve()
            assert "runs" in str(resolved_dir), "Stage-A out_dir must live under runs/<SID>"
        except Exception:
            pass

        full_tsv = accounts_dir / "_debug_full.tsv"
        count = _dump_full_tsv(layout, full_tsv)
        logger.info("Stage-A: wrote full TSV: %s", full_tsv)
        sample_tokens = _first_line_tokens(full_tsv)

        general_json = accounts_dir / "general_info_from_full.json"
        split_general_info(full_tsv, general_json)
        logger.info("Stage-A: wrote general info JSON: %s", general_json)
        _log_join_sample(session_id, sample_tokens, str(general_json))

        json_out = accounts_dir / "accounts_from_full.json"
        result = split_accounts_from_tsv(
            full_tsv,
            json_out,
            write_tsv=True,
            session_id=session_id,
            layout_pages=layout_pages,
            block_windows=block_windows,
        )
        logger.info("Stage-A: wrote accounts JSON: %s", json_out)
        _log_join_sample(session_id, sample_tokens, str(json_out))

        enriched_json = None
        try:
            stem = json_out.stem
            for suffix in ("enriched", "normalized"):
                candidate = json_out.with_name(f"{stem}.{suffix}.json")
                if candidate.exists():
                    enriched_json = candidate
                    logger.info(
                        "Stage-A: found enriched accounts JSON: %s", candidate
                    )
                    break
            if enriched_json is None:
                extra_candidates = [
                    p for p in accounts_dir.glob(f"{stem}.*.json") if p != json_out
                ]
                if len(extra_candidates) == 1:
                    enriched_json = extra_candidates[0]
                    logger.info(
                        "Stage-A: found enriched accounts JSON: %s", enriched_json
                    )
        except Exception:
            enriched_json = None

        accounts = result.get("accounts") or []
        collections = sum(1 for a in accounts if a.get("section") == "collections")
        logger.info(
            "Stage-A: accounts summary: total=%d collections=%d stop_marker_seen=%s",
            len(accounts),
            collections,
            result.get("stop_marker_seen"),
        )

        # Register artifacts in the accounts table index
        idx_path = accounts_dir / "_table_index.json"
        try:
            idx_obj = (
                json.loads(idx_path.read_text(encoding="utf-8"))
                if idx_path.exists()
                else {}
            )
        except Exception:
            idx_obj = {}

        idx_obj.setdefault("session_id", session_id)
        idx_obj["general_info"] = str(general_json)

        extras = [
            {"type": "full_tsv", "path": str(full_tsv)},
            {"type": "accounts_from_full", "path": str(json_out)},
        ]
        if enriched_json is not None:
            extras.append(
                {"type": "accounts_from_full_enriched", "path": str(enriched_json)}
            )

        existing = idx_obj.get("extras")
        if isinstance(existing, list):
            skip = {e.get("type") for e in extras}
            idx_obj["extras"] = [
                e for e in existing if isinstance(e, dict) and e.get("type") not in skip
            ]
            idx_obj["extras"].extend(extras)
        else:
            idx_obj["extras"] = extras

        idx_path.write_text(
            json.dumps(idx_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("BLOCK: accounts_table built sid=%s tokens=%d", session_id, count)
        return {
            "full_tsv": str(full_tsv),
            "accounts_json": str(json_out),
            "general_info": str(general_json),
        }
    except Exception:
        logger.exception("BLOCK: failed to build accounts_table sid=%s", session_id)
        return {}


def export_account_blocks(
    session_id: str, pdf_path: str | Path, *, accounts_out_dir: Path | None = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Extract account blocks from ``pdf_path`` and export them to JSON files.

    Parameters
    ----------
    session_id:
        Identifier used for the output directory ``traces/blocks/<session_id>``.
    pdf_path:
        Path to the PDF to parse.

    Returns
    -------
    tuple[list[dict], dict]
        A tuple of the exported account block dictionaries (each with
        ``heading`` and ``lines`` keys) and a metadata mapping with paths to
        Stage-A artifacts.
    """
    cached = load_cached_text(session_id)
    if not cached:
        raise ValueError("no_cached_text_for_session")
    text = cached["full_text"]

    # TP3: Preload layout tokens per page (best-effort, soft-fail)
    layout_pages: list[dict] = []
    TRY_LAYOUT = USE_LAYOUT_TEXT
    if TRY_LAYOUT:
        try:
            layout = load_text_with_layout(str(pdf_path))
            layout_pages = list(layout.get("pages") or [])
        except Exception:
            layout_pages = []
    logger.info("BLOCK: segmentation start sid=%s", session_id)
    fbk_blocks: List[Dict[str, Any]] = segment_account_blocks(text)

    if not fbk_blocks:
        logger.error(
            "BLOCKS_FAIL_FAST: 0 blocks extracted sid=%s file=%s",
            session_id,
            str(pdf_path),
        )
        raise ValueError("No blocks extracted")

    blocks_by_account_fuzzy = build_block_fuzzy(fbk_blocks) if fbk_blocks else {}
    logger.warning(
        "ANZ: pre-save fbk=%d fuzzy=%d sid=%s",
        len(fbk_blocks),
        len(blocks_by_account_fuzzy or {}),
        session_id,
    )

    # Determine canonical accounts_table directory and base output dir
    if accounts_out_dir is not None:
        accounts_dir = accounts_out_dir.resolve()
        # Guardrail: ensure caller provided a canonical runs path
        assert "runs" in str(accounts_dir), "Stage-A out_dir must live under runs/<SID>"
        # Place block exports under runs/<SID>/traces/blocks via RunManifest
        m = RunManifest.for_sid(session_id, allow_create=False)
        out_dir = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # Guardrail for blocks export dir as well
        low = str(out_dir.resolve()).lower()
        assert ("/runs/" in low) or ("\\runs\\" in low), "Stage-A out_dir must live under runs/<SID>"
    else:
        # Fallback legacy base (kept for non-integrated usages)
        out_dir = (Path("traces") / "blocks" / session_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        accounts_dir = (out_dir / "accounts_table").resolve()
    logger.warning("BLOCK_ENRICH: enabled=%s sid=%s", ENRICH_ENABLED, session_id)
    stage_a_meta: Dict[str, str] = {}

    # Stage A: Write a consolidated layout snapshot with all pages and tokens
    if RAW_TWO_STAGE and layout_pages:
        try:
            pages_out: List[Dict[str, Any]] = []
            for idx, pg in enumerate(layout_pages, start=1):
                try:
                    number = int(pg.get("number", idx))
                except Exception:
                    number = idx
                toks_in = list(pg.get("tokens") or [])
                toks_out: List[Dict[str, Any]] = []
                for t in toks_in:
                    # Keep stable fields and attach explicit page
                    toks_out.append(
                        {
                            "x0": t.get("x0"),
                            "x1": t.get("x1"),
                            "y0": t.get("y0"),
                            "y1": t.get("y1"),
                            "text": t.get("text"),
                            **({"line": t.get("line")} if "line" in t else {}),
                            **({"col": t.get("col")} if "col" in t else {}),
                            "page": number,
                        }
                    )
                pages_out.append(
                    {
                        "number": number,
                        "width": float(pg.get("width", 0.0) or 0.0),
                        "height": float(pg.get("height", 0.0) or 0.0),
                        "tokens": toks_out,
                    }
                )
            snap = {"session_id": session_id, "pages": pages_out}
            (out_dir / "layout_snapshot.json").write_text(
                json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Defer _build_accounts_table until after block_windows.json is ready
        except Exception:
            logger.exception(
                "BLOCK: failed to write layout_snapshot.json for sid=%s", session_id
            )

    # Task B: Load existing index map (non-disruptive); used to add index_headline into packages
    index_headlines: dict[int, str] = {}
    try:
        idx_path = out_dir / "_index.json"
        if idx_path.exists():
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            for row in idx_data or []:
                try:
                    bi = int(row.get("i"))
                    head = str(row.get("heading") or "")
                    if bi:
                        index_headlines[bi] = head
                except Exception:
                    continue
    except Exception:
        index_headlines = {}

    # TP4: Optional debug dump of layout tokens per page
    if BLOCK_DEBUG and layout_pages:
        for idx, pg in enumerate(layout_pages, start=1):
            try:
                dbg_path = out_dir / f"_debug_layout_page{idx:03d}.tsv.json"
                with dbg_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "number": pg.get("number", idx),
                            "width": pg.get("width"),
                            "height": pg.get("height"),
                            "tokens": pg.get("tokens", []),
                        },
                        f,
                        ensure_ascii=False,
                    )
            except Exception:
                pass

    out_blocks: List[Dict[str, Any]] = []
    raw_index_entries: List[Dict[str, Any]] = []
    window_index_entries: List[Dict[str, Any]] = []
    windows_by_block: Dict[str, List[Dict[str, Any]]] = {}
    page_tokens_map: Dict[int, List[dict]] = {}
    page_dims: Dict[int, Tuple[float, float]] = {}
    for pg in layout_pages or []:
        try:
            pnum = int(pg.get("number", 0) or 0)
        except Exception:
            pnum = 0
        page_tokens_map[pnum] = list(pg.get("tokens") or [])
        try:
            pw = float(pg.get("width") or 0.0)
        except Exception:
            pw = 0.0
        try:
            ph = float(pg.get("height") or 0.0)
        except Exception:
            ph = 0.0
        page_dims[pnum] = (pw, ph)

    def _window_from_tokens(tokens: List[dict], pnum: int) -> Dict[str, Any] | None:
        if not tokens:
            return None
        try:
            x0 = min(float(t.get("x0", 0.0) or 0.0) for t in tokens)
            x1 = max(float(t.get("x1", 0.0) or 0.0) for t in tokens)
            y0 = min(float(t.get("y0", 0.0) or 0.0) for t in tokens)
            y1 = max(float(t.get("y1", 0.0) or 0.0) for t in tokens)
        except Exception:
            return None
        pw, ph = page_dims.get(pnum, (0.0, 0.0))
        pad = 3.0
        return {
            "page": int(pnum),
            "x_min": float(max(0.0, x0 - pad)),
            "x_max": float(min(pw or (x1 + pad), x1 + pad)),
            "y_top": float(max(0.0, y0 - pad)),
            "y_bottom": float(min(ph or (y1 + pad), y1 + pad)),
        }

    def _fallback_windows_from_layout_window(
        blk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        dbg_win = ((blk.get("meta") or {}).get("debug") or {}).get("layout_window")
        if not isinstance(dbg_win, dict):
            return []
        try:
            pnum = int(dbg_win.get("page", 0) or 0)
        except Exception:
            pnum = 0
        toks_all = page_tokens_map.get(pnum, [])
        try:
            x_min = float(dbg_win.get("x_min", 0.0) or 0.0)
            x_max = float(dbg_win.get("x_max", 0.0) or 0.0)
            y_top = float(dbg_win.get("y_top", 0.0) or 0.0)
            y_bottom = float(dbg_win.get("y_bottom", 0.0) or 0.0)
        except Exception:
            return []
        sel: List[dict] = []
        for t in toks_all:
            try:
                mx = (float(t.get("x0", 0.0)) + float(t.get("x1", 0.0))) / 2.0
                my = (float(t.get("y0", 0.0)) + float(t.get("y1", 0.0))) / 2.0
            except Exception:
                continue
            if x_min <= mx <= x_max and y_top <= my <= y_bottom:
                sel.append(t)
        win = _window_from_tokens(sel, pnum)
        return [win] if win else []

    # Track canonical -> variants mapping to summarize at the end
    canon_variants: dict[str, set[str]] = {}
    idx_info = []
    for i, blk in enumerate(fbk_blocks, 1):
        # Log basic block info before enrichment
        btype = (
            (blk.get("meta") or {}).get("block_type") if isinstance(blk, dict) else None
        )
        logger.info(
            "BLOCK: new block i=%d type=%s heading=%r lines=%d",
            i,
            btype or "unknown",
            blk.get("heading") if isinstance(blk, dict) else "",
            len(blk.get("lines") or []) if isinstance(blk, dict) else 0,
        )
        # RAW mode: do not enrich fields; work off the original block
        out_blk = dict(blk)
        per_page_windows: List[Dict[str, Any]] = []

        # TP3: Column-reader integration to ensure full field sets per bureau
        try:
            # Map block line range to page indices using cached pages line counts
            page_texts = list((cached.get("pages") or []))
            starts: list[int] = []  # 1-based line start per page
            ends: list[int] = []  # 1-based line end per page (inclusive)
            acc = 1
            for ptxt in page_texts:
                n_lines = len(str(ptxt or "").splitlines())
                if n_lines <= 0:
                    n_lines = 0
                starts.append(acc)
                ends.append(acc + max(0, n_lines) - 1)
                acc += max(0, n_lines)

            meta = out_blk.get("meta") or {}
            sline = int(meta.get("start_line", 1))
            eline = int(meta.get("end_line", sline))
            # Determine overlapping pages
            page_idxs: list[int] = []  # 0-based index into cached pages / layout_pages
            for idx, (ps, pe) in enumerate(zip(starts, ends)):
                if ps == 0 and pe == 0:
                    continue
                # Overlap [sline, eline) with [ps, pe]
                if sline <= pe and (eline - 1) >= ps:
                    page_idxs.append(idx)

            # L1: Build a focused layout window per block using the tri-header
            layout_tokens: list[dict] = []
            layout_window: dict | None = None
            try:
                # Find bureau header line within the block
                hdr_idx, _order = _find_bureau_header_idx(
                    [str(x or "") for x in (out_blk.get("lines") or [])]
                )
            except Exception:
                hdr_idx = None

            # Map header line to a specific page
            header_page_idx: int | None = None
            if hdr_idx is not None:
                header_global_line = sline + hdr_idx  # 1-based within document
                for pidx, (ps, pe) in enumerate(zip(starts, ends)):
                    if ps <= header_global_line <= pe:
                        header_page_idx = pidx
                        break

            # Choose candidate page tokens to search for headers (prefer the header's page)
            page_candidate_idxs = (
                [header_page_idx] if header_page_idx is not None else list(page_idxs)
            )
            page_candidate_idxs = [i for i in page_candidate_idxs if i is not None]

            def _norm_hdr_token(txt: str) -> str:
                t = re.sub(r"[^A-Za-z]", "", (txt or "").lower())
                return t

            def _mid_x(t: dict) -> float:
                try:
                    x0 = float(t.get("x0", 0.0))
                    x1 = float(t.get("x1", x0))
                    return (x0 + x1) / 2.0
                except Exception:
                    return 0.0

            def _mid_y(t: dict) -> float:
                try:
                    y0 = float(t.get("y0", 0.0))
                    y1 = float(t.get("y1", y0))
                    return (y0 + y1) / 2.0
                except Exception:
                    return 0.0

            # Scan candidate pages to construct a window around the TU/EX/EQ header
            for pidx in page_candidate_idxs:
                if pidx is None or pidx >= len(layout_pages):
                    continue
                page = layout_pages[pidx]
                toks = list(page.get("tokens") or [])
                # Locate header tokens
                heads = {"transunion": [], "experian": [], "equifax": []}
                for t in toks:
                    name = _norm_hdr_token(str(t.get("text", "")))
                    if name == "transunion":
                        heads["transunion"].append(t)
                    elif name == "experian":
                        heads["experian"].append(t)
                    elif name == "equifax":
                        heads["equifax"].append(t)
                if not all(heads[k] for k in ("transunion", "experian", "equifax")):
                    continue
                # Pick first occurrence for each header (top-most)
                pick = {
                    k: sorted(v, key=lambda z: float(z.get("y0", 0.0)))[0]
                    for k, v in heads.items()
                    if v
                }
                x_min = min(float(pick[k].get("x0", 0.0)) for k in pick)
                x_max = max(float(pick[k].get("x1", 0.0)) for k in pick)
                y_top = min(_mid_y(pick[k]) for k in pick)
                # Find next section start
                y_candidates: list[float] = []
                for t in toks:
                    norm = _norm_hdr_token(str(t.get("text", "")))
                    if norm.startswith("twoyearpaymenthistory") or norm.startswith(
                        "dayslate7yearhistory"
                    ):
                        y_candidates.append(_mid_y(t))
                y_bottom = min(
                    (y for y in y_candidates if y > y_top),
                    default=float(page.get("height") or 1e9),
                )
                # Small safety padding to catch thin headers near window edges
                try:
                    page_h = float(page.get("height") or 1e9)
                except Exception:
                    page_h = 1e9
                y_top = max(0.0, y_top - 10.0)
                y_bottom = min(page_h, y_bottom + 10.0)
                # Collect tokens fully within the window by midpoint
                sel: list[dict] = []
                for t in toks:
                    mx = _mid_x(t)
                    my = _mid_y(t)
                    if x_min <= mx <= x_max and y_top <= my <= y_bottom:
                        sel.append(t)
                if not sel and BLOCK_DEBUG:
                    try:
                        sample = [
                            {
                                "text": tt.get("text"),
                                "x0": tt.get("x0"),
                                "y0": tt.get("y0"),
                                "x1": tt.get("x1"),
                                "y1": tt.get("y1"),
                            }
                            for tt in (toks[:3] if isinstance(toks, list) else [])
                        ]
                        logger.warning(
                            "RAW_COORDS: empty layout_tokens after window filter page=%s window=%s sample=%s",
                            page.get("number"),
                            {
                                "x_min": x_min,
                                "x_max": x_max,
                                "y_top": y_top,
                                "y_bottom": y_bottom,
                            },
                            sample,
                        )
                    except Exception:
                        pass
                layout_tokens = sel
                # Determine y_bottom from next block start on the same page if possible
                try:
                    y_bottom_candidate = None
                    # sline/eline are available from meta for this block, computed earlier
                    next_global_line = None
                    if i < len(fbk_blocks):
                        nxt = fbk_blocks[i]  # next block (since i is 1-based)
                        next_global_line = sline + len(blk.get("lines") or [])
                    # Approximate: find earliest token on this page that matches next block heading
                    if next_global_line is not None:
                        toks_on_page = list(page.get("tokens") or [])
                        ys_next: list[float] = []
                        for tt in toks_on_page:
                            try:
                                my = (
                                    float(tt.get("y0", 0.0)) + float(tt.get("y1", 0.0))
                                ) / 2.0
                            except Exception:
                                continue
                            if isinstance(nxt, dict) and isinstance(
                                nxt.get("heading"), str
                            ):
                                txt = str(tt.get("text", "")).strip().lower()
                                if txt and nxt["heading"].lower() in txt and my > y_top:
                                    ys_next.append(my)
                        if ys_next:
                            y_bottom_candidate = min(ys_next)
                    # Override y_bottom if candidate found; keep padding
                    if y_bottom_candidate:
                        y_bottom2 = max(y_top, min(page_h, y_bottom_candidate))
                        # re-apply padding consistently
                        y_bottom = min(page_h, y_bottom2 + 10.0)
                except Exception:
                    pass
                layout_window = {
                    "page": int(page.get("number", pidx + 1)),
                    "y_top": y_top,
                    "y_bottom": y_bottom,
                    "x_min": x_min,
                    "x_max": x_max,
                }
                break

            # Attach layout tokens + window to block, mark mode
            if layout_tokens:
                out_blk["layout_tokens"] = layout_tokens
                out_blk["mode"] = "raw_coords"
                # Always embed the computed layout_window for UI/debug ingestion
                if layout_window:
                    out_blk.setdefault("meta", {}).setdefault("debug", {})[
                        "layout_window"
                    ] = layout_window

            # Build and write RAW coordinate package (guarded by RAW_TWO_STAGE)
            if not RAW_TWO_STAGE:
                layout_tokens2 = out_blk.get("layout_tokens") or []
                header_tokens = [
                    t
                    for t in layout_tokens2
                    if str(t.get("text", "")).strip().lower()
                    in ("transunion", "experian", "equifax")
                ]
                bureau_cols = (
                    detect_bureau_columns(header_tokens) if header_tokens else {}
                )
                # Always write RAW, even when bands are missing (empty bands)
                # Build a minimal package input with explicit window + tokens
                index_headline = index_headlines.get(i)
                blk_for_pkg = {
                    "block_id": i,
                    "block_filename": f"block_{i:02d}.json",
                    "index_headline": index_headline,
                    "layout_tokens": out_blk.get("layout_tokens", []),
                    "meta": {
                        "debug": {
                            "layout_window": (
                                (out_blk.get("meta") or {}).get("debug") or {}
                            ).get("layout_window")
                        }
                    },
                    # For convenience, also expose heading for packager
                    "heading": out_blk.get("heading"),
                }

                # Hard-guard: ensure window and tokens exist; reconstruct if needed
                def _mid(x0, x1):
                    try:
                        return (float(x0) + float(x1)) / 2.0
                    except Exception:
                        return 0.0

                win_obj = (
                    (blk_for_pkg.get("meta") or {})
                    .get("debug", {})
                    .get("layout_window")
                )
                if not win_obj:
                    # Reconstruct a minimal window from the first overlapping page
                    sel_page = None
                    try:
                        if page_idxs:
                            p0 = page_idxs[0]
                            if p0 is not None and p0 < len(layout_pages):
                                sel_page = layout_pages[p0]
                        if sel_page is None and layout_pages:
                            sel_page = layout_pages[0]
                    except Exception:
                        sel_page = None
                    if sel_page:
                        toks_all = list(sel_page.get("tokens") or [])
                        xs = [_mid(t.get("x0", 0), t.get("x1", 0)) for t in toks_all]
                        ys = [_mid(t.get("y0", 0), t.get("y1", 0)) for t in toks_all]
                        if xs and ys:
                            win_obj = {
                                "page": int(sel_page.get("number", 1)),
                                "x_min": float(min(xs)),
                                "x_max": float(max(xs)),
                                "y_top": float(min(ys)),
                                "y_bottom": float(max(ys)),
                            }
                            blk_for_pkg["meta"]["debug"]["layout_window"] = win_obj

                if not win_obj:
                    logger.error(
                        "RAW_COORDS: missing layout_window for block %s (hard-guard) -> abort packaging this block",
                        i,
                    )
                    continue

                if not blk_for_pkg.get("layout_tokens"):
                    # Filter tokens from selected page by window
                    try:
                        sel_num = int(win_obj.get("page", 1) or 1)
                        sel_page = next(
                            (
                                p
                                for p in (layout_pages or [])
                                if int(p.get("number", 0)) == sel_num
                            ),
                            None,
                        )
                        page_tokens_all = (
                            list(sel_page.get("tokens") or []) if sel_page else []
                        )
                        x_min = float(win_obj.get("x_min", 0.0) or 0.0)
                        x_max = float(win_obj.get("x_max", 0.0) or 0.0)
                        y_top = float(win_obj.get("y_top", 0.0) or 0.0)
                        y_bottom = float(win_obj.get("y_bottom", 0.0) or 0.0)
                        blk_for_pkg["layout_tokens"] = [
                            tt
                            for tt in page_tokens_all
                            if (
                                x_min <= _mid(tt.get("x0", 0), tt.get("x1", 0)) <= x_max
                            )
                            and (
                                y_top
                                <= _mid(tt.get("y0", 0), tt.get("y1", 0))
                                <= y_bottom
                            )
                        ]
                    except Exception:
                        blk_for_pkg["layout_tokens"] = []

                if not blk_for_pkg.get("layout_tokens"):
                    logger.error(
                        "RAW_COORDS: no tokens in window for block %s (bounds=%s)",
                        i,
                        win_obj,
                    )
                    continue

                raw_pkg = package_block_raw_coords(
                    blk_for_pkg, bureau_cols if bureau_cols else {}
                )
                raw_path = write_block_raw_coords(
                    session_id, i, raw_pkg, index_headline
                )
                out_blk.setdefault("artifacts", {})["raw_coords_path"] = raw_path
                # Persist stats on block artifacts for indexing
                try:
                    stats = raw_pkg.get("stats") or {}
                    out_blk.setdefault("artifacts", {})["raw_row_count"] = int(
                        stats.get("row_count", 0) or 0
                    )
                    out_blk.setdefault("artifacts", {})["raw_token_count"] = int(
                        stats.get("token_count", 0) or 0
                    )
                except Exception:
                    pass
                logger.info(
                    "RAW_COORDS: wrote %s%s",
                    raw_path,
                    " (no bands)" if not bureau_cols else "",
                )
                # Clean fields/presence/tokens only after RAW was written
                try:
                    out_blk.pop("fields", None)
                    meta_tmp = out_blk.get("meta") or {}
                    if isinstance(meta_tmp, dict):
                        meta_tmp.pop("bureau_presence", None)
                        out_blk["meta"] = meta_tmp
                    out_blk.pop("layout_tokens", None)
                except Exception:
                    pass
        except Exception:
            # Keep out_blk as generated by original enrich path when column-reader fails
            pass
        # Track RAW artifact for index (always include entry)
        try:
            artifacts = out_blk.get("artifacts") or {}
            raw_path_idx = artifacts.get("raw_coords_path")
            row_count_idx = int(artifacts.get("raw_row_count") or 0)
            token_count_idx = int(artifacts.get("raw_token_count") or 0)
            raw_index_entries.append(
                {
                    "block_id": i,
                    "heading": out_blk.get("heading"),
                    "index_headline": index_headlines.get(i),
                    "raw_coords_path": raw_path_idx,
                    "row_count": row_count_idx,
                    "token_count": token_count_idx,
                }
            )
        except Exception:
            pass
        # Remove any residual fields/presence scaffolding before writing the block JSON
        try:
            if "fields" in out_blk:
                del out_blk["fields"]
            meta_tmp = out_blk.get("meta") or {}
            if isinstance(meta_tmp, dict) and "bureau_presence" in meta_tmp:
                meta_tmp.pop("bureau_presence", None)
                out_blk["meta"] = meta_tmp
        except Exception:
            pass
        out_blocks.append(out_blk)
        jpath = out_dir / f"block_{i:02d}.json"
        # Write a lightweight block envelope for UI: heading, lines, artifacts.raw_coords_path, and debug layout_window
        try:
            artifacts = out_blk.get("artifacts") or {}
            dbg_win = ((out_blk.get("meta") or {}).get("debug") or {}).get(
                "layout_window"
            )
            light_blk = {
                "heading": out_blk.get("heading"),
                "lines": out_blk.get("lines") or [],
                "artifacts": {
                    "raw_coords_path": (
                        None if RAW_TWO_STAGE else artifacts.get("raw_coords_path")
                    )
                },
                "debug": {"layout_window": dbg_win},
            }
            # Stage A: capture per-block window entry (allow None when unavailable)
            if RAW_TWO_STAGE:
                # Build per-page spans using cached page line ranges and layout tokens
                spans: List[Dict[str, Any]] = []
                per_page_windows = []
                try:
                    # Determine overlapping pages for this block via text line ranges
                    spans_pages = []
                    for pidx, (ps, pe) in enumerate(zip(starts, ends)):
                        if ps == 0 and pe == 0:
                            continue
                        if sline <= pe and (eline - 1) >= ps:
                            spans_pages.append(pidx)
                    for pidx in spans_pages:
                        page_obj = (
                            layout_pages[pidx] if pidx < len(layout_pages) else None
                        )
                        if not page_obj:
                            continue
                        try:
                            page_number = int(page_obj.get("number", pidx + 1))
                        except Exception:
                            page_number = pidx + 1
                        # Page-strict guard: reset state on page change
                        try:
                            logger.debug(
                                "StageA: RESET_STATE sid=%s block_id=%s page=%s",
                                session_id,
                                i,
                                page_number,
                            )
                        except Exception:
                            pass
                        toks_all = list(page_obj.get("tokens") or [])
                        # Compute per-page line overlap in page-local numbering (assuming 1-based lines)
                        try:
                            page_start_global = starts[pidx]
                            page_end_global = ends[pidx]
                            # Block overlap in global lines
                            ov_start = max(sline, page_start_global)
                            ov_end = min(eline - 1, page_end_global)
                            has_overlap = ov_start <= ov_end
                        except Exception:
                            has_overlap = False
                        toks_used = []
                        used_fallback = False
                        line_min = None
                        line_max = None
                        if has_overlap:
                            try:
                                # Map to page-local lines (1-based)
                                pl_min = max(1, ov_start - page_start_global + 1)
                                pl_max = max(pl_min, ov_end - page_start_global + 1)
                                for tt in toks_all:
                                    ln = tt.get("line")
                                    try:
                                        ln_i = int(ln)
                                    except Exception:
                                        continue
                                    if pl_min <= ln_i <= pl_max:
                                        toks_used.append(tt)
                                if toks_used:
                                    line_min = pl_min
                                    line_max = pl_max
                            except Exception:
                                toks_used = []
                        # Fallback: if no line-based tokens were found, use all tokens on the page (best-effort)
                        if not toks_used and toks_all:
                            toks_used = toks_all
                            used_fallback = True
                        if toks_used:
                            try:
                                y_min = min(
                                    float(tt.get("y0", 0.0) or 0.0) for tt in toks_used
                                )
                            except Exception:
                                y_min = 0.0
                            try:
                                y_max = max(
                                    float(tt.get("y1", 0.0) or 0.0) for tt in toks_used
                                )
                            except Exception:
                                y_max = 0.0
                            try:
                                x_min = min(
                                    float(tt.get("x0", 0.0) or 0.0) for tt in toks_used
                                )
                            except Exception:
                                x_min = 0.0
                            try:
                                x_max = max(
                                    float(tt.get("x1", 0.0) or 0.0) for tt in toks_used
                                )
                            except Exception:
                                x_max = 0.0
                            # Hard boundary rule: snap to midline of first/last lines when available
                            try:
                                # collect midY per line for this page
                                mids_by_line: Dict[int, List[float]] = {}
                                for tt in toks_all:
                                    ln = tt.get("line")
                                    try:
                                        ln_i = int(ln)
                                    except Exception:
                                        ln_i = None
                                    if ln_i is None:
                                        continue
                                    my = (
                                        float(tt.get("y0", 0.0) or 0.0)
                                        + float(tt.get("y1", 0.0) or 0.0)
                                    ) / 2.0
                                    mids_by_line.setdefault(ln_i, []).append(my)
                                # If line_min/line_max known, snap to their midlines
                                if line_min is not None and line_min in mids_by_line:
                                    vals = mids_by_line[line_min]
                                    if vals:
                                        y_min = sum(vals) / len(vals)
                                else:
                                    # If this is the first page touched by the block, allow page top
                                    if spans_pages and pidx == min(spans_pages):
                                        y_min = 0.0
                                if line_max is not None and line_max in mids_by_line:
                                    vals = mids_by_line[line_max]
                                    if vals:
                                        y_max = sum(vals) / len(vals)
                                else:
                                    # If this is the last page touched by the block, allow page bottom
                                    try:
                                        ph_local = float(page_obj.get("height") or 0.0)
                                    except Exception:
                                        ph_local = 0.0
                                    if (
                                        spans_pages
                                        and pidx == max(spans_pages)
                                        and ph_local
                                    ):
                                        y_max = ph_local
                            except Exception:
                                pass
                            # Count all tokens on this page whose midY falls within [y_min, y_max]
                            try:

                                def _midy(tt: dict) -> float:
                                    return (
                                        float(tt.get("y0", 0.0) or 0.0)
                                        + float(tt.get("y1", 0.0) or 0.0)
                                    ) / 2.0

                                token_count_in_span = sum(
                                    1 for tt in toks_all if y_min <= _midy(tt) <= y_max
                                )
                            except Exception:
                                token_count_in_span = len(toks_used)

                            # small padding and clamp to page bounds for X
                            pad = 3.0
                            try:
                                pw = float(page_obj.get("width") or 0.0) or 0.0
                            except Exception:
                                pw = 0.0

                            x_min_w = max(0.0, x_min - pad)
                            x_max_w = min(pw or (x_max + pad), x_max + pad)

                            span_entry: Dict[str, Any] = {
                                "page": page_number,
                                "y_min": y_min,
                                "y_max": y_max,
                                "token_count": len(toks_used),
                                "token_count_in_span": int(token_count_in_span),
                                "token_count_assigned": int(
                                    0 if used_fallback else len(toks_used)
                                ),
                                "assigned": (not used_fallback),
                                "x_min": float(x_min_w),
                                "x_max": float(x_max_w),
                            }
                            if line_min is not None and line_max is not None:
                                span_entry["line_min"] = int(line_min)
                                span_entry["line_max"] = int(line_max)
                            spans.append(span_entry)
                            # per-page window with small padding
                            try:
                                pw = float(page_obj.get("width") or 0.0) or 0.0
                                ph = float(page_obj.get("height") or 0.0) or 0.0
                            except Exception:
                                pw = ph = 0.0
                            per_page_windows.append(
                                {
                                    "page": int(page_number),
                                    "x_min": float(max(0.0, x_min - pad)),
                                    "x_max": float(
                                        min(pw or (x_max + pad), x_max + pad)
                                    ),
                                    "y_top": float(max(0.0, y_min - pad)),
                                    "y_bottom": float(
                                        min(ph or (y_max + pad), y_max + pad)
                                    ),
                                }
                            )
                except Exception:
                    spans = []
                    per_page_windows = []

                # Derive pages list and continued flag from per-page windows/spans
                try:
                    pages_list = sorted(
                        {
                            int(e.get("page"))
                            for e in (per_page_windows or [])
                            if isinstance(e, dict) and e.get("page") is not None
                        }
                    )
                    if not pages_list and spans:
                        pages_list = sorted(
                            {
                                int(e.get("page"))
                                for e in spans
                                if isinstance(e, dict) and e.get("page") is not None
                            }
                        )
                    if (
                        not pages_list
                        and isinstance(dbg_win, dict)
                        and dbg_win.get("page")
                    ):
                        pages_list = [int(dbg_win.get("page"))]
                except Exception:
                    pages_list = []
                continued_flag = bool(len(pages_list) > 1)

                # Compute explicit head/tail anchors by scanning heading tokens
                head_anchor = None
                tail_anchor = None
                try:
                    import re as _re

                    def _norm_txt2(s: str) -> str:
                        return _re.sub(r"\W+", "", (s or "").lower())

                    def _find_anchor(h: str | None):
                        if not h:
                            return None
                        hn = _norm_txt2(h)
                        best = None
                        for pidx, pg in enumerate(layout_pages):
                            toks = list(pg.get("tokens") or [])
                            for t in toks:
                                z = _norm_txt2(str(t.get("text", "")))
                                if not z:
                                    continue
                                if hn in z:
                                    try:
                                        pnum = int(pg.get("number", pidx + 1))
                                    except Exception:
                                        pnum = pidx + 1
                                    try:
                                        y = (
                                            float(t.get("y0", 0.0))
                                            + float(t.get("y1", 0.0))
                                        ) / 2.0
                                    except Exception:
                                        y = 0.0
                                    ln = t.get("line")
                                    try:
                                        ln_i = int(ln) if ln is not None else None
                                    except Exception:
                                        ln_i = None
                                    tup = (pidx, pnum, y, ln_i)
                                    if best is None or (tup[0], tup[2]) < (
                                        best[0],
                                        best[2],
                                    ):
                                        best = tup
                        if best is None:
                            return None
                        return {
                            "page": int(best[1]),
                            "y": float(best[2]),
                            "line": best[3],
                        }

                    head_anchor = _find_anchor(str(out_blk.get("heading") or ""))
                    nxt_heading = None
                    if i < len(fbk_blocks):
                        try:
                            nxt_heading = str(
                                (fbk_blocks[i] or {}).get("heading") or ""
                            )
                        except Exception:
                            nxt_heading = None
                    nxt = _find_anchor(nxt_heading)
                    if nxt is not None:
                        tail_anchor = nxt
                    else:
                        # fall back to last page bottom
                        if layout_pages:
                            try:
                                lp = layout_pages[-1]
                                pnum = int(lp.get("number", len(layout_pages)))
                            except Exception:
                                pnum = len(layout_pages)
                            try:
                                yb = float(lp.get("height") or 0.0)
                            except Exception:
                                yb = 0.0
                            tail_anchor = {
                                "page": int(pnum),
                                "y": float(yb),
                                "line": None,
                            }
                except Exception:
                    head_anchor = head_anchor or None
                    tail_anchor = tail_anchor or None

                window_index_entries.append(
                    {
                        "block_id": i,
                        "heading": out_blk.get("heading"),
                        "index_headline": index_headlines.get(i),
                        "window": dbg_win if dbg_win else None,
                        "spans": spans,
                        "pages": pages_list,
                        "continued": continued_flag,
                        "head_anchor": head_anchor,
                        "tail_anchor": tail_anchor,
                    }
                )
        except Exception:
            light_blk = {
                "heading": out_blk.get("heading"),
                "lines": out_blk.get("lines") or [],
                "artifacts": {"raw_coords_path": None},
                "debug": {"layout_window": None},
            }
            if RAW_TWO_STAGE:
                window_index_entries.append(
                    {
                        "block_id": i,
                        "heading": out_blk.get("heading"),
                        "index_headline": index_headlines.get(i),
                        "window": None,
                        "spans": [],
                        "pages": [],
                        "continued": False,
                    }
                )
        if RAW_TWO_STAGE:
            if not per_page_windows:
                per_page_windows = _fallback_windows_from_layout_window(out_blk)
            windows_by_block[str(i)] = per_page_windows
        with jpath.open("w", encoding="utf-8") as f:
            json.dump(light_blk, f, ensure_ascii=False, indent=2)
        idx_info.append({"i": i, "heading": out_blk["heading"], "file": str(jpath)})
        # Accumulate canonical variant counts
        try:
            meta = out_blk.get("meta", {})
            canon = meta.get("issuer_canonical")
            variant = meta.get("issuer_variant")
            if canon and isinstance(variant, str):
                canon_variants.setdefault(canon, set()).add(variant)
        except Exception:
            pass

    with (out_dir / "_index.json").open("w", encoding="utf-8") as f:
        json.dump(idx_info, f, ensure_ascii=False, indent=2)

    # Stage A: write block_windows.json
    if RAW_TWO_STAGE:
        try:
            # Ensure window.page exists for each entry when available
            blocks_fixed: List[Dict[str, Any]] = []
            for row in window_index_entries:
                win = row.get("window") if isinstance(row, dict) else None
                if isinstance(win, dict):
                    if "page" not in win or win.get("page") in (None, 0, ""):
                        # Best-effort default to 1 when unavailable
                        try:
                            win["page"] = int(win.get("page") or 1)
                        except Exception:
                            win["page"] = 1
                blocks_fixed.append(row)
            # QA: per-page overlap/gap checks between consecutive blocks
            try:
                EPS_Y = float(os.getenv("OVERLAP_EPS_Y", "0.5"))
            except Exception:
                EPS_Y = 0.5
            qa_map: Dict[int, Dict[str, Any]] = {}
            page_blocks: Dict[int, List[Tuple[int, float, float, float]]] = {}
            for row in blocks_fixed:
                try:
                    bid = int(row.get("block_id"))
                except Exception:
                    continue
                qa_map[bid] = {
                    "overlap_with_next": False,
                    "overlap_pages": set(),
                    "overlap_delta_max": 0.0,
                    "gap_to_next_min": None,
                }
                head_anchor = row.get("head_anchor") if isinstance(row, dict) else None
                ha_page = ha_y = None
                if isinstance(head_anchor, dict):
                    try:
                        ha_page = int(head_anchor.get("page"))
                        ha_y = float(head_anchor.get("y"))
                    except Exception:
                        ha_page = ha_y = None
                processed_pages: set[int] = set()
                for w in windows_by_block.get(str(bid), []) or []:
                    try:
                        p = int(w.get("page"))
                        y_top = float(w.get("y_top"))
                        y_bottom = float(w.get("y_bottom"))
                    except Exception:
                        continue
                    sort_y = ha_y if ha_page == p else y_top
                    page_blocks.setdefault(p, []).append((bid, sort_y, y_top, y_bottom))
                    processed_pages.add(p)
                for sp in row.get("spans") or []:
                    try:
                        p = int(sp.get("page"))
                        if p in processed_pages:
                            continue
                        y_top = float(sp.get("y_min"))
                        y_bottom = float(sp.get("y_max"))
                    except Exception:
                        continue
                    sort_y = ha_y if ha_page == p else y_top
                    page_blocks.setdefault(p, []).append((bid, sort_y, y_top, y_bottom))

            for p, blks in page_blocks.items():
                blks_sorted = sorted(blks, key=lambda t: t[1])
                for idx in range(len(blks_sorted) - 1):
                    bid_a, _, _, yb_a = blks_sorted[idx]
                    bid_b, _, yt_b, _ = blks_sorted[idx + 1]
                    delta = yt_b - yb_a
                    qa = qa_map.get(bid_a)
                    if qa is None:
                        continue
                    if delta < -EPS_Y:
                        ov = -(delta)
                        qa["overlap_with_next"] = True
                        qa["overlap_pages"].add(p)
                        if ov > qa["overlap_delta_max"]:
                            qa["overlap_delta_max"] = ov
                        try:
                            logger.warning(
                                "StageA: overlap sid=%s page=%d blockA=%d blockB=%d delta=%.2f",
                                session_id,
                                p,
                                bid_a,
                                bid_b,
                                ov,
                            )
                        except Exception:
                            pass
                    elif delta > EPS_Y:
                        gap = delta
                        cur = qa.get("gap_to_next_min")
                        if cur is None or gap < cur:
                            qa["gap_to_next_min"] = gap

            qa_map_final: Dict[int, Dict[str, Any]] = {}
            for bid, qa in qa_map.items():
                out = {"overlap_with_next": qa.get("overlap_with_next", False)}
                pages = qa.get("overlap_pages")
                if pages:
                    out["overlap_pages"] = sorted(pages)
                    out["overlap_delta_max"] = qa.get("overlap_delta_max", 0.0)
                gap_min = qa.get("gap_to_next_min")
                if gap_min is not None:
                    out["gap_to_next_min"] = gap_min
                qa_map_final[bid] = out
            for row in blocks_fixed:
                try:
                    bid = int(row.get("block_id"))
                except Exception:
                    continue
                if bid in qa_map_final:
                    row["qa"] = qa_map_final[bid]
            # Compute spans checksum for stability tracking
            try:

                def _as_int(v):
                    try:
                        return int(v)
                    except Exception:
                        return 0

                def _as_float(v):
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                hasher = hashlib.sha1()
                for row in sorted(
                    blocks_fixed, key=lambda r: _as_int(r.get("block_id", 0))
                ):
                    bid = _as_int(row.get("block_id", 0))
                    spans = (
                        list(row.get("spans") or []) if isinstance(row, dict) else []
                    )
                    spans_sorted = sorted(
                        spans,
                        key=lambda s: (
                            _as_int((s or {}).get("page", 0)),
                            _as_float((s or {}).get("y_min", 0.0)),
                            _as_float((s or {}).get("y_max", 0.0)),
                        ),
                    )
                    for sp in spans_sorted:
                        pg = _as_int(sp.get("page", 0))
                        xm = _as_float(sp.get("x_min", 0.0))
                        xM = _as_float(sp.get("x_max", 0.0))
                        y0 = _as_float(sp.get("y_min", 0.0))
                        y1 = _as_float(sp.get("y_max", 0.0))
                        ln0 = sp.get("line_min", None)
                        ln1 = sp.get("line_max", None)
                        ln0s = "" if ln0 is None else str(_as_int(ln0))
                        ln1s = "" if ln1 is None else str(_as_int(ln1))
                        rec = f"{bid}|{pg}|{xm:.3f}|{xM:.3f}|{y0:.3f}|{y1:.3f}|{ln0s}|{ln1s}\n"
                        hasher.update(rec.encode("utf-8"))
                spans_checksum = hasher.hexdigest()
            except Exception:
                spans_checksum = ""

            windows_obj = {
                "session_id": session_id,
                "schema_version": "2.0",
                "spans_checksum": spans_checksum,
                "blocks": blocks_fixed,
                "windows_by_block": windows_by_block,
            }
            (out_dir / "block_windows.json").write_text(
                json.dumps(windows_obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Now build accounts table with windows available
            try:
                snap_path = out_dir / "layout_snapshot.json"
                if snap_path.exists():
                    snap = json.loads(snap_path.read_text(encoding="utf-8"))
                else:
                    snap = {"session_id": session_id, "pages": []}
                stage_a_meta = _build_accounts_table(
                    session_id, accounts_dir, snap, layout_pages=layout_pages, block_windows=windows_obj
                )
            except Exception:
                logger.exception("BLOCK: failed to build accounts_table with windows sid=%s", session_id)
            # Write per-block debug TSVs with tokens filtered by spans (page-aware)
            try:
                debug_dir = accounts_dir
                debug_dir.mkdir(parents=True, exist_ok=True)
                # Build page number -> tokens map from layout_pages
                page_tokens_map: Dict[int, List[dict]] = {}
                for pg in layout_pages or []:
                    try:
                        pnum = int(pg.get("number", 0) or 0)
                    except Exception:
                        pnum = 0
                    page_tokens_map[pnum] = list(pg.get("tokens") or [])

                for row in blocks_fixed:
                    bid = row.get("block_id")
                    try:
                        bid_i = int(bid or 0)
                    except Exception:
                        bid_i = 0
                    if not bid_i:
                        continue
                    spans = (
                        list(row.get("spans") or []) if isinstance(row, dict) else []
                    )
                    if not spans:
                        # still create an empty file for consistency
                        (debug_dir / f"_debug_block_{bid_i}.tsv").write_text(
                            "page\tline\ty0\ty1\tx0\tx1\ttext\n", encoding="utf-8"
                        )
                        continue
                    seen = set()
                    lines_out: List[str] = ["page\tline\ty0\ty1\tx0\tx1\ttext\n"]
                    for sp in spans:
                        try:
                            sp_page = int(sp.get("page", 0) or 0)
                        except Exception:
                            sp_page = 0
                        toks_page = page_tokens_map.get(sp_page) or []
                        try:
                            y0 = float(sp.get("y_min", 0.0) or 0.0)
                            y1 = float(sp.get("y_max", 0.0) or 0.0)
                        except Exception:
                            y0 = 0.0
                            y1 = 0.0
                        try:
                            xs0 = float(sp.get("x_min", 0.0) or 0.0)
                            xs1 = float(sp.get("x_max", 0.0) or 0.0)
                        except Exception:
                            xs0 = xs1 = 0.0
                        for t in toks_page:
                            try:
                                ty0 = float(t.get("y0", 0.0) or 0.0)
                                ty1 = float(t.get("y1", 0.0) or 0.0)
                                my = (ty0 + ty1) / 2.0
                            except Exception:
                                continue
                            if not (y0 <= my <= y1):
                                continue
                            try:
                                tx0 = float(t.get("x0", 0.0) or 0.0)
                                tx1 = float(t.get("x1", 0.0) or 0.0)
                                mx = (tx0 + tx1) / 2.0
                            except Exception:
                                mx = tx0 = tx1 = 0.0
                            # If x bounds exist for the span, apply them; otherwise accept all
                            if xs0 or xs1:
                                if not (xs0 <= mx <= xs1):
                                    continue
                            ln = t.get("line")
                            try:
                                ln_s = str(int(ln)) if ln is not None else ""
                            except Exception:
                                ln_s = ""
                            key = (
                                sp_page,
                                ln_s,
                                f"{ty0:.3f}",
                                f"{ty1:.3f}",
                                f"{tx0:.3f}",
                                f"{tx1:.3f}",
                                str(t.get("text", "")),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            text = (
                                str(t.get("text", ""))
                                .replace("\t", " ")
                                .replace("\n", " ")
                            )
                            lines_out.append(
                                f"{sp_page}\t{ln_s}\t{ty0:.3f}\t{ty1:.3f}\t{tx0:.3f}\t{tx1:.3f}\t{text}\n"
                            )
                    (debug_dir / f"_debug_block_{bid_i}.tsv").write_text(
                        "".join(lines_out), encoding="utf-8"
                    )
            except Exception:
                logger.debug(
                    "StageA: failed to write per-block debug TSVs (sid=%s)",
                    session_id,
                    exc_info=True,
                )
            try:
                pages_n = len(layout_pages) if isinstance(layout_pages, list) else 0
            except Exception:
                pages_n = 0
            try:
                blocks_n = len(window_index_entries)
            except Exception:
                blocks_n = 0
            logger.info(
                "BLOCK: StageA snapshot+windows written sid=%s pages=%d blocks=%d",
                session_id,
                pages_n,
                blocks_n,
            )
        except Exception:
            logger.exception(
                "BLOCK: failed to write block_windows.json for sid=%s", session_id
            )
    else:
        # Legacy: Write compact RAW index per session
        try:
            raw_dir = out_dir / "accounts_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_index = {"session_id": session_id, "blocks": raw_index_entries}
            (raw_dir / "_raw_index.json").write_text(
                json.dumps(raw_index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # Log canonicalization summary + segmentation summary
    try:
        summary = {
            k: len(v) for k, v in sorted(canon_variants.items(), key=lambda x: x[0])
        }
        logger.debug(
            "BLOCK: canonicalization_summary %s",
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
        )
    except Exception:
        pass

    try:
        accounts = sum(
            1
            for b in out_blocks
            if (b.get("meta") or {}).get("block_type") == "account"
        )
        summaries = sum(
            1
            for b in out_blocks
            if (b.get("meta") or {}).get("block_type") == "summary"
        )
        logger.info(
            "BLOCK: segmentation summary accounts=%d summaries=%d total=%d",
            accounts,
            summaries,
            len(out_blocks),
        )
    except Exception:
        pass

    logger.warning(
        "ANZ: export blocks sid=%s dir=%s files=%d",
        session_id,
        str(out_dir),
        len(out_blocks),
    )
    try:
        # Register export blocks dir in manifest for discoverability
        m = RunManifest.for_sid(session_id, allow_create=False)
        m.set_base_dir("traces_blocks_dir", out_dir)
        m.set_artifact("traces.blocks", "export_dir", out_dir)
    except Exception:
        pass

    if stage_a_meta and getattr(config, "PURGE_TRACE_AFTER_EXPORT", False):
        from .trace_cleanup import purge_trace_except_artifacts

        project_root = Path(__file__).resolve().parents[4]
        try:
            purge_trace_except_artifacts(
                sid=session_id,
                root=project_root,
                dry_run=False,
                delete_texts_sid=not getattr(config, "PURGE_TRACE_KEEP_TEXTS", False),
            )
        except Exception:
            logger.exception("BLOCK: purge trace failed sid=%s", session_id)

    return out_blocks, stage_a_meta


# ----------------------------------------------------------------------------
# H1 override: enrich_block with TOP-group alignment and continuation joining
# ----------------------------------------------------------------------------
def enrich_block(blk: dict) -> dict:
    """Add structured ``fields`` map parsed from ``blk['lines']``.

    Implements H1: Align TOP-group (N=12) values per bureau with continuation-line
    joining and strict stop conditions. Falls back to the simple column-split
    parser when a TOP label group (>=3 labels) isn't detected.
    """

    heading = blk.get("heading", "")
    if BLOCK_DEBUG:
        logger.warning("ENRICH: start heading=%r", heading)

    # Normalize anomalies in potential bureau header lines
    raw_lines = [str(line or "") for line in (blk.get("lines") or [])]
    lines = [(line or "").replace("Â®", "").strip() for line in raw_lines]

    # Local H1 helpers (scoped to this function)
    TOP_LABELS = [
        "Account #",
        "High Balance",
        "Last Verified",
        "Date of Last Activity",
        "Date Reported",
        "Date Opened",
        "Balance Owed",
        "Closed Date",
        "Account Rating",
        "Account Description",
        "Dispute Status",
        "Creditor Type",
        "Original Creditor",
        "Original Creditor 01",
        "Original Creditor 02",
        "Orig. Creditor",
    ]
    BOTTOM_LABELS = [
        "Account Status",
        "Payment Status",
        "Creditor Remarks",
        "Payment Amount",
        "Last Payment",
        "Term Length",
        "Past Due Amount",
        "Account Type",
        "Payment Frequency",
        "Credit Limit",
    ]

    def _norm(s: str) -> str:
        s = (s or "").replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s.strip())
        s = s.rstrip(":")
        s = re.sub(r"[^A-Za-z0-9 #]+", " ", s)  # keep letters/numbers/space/#
        return s.upper()

    def _is_bureau(s: str) -> str | None:
        t = _norm(s)
        if t == "TRANSUNION":
            return "transunion"
        if t == "EXPERIAN":
            return "experian"
        if t == "EQUIFAX":
            return "equifax"
        return None

    def _is_label(s: str) -> tuple[str | None, str | None]:
        t = _norm(s)
        for lab in TOP_LABELS:
            if t == _norm(lab):
                return "top", lab
        for lab in BOTTOM_LABELS:
            if t == _norm(lab):
                return "bottom", lab
        return None, None

    # 1) Find TOP label group (>=3 consecutive TOP labels)
    top_start_idx: int | None = None
    top_labels: list[str] = []
    i = 0
    while i < len(lines):
        k, lab = _is_label(lines[i])
        if k == "top":
            j = i
            acc: list[str] = []
            while j < len(lines):
                k2, lab2 = _is_label(lines[j])
                if k2 == "top" and lab2:
                    acc.append(lab2)
                    j += 1
                else:
                    break
            if len(acc) >= 3:
                top_start_idx = i
                top_labels = acc
                break
            i = j
            continue
        i += 1

    # Determine bureau order (prefer explicit single-token lines)
    order: list[str] = []
    bureau_idx: dict[str, int] = {}
    if top_start_idx is not None:
        scan_start = max(0, top_start_idx - 2)
        for idx in range(scan_start, len(lines)):
            b = _is_bureau(lines[idx])
            if b and b not in bureau_idx:
                bureau_idx[b] = idx
        order = [b for b, _ in sorted(bureau_idx.items(), key=lambda x: x[1])]
    if not order:
        try:
            order = detect_bureau_order(lines) or []
        except Exception:
            order = []

    # Initialise fields map with existing content preserved
    existing_fields = blk.get("fields") or {}
    field_keys = list(FIELD_LABELS.values())
    fields: dict[str, dict[str, str]] = {
        b: {k: existing_fields.get(b, {}).get(k, "") for k in field_keys}
        for b in ("transunion", "experian", "equifax")
    }

    # TOP label -> snake_case mapping
    top_key_map = {
        "Account #": "account_number_display",
        "High Balance": "high_balance",
        "Last Verified": "last_verified",
        "Date of Last Activity": "date_of_last_activity",
        "Date Reported": "date_reported",
        "Date Opened": "date_opened",
        "Balance Owed": "balance_owed",
        "Closed Date": "closed_date",
        "Account Rating": "account_rating",
        "Account Description": "account_description",
        "Dispute Status": "dispute_status",
        "Creditor Type": "creditor_type",
        "Original Creditor": "original_creditor",
        "Original Creditor 01": "original_creditor",
        "Original Creditor 02": "original_creditor",
        "Orig. Creditor": "original_creditor",
    }

    CONTINUATION_SINGLETONS = {":", "mortgage", "balance", "account", "finance"}

    def _should_join(prev_val: str, current_line: str, next_line: str | None) -> bool:
        s = (current_line or "").strip()
        if not s:
            return False
        low = s.lower()
        if low in CONTINUATION_SINGLETONS:
            return True
        if low == s and len(low) <= 16 and len(low.split()) <= 2:
            if re.match(r"^[0-9$]", low):
                return False
            return True
        if prev_val and re.search(r"[-/]\s*$", prev_val):
            return True
        if next_line is not None and next_line.strip() == ":":
            return True
        return False

    STOP_SET = {_norm("TWO-YEAR PAYMENT HISTORY"), _norm("DAYS LATE - 7 YEAR HISTORY")}

    def _is_stop_token(s: str) -> bool:
        if _is_bureau(s):
            return True
        kind, _ = _is_label(s)
        if kind is not None:
            return True
        t = _norm(s)
        if t in STOP_SET:
            return True
        return False

    h1_applied = False
    if top_start_idx is not None and order and bureau_idx:
        n_needed = len(top_labels)
        for b in order:
            if b not in bureau_idx:
                continue
            start_i = bureau_idx[b] + 1
            vals: list[str] = []
            idx2 = start_i
            while idx2 < len(lines) and len(vals) < n_needed:
                t = lines[idx2].strip()
                if _is_stop_token(t):
                    break
                if t == "":
                    vals.append("")
                    idx2 += 1
                    continue
                if not vals:
                    vals.append(t)
                else:
                    nxt = lines[idx2 + 1] if (idx2 + 1) < len(lines) else None
                    if _should_join(vals[-1], t, nxt):
                        vals[-1] = (vals[-1] + " " + t).strip() if vals[-1] else t
                    else:
                        vals.append(t)
                idx2 += 1
            while len(vals) < n_needed:
                vals.append("")
            for lab, val in zip(top_labels, vals[:n_needed]):
                key = top_key_map.get(lab)
                if not key:
                    continue
                v = clean_value(val)
                fields[b][key] = v
        h1_applied = any(
            any(
                not is_effectively_blank(fields[b].get(top_key_map[lab], ""))
                for lab in top_labels
                if lab in top_key_map
            )
            for b in order
        )

    # ---------------- H2: BOTTOM group alignment (N=10) per bureau ----------------
    # Find BOTTOM label group (>=3 consecutive labels)
    bottom_start_idx: int | None = None
    bottom_labels: list[str] = []
    i2 = 0
    while i2 < len(lines):
        k, lab = _is_label(lines[i2])
        if k == "bottom":
            j2 = i2
            acc2: list[str] = []
            while j2 < len(lines):
                k3, lab3 = _is_label(lines[j2])
                if k3 == "bottom" and lab3:
                    acc2.append(lab3)
                    j2 += 1
                else:
                    break
            if len(acc2) >= 3:
                bottom_start_idx = i2
                bottom_labels = acc2
                break
            i2 = j2
            continue
        i2 += 1

    # Determine bureau order for BOTTOM (reuse TOP order if available)
    order_bottom: list[str] = order[:] if order else []
    bureau_idx_bottom: dict[str, int] = {}
    if bottom_start_idx is not None:
        scan_from = max(0, bottom_start_idx - 2)
        for idxb in range(scan_from, len(lines)):
            b2 = _is_bureau(lines[idxb])
            if b2 and b2 not in bureau_idx_bottom:
                bureau_idx_bottom[b2] = idxb
        if not order_bottom:
            order_bottom = [
                b for b, _ in sorted(bureau_idx_bottom.items(), key=lambda x: x[1])
            ]
        if not order_bottom:
            try:
                order_bottom = detect_bureau_order(lines) or []
            except Exception:
                order_bottom = []

    # BOTTOM label -> snake_case mapping
    bottom_key_map = {
        "Account Status": "account_status",
        "Payment Status": "payment_status",
        "Creditor Remarks": "creditor_remarks",
        "Payment Amount": "payment_amount",
        "Last Payment": "last_payment",
        "Term Length": "term_length",
        "Past Due Amount": "past_due_amount",
        "Account Type": "account_type",
        "Payment Frequency": "payment_frequency",
        "Credit Limit": "credit_limit",
    }

    if bottom_start_idx is not None and (order_bottom or bureau_idx_bottom):
        n_needed_bottom = len(bottom_labels)
        for b in order_bottom or list(bureau_idx_bottom.keys()):
            if b not in bureau_idx_bottom:
                # No explicit bureau token near bottom; skip to avoid spillover
                continue
            start_b = bureau_idx_bottom[b] + 1
            vals_b: list[str] = []
            p = start_b
            while p < len(lines) and len(vals_b) < n_needed_bottom:
                tline = lines[p].strip()
                if _is_stop_token(tline):
                    break
                if tline == "":
                    vals_b.append("")
                    p += 1
                    continue
                if not vals_b:
                    vals_b.append(tline)
                else:
                    nxt2 = lines[p + 1] if (p + 1) < len(lines) else None
                    curr_label = (
                        bottom_labels[len(vals_b) - 1]
                        if (len(vals_b) - 1) < len(bottom_labels)
                        else None
                    )
                    # For BOTTOM section, only allow joining for Creditor Remarks;
                    # avoid accidental concatenation for status/type fields.
                    join = (
                        _should_join(vals_b[-1], tline, nxt2)
                        if curr_label in {"Creditor Remarks"}
                        else False
                    )
                    if join:
                        vals_b[-1] = (
                            (vals_b[-1] + " " + tline).strip() if vals_b[-1] else tline
                        )
                    else:
                        vals_b.append(tline)
                p += 1
            while len(vals_b) < n_needed_bottom:
                vals_b.append("")
            # Merge into fields[b]
            dst = fields.setdefault(b, {})
            for lab, val in zip(bottom_labels, vals_b[:n_needed_bottom]):
                keyb = bottom_key_map.get(lab)
                if not keyb:
                    continue
                v2 = clean_value(val)
                if keyb == "creditor_remarks" and dst.get(keyb) and not is_effectively_blank(v2):
                    dst[keyb] = f"{dst[keyb]} {v2}".strip()
                elif not dst.get(keyb):
                    dst[keyb] = v2

    # Fallback: simple column-split parsing when H1 didn't execute
    if not h1_applied:
        order2: list[str] = []
        try:
            order2 = detect_bureau_order(lines) or []
        except Exception:
            order2 = []
        if order2:
            in_section = False
            for line in lines:
                clean = line.strip()
                if not clean:
                    continue
                if not in_section:
                    norm = re.sub(r"[^a-z]+", " ", clean.lower())
                    if all(b in norm for b in order2):
                        in_section = True
                    continue
                norm_line = clean.lower()
                for label, key in FIELD_LABELS.items():
                    if norm_line.startswith(label):
                        rest = clean[len(label) :].strip()
                        if rest.startswith(":"):
                            rest = rest[1:].strip()
                        vals = _split_vals(rest, len(order2))
                        for idx3, bureau in enumerate(order2):
                            v = vals[idx3] if idx3 < len(vals) else ""
                            normalized = clean_value(v)
                            dst2 = fields.setdefault(bureau, {})
                            if (
                                key == "creditor_remarks"
                                and dst2.get(key)
                                and not is_effectively_blank(normalized)
                            ):
                                dst2[key] = f"{dst2[key]} {normalized}".strip()
                            elif not dst2.get(key):
                                dst2[key] = normalized
                        break

    # T4: If layout tokens are provided on the block, use column-reader (T2+T3)
    try:
        layout_tokens = blk.get("layout_tokens") or []
    except Exception:
        layout_tokens = []
    if layout_tokens:
        # Detect header bands (for stability + optional debug)
        try:
            bureau_cols = detect_bureau_columns(layout_tokens)
        except Exception:
            bureau_cols = {}
        if BLOCK_DEBUG:
            try:
                logger.info(
                    "ENRICH: layout header bands detected=%s", bool(bureau_cols)
                )
            except Exception:
                pass
        # Build full 3-bureau table from layout
        try:
            fields_from_layout = extract_bureau_table(
                {
                    "layout_tokens": layout_tokens,
                    "meta": {"debug": {"bureau_cols": bureau_cols}},
                }
            )
            if isinstance(fields_from_layout, dict) and all(
                b in fields_from_layout for b in ("transunion", "experian", "equifax")
            ):
                fields = fields_from_layout
            elif not bureau_cols:
                # Safe fallback: keep empty fields and log
                logger.warning("ENRICH: no bureau columns; filled empty fields.")
        except Exception:
            pass
    else:
        # No layout available — keep empty fields and log once for visibility
        if BLOCK_DEBUG:
            try:
                logger.warning("ENRICH: no bureau columns; filled empty fields.")
            except Exception:
                pass

    # H3: Ensure all 22 keys exist per bureau, clean money fields, and set meta
    ALL_KEYS = list(FIELD_LABELS.values())
    for b in ("transunion", "experian", "equifax"):
        dst = fields.setdefault(b, {})
        for k in ALL_KEYS:
            dst.setdefault(k, "")

    # Apply standard cleaning + presence + account tail extraction
    base_meta = dict(blk.get("meta") or {})
    cleaned_fields, cleaned_meta = _g4_apply(fields, base_meta, raw_lines)

    # T4: Explicitly set bureau_presence based on non-empty values (post-clean)
    try:
        bp: dict[str, bool] = {}
        for b in ("transunion", "experian", "equifax"):
            vals = cleaned_fields.get(b, {})
            bp[b] = any(str(v or "").strip() for v in vals.values())
        cleaned_meta.setdefault("bureau_presence", {}).update(bp)
    except Exception:
        pass

    # T6: Debug artifacts when enabled
    if BLOCK_DEBUG:
        try:
            cleaned_meta.setdefault("debug", {})
            # Bureau header line (1-based index in lines)
            try:
                hdr_idx, _order = _find_bureau_header_idx(
                    [str(x or "") for x in (blk.get("lines") or [])]
                )
            except Exception:
                hdr_idx = None
            cleaned_meta["debug"]["bureau_header_line"] = (
                (hdr_idx + 1) if hdr_idx is not None else None
            )
            # Bureau columns (x bands)
            if layout_tokens:
                try:
                    cols = detect_bureau_columns(layout_tokens)
                except Exception:
                    cols = {}
                cleaned_meta["debug"]["bureau_cols"] = cols
                # First 8 rows preview
                cleaned_meta["debug"]["rows"] = (
                    build_debug_rows(layout_tokens, cols, 8) if cols else []
                )
        except Exception:
            pass

    return {**blk, "fields": cleaned_fields, "meta": cleaned_meta}


def export_stage_a(session_id: str, *, accounts_out_dir: Path | None = None) -> dict:
    """Run Stage A export and return artifact paths."""
    try:
        pdf_path = require_pdf_for_sid(session_id)
        _, meta = export_account_blocks(session_id, pdf_path, accounts_out_dir=accounts_out_dir)
        accounts_dir = Path(meta["accounts_json"]).parent
        artifacts = {
            "full_tsv": Path(meta["full_tsv"]),
            "accounts_json": Path(meta["accounts_json"]),
            "general_info_json": Path(meta["general_info"]),
        }
        return {
            "sid": session_id,
            "accounts_table_dir": accounts_dir,
            "artifacts": artifacts,
            "ok": True,
        }
    except Exception as e:
        logger.exception(
            "export_stage_a_failed", extra={"sid": session_id, "error": str(e)}
        )
        return {"sid": session_id, "ok": False, "error": str(e)}


def run_stage_a(*, sid: str, accounts_out_dir: Path) -> dict:
    """
    Canonical entry point: write Stage-A artifacts strictly under ``accounts_out_dir``.

    All artifacts (TSV/JSON) are emitted under this directory. No legacy
    ``traces/blocks`` derivation is performed.
    """
    assert "runs" in str(accounts_out_dir.resolve()), "Stage-A out_dir must live under runs/<SID>"
    return export_stage_a(session_id=sid, accounts_out_dir=accounts_out_dir)
