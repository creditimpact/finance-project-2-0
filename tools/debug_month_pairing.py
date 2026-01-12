import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

HEBREW_MONTHS = {
    "ינו׳", "פבר׳", "מרץ", "אפר׳", "מאי", "יוני",
    "יולי", "אוג׳", "ספט׳", "אוק׳", "נוב׳", "דצמ׳",
}

STATUS_RE = re.compile(r"^(?:ok|co|[0-9]{2,3})$", re.IGNORECASE)

@dataclass
class Tok:
    page: int
    line: int
    y0: float
    y1: float
    x0: float
    x1: float
    text: str

    @property
    def y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def x(self) -> float:
        return (self.x0 + self.x1) / 2.0


def load_per_account_tsv(tsv_path: str) -> List[Tok]:
    toks: List[Tok] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                toks.append(
                    Tok(
                        page=int(float(row["page"])),
                        line=int(float(row["line"])),
                        y0=float(row["y0"]),
                        y1=float(row["y1"]),
                        x0=float(row["x0"]),
                        x1=float(row["x1"]),
                        text=str(row["text"]),
                    )
                )
            except Exception:
                # Skip malformed rows
                continue
    # Sort by page, then line, then x
    toks.sort(key=lambda t: (t.page, t.line, t.x0))
    return toks


def find_region_indices(tokens: List[Tok]) -> Tuple[int, int]:
    """Return (start_idx, end_idx) for 2Y region within tokens.
    Start at first token containing 'Two-Year' or 'Payment History';
    End at first token containing 'Days Late' and 'History' or '7' and 'Year'.
    If end not found, use len(tokens).
    """
    start = None
    for i, t in enumerate(tokens):
        txt = t.text.lower().strip()
        if ("two" in txt and "year" in txt) or ("payment" in txt and "history" in txt):
            start = i
            break
    if start is None:
        # Fallback: first 'history'
        for i, t in enumerate(tokens):
            txt = t.text.lower().strip()
            if "history" in txt:
                start = i
                break
    if start is None:
        return (0, 0)
    end = len(tokens)
    for i in range(start + 1, len(tokens)):
        txt = tokens[i].text.lower().strip()
        if ("days" in txt and "history" in txt) or ("7" in txt and "year" in txt and "history" in txt):
            end = i
            break
    return (start, end)


def find_bureau_indices(two_year_tokens: List[Tok]) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    for i, t in enumerate(two_year_tokens):
        s = t.text.lower().strip()
        if "transunion" in s and "transunion" not in indices:
            indices["transunion"] = i
        if "experian" in s and "experian" not in indices:
            indices["experian"] = i
        if "equifax" in s and "equifax" not in indices:
            indices["equifax"] = i
        if len(indices) == 3:
            break
    return indices


def slice_by_bureau(two_year_tokens: List[Tok], bureau_indices: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    order = sorted(bureau_indices.items(), key=lambda kv: kv[1])
    slices: Dict[str, Tuple[int, int]] = {}
    for i, (b, idx) in enumerate(order):
        next_idx = order[i + 1][1] if i + 1 < len(order) else len(two_year_tokens)
        slices[b] = (idx, next_idx)
    return slices


def cluster_rows(tokens: List[Tok]) -> List[List[Tok]]:
    """Group tokens by (page, line)."""
    groups: Dict[Tuple[int, int], List[Tok]] = defaultdict(list)
    for t in tokens:
        groups[(t.page, t.line)].append(t)
    rows = [sorted(v, key=lambda t: t.x) for _, v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    return rows


def is_month(t: Tok) -> bool:
    return t.text.strip() in HEBREW_MONTHS


def is_status(t: Tok) -> bool:
    return bool(STATUS_RE.match(t.text.strip()))


def pair_rows(month_rows: List[List[Tok]], status_rows: List[List[Tok]], dy_limit: float = 40.0) -> Dict[int, Optional[int]]:
    """For each month row index, pick closest status row index above within dy_limit.
    Ensures one-to-one: a status row can be used at most once.
    Returns mapping month_row_idx -> status_row_idx (or None).
    """
    assigned: Dict[int, Optional[int]] = {}
    used_status = set()
    # Precompute row y centers
    m_y = [sum(t.y for t in row) / max(len(row), 1) for row in month_rows]
    s_y = [sum(t.y for t in row) / max(len(row), 1) for row in status_rows]
    for mi in range(len(month_rows)):
        best_j = None
        best_dy = None
        for sj in range(len(status_rows)):
            if sj in used_status:
                continue
            dy = m_y[mi] - s_y[sj]
            if dy <= 0 or dy > dy_limit:
                continue
            if best_dy is None or dy < best_dy:
                best_dy = dy
                best_j = sj
        if best_j is not None:
            assigned[mi] = best_j
            used_status.add(best_j)
        else:
            assigned[mi] = None
    return assigned


def pair_cells(month_row: List[Tok], status_row: Optional[List[Tok]]) -> List[Tuple[str, str]]:
    """Return list of (month_text, status_text or "--") pairs by nearest x ordering.
    Sort both rows by x ascending and pair index-wise.
    """
    months = sorted([t for t in month_row], key=lambda t: t.x)
    statuses = sorted([t for t in (status_row or [])], key=lambda t: t.x)
    out: List[Tuple[str, str]] = []
    for i, mt in enumerate(months):
        st_text = "--"
        if i < len(statuses):
            st_text = statuses[i].text.strip()
        out.append((mt.text.strip(), st_text))
    return out


def analyze_account(sid: str, account_idx: int) -> Dict[str, Dict[str, object]]:
    base = os.path.join("runs", sid)
    per_acc_dir = os.path.join(base, "traces", "accounts_table", "per_account_tsv")
    tsv_path = os.path.join(per_acc_dir, f"_debug_account_{account_idx}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    tokens = load_per_account_tsv(tsv_path)
    # Use same bounded 2Y slice
    start_idx, end_idx = find_region_indices(tokens)
    two_year_tokens = tokens[start_idx:end_idx]
    b_ix = find_bureau_indices(two_year_tokens)
    slices = slice_by_bureau(two_year_tokens, b_ix) if b_ix else {}

    results: Dict[str, Dict[str, object]] = {}

    for bureau in ("transunion", "experian", "equifax"):
        b_slice = slices.get(bureau)
        if not b_slice:
            results[bureau] = {
                "months_detected": 0,
                "status_tokens_detected": 0,
                "month_rows_detected": 0,
                "status_rows_detected": 0,
                "paired_months_with_status": 0,
                "pairs_sample_first5": [],
                "pairs_sample_last5": [],
                "eq_range": None,
            }
            continue
        s, e = b_slice
        b_tokens = two_year_tokens[s:e]
        # Cluster months and statuses by rows (page,line)
        month_tokens = [t for t in b_tokens if is_month(t)]
        status_tokens = [t for t in b_tokens if is_status(t)]
        month_rows = [row for row in cluster_rows(month_tokens) if row]
        status_rows = [row for row in cluster_rows(status_tokens) if row]
        mapping = pair_rows(month_rows, status_rows, dy_limit=40.0)
        # Build paired cells across all month rows
        all_pairs: List[Tuple[str, str]] = []
        paired_count = 0
        for mi, mrow in enumerate(month_rows):
            sj = mapping.get(mi)
            srow = status_rows[sj] if sj is not None and sj < len(status_rows) else None
            pairs = pair_cells(mrow, srow)
            for mtxt, stxt in pairs:
                all_pairs.append((mtxt, stxt))
                if stxt != "--":
                    paired_count += 1
        eq_range = None
        if bureau == "equifax":
            if b_tokens:
                pages = sorted({t.page for t in b_tokens})
                eq_range = {
                    "pages": [min(pages), max(pages)],
                    "token_idx_range": [start_idx + s, start_idx + e],
                }
        results[bureau] = {
            "months_detected": len(month_tokens),
            "status_tokens_detected": len(status_tokens),
            "month_rows_detected": len(month_rows),
            "status_rows_detected": len(status_rows),
            "paired_months_with_status": paired_count,
            "pairs_sample_first5": all_pairs[:5],
            "pairs_sample_last5": all_pairs[-5:],
            "eq_range": eq_range,
        }

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/debug_month_pairing.py <SID> <account_idx>")
        sys.exit(2)
    sid = sys.argv[1]
    try:
        acc_idx = int(sys.argv[2])
    except ValueError:
        print("account_idx must be an integer")
        sys.exit(2)
    res = analyze_account(sid, acc_idx)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
