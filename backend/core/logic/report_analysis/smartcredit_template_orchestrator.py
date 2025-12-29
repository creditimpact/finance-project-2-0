from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.core.logic.report_analysis.column_reader import detect_bureau_columns
from backend.core.logic.report_analysis.history_extractor import (
    extract_two_year_payment_history,
    extract_seven_year_delinquency,
)
from backend.core.logic.report_analysis.account_packager_coords import _slug
from backend.core.logic.report_analysis.personal_info_extractor import (
    extract_personal_info_rows,
)
from backend.core.logic.report_analysis.canonical_labels import (
    extract_canonical_labels,
    find_section_cut_index,
    LABEL_SCHEMA,
    detect_value_type,
)

logger = logging.getLogger(__name__)

# Tuning knobs (env-overridable)
GRID_DY = float(os.getenv("GRID_DY", "3.5"))  # row clustering dy
GRID_LABEL_MARGIN = float(os.getenv("GRID_LABEL_MARGIN", "8.0"))
MERGE_DY = float(os.getenv("MERGE_DY", "3.0"))  # vertical merge tolerance
LABEL_ATTACH_MAX_DY = float(os.getenv("LABEL_ATTACH_MAX_DY", "14.0"))
BUREAU_HDRS = {"transunion", "experian", "equifax"}
# Emit per-cell token provenance
EMIT_PROVENANCE = os.getenv("EMIT_PROVENANCE", "0") == "1"
# Known left-side labels that may appear without a trailing colon
NON_COLON_LABELS = {
    "account#",
    "account",  # normalized alias for Account #
    "highbalance",
    "lastverified",
    "dateoflastactivity",
    "datereported",
    "dateopened",
    "balanceowed",
    "closeddate",
    "accountrating",
    "accountdescription",
    "disputestatus",
    "creditortype",
    "accountstatus",
    "paymentstatus",
    "creditorremarks",
    "paymentamount",
    "lastpayment",
    "termlength",
    "pastdueamount",
}


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _join_text(tokens: List[dict]) -> str:
    # Join tokens left-to-right; filter symbol-only (keep '-') and prevent glued words
    import re

    _SYM_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)

    def _is_wordy(ch: str) -> bool:
        try:
            return ch.isalnum()
        except Exception:
            return False

    def _need_space(prev_txt: str, cur_txt: str, gap: float) -> bool:
        if gap >= 1.0:
            return True
        if prev_txt and cur_txt and _is_wordy(prev_txt[-1]) and _is_wordy(cur_txt[0]):
            return True
        return False

    def midx(t: dict) -> float:
        try:
            return (float(t.get("x0", 0.0)) + float(t.get("x1", 0.0))) / 2.0
        except Exception:
            return 0.0

    toks = sorted(tokens or [], key=midx)
    out: List[str] = []
    prev_tok: dict | None = None
    for t in toks:
        txt = str(t.get("text", "")).strip()
        if not txt:
            continue
        if _SYM_ONLY.match(txt) and txt != "-":
            continue
        if prev_tok is None:
            out.append(txt)
            prev_tok = t
            continue
        try:
            cur_x0 = float(t.get("x0", 0.0))
            prev_x1 = float(prev_tok.get("x1", 0.0))
            gap = cur_x0 - prev_x1
        except Exception:
            gap = 0.0
        if _need_space(out[-1], txt, gap):
            out.append(" ")
        out.append(txt)
        prev_tok = t
    return "".join(out).strip()


def _cluster_rows(tokens: List[dict], dy: float) -> List[List[dict]]:
    # Reading-order cluster: sort by (midY, midX), then group by Y distance
    toks = sorted(tokens or [], key=lambda t: (_mid(t.get("y0", 0.0), t.get("y1", 0.0)), _mid(t.get("x0", 0.0), t.get("x1", 0.0))))
    rows: List[List[dict]] = []
    cur: List[dict] = []
    cur_y: float | None = None
    for t in toks:
        y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
        if cur_y is None or abs(y - cur_y) <= dy:
            cur.append(t)
            try:
                cur_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in cur) / len(cur)
            except Exception:
                cur_y = y
        else:
            rows.append(cur)
            cur = [t]
            cur_y = y
    if cur:
        rows.append(cur)
    rows.sort(key=lambda r: (sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in r) / len(r)) if r else 0.0)
    return rows


def _group_rows_by_page_line(tokens: List[dict], dy: float) -> List[List[dict]]:
    # Prefer grouping by (page, line); fallback to per-page Y clustering for leftovers
    toks = list(tokens or [])
    by_pl: Dict[tuple[int, int], List[dict]] = {}
    leftovers_by_page: Dict[int, List[dict]] = {}
    for t in toks:
        try:
            p = int(t.get("page", 0) or 0)
        except Exception:
            p = 0
        ln = t.get("line", None)
        try:
            ln = int(ln) if ln is not None else None
        except Exception:
            ln = None
        if ln is not None:
            by_pl.setdefault((p, ln), []).append(t)
        else:
            leftovers_by_page.setdefault(p, []).append(t)

    # Build initial rows with (page,line)
    rows_meta: List[tuple[int, float, float, List[dict]]] = []  # (page, y_min, y_max, toks)
    for (p, ln), group in by_pl.items():
        ys = [(_mid(g.get("y0", 0.0), g.get("y1", 0.0))) for g in group]
        y_min = min(float(g.get("y0", 0.0) or 0.0) for g in group) if group else 0.0
        y_max = max(float(g.get("y1", 0.0) or 0.0) for g in group) if group else 0.0
        rows_meta.append((p, y_min, y_max, group))

    # Helper to compute vertical overlap ratio of a token vs row span
    def _overlap_ratio(tok: dict, y0: float, y1: float) -> float:
        try:
            ty0 = float(tok.get("y0", 0.0) or 0.0)
            ty1 = float(tok.get("y1", 0.0) or 0.0)
        except Exception:
            ty0 = ty1 = 0.0
        top = max(ty0, y0)
        bot = min(ty1, y1)
        ov = max(0.0, bot - top)
        h = max(1e-6, ty1 - ty0)
        return ov / h

    # Assign leftovers to nearest row on same page if vertical overlap >= 0.5
    for p, lst in list(leftovers_by_page.items()):
        remain: List[dict] = []
        for t in lst:
            # find best row on same page
            best_idx = -1
            best_ov = 0.0
            for idx, (rp, y0, y1, group) in enumerate(rows_meta):
                if rp != p:
                    continue
                ov = _overlap_ratio(t, y0, y1)
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            if best_idx >= 0 and best_ov >= 0.5:
                rows_meta[best_idx][3].append(t)  # type: ignore[index]
                # update row span
                try:
                    rows_meta[best_idx] = (
                        rows_meta[best_idx][0],
                        min(rows_meta[best_idx][1], float(t.get("y0", 0.0) or 0.0)),
                        max(rows_meta[best_idx][2], float(t.get("y1", 0.0) or 0.0)),
                        rows_meta[best_idx][3],
                    )
                except Exception:
                    pass
            else:
                remain.append(t)
        leftovers_by_page[p] = remain

    # Fallback cluster remaining tokens per page by Y
    for p, lst in leftovers_by_page.items():
        page_rows = _cluster_rows(lst, dy) if lst else []
        for r in page_rows:
            y_min = min(float(t.get("y0", 0.0) or 0.0) for t in r) if r else 0.0
            y_max = max(float(t.get("y1", 0.0) or 0.0) for t in r) if r else 0.0
            rows_meta.append((p, y_min, y_max, r))

    # Sort rows by (page, y_center) and return only token lists
    rows_meta.sort(key=lambda it: (int(it[0] or 0), (it[1] + it[2]) / 2.0))
    return [r for (_p, _y0, _y1, r) in rows_meta]

def _merge_vertical_splits(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Merge vertically split values within the same column when rows are very close
    cols = ("label", "tu", "ex", "eq")
    
    def _is_placeholder(v: str | None) -> bool:
        z = (v or "").strip()
        if not z:
            return True
        if z in {"$0", "0"}:
            return True
        # Single token (e.g., a short month or single word) treated as placeholder to allow safe merging
        try:
            return len(z.split()) <= 1
        except Exception:
            return False

    def _need_space_cells(prev_txt: str, cur_txt: str) -> bool:
        try:
            return bool(prev_txt and cur_txt and prev_txt[-1].isalnum() and cur_txt[0].isalnum())
        except Exception:
            return False
    try:
        rows_sorted = sorted(rows or [], key=lambda r: (int(r.get("page", 0) or 0), float(r.get("y", 0.0))))
    except Exception:
        rows_sorted = list(rows or [])
    out: List[Dict[str, Any]] = []
    for r in rows_sorted:
        if out:
            p = out[-1]
            try:
                close_y = abs(float(r.get("y", 0.0)) - float(p.get("y", 0.0))) <= MERGE_DY
            except Exception:
                close_y = False
            merged = False
            if close_y:
                for col in cols:
                    pv = str(p.get(col, "")).strip()
                    rv = str(r.get(col, "")).strip()
                    if pv and rv:
                        other_cols = [c for c in cols if c != col]
                        # Allow merge when other columns are placeholders in both rows
                        # Only merge within the same page
                        if (int(p.get("page", 0) or 0) == int(r.get("page", 0) or 0)) and all(
                            _is_placeholder(p.get(oc, "")) and _is_placeholder(r.get(oc, "")) for oc in other_cols
                        ):
                            sep = " " if _need_space_cells(pv, rv) else ""
                            p[col] = (pv + sep + rv).strip()
                            merged = True
                            break
            if not merged:
                out.append(r)
        else:
            out.append(r)
    return out


# --- Label attachment helpers ---
EPS_Y = 0.8  # small vertical epsilon to collapse duplicate anchors


def _clean_txt(s: str | None) -> str:
    return (s or "").strip()

def _norm(s: str | None) -> str:
    import re
    return re.sub(r"\W+", "", (s or "").lower())


def _collect_label_anchors(tokens: List[dict], label_max_x: float) -> Dict[int, List[Tuple[float, str]]]:
    # Return anchors per page: {page: [(y,label), ...]} (sorted, deduped per page)
    # 1) filter to left-column tokens within label area, bucket by page
    by_page: Dict[int, List[dict]] = {}
    for t in tokens or []:
        try:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
        except Exception:
            mx = 0.0
        if float(mx) <= float(label_max_x) + 4.0:
            try:
                p = int(t.get("page", 0) or 0)
            except Exception:
                p = 0
            by_page.setdefault(p, []).append(t)

    # 2) per page, cluster by Y lines and detect labels
    CLUST_DY = min(GRID_DY, 1.0)
    anchors_by_page: Dict[int, List[Tuple[float, str]]] = {}
    for p, toks in by_page.items():
        toks_sorted = sorted(toks, key=lambda t: _mid(t.get("y0", 0.0), t.get("y1", 0.0)))
        clusters: List[List[dict]] = []
        cur: List[dict] = []
        cur_y: float | None = None
        for t in toks_sorted:
            y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
            if cur_y is None or abs(y - cur_y) <= CLUST_DY:
                cur.append(t)
                try:
                    cur_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in cur) / len(cur)
                except Exception:
                    cur_y = y
            else:
                clusters.append(cur)
                cur = [t]
                cur_y = y
        if cur:
            clusters.append(cur)

        anchors: List[Tuple[float, str]] = []
        for group in clusters:
            joined = _join_text(group)
            jt = _clean_txt(joined)
            if not jt:
                continue
            is_colon = jt.endswith(":")
            try:
                nz = _norm(jt)
            except Exception:
                nz = ""
            is_known = nz in NON_COLON_LABELS
            if is_colon or is_known:
                try:
                    y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in group) / len(group)
                except Exception:
                    y = _mid(group[0].get("y0", 0.0), group[0].get("y1", 0.0))
                lab = jt[:-1].strip() if is_colon else jt
                anchors.append((float(y), lab))

        anchors.sort(key=lambda it: it[0])
        dedup: List[Tuple[float, str]] = []
        for y, lab in anchors:
            if not dedup or abs(y - dedup[-1][0]) > EPS_Y:
                dedup.append((y, lab))
        anchors_by_page[p] = dedup
    return anchors_by_page


def _is_bureau_header_cell(val: str | None) -> bool:
    z = (val or "").strip().lower()
    return z in BUREAU_HDRS


def _attach_labels_to_rows(
    out_rows: List[Dict[str, Any]],
    anchors_by_page: Dict[int, List[Tuple[float, str]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> List[Dict[str, Any]]:
    if not anchors_by_page or not out_rows:
        return out_rows
    # Ensure rows are processed in (page,y) order and anchors are sorted per page
    out_rows = sorted(out_rows, key=lambda r: (int(r.get("page", 0) or 0), float(r.get("y", 0.0))))
    ai_by_page: Dict[int, int] = {}
    for r in out_rows:
        try:
            y_row = float(r.get("y", 0.0))
        except Exception:
            y_row = 0.0
        try:
            rp = int(r.get("page", 0) or 0)
        except Exception:
            rp = 0
        anchors = anchors_by_page.get(rp) or []
        if not anchors:
            continue
        ai = ai_by_page.get(rp, 0)
        while ai + 1 < len(anchors) and anchors[ai + 1][0] <= y_row:
            ai += 1
        ai_by_page[rp] = ai
        anch_y, anch_lab = anchors[ai]
        if y_min is not None and y_row < y_min:
            continue
        if y_max is not None and y_row > y_max:
            continue
        if y_row >= anch_y and (y_row - anch_y) <= LABEL_ATTACH_MAX_DY:
            if any(_is_bureau_header_cell(r.get(k, "")) for k in ("tu", "ex", "eq")):
                continue
            cur_lab = str(r.get("label", "")).strip()
            if cur_lab:
                try:
                    cur_norm = _norm(cur_lab)
                    anch_norm = _norm(anch_lab)
                except Exception:
                    cur_norm = cur_lab.lower()
                    anch_norm = anch_lab.lower()
                # Do not overwrite if the row already has a known/correct label
                if (cur_norm in NON_COLON_LABELS) or (cur_norm == anch_norm):
                    continue
                r["label"] = anch_lab
    return out_rows


def _autofill_labels_from_block_lines(
    block_lines: List[str] | None,
    rows: List[Dict[str, Any]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[List[Dict[str, Any]], int, int, int]:
    """Autofill missing labels using canonical label order derived from block lines.
    Returns (rows_mutated, used_count, total_canonical, changed_rows).
    """
    try:
        raw_lines = list(block_lines or [])
        cut = find_section_cut_index(raw_lines)
        canonical = extract_canonical_labels(raw_lines[:cut])
    except Exception:
        canonical = []
    i = 0
    changed = 0
    rows_sorted = sorted(rows or [], key=lambda r: float(r.get("y", 0.0)))
    for r in rows_sorted:
        if i >= len(canonical):
            break
        try:
            y = float(r.get("y", 0.0))
        except Exception:
            y = 0.0
        if y_min is not None and y < y_min:
            continue
        if y_max is not None and y > y_max:
            continue
        # Only fill if any value present
        has_value = any(str(r.get(k, "")).strip() for k in ("tu", "ex", "eq"))
        if not has_value:
            continue
        cur_lab = str(r.get("label", "")).strip()
        if cur_lab:
            # Respect existing labels; do not consume canonical slot
            continue
        r["label"] = canonical[i]
        i += 1
        changed += 1
    return rows_sorted, i, len(canonical), changed


def _relabel_by_value_type(
    rows: List[Dict[str, Any]],
    block_lines: List[str] | None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[List[Dict[str, Any]], int]:
    """Adjust labels among early 'overview' rows so their expected types match detected value types.
    Only reassigns labels; does not reorder rows. Limited to the first N value rows,
    where N is the number of canonical overview labels.
    Returns (rows_mutated, changed_count).
    """
    if not rows:
        return rows, 0
    try:
        raw_lines = list(block_lines or [])
        cut = find_section_cut_index(raw_lines)
        canon = extract_canonical_labels(raw_lines[:cut])
    except Exception:
        canon = []
    if not canon:
        return rows, 0

    def label_expected_type(lbl: str) -> str:
        return LABEL_SCHEMA.get(lbl, "text")

    def row_value_text(r: Dict[str, Any]) -> str:
        return " ".join([(str(r.get("tu", "")) or ""), (str(r.get("ex", "")) or ""), (str(r.get("eq", "")) or "")]).strip()

    def has_value(r: Dict[str, Any]) -> bool:
        return any(str(r.get(k, "")).strip() for k in ("tu", "ex", "eq"))

    def mismatch(r: Dict[str, Any]) -> bool:
        t = detect_value_type(row_value_text(r))
        et = label_expected_type(str(r.get("label", "")))
        return (t != "empty") and (et != t)

    # Work on rows sorted by Y; collect indices of first len(canon) value rows
    rs = list(sorted(rows or [], key=lambda r: float(r.get("y", 0.0))))
    def _in_bounds(r: Dict[str, Any]) -> bool:
        try:
            y = float(r.get("y", 0.0))
        except Exception:
            y = 0.0
        if y_min is not None and y < y_min:
            return False
        if y_max is not None and y > y_max:
            return False
        return True

    idxs: List[int] = [i for i, r in enumerate(rs) if has_value(r) and _in_bounds(r)][: len(canon)]
    if not idxs:
        return rs, 0

    # Track used canonical labels (normalized) from current labels in the region
    def _nrm(lbl: str) -> str:
        return (lbl or "").strip().lower().replace(" ", "")

    used = set(_nrm(rs[i].get("label", "")) for i in idxs if rs[i].get("label"))
    canon_norm = [(_nrm(c), c) for c in canon]

    changed = 0
    for pos, i in enumerate(idxs):
        r = rs[i]
        if not r.get("label"):
            continue
        if not mismatch(r):
            continue
        # Try swap with neighbor inside allowed region
        for off in (-1, 1):
            jpos = pos + off
            if jpos < 0 or jpos >= len(idxs):
                continue
            j = idxs[jpos]
            rj = rs[j]
            if not rj.get("label"):
                continue
            t_i = detect_value_type(row_value_text(r))
            t_j = detect_value_type(row_value_text(rj))
            et_i = label_expected_type(str(r.get("label") or ""))
            et_j = label_expected_type(str(rj.get("label") or ""))
            after_match = int(et_j == t_i) + int(et_i == t_j)
            before_match = int(et_i == t_i) + int(et_j == t_j)
            if after_match > before_match:
                r["label"], rj["label"] = rj.get("label"), r.get("label")
                changed += 1
                break
        # If still mismatched, pick an unused canonical label matching detected type
        if mismatch(r):
            t_i = detect_value_type(row_value_text(r))
            for nrm, original in canon_norm:
                if nrm in used:
                    continue
                if label_expected_type(original) == t_i:
                    r["label"] = original
                    used.add(nrm)
                    changed += 1
                    break

    # Map back to original order correspondingly
    return rs, changed


def _compute_overview_bounds(
    ltoks: List[dict],
    out_rows: List[Dict[str, Any]],
    window: Dict[str, Any],
) -> Tuple[float, float]:
    """Compute Y bounds for the overview section using tokens and built rows.
    Returns (y_overview_top, y_overview_bottom).
    """
    # 1) header line Y (first occurrence of bureau headers)
    try:
        hdr_y = None
        for t in ltoks:
            z = str(t.get("text", "")).strip().lower()
            if z in ("transunion", "experian", "equifax"):
                y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
                hdr_y = y if hdr_y is None else min(hdr_y, y)
        # 2) first value row Y
        val_y = None
        for r in sorted(out_rows or [], key=lambda r: float(r.get("y", 0.0))):
            if any(str(r.get(k, "")).strip() for k in ("tu", "ex", "eq")):
                val_y = float(r.get("y", 0.0))
                break
        # 3) section heading (Two-Year / Days Late) bottom bound
        # Cluster tokens into lines and search normalized text
        toks = sorted(ltoks or [], key=lambda t: _mid(t.get("y0", 0.0), t.get("y1", 0.0)))
        lines: List[List[dict]] = []
        cur: List[dict] = []
        cur_y: float | None = None
        for t in toks:
            y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
            if cur_y is None or abs(y - cur_y) <= 1.5:
                cur.append(t)
                try:
                    cur_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in cur) / len(cur)
                except Exception:
                    cur_y = y
            else:
                lines.append(cur)
                cur = [t]
                cur_y = y
        if cur:
            lines.append(cur)
        import re as _re

        def _norm(s: str) -> str:
            return _re.sub(r"\W+", "", (s or "").lower())

        sec_y = None
        for line in lines:
            txt = _join_text(line)
            nz = _norm(txt)
            if nz in ("twoyearpaymenthistory", "dayslate7yearhistory"):
                y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in line) / max(1, len(line))
                sec_y = y
                break
        try:
            y_bottom = float(window.get("y_bottom", 0.0))
        except Exception:
            y_bottom = sec_y or 0.0
        y_overview_bottom = float(sec_y) if sec_y is not None else y_bottom
        # Overview top is max of header and first value row if both exist
        candidates = [v for v in [hdr_y, val_y] if v is not None]
        y_overview_top = float(max(candidates)) if candidates else float(val_y or (hdr_y or 0.0))
        return y_overview_top, y_overview_bottom
    except Exception:
        # Fallback to window bounds
        try:
            return float(window.get("y_top", 0.0)), float(window.get("y_bottom", 0.0))
        except Exception:
            return 0.0, 0.0


def run_template_first(session_id: str, root: Path | None = None) -> Dict[str, Any]:
    """
    Build SmartCredit 4-col grid tables directly from Stage-A artifacts.
    Returns: {ok: bool, confidence: 0.0, accounts_path: str|None, reason: str|None}
    """
    try:
        base = (root or Path.cwd()) / "traces" / "blocks" / session_id
        layout_path = base / "layout_snapshot.json"
        windows_path = base / "block_windows.json"

        if not layout_path.exists() or not windows_path.exists():
            logger.warning(
                "TEMPLATE: Stage-A artifacts missing (sid=%s) layout=%s windows=%s",
                session_id,
                str(layout_path),
                str(windows_path),
            )
            return {"ok": False, "confidence": 0.0, "accounts_path": None, "reason": "missing_artifacts"}

        layout = _read_json(layout_path)
        windows = _read_json(windows_path)
        if not layout or not windows:
            logger.warning("TEMPLATE: Stage-A artifacts unreadable (sid=%s)", session_id)
            return {"ok": False, "confidence": 0.0, "accounts_path": None, "reason": "missing_artifacts"}

        # Build pages map
        pages_by_num: Dict[int, dict] = {}
        for idx, p in enumerate(list(layout.get("pages") or []), start=1):
            try:
                num = int(p.get("number", idx) or idx)
            except Exception:
                num = idx
            pages_by_num[num] = p

        try:
            logger.info(
                "TEMPLATE: start sid=%s pages=%d blocks=%d",
                session_id,
                len(pages_by_num or {}),
                len(list(windows.get("blocks") or [])),
            )
        except Exception:
            pass

        accounts_dir = base / "accounts_table"
        accounts_dir.mkdir(parents=True, exist_ok=True)

        index_blocks: List[Dict[str, Any]] = []
        written = 0
        skipped = 0

        for row in list(windows.get("blocks") or []):
            try:
                block_id = int(row.get("block_id", 0) or 0)
                heading = row.get("heading")
                index_headline = row.get("index_headline")
                window = row.get("window") or None
                if not block_id or not window:
                    logger.info("TEMPLATE: skip block_id=%s no window", block_id)
                    skipped += 1
                    continue
                try:
                    page_no = int(window.get("page", 0) or 0)
                except Exception:
                    page_no = 0
                page = pages_by_num.get(page_no)
                if not page:
                    logger.info("TEMPLATE: skip block_id=%s reason=page_missing", block_id)
                    skipped += 1
                    continue

                # Slice tokens by window, but expand effective X min to include left-side labels
                toks_all = list(page.get("tokens") or [])
                try:
                    x_min = float(window.get("x_min", 0.0) or 0.0)
                    x_max = float(window.get("x_max", 0.0) or 0.0)
                    y_top = float(window.get("y_top", 0.0) or 0.0)
                    y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
                except Exception:
                    logger.info("TEMPLATE: skip block_id=%s reason=invalid_window", block_id)
                    skipped += 1
                    continue

                # 1) keep Y band (vertical overlap with window)
                try:
                    y_band_tokens: List[dict] = [
                        t for t in toks_all
                        if float(t.get("y0", 0.0)) <= float(y_bottom) and float(t.get("y1", 0.0)) >= float(y_top)
                    ]
                except Exception:
                    y_band_tokens = []

                # 2) label-like hints
                _LABEL_HINTS = (
                    "account",
                    "date",
                    "balance",
                    "opened",
                    "reported",
                    "status",
                    "payment",
                    "credit",
                    "high",
                    "remarks",
                    "description",
                    "past due",
                    "term",
                    "frequency",
                    "limit",
                )

                def _looks_like_label(s: str | None) -> bool:
                    try:
                        z = (s or "").strip().lower()
                        return len(z) > 2 and any(h in z for h in _LABEL_HINTS)
                    except Exception:
                        return False

                try:
                    label_x_candidates = [
                        float(t.get("x0", 0.0)) for t in y_band_tokens if _looks_like_label(t.get("text"))
                    ]
                except Exception:
                    label_x_candidates = []

                left_label_guess = min(label_x_candidates) if label_x_candidates else float("inf")

                # 3) effective X bounds (with protective floor)
                FLO0R = 96.0
                eff_x_min = max(
                    FLO0R,
                    min(
                        float(x_min),
                        (left_label_guess - 6.0) if left_label_guess != float("inf") else float(x_min),
                    ),
                )
                eff_x_max = float(x_max)

                # 4) pick in-window tokens using effective X bounds
                ltoks: List[dict] = []
                for t in y_band_tokens:
                    mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
                    if eff_x_min <= mx <= eff_x_max:
                        ltoks.append(t)

                # Note: label anchors will be collected after computing label_max_x

                if not ltoks:
                    logger.info("TEMPLATE: skip block_id=%s reason=empty_window_tokens", block_id)
                    skipped += 1
                    continue

                # Special-case: Personal Information block (two-column key:value)
                if str(heading or "").strip().lower() == "personal information":
                    # Heuristic label_max_x for personal info: near left, but use a generous width
                    width = float(eff_x_max) - float(eff_x_min)
                    pi_label_max_x = float(eff_x_min) + max(200.0, min(260.0, 0.45 * width))
                    pi_rows, pi_anchors = extract_personal_info_rows(ltoks, eff_x_min, eff_x_max, pi_label_max_x)
                    # Write key_value output
                    slug = _slug(heading or f"block-{block_id}")
                    out_path = accounts_dir / f"account_table_{block_id:02d}__{slug}.json"
                    obj = {
                        "session_id": session_id,
                        "block_id": block_id,
                        "block_heading": heading,
                        "mode": "key_value",
                        "rows": pi_rows,
                        "meta": {
                            "bands": {},
                            "label_max_x": float(pi_label_max_x),
                            "eff_x_min": float(eff_x_min),
                            "eff_x_max": float(eff_x_max),
                            "debug": {
                                "anchors": [{"y": float(y), "label": lab} for y, lab in pi_anchors],
                                "personal_info_pairs": int(len(pi_rows)),
                            },
                        },
                    }
                    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    index_blocks.append({
                        "block_id": block_id,
                        "heading": heading,
                        "index_headline": index_headline,
                        "table_path": str(out_path),
                        "row_count": len(pi_rows),
                        "column_count": 2,
                    })
                    written += 1
                    continue

                # Build rows by (page,line) when available; fallback to Y clustering per page
                rows = _group_rows_by_page_line(ltoks, GRID_DY)

                # Detect bureau bands per page from in-window tokens
                tokens_by_page: Dict[int, List[dict]] = {}
                for t in ltoks:
                    try:
                        tp = int(t.get("page", page_no) or page_no)
                    except Exception:
                        tp = page_no
                    tokens_by_page.setdefault(tp, []).append(t)

                def _detect_bands_for_tokens(toks_page: List[dict]) -> Dict[str, List[float]]:
                    header_tokens = [
                        t for t in toks_page if str(t.get("text", "")).strip().lower() in ("transunion", "experian", "equifax")
                    ]
                    b = detect_bureau_columns(header_tokens) if header_tokens else {}
                    if b:
                        return b
                    import re

                    def _norm(s: str | None) -> str:
                        try:
                            return re.sub(r"\W+", "", (s or "").lower())
                        except Exception:
                            return ""

                    HEADS = {"tu": "transunion", "ex": "experian", "eq": "equifax"}

                    def _mx(t: dict) -> float:
                        return (float(t.get("x0", 0.0)) + float(t.get("x1", 0.0))) / 2.0

                    centers: Dict[str, List[float]] = {}
                    for t in toks_page:
                        z = _norm(t.get("text"))
                        if z == HEADS["tu"]:
                            centers.setdefault("tu", []).append(_mx(t))
                        elif z == HEADS["ex"]:
                            centers.setdefault("ex", []).append(_mx(t))
                        elif z == HEADS["eq"]:
                            centers.setdefault("eq", []).append(_mx(t))

                    have = [k for k, v in centers.items() if v]
                    if len(have) >= 2:
                        xs = {k: (sum(v) / len(v)) for k, v in centers.items() if v}
                        ordered = sorted(xs.items(), key=lambda kv: kv[1])
                        mids = [(ordered[i][1] + ordered[i + 1][1]) / 2.0 for i in range(len(ordered) - 1)]
                        xmin, xmax = eff_x_min, eff_x_max
                        fb: Dict[str, List[float]] = {}
                        if len(ordered) == 3:
                            (k0, _x0), (k1, _x1), (k2, _x2) = ordered
                            m01, m12 = mids[0], mids[1]
                            fb[k0] = [xmin, m01]
                            fb[k1] = [m01, m12]
                            fb[k2] = [m12, xmax]
                        else:
                            (k0, _x0), (k1, _x1) = ordered
                            m01 = mids[0]
                            fb[k0] = [xmin, m01]
                            fb[k1] = [m01, xmax]
                        out: Dict[str, List[float]] = {}
                        if "tu" in fb:
                            out["transunion"] = fb["tu"]
                        if "ex" in fb:
                            out["experian"] = fb["ex"]
                        if "eq" in fb:
                            out["equifax"] = fb["eq"]
                        return out
                    return {}

                bands_by_page: Dict[int, Dict[str, List[float]]] = {}
                for pnum, toks_p in tokens_by_page.items():
                    try:
                        bands_by_page[pnum] = _detect_bands_for_tokens(toks_p)
                    except Exception:
                        bands_by_page[pnum] = {}
                bands = bands_by_page.get(int(page_no) if page_no else 0, {})
                # Compute label max x relative to effective window
                try:
                    base_left = float(eff_x_min) if "eff_x_min" in locals() else float(x_min)
                except Exception:
                    base_left = float(x_min)
                if bands:
                    try:
                        lefts = [float(v[0]) for v in bands.values()]
                        label_max_x = max(min(lefts) - GRID_LABEL_MARGIN, base_left + 8.0)
                    except Exception:
                        label_max_x = base_left + 16.0
                else:
                    label_max_x = base_left + 16.0
                    try:
                        logger.info(
                            "TEMPLATE: block_id=%s no_bands_detected using_relative_label_max_x=%.1f base_left=%.1f",
                            block_id,
                            label_max_x,
                            base_left,
                        )
                    except Exception:
                        pass

                def assign_cell(t: dict) -> str | None:
                    mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
                    try:
                        tp = int(t.get("page", page_no) or page_no)
                    except Exception:
                        tp = page_no
                    pb = bands_by_page.get(tp) or {}
                    if pb:
                        for name, (xL, xR) in pb.items():
                            if xL <= mx <= xR:
                                return name.lower()
                    return None

                out_rows: List[Dict[str, Any]] = []
                for r in rows:
                    label_t: List[dict] = []
                    tu_t: List[dict] = []
                    ex_t: List[dict] = []
                    eq_t: List[dict] = []
                    for t in r:
                        mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
                        if mx <= label_max_x:
                            label_t.append(t)
                        else:
                            cell = assign_cell(t)
                            if cell == "transunion":
                                tu_t.append(t)
                            elif cell == "experian":
                                ex_t.append(t)
                            elif cell == "equifax":
                                eq_t.append(t)
                    label_text = _join_text(label_t)
                    tu_text = _join_text(tu_t)
                    ex_text = _join_text(ex_t)
                    eq_text = _join_text(eq_t)
                    # Skip header row
                    if not label_text and {tu_text.lower(), ex_text.lower(), eq_text.lower()} == {"transunion", "experian", "equifax"}:
                        continue
                    # Skip fully empty rows
                    if not any([label_text.strip(), tu_text.strip(), ex_text.strip(), eq_text.strip()]):
                        continue
                    y = sum(_mid(t.get("y0", 0.0), t.get("y1", 0.0)) for t in r) / max(1, len(r))
                    try:
                        r_page = int((r[0].get("page", page_no) or page_no)) if r else int(page_no)
                    except Exception:
                        r_page = int(page_no) if page_no else 0
                    row_obj: Dict[str, Any] = {
                        "page": r_page,
                        "y": float(y),
                        "label": label_text,
                        "tu": tu_text,
                        "ex": ex_text,
                        "eq": eq_text,
                    }
                    if EMIT_PROVENANCE:
                        def _tok_minimal(t: dict) -> Dict[str, Any]:
                            try:
                                pg = int(t.get("page", r_page) or r_page)
                            except Exception:
                                pg = r_page
                            return {
                                "page": pg,
                                **({"line": int(t.get("line"))} if isinstance(t.get("line"), (int, float, str)) and str(t.get("line")).strip() != "" else {}),
                                "x0": t.get("x0"),
                                "y0": t.get("y0"),
                                "x1": t.get("x1"),
                                "y1": t.get("y1"),
                            }
                        row_obj["src"] = {
                            "label": [_tok_minimal(t) for t in label_t],
                            "tu": [_tok_minimal(t) for t in tu_t],
                            "ex": [_tok_minimal(t) for t in ex_t],
                            "eq": [_tok_minimal(t) for t in eq_t],
                        }
                    out_rows.append(row_obj)

                # Merge vertically split values within columns before writing
                out_rows = _merge_vertical_splits(out_rows)

                # Collect label anchors from left column (supports colon and known non-colon labels)
                anchors_by_page = _collect_label_anchors(ltoks, float(label_max_x))
                # Attach labels to rows by vertical range using collected anchors
                # Compute Y overview bounds to limit post-processing to the overview section
                y_overview_top, y_overview_bottom = _compute_overview_bounds(ltoks, out_rows, window)
                out_rows = _attach_labels_to_rows(out_rows, anchors_by_page, y_min=y_overview_top, y_max=y_overview_bottom)

                # Autofill missing labels from canonical block lines (do not overwrite existing)
                block_lines_path = base / f"block_{block_id:02d}.json"
                try:
                    block_obj = _read_json(block_lines_path) or {}
                    block_lines = list(block_obj.get("lines") or [])
                except Exception:
                    block_lines = []
                out_rows, canon_used, canon_total, autofill_changed = _autofill_labels_from_block_lines(
                    block_lines, out_rows, y_min=y_overview_top, y_max=y_overview_bottom
                )

                # Type-aware relabeling to correct misaligned labels among overview rows
                out_rows, relabel_changed = _relabel_by_value_type(out_rows, block_lines, y_min=y_overview_top, y_max=y_overview_bottom)

                # Write block output
                slug = _slug(heading or f"block-{block_id}")
                out_path = accounts_dir / f"account_table_{block_id:02d}__{slug}.json"
                meta_bands = {k: [float(v[0]), float(v[1])] for k, v in (bands or {}).items()}
                try:
                    meta_bands_by_page = {
                        str(k): {kk: [float(vv[0]), float(vv[1])] for kk, vv in (vb or {}).items()}
                        for k, vb in (bands_by_page or {}).items()
                    }
                except Exception:
                    meta_bands_by_page = {}
                try:
                    meta_eff_x_min = float(eff_x_min)
                    meta_eff_x_max = float(eff_x_max)
                except Exception:
                    # Fallback to raw window if effective bounds are not available
                    meta_eff_x_min = float(x_min)
                    meta_eff_x_max = float(x_max)
                # Extract extras (e.g., two-year payment history)
                extras: List[Dict[str, Any]] = []
                try:
                    hist_path = extract_two_year_payment_history(
                        session_id=session_id,
                        block_id=block_id,
                        heading=heading,
                        page_tokens=list(page.get("tokens") or []),
                        window=window,
                        bands=bands,
                        out_dir=accounts_dir,
                    )
                except Exception:
                    hist_path = None
                if hist_path:
                    extras.append({"type": "two_year_payment_history", "path": hist_path})
                # Extract 7-year delinquency summary
                try:
                    delinq_path = extract_seven_year_delinquency(
                        session_id=session_id,
                        block_id=block_id,
                        heading=heading,
                        page_tokens=list(page.get("tokens") or []),
                        window=window,
                        bands=bands,
                        out_dir=accounts_dir,
                    )
                except Exception:
                    delinq_path = None
                if delinq_path:
                    extras.append({"type": "seven_year_delinquency", "path": delinq_path})

                # Stable ordering by (page, y)
                try:
                    out_rows = sorted(out_rows, key=lambda r: (int(r.get("page", 0) or 0), float(r.get("y", 0.0) or 0.0)))
                except Exception:
                    pass

                # Diagnostics: label coverage counters
                attached = sum(1 for r in out_rows if (str(r.get("label", "")).strip()))
                total = len(out_rows)
                value_rows = sum(
                    1
                    for r in out_rows
                    if any(str(r.get(k, "")).strip() for k in ("tu", "ex", "eq"))
                )
                # Type mismatch counter after relabeling (grid_table only)
                def _row_val_text(rr: Dict[str, Any]) -> str:
                    return " ".join([(rr.get("tu") or ""), (rr.get("ex") or ""), (rr.get("eq") or "")]).strip()
                mismatch_count = 0
                try:
                    mismatch_count = sum(
                        1
                        for rr in out_rows
                        if str(rr.get("label", "")).strip()
                        and detect_value_type(_row_val_text(rr))
                        != LABEL_SCHEMA.get(str(rr.get("label") or ""), "text")
                    )
                except Exception:
                    mismatch_count = 0
                try:
                    if value_rows > attached:
                        delta = int(value_rows - attached)
                        logger.warning(
                            "TEMPLATE: coverage gap block_id=%s value_rows=%d label_attached=%d delta=%d",
                            block_id,
                            value_rows,
                            attached,
                            delta,
                        )
                except Exception:
                    pass

                # Prepare debug anchors for diagnostics
                debug_anchors = []
                debug_anchors_by_page = []
                try:
                    for pg, lst in (anchors_by_page or {}).items():
                        for (y, lab) in lst or []:
                            debug_anchors_by_page.append({"page": int(pg), "y": float(y), "label": str(lab)})
                            debug_anchors.append({"y": float(y), "label": str(lab)})
                except Exception:
                    pass

                obj = {
                    "session_id": session_id,
                    "block_id": block_id,
                    "block_heading": heading,
                    "mode": "grid_table",
                    "rows": out_rows,
                    "meta": {
                        "bands": meta_bands,
                        "bands_by_page": meta_bands_by_page,
                        "label_max_x": float(label_max_x),
                        "eff_x_min": meta_eff_x_min,
                        "eff_x_max": meta_eff_x_max,
                        "debug": {
                            "label_attached_rows": int(attached),
                            "label_total_rows": int(total),
                            "value_rows": int(value_rows),
                            "anchors": debug_anchors,
                            "canonical_labels_total": int(canon_total),
                            "canonical_labels_used": int(canon_used),
                            "autofill_rows_changed": int(autofill_changed),
                            "relabel_by_type_changed": int(relabel_changed),
                            "mismatch_count": int(mismatch_count),
                            "y_overview_top": float(y_overview_top),
                            "y_overview_bottom": float(y_overview_bottom),
                            "anchors_by_page": debug_anchors_by_page,
                        },
                    },
                }
                out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                index_blocks.append({
                    "block_id": block_id,
                    "heading": heading,
                    "index_headline": index_headline,
                    "table_path": str(out_path),
                    "row_count": len(out_rows),
                    "column_count": 4,
                    "extras": extras,
                })
                written += 1
            except Exception:
                logger.warning("TEMPLATE: failed to process block row sid=%s", session_id, exc_info=True)
                skipped += 1
                continue

        # Write index
        idx_path = accounts_dir / "_table_index.json"
        idx_obj = {"session_id": session_id, "blocks": index_blocks}
        idx_path.write_text(json.dumps(idx_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("TEMPLATE: tables written sid=%s written=%d skipped=%d", session_id, written, skipped)
        return {"ok": bool(written > 0), "confidence": 0.0, "accounts_path": str(idx_path) if written else None, "reason": None if written else "no_tables"}
    except Exception:
        logger.warning("TEMPLATE: unexpected failure (sid=%s)", session_id, exc_info=True)
        return {"ok": False, "confidence": 0.0, "accounts_path": None, "reason": "unexpected_error"}


__all__ = ["run_template_first"]
