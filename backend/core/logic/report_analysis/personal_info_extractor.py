from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.core.logic.report_analysis.canonical_labels import NON_COLON_LABELS


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _join_text(tokens: List[dict]) -> str:
    def midx(t: dict) -> float:
        return _mid(t.get("x0", 0.0), t.get("x1", 0.0))

    toks = sorted(tokens or [], key=midx)
    out: List[str] = []
    prev: dict | None = None
    for t in toks:
        txt = str(t.get("text", "")).strip()
        if not txt:
            continue
        if prev is None:
            out.append(txt)
            prev = t
            continue
        try:
            gap = float(t.get("x0", 0.0)) - float(prev.get("x1", 0.0))
        except Exception:
            gap = 0.0
        if gap >= 1.0 or (out[-1] and txt and out[-1][-1].isalnum() and txt[0].isalnum()):
            out.append(" ")
        out.append(txt)
        prev = t
    return "".join(out).strip()


def _norm(s: str | None) -> str:
    import re
    return re.sub(r"\W+", "", (s or "").lower())


def extract_personal_info_rows(
    tokens_in_window: List[dict],
    eff_x_min: float,
    eff_x_max: float,
    label_max_x: float,
    line_dy: float = 1.5,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, str]]]:
    """Extract key:value rows for the Personal Information block.
    Returns (rows, anchors) where rows are [{page,y,label,value}], anchors are [(y,label)].
    """
    try:
        label_x_max = float(label_max_x)
    except Exception:
        label_x_max = float(eff_x_min) + 200.0

    # Build per (page,line) groups
    by_pl: Dict[Tuple[int, int], List[dict]] = {}
    for t in tokens_in_window or []:
        try:
            p = int(t.get("page", 0) or 0)
        except Exception:
            p = 0
        ln = t.get("line", None)
        try:
            ln = int(ln) if ln is not None else None
        except Exception:
            ln = None
        if ln is None:
            # fallback to y-approx bucket using rounded y
            try:
                yb = int(round(_mid(t.get("y0", 0.0), t.get("y1", 0.0)) / max(1.0, line_dy)))
            except Exception:
                yb = 0
            ln = 10_000_000 + yb
        by_pl.setdefault((p, ln), []).append(t)

    # Small synonym map to normalize labels
    SYN = {
        "alsoknownas": "Also Known As",
        "aka": "Also Known As",
        "alias": "Also Known As",
        "name": "Name",
        "fullname": "Name",
        "dateofbirth": "Date of Birth",
        "dob": "Date of Birth",
        "currentaddress": "Current Address",
        "previousaddress": "Previous Address",
        "formeraddress": "Previous Address",
        "creditreportdate": "Credit Report Date",
        "employer": "Employer",
        "consumerstatement": "Consumer Statement",
    }

    rows: List[Dict[str, Any]] = []
    anchors: List[Tuple[float, str]] = []

    # Process groups ordered by (page,line)
    for (p, ln), group in sorted(by_pl.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        # Split by x threshold
        left = []
        right = []
        for t in group:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            if mx <= (label_x_max + 4.0):
                left.append(t)
            else:
                right.append(t)
        lab_txt = _join_text(left)
        val_txt = _join_text(right)
        lab_norm = _norm(lab_txt)
        # Skip known bureau/section headings
        if lab_norm in {"transunion", "experian", "equifax", "summary", "accounthistory"}:
            continue
        # Determine if this looks like a label candidate
        is_label = False
        if lab_txt.endswith(":"):
            is_label = True
            lab_txt = lab_txt[:-1].strip()
        elif lab_norm in NON_COLON_LABELS or lab_norm in SYN:
            is_label = True
        if not is_label:
            # Be permissive: if there is any left text and some right text, treat as label row
            if lab_txt and val_txt:
                is_label = True
            else:
                continue
        # Normalize via synonyms map if applicable
        if lab_norm in SYN:
            lab_txt = SYN[lab_norm]
        # y center of the line
        try:
            y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in group) / max(1, len(group))
        except Exception:
            y = _mid(group[0].get("y0", 0.0), group[0].get("y1", 0.0))
        anchors.append((float(y), lab_txt))
        # If empty value on same line, try next nearby line with same page
        if not val_txt:
            # Look-ahead small dy to find right-side tokens
            dy = max(3.0, line_dy * 2)
            for (p2, ln2), grp2 in sorted(by_pl.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                if p2 != p or ln2 <= ln:
                    continue
                try:
                    y2 = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in grp2) / max(1, len(grp2))
                except Exception:
                    y2 = _mid(grp2[0].get("y0", 0.0), grp2[0].get("y1", 0.0))
                if 0.0 < (y2 - y) <= dy:
                    right2 = [t for t in grp2 if _mid(t.get("x0", 0.0), t.get("x1", 0.0)) > (label_x_max + 4.0) and _mid(t.get("x0", 0.0), t.get("x1", 0.0)) <= eff_x_max]
                    val_txt = _join_text(right2)
                    if val_txt:
                        break
        rows.append({"page": int(p), "y": float(y), "label": lab_txt, "value": val_txt})

    return rows, anchors
