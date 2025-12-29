from __future__ import annotations

from typing import Dict, Any, List

from .normalize_fields import (
    LABELS,
    LABEL_TO_KEY,
    CANONICAL_KEYS,
    clean_value,
    ensure_all_keys,
)


def map_raw_to_fields(raw_pkg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map a RAW coordinate package into per-bureau fields dicts.

    Strategy (label-free): assume rows are top-to-bottom aligned to LABELS order
    within the account window. When there are fewer rows than labels or rows
    include non-field content, we map what we can by index and leave the rest
    empty. No guessing beyond ordering; placeholders are cleaned to empty.
    """

    rows: List[Dict[str, Any]] = list(raw_pkg.get("rows") or [])
    # Sort by row_y defensively
    rows = sorted(rows, key=lambda r: float(r.get("row_y") or 0.0))

    result: Dict[str, Dict[str, str]] = {
        "transunion": {k: "" for k in CANONICAL_KEYS},
        "experian": {k: "" for k in CANONICAL_KEYS},
        "equifax": {k: "" for k in CANONICAL_KEYS},
    }

    # Build per-row concatenated strings if not present
    def _row_text(r: Dict[str, Any], bureau: str) -> str:
        key = f"{bureau}_text"
        if key in r and isinstance(r[key], str):
            return r[key]
        toks = r.get(bureau) or []
        s = " ".join(str(t.get("text", "")) for t in toks)
        return s

    # Map rows to labels by index, up to the available rows and 22 labels
    n = min(len(rows), len(LABELS))
    for i in range(n):
        label = LABELS[i]
        key = LABEL_TO_KEY.get(label)
        if not key:
            continue
        r = rows[i]
        tu = clean_value(_row_text(r, "transunion"))
        ex = clean_value(_row_text(r, "experian"))
        eq = clean_value(_row_text(r, "equifax"))
        result["transunion"][key] = tu
        result["experian"][key] = ex
        result["equifax"][key] = eq

    # Ensure all keys (idempotent) and return
    for b in ("transunion", "experian", "equifax"):
        result[b] = ensure_all_keys(result[b])
    return result


__all__ = ["map_raw_to_fields"]

