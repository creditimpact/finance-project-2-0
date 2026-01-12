#!/usr/bin/env python3
"""Unit test: year marker tokens (e.g., '24) should normalize to January in monthly v2 output."""

from backend.core.logic.report_analysis.tsv_v2_monthly_extractor import extract_tsv_v2_monthly


def _tok(text: str, x0: float, x1: float, y0: float, y1: float, page: int = 1, line: int = 1):
    return {"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "page": page, "line": line}


def test_year_token_becomes_january_with_status_preserved():
    # Build minimal token map for a single account with TU months: Nov, Dec, '24, Feb
    tokens_by_line = {
        (1, 1): [_tok("Two-Year", 0, 10, 0, 1, page=1, line=1)],
        (1, 2): [_tok("Payment", 0, 10, 0, 1, page=1, line=2)],
        (1, 3): [_tok("History", 0, 10, 0, 1, page=1, line=3)],
        (1, 4): [_tok("Transunion", 0, 10, 0, 1, page=1, line=4)],
        # Status row (above months, same bureau slice)
        (1, 5): [
            _tok("OK", 10, 20, 0, 1, page=1, line=5),
            _tok("OK", 30, 40, 0, 1, page=1, line=5),
            _tok("CO", 50, 60, 0, 1, page=1, line=5),
            _tok("--", 70, 80, 0, 1, page=1, line=5),
        ],
        # Month row
        (1, 6): [
            _tok("Nov", 10, 20, 10, 11, page=1, line=6),
            _tok("Dec", 30, 40, 10, 11, page=1, line=6),
            _tok("'24", 50, 60, 10, 11, page=1, line=6),
            _tok("Feb", 70, 80, 10, 11, page=1, line=6),
        ],
        (1, 7): [_tok("Experian", 0, 10, 0, 1, page=1, line=7)],
        (1, 8): [_tok("Equifax", 0, 10, 0, 1, page=1, line=8)],
        (1, 9): [_tok("Seven", 0, 10, 0, 1, page=1, line=9)],
    }

    # Minimal lines list for bounds
    lines = [{"page": 1, "line": i} for i in range(1, 10)]

    # months_v2 from the earlier extractor (keeps raw token for the year marker)
    tsv_v2_months = {
        "transunion": ["Nov", "Dec", "'24", "Feb"],
        "experian": [],
        "equifax": [],
    }

    monthly = extract_tsv_v2_monthly(
        session_id="sid-test",
        heading="TestHeading",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    assert monthly is not None
    tu = monthly["transunion"]
    assert len(tu) == 4  # no extra marker entry

    jan_entry = tu[2]
    assert jan_entry["month"] == "'24"
    assert jan_entry["status"] == "co"  # status preserved and lowercased
    assert jan_entry.get("month_label_normalized") == "Jan"
    assert jan_entry.get("derived_month_num") == 1
    assert jan_entry.get("derived_year") == 2024
    assert jan_entry.get("year_token_raw") == "'24"

    # Ensure other months remain untouched
    assert tu[0]["month"] == "Nov"
    assert tu[1]["month"] == "Dec"
    assert tu[3]["month"] == "Feb"
*** End Patch