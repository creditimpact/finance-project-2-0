import json
import os
from pathlib import Path

import pytest

from backend.core.logic.report_analysis.history_extractor import (
    extract_two_year_payment_history,
)


def _tok(text, x0, y0, x1=None, y1=None, page=1):
    if x1 is None:
        x1 = x0 + 10
    if y1 is None:
        y1 = y0 + 8
    return {"text": text, "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "page": page}


@pytest.mark.parametrize("use_two_rows", [False, True])
def test_two_year_history_x_aligns_months_and_statuses(tmp_path: Path, monkeypatch, use_two_rows: bool):
    # Enable feature flag
    monkeypatch.setenv("HISTORY_X_MATCH_ENABLED", "1")
    monkeypatch.setenv("HISTORY_Y_CLUSTER_DY", "2.5")
    monkeypatch.setenv("HISTORY_Y_PAIR_MAX_DY", "12.0")
    monkeypatch.setenv("HISTORY_X_SEAM_GUARD", "2.0")

    # Window and bands
    window = {"y_top": 100.0, "y_bottom": 300.0, "x_min": 0.0, "x_max": 600.0}
    bands = {
        "transunion": (150.0, 300.0),
        "experian": (300.0, 450.0),
        "equifax": (450.0, 600.0),
    }

    tokens = []
    # Heading line
    tokens.append(_tok("Two-Year Payment History", 20, 120, 180, 128))

    # TU status row above months
    # Assign statuses near month centers
    if not use_two_rows:
        # Single row: Jan, Feb, Mar with statuses OK, 30, OK
        tokens.extend([
            _tok("OK", 170, 145, 178, 152),
            _tok("30", 200, 145, 208, 152),
            _tok("OK", 230, 145, 238, 152),
        ])
        tokens.extend([
            _tok("Jan", 170, 150, 182, 158),
            _tok("Feb", 200, 150, 212, 158),
            _tok("Mar", 230, 150, 242, 158),
        ])
    else:
        # Two rows of months: Jan Feb on row1, Mar Apr on row2
        tokens.extend([
            _tok("OK", 170, 145, 178, 152),
            _tok("30", 200, 145, 208, 152),
        ])
        tokens.extend([
            _tok("Jan", 170, 150, 182, 158),
            _tok("Feb", 200, 150, 212, 158),
        ])
        # Second pair: statuses row above months row 2
        tokens.extend([
            _tok("60", 230, 165, 238, 172),
            _tok("OK", 260, 165, 268, 172),
        ])
        tokens.extend([
            _tok("Mar", 230, 170, 242, 178),
            _tok("Apr", 260, 170, 272, 178),
        ])

    # XP simple single row mirrors TU
    tokens.extend([
        _tok("OK", 320, 145, 328, 152),
        _tok("OK", 350, 145, 358, 152),
        _tok("OK", 380, 145, 388, 152),
        _tok("Jan", 320, 150, 332, 158),
        _tok("Feb", 350, 150, 362, 158),
        _tok("Mar", 380, 150, 392, 158),
    ])

    # EQ missing to test gaps

    out_path = tmp_path
    res = extract_two_year_payment_history(
        session_id="sid123",
        block_id=3,
        heading="Sample Block",
        page_tokens=tokens,
        window=window,
        bands=bands,
        out_dir=out_path,
    )
    assert res is not None
    p = Path(res)
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))

    # monthly present when feature enabled
    assert "monthly" in data
    monthly_tu = data["monthly"].get("transunion", [])
    assert monthly_tu, "expected TU monthly list"

    # values array should reflect just the values in order
    vals_tu = data["values"].get("transunion", [])
    assert vals_tu, "expected TU values list"

    # Validate mapping for first three months
    if not use_two_rows:
        assert vals_tu[:3] == ["OK", "30", "OK"]
    else:
        # Two rows produce four months: OK,30,60,OK
        assert vals_tu[:4] == ["OK", "30", "60", "OK"]

    # Experian simple OKs
    vals_xp = data["values"].get("experian", [])
    assert vals_xp[:3] == ["OK", "OK", "OK"]
