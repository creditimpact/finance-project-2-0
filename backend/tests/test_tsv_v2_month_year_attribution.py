from backend.core.logic.report_analysis.tsv_v2_monthly_extractor import extract_tsv_v2_monthly


def _tok(text: str, x0: float, x1: float, y0: float, y1: float, page: int = 1, line: int = 1):
    return {"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "page": page, "line": line}


def _build_tokens(months):
    # Single-line month row and status row aligned left-to-right
    status_tokens = [_tok("OK", 10 + i * 20, 20 + i * 20, 0, 1, line=5) for i in range(len(months))]
    month_tokens = [_tok(m, 10 + i * 20, 20 + i * 20, 10, 11, line=6) for i, m in enumerate(months)]

    tokens_by_line = {
        (1, 1): [_tok("Two-Year", 0, 10, 0, 1, line=1)],
        (1, 2): [_tok("Payment", 0, 10, 0, 1, line=2)],
        (1, 3): [_tok("History", 0, 10, 0, 1, line=3)],
        (1, 4): [_tok("Transunion", 0, 10, 0, 1, line=4)],
        (1, 5): status_tokens,
        (1, 6): month_tokens,
        (1, 7): [_tok("Experian", 0, 10, 0, 1, line=7)],
        (1, 8): [_tok("Equifax", 0, 10, 0, 1, line=8)],
        (1, 9): [_tok("Seven", 0, 10, 0, 1, line=9)],
    }

    lines = [{"page": 1, "line": i} for i in range(1, 10)]
    return tokens_by_line, lines


def _tu_months(monthly_result):
    assert monthly_result is not None
    return monthly_result["transunion"]


def _assert_months(months, expected):
    # expected is list of tuples (month_num, year)
    assert len(months) == len(expected)
    for entry, (mnum, year) in zip(months, expected):
        assert entry.get("derived_month_num") == mnum
        assert entry.get("derived_year") == year
        assert entry.get("month_year_key") == f"{year:04d}-{mnum:02d}"


def test_months_before_first_anchor_forward_rollover():
    seq = ["Sep", "Oct", "Nov", "Dec", "'25", "Feb", "Mar"]
    tokens_by_line, lines = _build_tokens(seq)
    tsv_v2_months = {"transunion": seq, "experian": [], "equifax": []}

    monthly = extract_tsv_v2_monthly(
        session_id="sid-1",
        heading="H1",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    tu = _tu_months(monthly)
    _assert_months(
        tu,
        [
            (9, 2024),
            (10, 2024),
            (11, 2024),
            (12, 2024),
            (1, 2025),
            (2, 2025),
            (3, 2025),
        ],
    )


def test_multiple_year_anchors_resets_years():
    seq = ["Oct", "'24", "Nov", "Dec", "'25", "Jan", "Feb", "'26", "Mar"]
    tokens_by_line, lines = _build_tokens(seq)
    tsv_v2_months = {"transunion": seq, "experian": [], "equifax": []}

    monthly = extract_tsv_v2_monthly(
        session_id="sid-2",
        heading="H2",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    tu = _tu_months(monthly)
    _assert_months(
        tu,
        [
            (10, 2023),
            (1, 2024),
            (11, 2024),
            (12, 2024),
            (1, 2025),
            (1, 2025),
            (2, 2025),
            (1, 2026),
            (3, 2026),
        ],
    )


def test_reverse_order_backward_direction():
    seq = ["Mar", "Feb", "'24", "Dec", "Nov"]
    tokens_by_line, lines = _build_tokens(seq)
    tsv_v2_months = {"transunion": seq, "experian": [], "equifax": []}

    monthly = extract_tsv_v2_monthly(
        session_id="sid-3",
        heading="H3",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    tu = _tu_months(monthly)
    _assert_months(
        tu,
        [
            (3, 2024),
            (2, 2024),
            (1, 2024),
            (12, 2023),
            (11, 2023),
        ],
    )


def test_duplicate_months_get_distinct_years():
    seq = ["Oct", "Nov", "'25", "Nov"]
    tokens_by_line, lines = _build_tokens(seq)
    tsv_v2_months = {"transunion": seq, "experian": [], "equifax": []}

    monthly = extract_tsv_v2_monthly(
        session_id="sid-4",
        heading="H4",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    tu = _tu_months(monthly)
    _assert_months(
        tu,
        [
            (10, 2024),
            (11, 2024),
            (1, 2025),
            (11, 2025),
        ],
    )


def test_backward_rollover_one_to_twelve():
    seq = ["Mar", "Feb", "'25", "Dec"]
    tokens_by_line, lines = _build_tokens(seq)
    tsv_v2_months = {"transunion": seq, "experian": [], "equifax": []}

    monthly = extract_tsv_v2_monthly(
        session_id="sid-5",
        heading="H5",
        idx=0,
        tokens_by_line=tokens_by_line,
        lines=lines,
        tsv_v2_months=tsv_v2_months,
    )

    tu = _tu_months(monthly)
    _assert_months(
        tu,
        [
            (3, 2025),
            (2, 2025),
            (1, 2025),
            (12, 2024),
        ],
    )