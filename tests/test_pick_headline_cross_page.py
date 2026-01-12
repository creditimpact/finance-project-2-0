import os
import sys
import pytest

# Ensure project root is on sys.path for direct module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.split_accounts_from_tsv import _pick_headline


def _mk_line(page, line, text):
    return {"page": page, "line": line, "text": text}


def test_cross_page_recovery_positive():
    # Page 5 tail ends with a valid ALL-CAPS issuer name
    lines = [
        _mk_line(5, 34, "SOME OTHER LINE"),
        _mk_line(5, 35, "IGNORED TRAILING"),
        _mk_line(5, 36, "DISCOVERCARD"),
        # Page break; start of page 6 has triad then anchor
        _mk_line(6, 1, "Transunion ® Experian ® Equifax ®"),
        _mk_line(6, 2, "Account # 601100******"),
    ]
    anchor_idx = 4  # index of page 6, line 2

    start_idx, heading, source = _pick_headline(lines, anchor_idx)

    assert heading == "DISCOVERCARD"
    assert source == "cross_page_backtrack"
    # start_idx must remain on current page (page 6)
    assert lines[start_idx]["page"] == 6


def test_cross_page_negative_footer_not_used():
    lines = [
        _mk_line(3, 35, "Page 5 of 15"),
        _mk_line(3, 36, "https://example.com/foo"),
        _mk_line(4, 1, "Transunion Experian Equifax"),
        _mk_line(4, 2, "Account # 1234"),
    ]
    anchor_idx = 3

    start_idx, heading, source = _pick_headline(lines, anchor_idx)

    assert heading is None
    assert source == "anchor_no_heading"
    assert lines[start_idx]["page"] == 4


def test_cross_page_negative_section_header_not_used():
    lines = [
        _mk_line(7, 35, "PUBLIC INFORMATION"),
        _mk_line(7, 36, "TOTAL ACCOUNTS"),
        _mk_line(8, 1, "Transunion ® Experian ® Equifax ®"),
        _mk_line(8, 2, "Account # 9999"),
    ]
    anchor_idx = 3

    start_idx, heading, source = _pick_headline(lines, anchor_idx)

    assert heading is None
    assert source == "anchor_no_heading"
    assert lines[start_idx]["page"] == 8
