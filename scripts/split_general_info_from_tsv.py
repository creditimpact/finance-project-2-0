"""Extract general information sections from a full token TSV dump.

This script reads a TSV file produced from a PDF token stream
(``_debug_full.tsv``) and emits a JSON file with the well known
"general information" sections that appear before the accounts table of a
credit report.  Section boundaries are detected purely from heading text and
*not* from fixed line numbers so the splitter works across differently
formatted reports.

The splitter reconstructs logical lines from individual tokens, normalises
them and then walks through them with a small finite state machine.  The
resulting JSON mirrors what the backend expects for further processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from backend.config import RAW_JOIN_TOKENS_WITH_SPACE

if TYPE_CHECKING:  # pragma: no cover
    from backend.core.logic.report_analysis.block_exporter import (
        join_tokens_with_space as join_tokens_with_space,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helpers


def norm_line(s: str) -> str:
    """Return a normalised representation of ``s`` suitable for matching.

    The normalisation is intentionally strict: lowercase, remove common
    symbols and punctuation and collapse all whitespace.  Only ``a-z``
    characters are retained which makes the matching robust against spacing
    or tokenisation differences.
    """

    s = s.lower()
    s = s.replace("®", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z]", "", s)
    return s


# Normalised marker strings for detecting section boundaries.  Multiple
# candidates are provided for robustness (e.g. "chargeof" covers truncated
# headings).
MARKERS: Dict[str, set[str]] = {
    "personal_info": {"personalinformation"},
    "summary": {"summary"},
    "account_history": {"accounthistory"},
    "collection_chargeoff": {"collectionchargeoff", "collectionchargeof"},
    "public_info": {"publicinformation"},
    "inquiries": {"inquiries"},
    "creditor_contacts": {"creditorcontacts"},
    # footer markers – match both the domain and the typical legal links
    "footer": {
        "smartcreditcom",
        "smartcredit",
        "serviceagreement",
        "privacypolicy",
        "termsofuse",
    },
}


HEADING_ROLES = {
    "personal_info": "emit",
    "summary": "emit",
    "account_history": "anchor_only",
    "collection_chargeoff": "anchor_only",
    "public_info": "emit",
    "inquiries": "emit",
    "creditor_contacts": "emit",
    "footer": "stop",
}


HEADINGS = {
    "personal_info": "Personal Information",
    "summary": "Summary",
    "public_info": "Public Information",
    "inquiries": "Inquiries",
    "creditor_contacts": "Creditor Contacts",
}


# ---------------------------------------------------------------------------
# TSV reading


def read_logical_lines(tsv_path: Path) -> List[Dict[str, Any]]:
    """Reconstruct logical lines from the token TSV ``tsv_path``."""

    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            page_str = row.get("page")
            line_str = row.get("line")
            if not page_str or not line_str:
                continue
            try:
                page = int(float(page_str))
                line = int(float(line_str))
            except Exception:
                continue
            tokens_by_line[(page, line)].append(row)

    logical_lines: List[Dict[str, Any]] = []
    for (page, line), tokens in sorted(tokens_by_line.items()):
        try:
            tokens_sorted = sorted(tokens, key=lambda t: float(t.get("x0") or 0.0))
        except Exception:  # pragma: no cover - very defensive
            tokens_sorted = tokens
        tokens_list = [tok.get("text", "") for tok in tokens_sorted]
        if RAW_JOIN_TOKENS_WITH_SPACE:
            from backend.core.logic.report_analysis.block_exporter import (
                join_tokens_with_space,
            )

            text = join_tokens_with_space(tokens_list)
        else:
            text = "".join(tokens_list)
        text_norm = norm_line(text)
        if not text_norm:
            continue
        logical_lines.append(
            {
                "page": page,
                "line": line,
                "text": text,
                "text_norm": text_norm,
            }
        )

    # Ensure deterministic ordering
    logical_lines.sort(key=lambda d: (d["page"], d["line"]))
    return logical_lines


# ---------------------------------------------------------------------------
# Section splitter


def split_general_info(tsv_path: Path, json_out: Path) -> Dict[str, Any]:
    """Split general information sections from ``tsv_path`` and write JSON."""

    lines = read_logical_lines(tsv_path)

    sections: List[Dict[str, Any]] = []
    current_key: str | None = None
    current_heading: str | None = None
    current_lines: List[Dict[str, Any]] = []

    summary_start: Tuple[int, int] | None = None
    acct_hist_start: Tuple[int, int] | None = None
    summary_filter_applied = False

    def start_section(key: str, line: Dict[str, Any]) -> None:
        nonlocal current_key, current_heading, current_lines
        current_key = key
        current_heading = HEADINGS[key]
        current_lines = [
            {"page": line["page"], "line": line["line"], "text": line["text"]}
        ]
        logger.info(
            "Detected start of %s at (page=%s,line=%s)",
            current_heading,
            line["page"],
            line["line"],
        )

    def append_line(line: Dict[str, Any]) -> None:
        current_lines.append(
            {"page": line["page"], "line": line["line"], "text": line["text"]}
        )

    def close_section() -> None:
        nonlocal current_key, current_heading, current_lines
        if not current_key or not current_lines:
            current_key = None
            current_heading = None
            current_lines = []
            return
        last = current_lines[-1]
        sections.append(
            {
                "section_index": len(sections),
                "heading": current_heading,
                "page_start": current_lines[0]["page"],
                "line_start": current_lines[0]["line"],
                "page_end": last["page"],
                "line_end": last["line"],
                "lines": current_lines.copy(),
            }
        )
        logger.info(
            "Closed %s at (page=%s,line=%s)",
            current_heading,
            last["page"],
            last["line"],
        )
        current_key = None
        current_heading = None
        current_lines = []

    def match(key: str, norm: str) -> bool:
        """Return True if ``norm`` matches any marker for ``key``."""

        return any(norm == m or m in norm for m in MARKERS[key])

    # ------------------------------------------------------------------
    # Anchor detection for the Summary range
    for line in lines:
        norm = line["text_norm"]
        if summary_start is None and match("summary", norm):
            summary_start = (line["page"], line["line"])
        if acct_hist_start is None and match("account_history", norm):
            acct_hist_start = (line["page"], line["line"])
        if summary_start and acct_hist_start:
            break

    if summary_start and acct_hist_start:
        summary_filter_applied = True

    def detect_heading(line: Dict[str, Any]) -> str | None:
        """Return the canonical heading key for ``line`` if it matches."""

        norm = line["text_norm"]
        for key, markers in MARKERS.items():
            if any(norm == m or m in norm for m in markers):
                if summary_filter_applied and key in {"public_info", "inquiries"}:
                    pos = (line["page"], line["line"])
                    if summary_start <= pos < acct_hist_start:  # type: ignore[arg-type]
                        return None
                return key
        return None

    for line in lines:
        key = detect_heading(line)
        if key:
            role = HEADING_ROLES[key]
            if role == "emit":
                close_section()
                start_section(key, line)
            elif role == "anchor_only":
                close_section()
            elif role == "stop":
                close_section()
                break
            continue

        if current_key:
            append_line(line)

    # Close any trailing section if we reach EOF without an end marker.
    if current_key:
        close_section()

    result = {"sections": sections, "summary_filter_applied": summary_filter_applied}
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "Wrote general info sections to %s (%d sections)", json_out, len(sections)
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Split general information sections from the full TSV"
    )
    ap.add_argument("--full", required=True, help="Path to _debug_full.tsv")
    ap.add_argument(
        "--json_out", required=True, help="Path to write general_info_from_full.json"
    )
    args = ap.parse_args(argv)

    tsv_path = Path(args.full)
    json_out = Path(args.json_out)
    split_general_info(tsv_path, json_out)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
