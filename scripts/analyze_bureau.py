#!/usr/bin/env python3
"""CLI tool to manually run bureau analysis for a text segment."""

import argparse
import json
from pathlib import Path

from backend.core.logic.report_analysis.report_prompting import analyze_bureau, _generate_prompt
from backend.core.services.ai_client import get_ai_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a bureau text segment")
    parser.add_argument("text_file", help="Path to text file containing the bureau segment")
    parser.add_argument("output", help="Where to write analysis JSON")
    parser.add_argument("--identity-theft", action="store_true", dest="identity_theft")
    parser.add_argument("--strategic-context", dest="strategic_context")
    parser.add_argument(
        "--expected-account",
        action="append",
        dest="expected_accounts",
        default=[],
        help="Expected account name (repeatable)",
    )
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    client = get_ai_client()

    prompt, late_summary, inquiry_summary = _generate_prompt(
        text,
        is_identity_theft=args.identity_theft,
        strategic_context=args.strategic_context,
    )
    hints = {}
    if args.expected_accounts:
        hints["expected_account_names"] = args.expected_accounts

    data, error = analyze_bureau(
        text,
        is_identity_theft=args.identity_theft,
        output_json_path=Path(args.output),
        ai_client=client,
        strategic_context=args.strategic_context,
        prompt=prompt,
        late_summary_text=late_summary,
        inquiry_summary=inquiry_summary,
        hints=hints or None,
    )
    result = {"data": data, "error_code": error}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual tool
    main()
