#!/usr/bin/env python
"""Helper to verify blocks_detected vs accounts persisted."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.case_store import api
from backend.core.logic.report_analysis.extractors import accounts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("input", type=Path, help="Path to report lines text")
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    block_count = len(accounts._split_blocks(lines))
    case = api.load_session_case(args.session_id)
    account_count = len(case.accounts)
    print(f"VERIFY 1:1 blocks_detected={block_count} accounts_persisted={account_count}")


if __name__ == "__main__":
    main()
