import argparse
from pathlib import Path

from backend.core.logic.report_analysis.smartcredit_template_orchestrator import (
    run_template_first,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SmartCredit template tables from Stage-A artifacts")
    parser.add_argument("--sid", required=True, help="Session ID (folder under traces/blocks)")
    args = parser.parse_args()

    result = run_template_first(args.sid, Path.cwd())
    print("TEMPLATE_OK:", result)


if __name__ == "__main__":
    main()

