import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.report_analysis.process_accounts import (
    process_analyzed_report,
    save_bureau_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a SmartCredit analysis report into bureau payloads"
    )
    parser.add_argument("input", type=Path, help="Path to analyzed report JSON")
    parser.add_argument(
        "output", type=Path, help="Directory where payloads will be written"
    )
    args = parser.parse_args()

    result = process_analyzed_report(args.input)
    save_bureau_outputs(result, args.output)
    print("[✅] Bureau-level segmentation complete.")


if __name__ == "__main__":
    main()
