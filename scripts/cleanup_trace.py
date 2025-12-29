import argparse
from pathlib import Path

from backend.core.logic.report_analysis.trace_cleanup import purge_after_export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    summary = purge_after_export(args.sid, Path(args.root))
    print(summary)


if __name__ == "__main__":
    main()
