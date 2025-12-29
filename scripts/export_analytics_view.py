import argparse
import sqlite3
import sys

from backend.analytics.view_exporter import (
    ExportFilters,
    fetch_joined,
    stream_csv,
    stream_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export analytics planner view")
    parser.add_argument("--db", required=True, help="Path to analytics database")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--tag", action="append", dest="action_tags")
    parser.add_argument("--family", action="append", dest="family_ids")
    parser.add_argument("--cycle-min", type=int)
    parser.add_argument("--cycle-max", type=int)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--view-name", default="analytics_planner_outcomes")
    args = parser.parse_args()

    cycle_range = None
    if args.cycle_min is not None and args.cycle_max is not None:
        cycle_range = (args.cycle_min, args.cycle_max)

    filters = ExportFilters(
        action_tags=args.action_tags,
        family_ids=args.family_ids,
        cycle_range=cycle_range,
        start_ts=args.start,
        end_ts=args.end,
    )

    conn = sqlite3.connect(args.db)
    try:
        rows = fetch_joined(conn, filters, view_name=args.view_name)
        streamer = stream_csv if args.format == "csv" else stream_json
        for chunk in streamer(rows):
            sys.stdout.write(chunk)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
