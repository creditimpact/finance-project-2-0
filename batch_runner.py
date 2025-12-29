import argparse
from pathlib import Path

from backend.analytics.batch_runner import BatchFilters, BatchRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or schedule batch analytics jobs")
    parser.add_argument("--schedule", help="Cron expression to schedule a job")
    parser.add_argument(
        "--tag", action="append", dest="action_tags", help="Action tag filter"
    )
    parser.add_argument("--family", action="append", dest="family_ids")
    parser.add_argument("--start_ts", type=int)
    parser.add_argument("--end_ts", type=int)
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--job-store", default="backend/analytics/batch_jobs.sqlite")
    parser.add_argument("--output-dir", default="backend/analytics/batch_reports")
    args = parser.parse_args()

    runner = BatchRunner(
        job_store=Path(args.job_store), output_dir=Path(args.output_dir)
    )

    if args.schedule:
        job_id = runner.schedule(args.schedule)
        print(job_id)
        return

    if not args.action_tags:
        parser.error("--tag required when not scheduling")

    filters = BatchFilters(
        action_tags=args.action_tags,
        family_ids=args.family_ids,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )
    job_id = runner.run(filters, args.format)
    print(job_id)


if __name__ == "__main__":
    main()
