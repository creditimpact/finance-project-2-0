from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def extract_metrics(path: Path) -> dict[str, float]:
    start = time.perf_counter()
    data = json.loads(path.read_text())
    latency = time.perf_counter() - start
    metrics = {
        "accounts_found": len(data.get("accounts", [])),
        "inquiries_found": len(data.get("inquiries", [])),
        "dup_removed": len(data.get("duplicates_removed", [])),
        "latency": latency,
        "cost": float(data.get("cost", 0)),
    }
    return metrics


def aggregate(results: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for key in results[0].keys():
        aggregated[key] = statistics.fmean(r[key] for r in results)
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect evaluation metrics for report files"
    )
    parser.add_argument(
        "reports", nargs="+", type=Path, help="Paths to sampled report JSON files"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to write aggregated metrics JSON"
    )
    args = parser.parse_args()
    metrics = [extract_metrics(p) for p in args.reports]
    aggregated = aggregate(metrics)
    output = json.dumps(aggregated, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output)
