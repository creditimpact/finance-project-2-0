import argparse
import pathlib
import sys
import time
from typing import Dict, Any

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from services.outcome_ingestion.ingest_report import ingest_report


def _make_report(idx: int) -> Dict[str, Any]:
    return {
        "Experian": {
            "accounts": [
                {
                    "name": f"Cred{idx}",
                    "account_number": str(idx),
                    "account_id": str(idx),
                    "balance": idx * 10,
                }
            ]
        }
    }


def benchmark(count: int) -> None:
    start = time.perf_counter()
    for i in range(count):
        ingest_report(None, _make_report(i))
    duration = time.perf_counter() - start
    rps = count / duration if duration else float("inf")
    latency_ms = (duration / count) * 1000 if count else 0
    print(f"Processed {count} reports in {duration:.2f}s -> {rps:.2f} rps ({latency_ms:.2f} ms/report)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark report ingestion throughput")
    parser.add_argument("-n", "--num-reports", type=int, default=100, help="Number of synthetic reports to ingest")
    args = parser.parse_args()
    benchmark(args.num_reports)
