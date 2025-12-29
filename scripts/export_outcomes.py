import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Allow importing the project modules when this script is run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.letters.outcomes_store import get_outcomes

EXPORT_DIR = Path("exports")


def export_outcomes() -> None:
    """Export outcomes from the last 7 days to JSON and CSV files."""
    now = datetime.now(UTC)
    start = now - timedelta(days=7)
    all_outcomes = get_outcomes()
    recent = []
    for o in all_outcomes:
        ts = o.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if start <= dt <= now:
            recent.append(o)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = now.strftime("%Y-%m-%d")
    json_path = EXPORT_DIR / f"outcomes_{date_str}.json"
    csv_path = EXPORT_DIR / f"outcomes_{date_str}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recent, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "timestamp",
        "session_id",
        "account_id",
        "bureau",
        "letter_version",
        "result",
        "days_to_response",
        "reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recent)


if __name__ == "__main__":
    export_outcomes()
