import json
import os
import threading
from datetime import UTC, datetime
from typing import Dict, List, Optional

from backend.assets.paths import data_path

OUTCOMES_FILE = data_path("outcomes.json")
_lock = threading.Lock()


def _load_outcomes() -> List[Dict]:
    if not os.path.exists(OUTCOMES_FILE):
        return []
    try:
        with open(OUTCOMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _write_outcomes(outcomes: List[Dict]) -> None:
    os.makedirs(os.path.dirname(OUTCOMES_FILE), exist_ok=True)
    tmp_path = OUTCOMES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(outcomes, f)
    os.replace(tmp_path, OUTCOMES_FILE)


def record_outcome(
    session_id: str,
    account_id: str,
    bureau: str,
    letter_version: int,
    result: str,
    days_to_response: Optional[int] | None = None,
    reason: Optional[str] | None = None,
) -> None:
    """Append an outcome record with timestamp to persistent storage."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "account_id": account_id,
        "bureau": bureau,
        "letter_version": letter_version,
        "result": result,
        "days_to_response": days_to_response,
        "reason": reason,
    }
    with _lock:
        outcomes = _load_outcomes()
        outcomes.append(record)
        _write_outcomes(outcomes)


def get_outcomes(filter_by: Optional[Dict] | None = None) -> List[Dict]:
    """Return stored outcomes optionally filtered by keys (e.g., bureau, result)."""
    with _lock:
        outcomes = list(_load_outcomes())
    if not filter_by:
        return outcomes
    return [o for o in outcomes if all(o.get(k) == v for k, v in filter_by.items())]
