from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def to_json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable primitives.

    - Dates → ISO string
    - Decimal → float
    - set/tuple → list
    - dict → dict with stringified keys, values sanitized
    - Fallback: str(obj)
    """

    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    # Fallback for custom objects
    return str(obj)

