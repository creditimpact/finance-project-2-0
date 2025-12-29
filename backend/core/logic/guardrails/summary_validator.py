import re
from typing import Any, Dict

_ADMISSION = re.compile(
    r"\b(i\s+admit|i\s+acknowledge|it\s+was\s+my\s+account|i\s+am\s+responsible)\b",
    re.I,
)
_EMOTION = re.compile(
    r"\b(i\s+am\s+angry|i\s+am\s+devastated|i\s+will\s+sue|demand\s+deletion)\b", re.I
)


def validate_structured_summaries(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        entry = dict(v)
        para = str(entry.get("paragraph", "")).strip()
        if _ADMISSION.search(para) or _EMOTION.search(para):
            entry["paragraph"] = (
                "I request accurate reporting and clarification under applicable law."
            )
            entry["flagged"] = True
        cleaned[k] = entry
    return cleaned
