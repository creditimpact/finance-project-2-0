import json
import re
from typing import Any, Dict


def try_fix_to_json(text: str | None) -> Dict[str, Any] | None:
    """Extract and parse a JSON object from ``text`` if possible.

    Args:
        text: Raw model output that should contain a JSON object.

    Returns:
        dict | None: Parsed JSON object when extraction succeeds, otherwise ``None``.

    Raises:
        ValueError: If ``text`` is ``None``.
    """

    if text is None:
        raise ValueError("Cannot coerce JSON from None input")

    if not isinstance(text, str):  # Defensive guard for unexpected payloads.
        return None

    stripped_text = text.strip()
    if not stripped_text:
        return None

    candidates = []
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidates.append(fenced.group(1))

    loose = re.search(r"(\{.*\})", text, flags=re.S)
    if loose:
        candidates.append(loose.group(1))

    candidates.append(text)

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
