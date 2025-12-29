from __future__ import annotations

from collections.abc import Mapping, Iterable


def _json_safe(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj

