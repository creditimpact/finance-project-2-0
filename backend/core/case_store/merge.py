from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Dict


def _merge_lists(base: List[Any], patch: List[Any]) -> List[Any]:
    """Merge two lists according to safe_deep_merge semantics."""
    # Keyed merge for payment_history by 'date'
    if all(isinstance(item, dict) and "date" in item for item in base + patch):
        result = [deepcopy(item) for item in base]
        index_map = {
            item["date"]: idx for idx, item in enumerate(base) if isinstance(item, dict) and "date" in item
        }
        for p_item in patch:
            key = p_item["date"]
            if key in index_map:
                idx = index_map[key]
                result[idx] = safe_deep_merge(result[idx], p_item)
            else:
                result.append(deepcopy(p_item))
        return result

    # Fallback: union preserving order, deduplicating by value
    result: List[Any] = []
    for item in base:
        result.append(deepcopy(item))
    for item in patch:
        if item not in result:
            result.append(deepcopy(item))
    return result


def safe_deep_merge(base: Any, patch: Any) -> Any:
    """Safely merge ``patch`` into ``base`` without mutating inputs."""
    if patch is None:
        return deepcopy(base)

    if isinstance(base, dict) and isinstance(patch, dict):
        result: Dict[Any, Any] = {}
        for key in base.keys() | patch.keys():
            if key in base and key in patch:
                result[key] = safe_deep_merge(base[key], patch[key])
            elif key in base:
                result[key] = deepcopy(base[key])
            else:
                result[key] = deepcopy(patch[key])
        return result

    if isinstance(base, list) and isinstance(patch, list):
        return _merge_lists(base, patch)

    if isinstance(base, (dict, list)) and not isinstance(patch, (dict, list)):
        return deepcopy(base)

    if not isinstance(base, (dict, list)) and isinstance(patch, (dict, list)):
        return deepcopy(patch)

    # Scalars
    if patch in (None, "") and base not in (None, ""):
        return deepcopy(base)
    return deepcopy(patch)
