"""Runtime flags controlling report analysis behavior.

Each flag can be overridden via an environment variable of the same name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class AnalysisFlags:
    """Container for report analysis feature flags."""

    chunk_by_bureau: bool = _env_bool("ANALYSIS_CHUNK_BY_BUREAU", True)
    inject_headings: bool = _env_bool("ANALYSIS_INJECT_HEADINGS", True)
    cache_enabled: bool = _env_bool("ANALYSIS_CACHE_ENABLED", True)
    debug_store_raw: bool = _env_bool("ANALYSIS_DEBUG_STORE_RAW", False)
    max_remediation_passes: int = _env_int("ANALYSIS_MAX_REMEDIATION_PASSES", 3)
    max_segment_tokens: int = _env_int("ANALYSIS_MAX_SEGMENT_TOKENS", 8000)


FLAGS = AnalysisFlags()
