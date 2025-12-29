"""Environment configuration for tradeline_check module."""

from __future__ import annotations

import os


def _bool_env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}


class TradlineCheckConfig:
    """Runtime configuration for tradeline_check module."""

    def __init__(self):
        self.enabled = _bool_env_flag("TRADELINE_CHECK_ENABLED", default=False)
        self.write_debug = _bool_env_flag("TRADELINE_CHECK_WRITE_DEBUG", default=False)
        # Q6 gate behavior flags
        self.gate_strict = _bool_env_flag("TRADELINE_CHECK_GATE_STRICT", default=False)
        self.write_empty_results = _bool_env_flag(
            "TRADELINE_CHECK_WRITE_EMPTY_RESULTS", default=False
        )
        # CSV of placeholder tokens (treated as missing)
        self.placeholder_tokens = _parse_placeholder_tokens(
            os.getenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS")
        )

    @classmethod
    def from_env(cls) -> TradlineCheckConfig:
        """Create config from environment variables."""
        return cls()


def _parse_placeholder_tokens(raw: str | None, default_csv: str = "--,") -> set[str]:
    """Parse CSV of placeholder tokens into a normalized set.

    Examples: "--,N/A,UNKNOWN" -> {"--", "n/a", "unknown"}
    Empty values and whitespace-only entries are ignored.
    """
    csv = raw if (raw is not None and raw.strip() != "") else default_csv
    tokens: set[str] = set()
    for part in csv.split(","):
        cleaned = part.strip()
        if cleaned:
            tokens.add(cleaned.lower())
    return tokens
