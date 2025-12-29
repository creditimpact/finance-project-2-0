"""Helpers for loading merge configuration from environment variables.

This module centralizes all parsing logic for ``MERGE_*`` environment variables
so that merge and deduplication behaviour can be controlled without code
changes. It ensures values such as booleans, numbers, and JSON payloads are
converted to native Python types and provides sensible defaults when variables
are missing.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Base prefix for all merge related environment variables.
MERGE_PREFIX = "MERGE_"

# Default configuration keeps backward compatible behaviour when flags are not
# provided via environment variables.
DEFAULT_FIELDS: List[str] = [
    "account_number",
    "date_opened",
    "balance_owed",
    "account_type",
    "account_status",
    "history_2y",
    "history_7y",
]

logger = logging.getLogger(__name__)


VALID_FIELDS: Set[str] = set(DEFAULT_FIELDS) | {"account_label"}


POINTS_MODE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "account_number": 1.0,
    "date_opened": 1.0,
    "balance_owed": 3.0,
    "account_type": 0.5,
    "account_status": 0.5,
    "history_2y": 1.0,
    "history_7y": 1.0,
    "account_label": 1.0,
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "fields": list(DEFAULT_FIELDS),
    # Allowlist defaults mirror the historic field sequence so enforcement can
    # be toggled on without requiring explicit overrides.
    "fields_override": list(DEFAULT_FIELDS),
    "allowlist_enforce": False,
    # Debug logging remains opt-in so production noise does not increase.
    "debug": False,
    # ``log_every`` controls the sampling cadence for debug logs when enabled.
    "log_every": 0,
    # Custom weights are opt-in to preserve legacy scoring when disabled.
    "use_custom_weights": False,
    # Optional merge fields stay disabled until toggled via MERGE_USE_* flags.
    "use_original_creditor": False,
    "use_creditor_name": False,
    "require_original_creditor_for_ai": False,
    "weights": {},
    "thresholds": {},
    "overrides": {},
    # Points-based scoring is disabled by default so legacy behaviour remains
    # active unless the new flag is explicitly enabled.
    "points_mode": False,
    "ai_points_threshold": 3.0,
    "direct_points_threshold": 5.0,
    # Emit per-pair diagnostics for the first N comparisons when points mode
    # is active so behaviour can be audited without enabling verbose logging.
    "points_diagnostics_limit": 3,
    "points_diagnostics": False,
    "points_persist_breakdown": False,
    "points_diag_dir": "ai_packs/merge/diagnostics",
    # Optional account label support remains disabled unless explicitly opted in.
    "use_account_label": False,
    "account_label_source": "meta.heading_guess",
    "account_label_normalize": True,
}

ALLOWLIST_FIELDS: List[str] = list(DEFAULT_FIELDS) + ["account_label"]

_MERGE_CONFIG_LOGGED = False


class InvalidConfigError(ValueError):
    """Raised when an environment variable contains an invalid value."""

    def __init__(self, key: str, message: str) -> None:
        self.key = key
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - representational only
        return f"{self.args[0]} ({self.key})"


class MergeConfig(Mapping[str, Any]):
    """Mapping-compatible wrapper exposing structured merge configuration."""

    def __init__(
        self,
        raw: Dict[str, Any],
        *,
        points_mode: bool,
        ai_points_threshold: float,
        direct_points_threshold: float,
        allowlist_enforce: bool,
        fields: List[str],
        weights: Dict[str, float],
        tolerances: Dict[str, Any],
        points_diagnostics_limit: int,
        points_diagnostics: bool,
        points_persist_breakdown: bool,
        points_diag_dir: str,
        use_account_label: bool,
        account_label_source: Optional[str],
        account_label_normalize: bool,
        require_original_creditor_for_ai: bool,
    ) -> None:
        self._raw: Dict[str, Any] = dict(raw)
        self.points_mode = points_mode
        self.ai_points_threshold = ai_points_threshold
        self.direct_points_threshold = direct_points_threshold
        self.allowlist_enforce = allowlist_enforce
        self.fields = list(fields)
        self.weights = dict(weights)
        self.tolerances = dict(tolerances)
        self.points_diagnostics_limit = int(max(points_diagnostics_limit, 0))
        self.points_diagnostics = bool(points_diagnostics)
        self.points_persist_breakdown = bool(points_persist_breakdown)
        self.points_diag_dir = str(points_diag_dir)
        self.use_account_label = bool(use_account_label)
        self.account_label_source = (
            str(account_label_source).strip()
            if account_label_source is not None
            else ""
        )
        self.account_label_normalize = bool(account_label_normalize)
        self.require_original_creditor_for_ai = bool(require_original_creditor_for_ai)

        # Surface structured values through the mapping interface for backward
        # compatibility with existing dictionary-based access patterns.
        self._raw.update(
            {
                "points_mode": self.points_mode,
                "ai_points_threshold": self.ai_points_threshold,
                "direct_points_threshold": self.direct_points_threshold,
                "allowlist_enforce": self.allowlist_enforce,
                "fields": list(self.fields),
                "weights": dict(self.weights),
                "tolerances": dict(self.tolerances),
                "points_diagnostics_limit": self.points_diagnostics_limit,
                "points_diagnostics": self.points_diagnostics,
                "points_persist_breakdown": self.points_persist_breakdown,
                "points_diag_dir": self.points_diag_dir,
                "use_account_label": self.use_account_label,
                "account_label_source": self.account_label_source,
                "account_label_normalize": self.account_label_normalize,
                "require_original_creditor_for_ai": self.require_original_creditor_for_ai,
            }
        )

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def keys(self) -> Iterable[str]:
        return self._raw.keys()

    def items(self):
        return self._raw.items()

    def values(self):
        return self._raw.values()



def _parse_env_value(env_key: str, raw_value: str) -> Any:
    """Convert a raw environment string into an appropriate Python type."""

    value = raw_value.strip()

    # Automatically decode *_JSON variables first to support structured config.
    if env_key.endswith("_JSON"):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidConfigError(
                env_key,
                f"Invalid JSON for {env_key}: {exc.msg}",
            ) from exc

    lowered = value.lower()
    if lowered in {"true", "false"}:
        # Translate human-friendly boolean strings into actual booleans.
        return lowered == "true"

    # Attempt integer parsing before floats to retain whole number semantics.
    if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
        try:
            return int(lowered)
        except ValueError:
            pass

    # Support floats that include decimal points or scientific notation.
    try:
        if any(char in lowered for char in [".", "e"]):
            return float(value)
    except ValueError:
        pass

    # Allow comma separated strings to represent field lists when JSON is not used.
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]

    # Fallback: retain original string for unhandled cases.
    return raw_value


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert arbitrary inputs into a boolean with a sensible default."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert input to float when possible, otherwise return the default."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Convert input to integer when possible, otherwise return the default."""

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _normalize_fields(value: Any) -> List[str]:
    """Normalize any field override input into a clean list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [segment.strip() for segment in value.split(",") if segment.strip()]
    elif isinstance(value, (set, tuple)):
        value = list(value)
    if isinstance(value, list):
        seen: Set[str] = set()
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            field = str(item).strip()
            if not field or field in seen:
                continue
            seen.add(field)
            normalized.append(field)
        return normalized
    return []


def _normalize_weights(
    raw_weights: Any,
    *,
    allowlist_enforce: bool,
    fields: List[str],
    default_weight: Optional[float] = 1.0,
) -> Dict[str, float]:
    """Normalize configured weights into a float mapping keyed by field name."""

    weights: Dict[str, float] = {}
    if isinstance(raw_weights, Mapping):
        for key, value in raw_weights.items():
            field = str(key).strip()
            if not field:
                continue
            if allowlist_enforce and field not in ALLOWLIST_FIELDS:
                continue
            coerced = _coerce_float(value)
            if coerced is None:
                continue
            weights[field] = float(coerced)

    resolved: Dict[str, float] = {}
    for field in fields:
        if field in weights:
            resolved[field] = weights[field]
        elif default_weight is not None:
            resolved[field] = float(default_weight)

    return resolved


def _parse_weights_map(
    raw_weights: Any, *, valid_fields: Sequence[str]
) -> Tuple[Dict[str, float], List[str]]:
    """Return sanitized weight mapping restricted to ``valid_fields``."""

    if not isinstance(raw_weights, Mapping):
        return {}, []

    allowed = {str(field).strip() for field in valid_fields if str(field).strip()}
    weights: Dict[str, float] = {}
    invalid: List[str] = []

    for key, value in raw_weights.items():
        if key is None:
            continue
        field = str(key).strip()
        if not field:
            continue
        coerced = _coerce_float(value)
        if coerced is None:
            continue
        if field not in allowed:
            invalid.append(field)
            continue
        weights[field] = float(coerced)

    return weights, invalid


def _resolve_fields(
    *,
    allowlist_enforce: bool,
    override: List[str],
    configured_fields: List[str],
) -> List[str]:
    """Resolve the effective field list respecting allowlist enforcement."""

    allowed = set(ALLOWLIST_FIELDS)
    source_fields = override if override else configured_fields
    if not source_fields:
        source_fields = list(ALLOWLIST_FIELDS)

    resolved = [field for field in source_fields if field in allowed]

    if not allowlist_enforce:
        for field in ALLOWLIST_FIELDS:
            if field not in resolved:
                resolved.append(field)

    return resolved


def _resolve_tolerances(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect tolerance-related values from the raw environment configuration."""

    source = config.get("tolerances")
    if not isinstance(source, Mapping):
        source = {}

    def _lookup(key: str, fallback_key: str) -> Any:
        if key in source:
            return source.get(key)
        return config.get(fallback_key)

    return {
        "MERGE_TOL_DATE_DAYS": _coerce_int(
            _lookup("MERGE_TOL_DATE_DAYS", "tol_date_days")
        ),
        "MERGE_TOL_BALANCE_ABS": _coerce_float(
            _lookup("MERGE_TOL_BALANCE_ABS", "tol_balance_abs")
        ),
        "MERGE_TOL_BALANCE_RATIO": _coerce_float(
            _lookup("MERGE_TOL_BALANCE_RATIO", "tol_balance_ratio")
        ),
        "MERGE_ACCOUNTNUMBER_MATCH_MINLEN": _coerce_int(
            _lookup("MERGE_ACCOUNTNUMBER_MATCH_MINLEN", "accountnumber_match_minlen")
        ),
        "MERGE_HISTORY_SIMILARITY_THRESHOLD": _coerce_float(
            _lookup("MERGE_HISTORY_SIMILARITY_THRESHOLD", "history_similarity_threshold")
        ),
    }


def _build_merge_config() -> Dict[str, Any]:
    """Construct the merge configuration from the dedicated ``MERGE_*`` block."""

    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    present_keys: Set[str] = set()
    raw_values: Dict[str, str] = {}
    issues: List[str] = []

    for key, raw_value in os.environ.items():
        if not key.startswith(MERGE_PREFIX):
            continue

        short_key = key[len(MERGE_PREFIX) :].lower()
        raw_values[short_key] = raw_value

        try:
            parsed_value = _parse_env_value(key, raw_value)
        except InvalidConfigError as exc:
            logger.warning("[MERGE] %s", exc)
            issues.append(str(exc))
            continue

        # Normalize *_json keys to expose cleaner dictionary names (e.g., weights).
        if short_key.endswith("_json"):
            trimmed_key = short_key[:-5]
            # Preserve the original raw string under the normalized key so
            # validation logic can surface accurate diagnostics when only the
            # *_JSON variant is provided.
            raw_values.setdefault(trimmed_key, raw_value)
            short_key = trimmed_key

        # Update the runtime configuration using the normalized key.
        config[short_key] = parsed_value
        # Track which keys were explicitly provided so that callers can
        # distinguish default values from environment overrides.
        present_keys.add(short_key)

    config["_present_keys"] = frozenset(present_keys)
    config["_raw_values"] = dict(raw_values)
    config["_issues"] = tuple(issues)

    return config


def _create_structured_config(raw_config: Dict[str, Any]) -> MergeConfig:
    """Produce the structured ``MergeConfig`` wrapper from raw environment data."""

    present_keys = set(raw_config.get("_present_keys", frozenset()))
    raw_values: Dict[str, str] = dict(raw_config.get("_raw_values", {}))
    validation_errors: List[str] = list(raw_config.get("_issues", []))
    validation_warnings: List[str] = []

    allowlist_enforce = _coerce_bool(raw_config.get("allowlist_enforce"))
    points_mode = _coerce_bool(raw_config.get("points_mode"))
    ai_points_threshold = _coerce_float(raw_config.get("ai_points_threshold"), 3.0) or 3.0
    direct_points_threshold = (
        _coerce_float(raw_config.get("direct_points_threshold"), 5.0) or 5.0
    )
    diagnostics_limit = _coerce_int(
        raw_config.get("points_diagnostics_limit"),
        int(DEFAULT_CONFIG["points_diagnostics_limit"]),
    )
    if diagnostics_limit is None:
        diagnostics_limit = int(DEFAULT_CONFIG["points_diagnostics_limit"])
    diagnostics_limit = max(int(diagnostics_limit), 0)

    points_diagnostics_flag = _coerce_bool(
        raw_config.get("points_diagnostics"),
        bool(DEFAULT_CONFIG.get("points_diagnostics", False)),
    )

    points_persist_breakdown_flag = _coerce_bool(
        raw_config.get("points_persist_breakdown"),
        bool(DEFAULT_CONFIG.get("points_persist_breakdown", False)),
    )

    points_diag_dir_raw = raw_config.get("points_diag_dir")
    if isinstance(points_diag_dir_raw, str) and points_diag_dir_raw.strip():
        points_diag_dir_value = points_diag_dir_raw.strip()
    elif points_diag_dir_raw is None:
        points_diag_dir_value = str(DEFAULT_CONFIG.get("points_diag_dir", "ai_packs/merge/diagnostics"))
    else:
        points_diag_dir_value = str(points_diag_dir_raw).strip()
        if not points_diag_dir_value:
            points_diag_dir_value = str(DEFAULT_CONFIG.get("points_diag_dir", "ai_packs/merge/diagnostics"))

    use_account_label = _coerce_bool(raw_config.get("use_account_label"))
    require_original_creditor_for_ai = _coerce_bool(
        raw_config.get("require_original_creditor_for_ai"),
        bool(DEFAULT_CONFIG["require_original_creditor_for_ai"]),
    )
    account_label_source_raw = raw_config.get("account_label_source")
    if isinstance(account_label_source_raw, str):
        account_label_source = account_label_source_raw.strip()
    elif account_label_source_raw is None:
        account_label_source = ""
    else:
        account_label_source = str(account_label_source_raw).strip()
    account_label_normalize = _coerce_bool(
        raw_config.get("account_label_normalize"),
        bool(DEFAULT_CONFIG["account_label_normalize"]),
    )

    fields_override = _normalize_fields(raw_config.get("fields_override"))
    recognized_override_fields = [field for field in fields_override if field in VALID_FIELDS]

    if "fields_override" in present_keys:
        raw_override = raw_values.get("fields_override", "")
        if not raw_override or not raw_override.strip():
            validation_errors.append(
                "MERGE_FIELDS_OVERRIDE must be a non-empty comma-separated list of merge fields."
            )
        unknown_fields = sorted(set(fields_override) - VALID_FIELDS)
        if unknown_fields:
            validation_warnings.append(
                "MERGE_FIELDS_OVERRIDE includes unrecognized fields: %s" % ", ".join(unknown_fields)
            )

    configured_fields = _normalize_fields(raw_config.get("fields"))
    effective_allowlist_enforce = allowlist_enforce or points_mode

    fields = _resolve_fields(
        allowlist_enforce=effective_allowlist_enforce,
        override=fields_override,
        configured_fields=configured_fields,
    )

    use_custom_weights = _coerce_bool(raw_config.get("use_custom_weights"))
    raw_weights: Any = raw_config.get("weights")
    if isinstance(raw_weights, Mapping):
        weights_candidate: Any = dict(raw_weights)
    elif raw_weights is None:
        weights_candidate = {}
    else:
        validation_errors.append(
            "MERGE_WEIGHTS_JSON must be a JSON object mapping fields to numeric weights."
        )
        weights_candidate = {}

    override_field_set: Set[str] = (
        set(recognized_override_fields) if recognized_override_fields else set(fields)
    )
    removed_weight_keys: Set[str] = set()
    if weights_candidate:
        extra_weight_keys = sorted(
            {str(key).strip() for key in weights_candidate.keys()} - override_field_set
        )
        if extra_weight_keys:
            validation_warnings.append(
                "MERGE_WEIGHTS_JSON provided weights for fields outside MERGE_FIELDS_OVERRIDE: %s"
                % ", ".join(extra_weight_keys)
            )
            removed_weight_keys.update(extra_weight_keys)
            for key in list(weights_candidate.keys()):
                if str(key).strip() in extra_weight_keys:
                    weights_candidate.pop(key, None)

    weights_from_env, invalid_weight_keys = _parse_weights_map(
        weights_candidate,
        valid_fields=recognized_override_fields or fields,
    )
    dropped_weight_keys = set(invalid_weight_keys) | removed_weight_keys
    if dropped_weight_keys:
        logger.warning(
            "[MERGE] Ignoring weights for non-configured fields: %s",
            sorted(dropped_weight_keys),
        )

    weights_source: Dict[str, float] = {}
    if points_mode:
        weights_source = dict(weights_from_env)
        if not weights_source:
            weights_source = {
                field: weight
                for field, weight in POINTS_MODE_DEFAULT_WEIGHTS.items()
                if field in fields
            }
    elif use_custom_weights:
        weights_source = dict(weights_from_env)

    if points_mode:
        weights = {
            field: float(weights_source[field])
            for field in fields
            if field in weights_source
        }
    else:
        weights = _normalize_weights(
            weights_source,
            allowlist_enforce=effective_allowlist_enforce,
            fields=fields,
            default_weight=1.0,
        )

    tolerances = _resolve_tolerances(raw_config)

    processed_raw = dict(raw_config)
    processed_raw.pop("_present_keys", None)
    processed_raw.pop("_raw_values", None)
    processed_raw.pop("_issues", None)
    processed_raw["use_custom_weights"] = bool(
        raw_config.get("use_custom_weights")
    ) or bool(points_mode)
    processed_raw["fields_override"] = fields_override
    processed_raw["use_account_label"] = use_account_label
    processed_raw["account_label_source"] = account_label_source
    processed_raw["account_label_normalize"] = account_label_normalize
    processed_raw["require_original_creditor_for_ai"] = require_original_creditor_for_ai
    processed_raw["points_diagnostics"] = points_diagnostics_flag
    processed_raw["points_persist_breakdown"] = points_persist_breakdown_flag
    processed_raw["points_diag_dir"] = points_diag_dir_value

    for warning in validation_warnings:
        logger.warning("[MERGE] %s", warning)

    if validation_errors:
        message = "; ".join(validation_errors)
        if points_mode:
            logger.error("[MERGE] %s", message)
            raise ValueError(f"Invalid merge configuration: {message}")
        logger.warning("[MERGE] %s", message)

    global _MERGE_CONFIG_LOGGED
    if not _MERGE_CONFIG_LOGGED:
        message = (
            "[MERGE] Config points_mode=%s ai_threshold=%.2f direct_threshold=%.2f "
            "fields=%s weights=%s tolerances=%s diagnostics_limit=%s diagnostics=%s"
            % (
                points_mode,
                float(ai_points_threshold),
                float(direct_points_threshold),
                fields,
                weights,
                tolerances,
                diagnostics_limit,
                points_diagnostics_flag,
            )
        )
        logger.info(message)
        print(message)
        logger.info(
            "[MERGE] points_diagnostics=%s limit=%s oc_gate=%s persist_breakdown=%s diag_dir=%s",
            "on" if points_diagnostics_flag else "off",
            diagnostics_limit,
            "on" if require_original_creditor_for_ai else "off",
            "on" if points_persist_breakdown_flag else "off",
            points_diag_dir_value,
        )
        _MERGE_CONFIG_LOGGED = True

    structured = MergeConfig(
        processed_raw,
        points_mode=points_mode,
        ai_points_threshold=float(ai_points_threshold),
        direct_points_threshold=float(direct_points_threshold),
        allowlist_enforce=allowlist_enforce,
        fields=fields,
        weights=weights,
        tolerances=tolerances,
        points_diagnostics_limit=diagnostics_limit,
        points_diagnostics=points_diagnostics_flag,
        points_persist_breakdown=points_persist_breakdown_flag,
        points_diag_dir=points_diag_dir_value,
        use_account_label=use_account_label,
        account_label_source=account_label_source,
        account_label_normalize=account_label_normalize,
        require_original_creditor_for_ai=require_original_creditor_for_ai,
    )

    setattr(structured, "weights_map", dict(weights_from_env))

    return structured


@lru_cache(maxsize=1)
def get_merge_config() -> MergeConfig:
    """Return cached merge configuration for reuse across the application."""

    # Cache ensures repeated calls are cheap while still reflecting the env state
    # from process startup. Using a helper makes it easy to reset in tests.
    return _create_structured_config(_build_merge_config())


def reset_merge_config_cache() -> None:
    """Clear cached merge configuration (primarily for testing support)."""

    # Providing a reset hook keeps behaviour explicit for unit tests.
    get_merge_config.cache_clear()

