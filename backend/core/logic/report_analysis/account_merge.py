"""Utilities for deterministic 0–100 merge scoring and tagging."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from backend import config as app_config
from backend.core.ai.adjudicator import AdjudicatorError, validate_ai_payload
from backend.core.ai.paths import get_merge_paths, pair_pack_path
from backend.core.io.json_io import _atomic_write_json
from backend.core.io.tags import read_tags, upsert_tag, write_tags_atomic
from backend.core.logic.normalize.accounts import normalize_acctnum as _normalize_acctnum_basic
from backend.core.merge import acctnum
from backend.core.merge.acctnum import acctnum_level
from backend.core.logic.report_analysis.ai_pack import build_ai_pack_for_pair
from backend.core.logic.report_analysis import config as merge_config
from backend.core.logic.summary_compact import compact_merge_sections
from backend.telemetry.metrics import emit_counter
from backend.core.runflow import runflow_event, steps_pair_topn
from backend.core.runflow_spans import end_span, span_step, start_span
from backend.config.merge_config import (
    DEFAULT_FIELDS,
    POINTS_MODE_DEFAULT_WEIGHTS,
    MergeConfig,
    get_merge_config,
)
# NOTE: do not import validation_builder at module import-time.
# We'll lazy-import inside the function to avoid circular imports.
from backend.core.logic.validation_requirements import (
    _raw_value_provider_for_account_factory,
    apply_validation_summary,
    build_summary_payload as build_validation_summary_payload,
    build_validation_requirements,
    sync_validation_tag,
    _redact_account_number_raw,
)
__all__ = [
    "load_bureaus",
    "get_merge_cfg",
    "gen_unordered_pairs",
    "score_pair_0_100",
    "score_all_pairs_0_100",
    "choose_best_partner",
    "persist_merge_tags",
    "score_and_tag_best_partners",
    "merge_v2_only_enabled",
    "build_merge_pair_tag",
    "build_merge_best_tag",
    "build_summary_merge_entry",
    "build_summary_ai_entries",
    "apply_summary_updates",
    "merge_summary_sections",
    "MergeDecision",
    "normalize_history_field",
    "history_similarity_score",
    "match_history_field",
    "resolve_identity_debt_fields",
    "coerce_score_value",
    "detect_points_mode_from_payload",
    "normalize_parts_for_serialization",
]


logger = logging.getLogger(__name__)
_candidate_logger = logging.getLogger("ai_packs")

_LOCKS_DIRNAME = ".locks"
_MERGE_INFLIGHT_LOCK_FILENAME = "merge_inflight.lock"


CANON: Dict[str, Dict[str, List[str]]] = {
    "same_account_same_debt": {"aliases": ["merge", "same_account"]},
    "same_account_diff_debt": {"aliases": ["same_account_debt_different"]},
    "same_account_debt_unknown": {"aliases": []},
    "same_debt_diff_account": {"aliases": ["same_debt_account_different"]},
    "same_debt_account_unknown": {"aliases": ["same_debt"]},
    "different": {"aliases": []},
    "duplicate": {"aliases": []},
    "not_duplicate": {"aliases": []},
}


class MergeDecision(str, Enum):
    """Canonical AI merge decisions with legacy aliases."""

    SAME_ACCOUNT_SAME_DEBT = "same_account_same_debt"
    SAME_ACCOUNT_DIFF_DEBT = "same_account_diff_debt"
    SAME_ACCOUNT_DEBT_UNKNOWN = "same_account_debt_unknown"
    SAME_DEBT_DIFF_ACCOUNT = "same_debt_diff_account"
    SAME_DEBT_ACCOUNT_UNKNOWN = "same_debt_account_unknown"
    DIFFERENT = "different"
    DUPLICATE = "duplicate"
    NOT_DUPLICATE = "not_duplicate"

    # Legacy aliases preserved for backwards compatibility.
    MERGE = "same_account_same_debt"
    SAME_ACCOUNT = "same_account_same_debt"
    SAME_ACCOUNT_DEBT_DIFFERENT = "same_account_diff_debt"
    SAME_DEBT_ACCOUNT_DIFFERENT = "same_debt_diff_account"
    SAME_DEBT = "same_debt_account_unknown"

    @classmethod
    def canonical_value(cls, value: Any) -> Optional[str]:
        """Return the canonical decision string for *value* or ``None``."""

        if value is None:
            return None
        decision_text = str(value).strip().lower()
        if not decision_text:
            return None
        if decision_text in CANON:
            return decision_text
        return _MERGE_DECISION_ALIASES.get(decision_text)


AI_PAIR_KIND_BY_DECISION = {
    "same_account_same_debt": "same_account_pair",
    "same_account_diff_debt": "same_account_pair",
    "same_account_debt_unknown": "same_account_pair",
    "same_debt_diff_account": "same_debt_pair",
    "same_debt_account_unknown": "same_debt_pair",
    "duplicate": "same_account_pair",
    "different": "same_account_pair",
}

_MERGE_DECISION_ALIASES: Dict[str, str] = {
    alias.lower(): canonical
    for canonical, meta in CANON.items()
    for alias in meta.get("aliases", [])
}


def _normalize_merge_decision(
    decision: Any,
    *,
    partner: Any | None = None,
    log_missing: bool = False,
    log_unknown: bool = False,
) -> Tuple[str, bool]:
    """Return a canonical decision string and whether normalization occurred."""

    canonical = MergeDecision.canonical_value(decision)
    if canonical is not None:
        original_text = str(decision).strip().lower()
        return canonical, original_text != canonical

    if decision is None or (isinstance(decision, str) and not decision.strip()):
        if log_missing:
            if partner is not None:
                logger.warning(
                    "AI_DECISION_MISSING partner=%s decision=%r; defaulting to 'different'",
                    partner,
                    decision,
                )
            else:
                logger.warning(
                    "AI_DECISION_MISSING decision=%r; defaulting to 'different'",
                    decision,
                )
        return "different", True

    decision_text = str(decision).strip().lower()
    if decision_text in _MERGE_DECISION_ALIASES:
        return _MERGE_DECISION_ALIASES[decision_text], True

    if log_unknown:
        if partner is not None:
            logger.warning(
                "AI_DECISION_UNKNOWN decision=%s partner=%s; falling back to 'different'",
                decision_text,
                partner,
            )
        else:
            logger.warning(
                "AI_DECISION_UNKNOWN decision=%s; falling back to 'different'",
                decision_text,
            )
    return "different", True


def _configure_candidate_logger(logs_path: Path) -> None:
    """Ensure the candidate logger writes to the provided ``logs.txt`` file."""

    try:
        logs_path = Path(logs_path)
    except TypeError:  # pragma: no cover - defensive
        return

    logs_dir = logs_path.parent
    if logs_dir and not logs_dir.exists():
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - defensive
            return

    target = logs_path.resolve()

    existing_handler: Optional[logging.FileHandler] = None
    for handler in list(_candidate_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve()
            if handler_path == target:
                existing_handler = handler
            else:
                _candidate_logger.removeHandler(handler)
                handler.close()

    if existing_handler is None:
        file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        _candidate_logger.addHandler(file_handler)

    _candidate_logger.setLevel(logging.INFO)
    _candidate_logger.propagate = False


def _sanitize_acct_level(v: object) -> str:
    s = str(v or "none").strip().lower()
    return (
        s
        if s
        in {
            "none",
            "unknown",
            "partial",
            "masked_match",
            "exact_or_known_match",
        }
        else "none"
    )


AI_PACK_SCORE_THRESHOLD = 27


_ACCT_LEVEL_PRIORITY = {
    "exact_or_known_match": 4,
    "masked_match": 3,
    "partial": 2,
    "unknown": 1,
    "none": 0,
}

_STRONG_MATCH_LEVELS = {"exact_or_known_match"}
_WEAK_MATCH_LEVELS = {"masked_match", "partial"}


def is_missing(value: Any) -> bool:
    """Return True when a value represents an explicit missing sentinel."""

    return value in {None, "", "--"}


def _read_env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _read_env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _read_env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return bool(default)


MERGE_SKIP_COUNTS_ENABLED = _read_env_flag(os.environ, "MERGE_SKIP_COUNTS_ENABLED", True)
MERGE_ZERO_PACKS_SIGNAL = _read_env_flag(os.environ, "MERGE_ZERO_PACKS_SIGNAL", True)
MERGE_POINTS_DIAGNOSTICS_EMIT_JSON = _read_env_flag(
    os.environ, "MERGE_POINTS_DIAGNOSTICS_EMIT_JSON", True
)


_SKIP_REASON_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_skip_reason(reason: Any) -> str:
    """Normalize skip reasons to ``snake_case`` identifiers."""

    if reason is None:
        return "unspecified"
    text = str(reason).strip().lower()
    if not text:
        return "unspecified"
    normalized = _SKIP_REASON_SANITIZE_RE.sub("_", text).strip("_")
    return normalized or "unspecified"


def _select_skip_reason_top(counts: Mapping[str, int]) -> Optional[str]:
    """Return the plurality skip reason with lexicographic tie-break."""

    if not counts:
        return None
    max_count = max(int(value) for value in counts.values())
    top_keys = [key for key, value in counts.items() if int(value) == max_count]
    if not top_keys:
        return None
    return sorted(top_keys)[0]


def merge_v2_only_enabled() -> bool:
    """Return True when legacy merge artefact writes must be skipped."""

    return merge_config.get_merge_v2_only()


def gen_unordered_pairs(indices: List[int]) -> List[Tuple[int, int]]:
    """Return all unordered pairs (i, j) with i < j and no duplicates."""

    unique = sorted(set(indices))
    return [(i, j) for pos, i in enumerate(unique) for j in unique[pos + 1 :]]


def load_bureaus(
    sid: str, idx: int, runs_root: Path = Path("runs")
) -> Dict[str, Dict[str, Any]]:
    """Load bureau data for a case account, normalizing missing values."""

    bureaus_path = runs_root / sid / "cases" / "accounts" / str(idx) / "bureaus.json"
    if not bureaus_path.exists():
        raise FileNotFoundError(
            f"bureaus.json not found for sid={sid!r} index={idx} under {runs_root}"
        )

    with bureaus_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, Mapping):
        logger.warning(
            "Unexpected bureaus payload type %s for sid=%s idx=%s; treating as empty",
            type(data).__name__,
            sid,
            idx,
        )
        data = {}

    result: Dict[str, Dict[str, Any]] = {}
    for bureau in ("transunion", "experian", "equifax"):
        branch = data.get(bureau) if isinstance(data, Mapping) else None
        if isinstance(branch, Mapping):
            cleaned = {
                key: value
                for key, value in branch.items()
                if not is_missing(value)
            }
        else:
            cleaned = {}
        result[bureau] = dict(cleaned)

    return result


def _load_case_meta(meta_path: Path) -> Dict[str, Any]:
    """Return parsed meta.json payload when available."""

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.debug("MERGE_META_LOAD_FAILED path=%s", meta_path, exc_info=True)
        return {}

    return data if isinstance(data, Mapping) else {}


def _load_case_history_maps(bureaus_path: Path) -> Tuple[Dict[str, Sequence[Any]], Dict[str, Mapping[str, Any]]]:
    """Return two-year and seven-year history maps from bureaus.json."""

    try:
        with bureaus_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}, {}
    except (OSError, json.JSONDecodeError):
        logger.debug("MERGE_HISTORY_LOAD_FAILED path=%s", bureaus_path, exc_info=True)
        return {}, {}

    if not isinstance(payload, Mapping):
        return {}, {}

    history_2y_raw = payload.get("two_year_payment_history")
    history_7y_raw = payload.get("seven_year_history")

    history_2y: Dict[str, Sequence[Any]] = {}
    if isinstance(history_2y_raw, Mapping):
        for bureau, entries in history_2y_raw.items():
            if isinstance(entries, Sequence):
                history_2y[str(bureau)] = list(entries)

    history_7y: Dict[str, Mapping[str, Any]] = {}
    if isinstance(history_7y_raw, Mapping):
        for bureau, entry in history_7y_raw.items():
            if isinstance(entry, Mapping):
                history_7y[str(bureau)] = dict(entry)

    return history_2y, history_7y


def _has_original_creditor(bureaus: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return True when any bureau branch provides an original creditor value."""

    if not isinstance(bureaus, Mapping):
        return False

    for branch in bureaus.values():
        if not isinstance(branch, Mapping):
            continue
        value = branch.get("original_creditor")
        if not is_missing(value):
            return True
    return False


def _pair_has_original_creditor(
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return True when either side includes a usable original creditor value."""

    return _has_original_creditor(left_bureaus) or _has_original_creditor(right_bureaus)


def _ai_pack_gate_allows(
    cfg: MergeCfg,
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
) -> Tuple[bool, str]:
    """Apply AI pack gating derived from configuration flags."""

    require_original = bool(getattr(cfg, "require_original_creditor_for_ai", False))
    if not require_original:
        return True, ""

    if _pair_has_original_creditor(left_bureaus, right_bureaus):
        return True, ""

    return False, "missing_original_creditor"


@dataclass
class MergeCfg:
    """Centralized configuration for deterministic account merging."""

    points: Mapping[str, int]
    weights: Mapping[str, float]
    thresholds: Mapping[str, int]
    triggers: Mapping[str, Union[int, str, bool]]
    tolerances: Mapping[str, Union[int, float]]
    fields: Sequence[str] = field(default_factory=tuple)
    overrides: Mapping[str, Any] = field(default_factory=dict)
    allowlist_enforce: bool = False  # Runtime flag to enable the field allowlist.
    allowlist_fields: Sequence[str] = field(
        default_factory=tuple
    )  # Explicit list of fields honoured when the allowlist is active.
    use_custom_weights: bool = False
    use_original_creditor: bool = False
    use_creditor_name: bool = False
    use_account_label: bool = False
    require_original_creditor_for_ai: bool = False
    points_diagnostics: bool = False
    account_label_source: str = ""
    account_label_normalize: bool = True
    debug: bool = False  # Runtime toggle controlling verbose merge logging.
    log_every: int = 0  # Optional cadence controlling how often debug logs fire.

    @property
    def threshold(self) -> int:
        """Return the configured AI threshold as an integer."""

        raw = self.thresholds.get("AI_THRESHOLD", 0)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def ai_threshold(self) -> int:
        """Alias for the AI threshold used by AI gating helpers."""

        return self.threshold

    @property
    def MERGE_ALLOWLIST_ENFORCE(self) -> bool:
        """Expose allowlist enforcement flag for compatibility with env keys."""

        return bool(self.allowlist_enforce)

    @property
    def MERGE_FIELDS_OVERRIDE(self) -> Tuple[str, ...]:
        """Expose the active field allowlist mirroring the ENV naming."""

        return tuple(self.allowlist_fields)

    @property
    def MERGE_WEIGHTS(self) -> Mapping[str, float]:
        """Expose the active per-field weights for runtime scoring."""

        return self.weights

    @property
    def MERGE_SCORE_THRESHOLD(self) -> int:
        """Return the active merge score cutoff with AUTO fallback."""

        def _coerce(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # Prefer the explicit MERGE_SCORE_THRESHOLD when provided via env config.
        primary = _coerce(self.thresholds.get("MERGE_SCORE_THRESHOLD"))
        if primary is not None:
            return primary

        # Fall back to the historic AUTO_MERGE_THRESHOLD for backward compatibility.
        fallback = _coerce(self.thresholds.get("AUTO_MERGE_THRESHOLD"))
        if fallback is not None:
            return fallback

        # Final defensive guard keeps the contract stable even on bad inputs.
        return 0

    @property
    def MERGE_USE_ORIGINAL_CREDITOR(self) -> bool:
        """Expose optional-field toggle mirroring the MERGE_* env flag."""

        return bool(self.use_original_creditor)

    @property
    def MERGE_USE_CREDITOR_NAME(self) -> bool:
        """Expose optional-field toggle mirroring the MERGE_* env flag."""

        return bool(self.use_creditor_name)

    @property
    def MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI(self) -> bool:
        """Return whether AI packs demand the original creditor field."""

        return bool(self.require_original_creditor_for_ai)

    @property
    def MERGE_DEBUG(self) -> bool:
        """Return whether debug logging for merge scoring is enabled."""

        return bool(self.debug)

    @property
    def MERGE_LOG_EVERY(self) -> int:
        """Return the configured debug logging cadence as a positive integer."""

        try:
            value = int(self.log_every)
        except (TypeError, ValueError):
            return 0
        return value if value >= 0 else 0


_TOLERANCE_DEFAULTS: Dict[str, Union[int, float]] = {
    "AMOUNT_TOL_ABS": 50.0,
    "AMOUNT_TOL_RATIO": 0.01,
    "LAST_PAYMENT_DAY_TOL": 5,
    "COUNT_ZERO_PAYMENT_MATCH": 0,
    "CA_DATE_MONTH_TOL": 6,
    # New merge tolerance knobs default to legacy behaviour (strict comparisons).
    "MERGE_TOL_DATE_DAYS": 0,
    "MERGE_TOL_BALANCE_ABS": 0.0,
    "MERGE_TOL_BALANCE_RATIO": 0.0,
    "MERGE_ACCOUNTNUMBER_MATCH_MINLEN": 0,
    "MERGE_HISTORY_SIMILARITY_THRESHOLD": 1.0,
}


def _merge_env_state(config: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(explicit_enabled, merge_enabled)`` for the new env config."""

    present_raw = config.get("_present_keys")
    if isinstance(present_raw, (set, frozenset, list, tuple)):
        present = {str(item).lower() for item in present_raw if item is not None}
    else:
        present = set()

    explicit_enabled = "enabled" in present
    if explicit_enabled:
        # The feature is explicitly controlled via MERGE_ENABLED; coerce truthiness.
        return True, bool(config.get("enabled"))

    # When MERGE_ENABLED is absent we preserve legacy behaviour (feature on).
    return False, True


def _sanitize_config_field_list(candidate: Any) -> Tuple[str, ...]:
    """Return merge field overrides sourced from ``MERGE_FIELDS_OVERRIDE``."""

    if isinstance(candidate, str):
        try:
            decoded = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = [item.strip() for item in candidate.split(",") if item.strip()]
        candidate = decoded

    if not isinstance(candidate, (list, tuple, set)):
        return ()

    sanitized: list[str] = []
    seen: Set[str] = set()
    for entry in candidate:
        if entry is None:
            continue
        name = str(entry).strip()
        if not name:
            continue
        lower = name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        sanitized.append(name)
    return tuple(sanitized)


def _sanitize_numeric_mapping(candidate: Any, *, upper_keys: bool = False) -> Dict[str, int]:
    """Coerce numeric mapping payloads into ``Dict[str, int]`` values."""

    if not isinstance(candidate, Mapping):
        return {}

    result: Dict[str, int] = {}
    for key, value in candidate.items():
        if key is None:
            continue
        name = str(key).strip()
        if not name:
            continue
        if upper_keys:
            name = name.upper()
        try:
            result[name] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _sanitize_weight_mapping(candidate: Any) -> Dict[str, float]:
    """Return ``Dict[str, float]`` for custom merge weight overrides."""

    if not isinstance(candidate, Mapping):
        return {}

    result: Dict[str, float] = {}
    for key, value in candidate.items():
        if key is None:
            continue
        name = str(key).strip()
        if not name:
            continue
        try:
            result[name] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _sanitize_tolerance_mapping(candidate: Any) -> Dict[str, Union[int, float]]:
    """Coerce tolerance overrides from the env config into native numbers."""

    if not isinstance(candidate, Mapping):
        return {}

    sanitized: Dict[str, Union[int, float]] = {}
    for key, value in candidate.items():
        if key is None:
            continue
        name = str(key).strip()
        if not name:
            continue

        default = _TOLERANCE_DEFAULTS.get(name)
        # Default to float parsing when we don't have a known baseline type.
        if isinstance(default, float):
            try:
                sanitized[name] = float(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(default, int):
            try:
                sanitized[name] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            try:
                sanitized[name] = float(value)
            except (TypeError, ValueError):
                continue

    return sanitized


def _coerce_env_bool(value: Any, default: bool = False) -> bool:
    """Coerce env-like values to bool with a safe default."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _coerce_env_int(value: Any, default: int = 0) -> int:
    """Return an integer parsed from env-style values with graceful fallback."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _sanitize_overrides_mapping(candidate: Any) -> Dict[str, Any]:
    """Return a shallow copy of overrides when a mapping is provided."""

    if not isinstance(candidate, Mapping):
        return {}

    return {str(key): value for key, value in candidate.items() if key is not None}


def _field_sequence_from_cfg(cfg: Optional[MergeCfg] = None) -> Tuple[str, ...]:
    """Resolve the active field sequence constrained to supported points fields."""

    current_cfg = cfg or get_merge_cfg()

    if bool(getattr(current_cfg, "points_mode", False)):
        return _resolve_points_mode_allowlist(current_cfg)

    configured_fields = tuple(getattr(current_cfg, "field_sequence", ()) or ())
    if not configured_fields:
        configured_fields = tuple(getattr(current_cfg, "fields", ()) or ())
    if not configured_fields:
        configured_fields = tuple(DEFAULT_FIELDS)

    allowlist_fields = tuple(getattr(current_cfg, "allowlist_fields", ()) or ())
    if allowlist_fields:
        defaults = set(DEFAULT_FIELDS)
        filtered = tuple(field for field in allowlist_fields if field in defaults)
        if filtered:
            logger.debug("Merge active fields: %s", list(filtered))
            return filtered

    defaults = set(DEFAULT_FIELDS)
    filtered = tuple(field for field in configured_fields if field in defaults)
    if filtered:
        return filtered

    return tuple(DEFAULT_FIELDS)


def resolve_identity_debt_fields(
    cfg: Optional[MergeCfg] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return the active identity/debt fields for the configured merge mode."""

    if cfg is None:
        cfg = get_merge_cfg()

    field_sequence = _field_sequence_from_cfg(cfg)
    points_mode_active = bool(getattr(cfg, "points_mode", False))

    if points_mode_active:
        identity_fields = tuple(
            field for field in field_sequence if field in _POINTS_MODE_IDENTITY_FIELDS
        )
        debt_fields = tuple(
            field for field in field_sequence if field in _POINTS_MODE_DEBT_FIELDS
        )
    else:
        identity_fields = tuple(
            field for field in field_sequence if field in _IDENTITY_FIELD_SET
        )
        debt_fields = tuple(
            field for field in field_sequence if field in _DEBT_FIELD_SET
        )

    return identity_fields, debt_fields


def get_merge_cfg(env: Optional[Mapping[str, str]] = None) -> MergeCfg:
    """Return merge configuration using environment overrides when provided."""

    env_mapping: Mapping[str, str]
    if env is None:
        env_mapping = os.environ
    else:
        env_mapping = env

    points_mode_active = False
    allowlist_enforce = False
    legacy_defaults_allowed = False
    points_mode_locked = False
    points_diagnostics_enabled = False

    points: Dict[str, int] = {}
    # Default weights act as neutral multipliers until custom overrides are enabled.
    weights_map: Dict[str, float] = {}
    custom_weights_enabled = False
    merge_use_original_creditor = False
    merge_use_creditor_name = False
    merge_use_account_label = False
    require_original_creditor_for_ai = False
    account_label_source = ""
    account_label_normalize = True
    merge_debug_enabled = False
    merge_log_every = 0
    ai_points_threshold = 3.0
    direct_points_threshold = 5.0
    points_diagnostics_limit = _POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK
    points_persist_breakdown_enabled = False
    points_diag_dir_setting = "ai_packs/merge/diagnostics"

    threshold_keys = ("AI_THRESHOLD", "AUTO_MERGE_THRESHOLD", "MERGE_SCORE_THRESHOLD")
    thresholds = {
        key: _read_env_int(env_mapping, key, 0) for key in threshold_keys
    }

    triggers: Dict[str, Union[int, str, bool]] = {}
    triggers["MERGE_AI_ON_BALOWED_EXACT"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_BALOWED_EXACT",
        True,
    )
    triggers["MERGE_AI_ON_HARD_ACCTNUM"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_HARD_ACCTNUM",
        True,
    )
    triggers["MERGE_AI_ON_MID_K"] = _read_env_int(
        env_mapping,
        "MERGE_AI_ON_MID_K",
        0,
    )
    triggers["MERGE_AI_ON_ALL_DATES"] = _read_env_flag(
        env_mapping,
        "MERGE_AI_ON_ALL_DATES",
        False,
    )

    tolerances = {
        key: (
            _read_env_int(env_mapping, key, int(default))
            if isinstance(default, int)
            else _read_env_float(env_mapping, key, float(default))
        )
        for key, default in _TOLERANCE_DEFAULTS.items()
    }

    # Ensure tolerance types remain float for ratio/absolute values
    if isinstance(tolerances["AMOUNT_TOL_ABS"], int):
        tolerances["AMOUNT_TOL_ABS"] = float(tolerances["AMOUNT_TOL_ABS"])
    if isinstance(tolerances["AMOUNT_TOL_RATIO"], int):
        tolerances["AMOUNT_TOL_RATIO"] = float(tolerances["AMOUNT_TOL_RATIO"])

    # Count-zero-payment match is an integer toggle but maintain numeric type explicitly.
    tolerances["COUNT_ZERO_PAYMENT_MATCH"] = int(
        tolerances["COUNT_ZERO_PAYMENT_MATCH"]
    )

    config_fields: Tuple[str, ...] = tuple()
    overrides_mapping: Dict[str, Any] = {}
    allowlist_fields: Tuple[str, ...] = tuple()
    fields_override: Tuple[str, ...] = tuple()

    if env is None:
        # Load merge-specific configuration when running against real env vars.
        merge_env_cfg = get_merge_config()
        points_mode_active = bool(getattr(merge_env_cfg, "points_mode", False))
        explicit_enabled, merge_enabled = _merge_env_state(merge_env_cfg)
        custom_weights_enabled = _coerce_env_bool(
            merge_env_cfg.get("use_custom_weights")
        )  # controlled by MERGE_USE_CUSTOM_WEIGHTS
        merge_use_original_creditor = _coerce_env_bool(
            merge_env_cfg.get("use_original_creditor")
        )  # controlled by MERGE_USE_ORIGINAL_CREDITOR
        merge_use_creditor_name = _coerce_env_bool(
            merge_env_cfg.get("use_creditor_name")
        )  # controlled by MERGE_USE_CREDITOR_NAME
        merge_use_account_label = _coerce_env_bool(
            merge_env_cfg.get("use_account_label"), False
        )  # controlled by MERGE_USE_ACCOUNT_LABEL
        require_original_creditor_for_ai = _coerce_env_bool(
            merge_env_cfg.get("require_original_creditor_for_ai"), False
        )
        points_diagnostics_enabled = _coerce_env_bool(
            merge_env_cfg.get("points_diagnostics"), False
        )
        points_persist_breakdown_enabled = _coerce_env_bool(
            merge_env_cfg.get("points_persist_breakdown"), False
        )
        raw_points_diag_dir = getattr(
            merge_env_cfg,
            "points_diag_dir",
            "ai_packs/merge/diagnostics",
        )
        points_diag_dir_setting = str(raw_points_diag_dir).strip() or "ai_packs/merge/diagnostics"
        raw_account_label_source = merge_env_cfg.get(
            "account_label_source", "meta.heading_guess"
        )
        if isinstance(raw_account_label_source, str):
            account_label_source = (
                raw_account_label_source.strip() or "meta.heading_guess"
            )
        elif raw_account_label_source is None:
            account_label_source = "meta.heading_guess"
        else:
            account_label_source = (
                str(raw_account_label_source).strip() or "meta.heading_guess"
            )
        account_label_normalize = _coerce_env_bool(
            merge_env_cfg.get("account_label_normalize"), True
        )
        merge_debug_enabled = _coerce_env_bool(
            merge_env_cfg.get("debug")
        )  # controlled by MERGE_DEBUG
        merge_log_every = _coerce_env_int(
            merge_env_cfg.get("log_every"), 0
        )  # controlled by MERGE_LOG_EVERY
        try:
            ai_points_threshold = float(getattr(merge_env_cfg, "ai_points_threshold", 3.0) or 3.0)
        except (TypeError, ValueError):
            ai_points_threshold = 3.0
        try:
            direct_points_threshold = float(
                getattr(merge_env_cfg, "direct_points_threshold", 5.0) or 5.0
            )
        except (TypeError, ValueError):
            direct_points_threshold = 5.0
        try:
            points_diagnostics_limit = int(
                getattr(
                    merge_env_cfg,
                    "points_diagnostics_limit",
                    _POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK,
                )
                or _POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK
            )
        except (TypeError, ValueError):
            points_diagnostics_limit = _POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK

        allowlist_enforce = bool(getattr(merge_env_cfg, "allowlist_enforce", False))
        points_mode_locked = bool(points_mode_active)
        legacy_defaults_allowed = not allowlist_enforce and not points_mode_active

        weights_map = dict(getattr(merge_env_cfg, "weights_map", {}))
        if legacy_defaults_allowed and not weights_map:
            weights_map = {field: 1.0 for field in _POINTS_MODE_FIELD_ALLOWLIST}

        if legacy_defaults_allowed:
            config_fields = _POINTS_MODE_FIELD_ALLOWLIST
            allowlist_fields = _POINTS_MODE_FIELD_ALLOWLIST
        else:
            config_fields = tuple()
            allowlist_fields = tuple()

        fields_override = _sanitize_config_field_list(merge_env_cfg.get("fields_override"))
        configured_fields = _sanitize_config_field_list(merge_env_cfg.get("fields"))

        if explicit_enabled and merge_enabled:
            if fields_override:
                # controlled by MERGE_FIELDS_OVERRIDE / MERGE_FIELDS_OVERRIDE_JSON
                config_fields = fields_override
            elif configured_fields:
                config_fields = configured_fields
            elif legacy_defaults_allowed:
                config_fields = _POINTS_MODE_FIELD_ALLOWLIST

            if points_mode_active:
                weights_map = dict(getattr(merge_env_cfg, "weights_map", {}))
            else:
                weights_override = _sanitize_weight_mapping(
                    merge_env_cfg.get("weights")
                )  # controlled by MERGE_WEIGHTS_JSON
                if custom_weights_enabled and weights_override:
                    weights_map.update(weights_override)

            thresholds_override = _sanitize_numeric_mapping(
                merge_env_cfg.get("thresholds"), upper_keys=True
            )  # controlled by MERGE_THRESHOLDS_JSON
            if thresholds_override:
                thresholds.update(thresholds_override)

            tolerance_override = _sanitize_tolerance_mapping(
                merge_env_cfg.get("tolerances")
            )  # controlled by MERGE_TOLERANCES_JSON
            if tolerance_override:
                tolerances.update(tolerance_override)

            overrides_mapping = _sanitize_overrides_mapping(
                merge_env_cfg.get("overrides")
            )  # controlled by MERGE_OVERRIDES_JSON
        else:
            # When MERGE_ENABLED is absent or disabled we retain legacy fields.
            if legacy_defaults_allowed:
                config_fields = _POINTS_MODE_FIELD_ALLOWLIST
            overrides_mapping = {}

        if fields_override:
            # controlled by MERGE_FIELDS_OVERRIDE / MERGE_FIELDS_OVERRIDE_JSON
            allowlist_fields = fields_override
        elif configured_fields:
            allowlist_fields = configured_fields

        # Allowlist enforcement can be toggled independently of MERGE_ENABLED.
        if merge_enabled:
            # controlled by MERGE_ALLOWLIST_ENFORCE
            allowlist_enforce = bool(getattr(merge_env_cfg, "allowlist_enforce", False))
        # Fall back to legacy fields when no override is provided.
        if not allowlist_fields and legacy_defaults_allowed and not points_mode_locked:
            allowlist_fields = _POINTS_MODE_FIELD_ALLOWLIST
    else:
        points_mode_active = _read_env_flag(env_mapping, "MERGE_POINTS_MODE", False)
        allowlist_enforce = _read_env_flag(env_mapping, "MERGE_ALLOWLIST_ENFORCE", False)
        points_mode_locked = bool(points_mode_active)
        legacy_defaults_allowed = not allowlist_enforce and not points_mode_active

        if legacy_defaults_allowed:
            weights_map = {field: 1.0 for field in _POINTS_MODE_FIELD_ALLOWLIST}
            config_fields = _POINTS_MODE_FIELD_ALLOWLIST
            allowlist_fields = _POINTS_MODE_FIELD_ALLOWLIST
        else:
            weights_map = {}
            config_fields = tuple()
            allowlist_fields = tuple()

        fields_override = _sanitize_config_field_list(
            env_mapping.get("MERGE_FIELDS_OVERRIDE_JSON")
        )
        if not fields_override:
            fields_override = _sanitize_config_field_list(
                env_mapping.get("MERGE_FIELDS_OVERRIDE")
            )
        if fields_override:
            allowlist_fields = fields_override

        overrides_mapping = {}
        allowlist_enforce = allowlist_enforce
        ai_points_threshold = _read_env_float(env_mapping, "MERGE_AI_POINTS_THRESHOLD", 3.0)
        direct_points_threshold = _read_env_float(
            env_mapping, "MERGE_DIRECT_POINTS_THRESHOLD", 5.0
        )
        merge_use_original_creditor = _read_env_flag(
            env_mapping,
            "MERGE_USE_ORIGINAL_CREDITOR",
            False,
        )
        merge_use_creditor_name = _read_env_flag(
            env_mapping,
            "MERGE_USE_CREDITOR_NAME",
            False,
        )
        merge_use_account_label = _read_env_flag(
            env_mapping,
            "MERGE_USE_ACCOUNT_LABEL",
            False,
        )
        require_original_creditor_for_ai = _read_env_flag(
            env_mapping,
            "MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI",
            False,
        )
        raw_account_label_source = env_mapping.get(
            "MERGE_ACCOUNT_LABEL_SOURCE", "meta.heading_guess"
        )
        if isinstance(raw_account_label_source, str):
            account_label_source = (
                raw_account_label_source.strip() or "meta.heading_guess"
            )
        elif raw_account_label_source is None:
            account_label_source = "meta.heading_guess"
        else:
            account_label_source = (
                str(raw_account_label_source).strip() or "meta.heading_guess"
            )
        account_label_normalize = _read_env_flag(
            env_mapping,
            "MERGE_ACCOUNT_LABEL_NORMALIZE",
            True,
        )
        merge_debug_enabled = _read_env_flag(
            env_mapping,
            "MERGE_DEBUG",
            False,
        )
        merge_log_every = _read_env_int(env_mapping, "MERGE_LOG_EVERY", 0)
        points_diagnostics_limit = _read_env_int(
            env_mapping,
            "MERGE_POINTS_DIAGNOSTICS_LIMIT",
            _POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK,
        )
        points_diagnostics_enabled = _read_env_flag(
            env_mapping,
            "MERGE_POINTS_DIAGNOSTICS",
            False,
        )
        points_persist_breakdown_enabled = _read_env_flag(
            env_mapping,
            "MERGE_POINTS_PERSIST_BREAKDOWN",
            False,
        )
        raw_points_diag_dir = env_mapping.get(
            "MERGE_POINTS_DIAGNOSTICS_DIR",
            "ai_packs/merge/diagnostics",
        )
        if isinstance(raw_points_diag_dir, str):
            points_diag_dir_setting = raw_points_diag_dir.strip() or "ai_packs/merge/diagnostics"
        elif raw_points_diag_dir is None:
            points_diag_dir_setting = "ai_packs/merge/diagnostics"
        else:
            points_diag_dir_setting = str(raw_points_diag_dir).strip() or "ai_packs/merge/diagnostics"

    fields_override = tuple(fields_override)
    allowlist_fields = tuple(allowlist_fields)
    allowlist_lookup: Set[str] = set(allowlist_fields)
    override_lookup: Set[str] = {str(name) for name in fields_override if name}

    points_mode_active = True
    points_mode_locked = True
    legacy_defaults_allowed = False

    allow_optional_original_creditor = bool(
        merge_use_original_creditor
        and "original_creditor" in allowlist_lookup
        and "original_creditor" in override_lookup
    )
    allow_optional_creditor_name = bool(
        merge_use_creditor_name
        and "creditor_name" in allowlist_lookup
        and "creditor_name" in override_lookup
    )
    allow_optional_account_label = bool(merge_use_account_label)
    if allow_optional_account_label:
        if "account_label" not in allowlist_lookup:
            allow_optional_account_label = False
        elif override_lookup and "account_label" not in override_lookup:
            allow_optional_account_label = False

    if not allow_optional_original_creditor:
        weights_map.pop("original_creditor", None)
        if not points_mode_active:
            points.pop("original_creditor", None)
    elif not points_mode_active and "original_creditor" not in points:
        # Optional field participates with zero points until weights are defined.
        points["original_creditor"] = 0
        weights_map.setdefault("original_creditor", 1.0)

    if not allow_optional_creditor_name:
        weights_map.pop("creditor_name", None)
        if not points_mode_active:
            points.pop("creditor_name", None)
    elif not points_mode_active and "creditor_name" not in points:
        # Optional field participates with zero points until weights are defined.
        points["creditor_name"] = 0
        weights_map.setdefault("creditor_name", 1.0)

    if not allow_optional_account_label:
        weights_map.pop("account_label", None)
        if not points_mode_active:
            points.pop("account_label", None)
    else:
        if not points_mode_active and "account_label" not in points:
            points["account_label"] = 0
        if "account_label" not in weights_map:
            weights_map["account_label"] = float(
                POINTS_MODE_DEFAULT_WEIGHTS.get("account_label", 1.0)
            )

    if points_mode_active:
        points = {}

    def _with_optional_fields(sequence: Sequence[str]) -> Tuple[str, ...]:
        """Return ``sequence`` plus any optional fields toggled on via config."""

        items = [
            item
            for item in sequence
            if not (
                item == "original_creditor" and not allow_optional_original_creditor
            )
            and not (item == "creditor_name" and not allow_optional_creditor_name)
            and not (item == "account_label" and not allow_optional_account_label)
        ]
        if allow_optional_original_creditor and "original_creditor" not in items:
            # Future toggle makes ``original_creditor`` opt-in without code churn.
            items.append("original_creditor")
        if allow_optional_creditor_name and "creditor_name" not in items:
            # Future toggle makes ``creditor_name`` opt-in without code churn.
            items.append("creditor_name")
        if allow_optional_account_label and "account_label" not in items:
            items.append("account_label")
        return tuple(items)

    config_fields = _with_optional_fields(config_fields)
    allowlist_fields = _with_optional_fields(allowlist_fields)

    if (allowlist_enforce or points_mode_locked) and allowlist_fields:
        config_fields = allowlist_fields

    if allowlist_enforce:
        allowed_field_set = set(allowlist_fields)
        weights_map = {name: weight for name, weight in weights_map.items() if name in allowed_field_set}

    if not weights_map:
        resolved_fields = allowlist_fields or config_fields or _POINTS_MODE_FIELD_ALLOWLIST
        weights_map = {
            field: float(POINTS_MODE_DEFAULT_WEIGHTS.get(field, 0.0))
            for field in resolved_fields
            if field in POINTS_MODE_DEFAULT_WEIGHTS
        }

    config_fields = tuple(config_fields)
    allowlist_fields = tuple(allowlist_fields)

    effective_use_account_label = bool(allow_optional_account_label)
    account_label_runtime_source = (
        account_label_source if effective_use_account_label else ""
    )
    account_label_runtime_normalize = bool(account_label_normalize)

    cfg_obj = MergeCfg(
        points=points,
        weights=weights_map,
        thresholds=thresholds,
        triggers=triggers,
        tolerances=tolerances,
        fields=config_fields,
        overrides=overrides_mapping,
        allowlist_enforce=allowlist_enforce,
        allowlist_fields=allowlist_fields,
        use_custom_weights=custom_weights_enabled,
        use_original_creditor=merge_use_original_creditor,
        use_creditor_name=merge_use_creditor_name,
        use_account_label=effective_use_account_label,
        require_original_creditor_for_ai=bool(require_original_creditor_for_ai),
        account_label_source=account_label_runtime_source,
        account_label_normalize=account_label_runtime_normalize,
        debug=merge_debug_enabled,
        log_every=merge_log_every,
    )

    setattr(cfg_obj, "points_mode", bool(points_mode_active))
    setattr(cfg_obj, "ai_points_threshold", float(ai_points_threshold))
    setattr(cfg_obj, "direct_points_threshold", float(direct_points_threshold))
    setattr(
        cfg_obj,
        "points_diagnostics_limit",
        int(max(points_diagnostics_limit or 0, 0)),
    )
    setattr(cfg_obj, "points_diagnostics", bool(points_diagnostics_enabled))
    setattr(cfg_obj, "points_persist_breakdown", bool(points_persist_breakdown_enabled))
    setattr(cfg_obj, "points_diag_dir", str(points_diag_dir_setting))

    active_field_sequence = (
        allowlist_fields if (allowlist_enforce or points_mode_locked) else config_fields
    )
    setattr(cfg_obj, "field_sequence", tuple(active_field_sequence))
    setattr(cfg_obj, "weights_map", dict(weights_map))

    return cfg_obj


# ---------------------------------------------------------------------------
# Deterministic merge helpers
# ---------------------------------------------------------------------------

_AMOUNT_SANITIZE_RE = re.compile(r"[\s$,/]")
_AMOUNT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_DECIMAL_ONE = Decimal("1")
_DECIMAL_HUNDRED = Decimal("100")
_ACCOUNT_LEVEL_ORDER = {
    "none": 0,
    "exact_or_known_match": 1,
}
_ACCOUNT_STRONG_LEVELS = {"exact_or_known_match"}
_MASK_CHARS = {"*", "x", "X", "•", "●"}

_ACCOUNT_LEVEL_PRIORITY = {
    "none": 0,
    "exact_or_known_match": 1,
}
_POINTS_MODE_ACCOUNT_NUMBER_MATCH_LEVELS = frozenset(
    {"exact", "known_match", "exact_or_known_match"}
)

_IDENTITY_FIELD_SET = {
    "account_number",
    "creditor_type",
    "date_opened",
    "closed_date",
    "date_of_last_activity",
    "date_reported",
    "last_verified",
    "account_type",
    "account_status",
    "account_label",
}

_DEBT_FIELD_SET = {
    "balance_owed",
    "high_balance",
    "past_due_amount",
    "last_payment",
}

_POINTS_MODE_IDENTITY_FIELDS: Tuple[str, ...] = (
    "account_number",
    "date_opened",
    "account_type",
    "account_status",
    "account_label",
)

_POINTS_MODE_DEBT_FIELDS: Tuple[str, ...] = (
    "balance_owed",
    "history_2y",
    "history_7y",
)


def _normalize_points_mode_acct_level(value: Any) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _POINTS_MODE_ACCOUNT_NUMBER_MATCH_LEVELS:
            return candidate
        if candidate in {"none", "unknown", "partial", "masked_match", "visible_digits_conflict"}:
            return "none"
    return "none"


def normalize_acctnum(raw: str | None) -> Dict[str, object]:
    """Return canonical metadata for account-number comparisons.

    Returns a dictionary with:

    ``raw``
        Original input string (or ``None``).
    ``digits``
        Concatenated digits extracted from the account number (may be empty).
    ``canon_mask``
        Canonical masked form after removing whitespace, dashes, dots, and
        collapsing mask runs.
    ``has_digits``
        Boolean indicating whether any digits were observed.
    ``has_mask``
        Boolean indicating whether any masking characters were present.
    ``visible_digits``
        Count of visible digits observed after removing masks and spacing.
    """

    raw_value = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    if raw_value is None:
        raw_value = ""

    stripped = re.sub(r"[\s\-.]+", "", raw_value)
    if not stripped:
        return {
            "raw": raw,
            "digits": "",
            "canon_mask": "",
            "has_digits": False,
            "has_mask": False,
            "visible_digits": 0,
        }

    translated_chars: List[str] = []
    digits_chars: List[str] = []
    visible_digit_count = 0
    for char in stripped:
        if char.isdigit():
            digits_chars.append(char)
            translated_chars.append(char)
            visible_digit_count += 1
        elif char in _MASK_CHARS:
            translated_chars.append("*")
        else:
            translated_chars.append(char.upper())

    canon_mask = re.sub(r"\*+", "*", "".join(translated_chars))
    digits = "".join(digits_chars)
    has_digits = bool(digits)

    has_mask = "*" in canon_mask

    return {
        "raw": raw,
        "digits": digits,
        "canon_mask": canon_mask,
        "has_digits": has_digits,
        "has_mask": has_mask,
        "visible_digits": visible_digit_count,
    }


DIGITS_RE = re.compile(r"\d")


def _digits_only(raw: str | None) -> str:
    return "".join(DIGITS_RE.findall(raw or ""))


def acctnum_match_level(
    a_raw: str | None, b_raw: str | None
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Return the visible-digits account-number level with debug payload."""

    a_value = str(a_raw) if a_raw is not None else ""
    b_value = str(b_raw) if b_raw is not None else ""
    a_digits = _digits_only(a_value)
    b_digits = _digits_only(b_value)

    level, detail_raw = acctnum.acctnum_match_level(a_value, b_value)
    detail = dict(detail_raw)

    min_length = _account_number_min_length()
    if (
        level == "exact_or_known_match"
        and min_length > 0
        and (len(a_digits) < min_length or len(b_digits) < min_length)
    ):
        # Preserve the debug payload while recording why the match was rejected.
        level = "none"
        detail = dict(detail)
        detail["why"] = "min_length"
        detail["min_length"] = str(min_length)

    debug: Dict[str, Dict[str, str]] = {
        "a": {
            "raw": a_value,
            "digits": a_digits,
        },
        "b": {
            "raw": b_value,
            "digits": b_digits,
        },
        "short": str(detail.get("short", "")),
        "long": str(detail.get("long", "")),
    }
    if "why" in detail:
        debug["why"] = str(detail["why"])

    return level, debug
_TYPE_ALIAS_MAP = {
    "us bk cacs": "u s bank",
    "us bk cac": "u s bank",
    "us bk cas": "u s bank",
    "us bk cc": "u s bank",
    "us bank cacs": "u s bank",
    "u.s. bank": "u s bank",
    "us bank": "u s bank",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%y",
    "%m-%d-%y",
    "%Y%m%d",
)

_AMOUNT_FIELDS = {"past_due_amount", "high_balance", "credit_limit"}
_ZERO_AMOUNT_FIELDS = {"balance_owed", "past_due_amount"}
_AMOUNT_ZERO_EPSILON = 1e-9
_DATE_FIELDS_DET = {
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "closed_date",
}
_AMOUNT_CONFLICT_FIELDS = {
    "balance_owed",
    "payment_amount",
    "past_due_amount",
    "high_balance",
    "credit_limit",
}
_TYPE_FIELDS = {"creditor_type", "account_type", "account_status"}

_POINTS_MODE_FIELD_ALLOWLIST: Tuple[str, ...] = (
    "account_number",
    "date_opened",
    "balance_owed",
    "account_type",
    "account_status",
    "history_2y",
    "history_7y",
    "account_label",
)

_POINTS_MODE_DIAGNOSTICS_LIMIT_FALLBACK = 3
_POINTS_MODE_DIAGNOSTICS_EMITTED = 0


def _resolve_points_mode_allowlist(cfg: MergeCfg) -> Tuple[str, ...]:
    """Return the active allowlist for points mode constrained to supported fields."""

    if cfg is None:
        return _POINTS_MODE_FIELD_ALLOWLIST

    candidates: Sequence[Sequence[str]] = (
        tuple(getattr(cfg, "allowlist_fields", ()) or ()),
        tuple(getattr(cfg, "field_sequence", ()) or ()),
        tuple(getattr(cfg, "fields", ()) or ()),
    )

    defaults = set(_POINTS_MODE_FIELD_ALLOWLIST)
    for sequence in candidates:
        filtered = [str(field) for field in sequence if str(field) in defaults]
        if filtered:
            # Preserve ordering but ensure uniqueness.
            return tuple(dict.fromkeys(filtered))

    return _POINTS_MODE_FIELD_ALLOWLIST


def _normalize_field_value(field: str, value: Any) -> Optional[Any]:
    """Normalize a merge field value according to deterministic rules."""

    if field == "balance_owed":
        return normalize_balance_owed(value)
    if field == "account_number":
        return normalize_account_number(value)
    if field == "payment_amount" or field in _AMOUNT_FIELDS:
        return normalize_amount_field(value)
    if field == "last_payment" or field in _DATE_FIELDS_DET:
        return to_date(value)
    if field in _TYPE_FIELDS:
        return normalize_type(value)
    if field in {"history_2y", "history_7y"}:
        return normalize_history_field(value)
    return value


def _serialize_normalized_value(value: Any) -> Any:
    """Convert normalized values into JSON/log friendly primitives."""

    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def _serialize_normalized_pair(a: Any, b: Any) -> Tuple[Any, Any]:
    return (_serialize_normalized_value(a), _serialize_normalized_value(b))


def _extract_account_number_string(
    bureaus: Mapping[str, Mapping[str, Any]],
    preferred_bureau: Optional[str] = None,
) -> str:
    """Return the best raw account-number string for a bureau mapping."""

    if not isinstance(bureaus, Mapping):
        return ""

    ordered_candidates: List[str] = []
    if preferred_bureau:
        ordered_candidates.append(str(preferred_bureau))
    ordered_candidates.extend(["transunion", "experian", "equifax"])

    seen: Set[str] = set()
    for bureau_key in ordered_candidates:
        if bureau_key in seen:
            continue
        seen.add(bureau_key)

        branch = bureaus.get(bureau_key)
        if not isinstance(branch, Mapping):
            continue

        for field_name in ("account_number_display", "account_number"):
            raw_value = branch.get(field_name)
            if is_missing(raw_value):
                continue
            return str(raw_value)

    return ""


def _normalize_account_display(branch: Mapping[str, Any] | None) -> acctnum.NormalizedAccountNumber:
    if not isinstance(branch, Mapping):
        return acctnum.normalize_display("")

    raw_value: Optional[str] = None
    for key in ("account_number_display", "account_number"):
        candidate = branch.get(key)
        if not is_missing(candidate):
            raw_value = str(candidate)
            break

    return acctnum.normalize_display(raw_value)


def _match_account_number_best_pair(
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    bureaus = ("transunion", "experian", "equifax")
    bureau_positions = {name: idx for idx, name in enumerate(bureaus)}

    hard_enabled = bool(cfg.triggers.get("MERGE_AI_ON_HARD_ACCTNUM", True))
    min_length = _resolve_tolerance_int(
        cfg,
        "MERGE_ACCOUNTNUMBER_MATCH_MINLEN",
        0,
    )

    normalized_a = {name: _normalize_account_display(A.get(name)) for name in bureaus}
    normalized_b = {name: _normalize_account_display(B.get(name)) for name in bureaus}

    best_match_aux: Dict[str, Any] | None = None
    best_match_rank = -1
    best_pair_rank: Tuple[int, int] | None = None
    best_any_aux: Dict[str, Any] | None = None
    best_any_rank = -1
    first_aux: Dict[str, Any] | None = None
    candidate_entries: List[Dict[str, Any]] = []

    for left in bureaus:
        left_norm = normalized_a[left]
        for right in bureaus:
            right_norm = normalized_b[right]

            reasons: List[str] = []
            left_has_digits = bool(left_norm.has_digits)
            right_has_digits = bool(right_norm.has_digits)

            candidate_entry: Dict[str, Any] = {
                "left_bureau": left,
                "right_bureau": right,
                "normalized_values": {
                    "a": left_norm.digits if left_has_digits else "",
                    "b": right_norm.digits if right_has_digits else "",
                },
                "raw_values": {
                    "a": left_norm.raw,
                    "b": right_norm.raw,
                },
                "visible_digits": {
                    "a": left_norm.visible_digits,
                    "b": right_norm.visible_digits,
                },
                "hard_match_enabled": bool(hard_enabled),
                "evaluated": False,
                "matched": False,
                "match_score": 0.0,
                "min_length_met": False,
            }

            if not left_has_digits:
                reasons.append("missing_left_digits")
            if not right_has_digits:
                reasons.append("missing_right_digits")

            if not (left_has_digits and right_has_digits):
                if reasons:
                    candidate_entry["reasons"] = reasons
                candidate_entries.append(candidate_entry)
                continue

            level_value, level_debug = acctnum_match_level(
                left_norm.raw,
                right_norm.raw,
            )
            level = _sanitize_acct_level(level_value)
            level_rank = _ACCOUNT_LEVEL_ORDER.get(level, 0)
            matched = hard_enabled and level == "exact_or_known_match"

            min_length_met = (
                left_norm.visible_digits >= min_length
                and right_norm.visible_digits >= min_length
            )

            result_aux: Dict[str, Any] = {
                "best_pair": (left, right),
                "normalized_values": (
                    left_norm.digits,
                    right_norm.digits,
                ),
                "acctnum_level": level,
                "acctnum_digits_len_a": len(left_norm.digits),
                "acctnum_digits_len_b": len(right_norm.digits),
                "raw_values": {"a": left_norm.raw, "b": right_norm.raw},
                # Surface whether the configured minimum length gate allowed the comparison.
                "min_length_met": min_length_met,
                "acctnum_debug": level_debug,
            }

            reasons_list: List[str] = []
            if not min_length_met:
                reasons_list.append("min_length_not_met")
            if not matched:
                reasons_list.append(f"level:{level}")
                if not hard_enabled:
                    reasons_list.append("hard_match_disabled")

            candidate_entry.update(
                {
                    "acctnum_level": level,
                    "min_length_met": bool(min_length_met),
                    "evaluated": True,
                    "match_score": 1.0 if matched and min_length_met else 0.0,
                    "matched": bool(matched and min_length_met),
                    "reasons": reasons_list,
                }
            )
            if isinstance(level_debug, Mapping):
                candidate_entry["debug"] = {
                    key: str(value)
                    for key, value in level_debug.items()
                }
            candidate_entries.append(candidate_entry)

            if first_aux is None:
                first_aux = dict(result_aux)

            if not min_length_met:
                # Respect the configured minimum by skipping matches that do not satisfy it.
                if level_rank > best_any_rank:
                    best_any_rank = level_rank
                    best_any_aux = dict(result_aux)
                continue

            if level_rank > best_any_rank:
                best_any_rank = level_rank
                best_any_aux = dict(result_aux)

            if not matched:
                continue

            pick = False
            if best_match_aux is None:
                pick = True
            else:
                prev_level = _sanitize_acct_level(best_match_aux.get("acctnum_level"))
                prev_rank = _ACCOUNT_LEVEL_ORDER.get(prev_level, 0)
                if level_rank > prev_rank:
                    pick = True
                elif level_rank == prev_rank:
                    pair_rank = (bureau_positions[left], bureau_positions[right])
                    if best_pair_rank is None or pair_rank < best_pair_rank:
                        pick = True

            if pick:
                best_match_aux = dict(result_aux)
                best_pair_rank = (bureau_positions[left], bureau_positions[right])
                best_match_rank = level_rank

    if best_match_aux is not None:
        best_payload = dict(best_match_aux)
        best_payload["matched"] = True
        best_payload["matched_bool"] = True
        best_payload["match_score"] = 1.0
        best_payload["candidate_pairs"] = list(candidate_entries)
        return True, best_payload

    if best_any_aux is not None:
        best_any_payload = dict(best_any_aux)
        best_any_payload["matched"] = False
        best_any_payload["matched_bool"] = False
        best_any_payload["match_score"] = 0.0
        best_any_payload["candidate_pairs"] = list(candidate_entries)
        return False, best_any_payload

    if first_aux is not None:
        first_payload = dict(first_aux)
        first_payload["matched"] = False
        first_payload["matched_bool"] = False
        first_payload["match_score"] = 0.0
        first_payload["candidate_pairs"] = list(candidate_entries)
        return False, first_payload

    return False, {"candidate_pairs": list(candidate_entries)}


def soft_acct_suspicious(a_display: str, b_display: str) -> bool:
    matched, _ = acctnum.acctnum_visible_match(a_display, b_display)
    return matched


def _detect_soft_acct_match(
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
) -> bool:
    bureaus = ("transunion", "experian", "equifax")

    for left_key in bureaus:
        left_branch = left_bureaus.get(left_key)
        if not isinstance(left_branch, Mapping):
            continue
        left_candidates = []
        for field_name in ("account_number_display", "account_number"):
            candidate = left_branch.get(field_name)
            if is_missing(candidate):
                continue
            left_candidates.append(str(candidate))
        if not left_candidates:
            continue

        for right_key in bureaus:
            right_branch = right_bureaus.get(right_key)
            if not isinstance(right_branch, Mapping):
                continue
            right_candidates = []
            for field_name in ("account_number_display", "account_number"):
                candidate = right_branch.get(field_name)
                if is_missing(candidate):
                    continue
                right_candidates.append(str(candidate))
            if not right_candidates:
                continue

            for left_display in left_candidates:
                for right_display in right_candidates:
                    if soft_acct_suspicious(left_display, right_display):
                        return True

    return False


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_zero_amount(value: Any) -> bool:
    try:
        return abs(float(value)) <= _AMOUNT_ZERO_EPSILON
    except (TypeError, ValueError):
        return False


def _both_amounts_positive(pair: Any) -> bool:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    try:
        return float(pair[0]) > 0 and float(pair[1]) > 0
    except (TypeError, ValueError):
        return False


def _resolve_tolerance_float(cfg: MergeCfg, key: str, default: float) -> float:
    """Return a float tolerance value guarded against bad inputs."""

    raw = cfg.tolerances.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _resolve_tolerance_int(cfg: MergeCfg, key: str, default: int) -> int:
    """Return an integer tolerance value guarded against bad inputs."""

    raw = cfg.tolerances.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _account_number_min_length(cfg: Optional[MergeCfg] = None) -> int:
    """Return the configured minimum visible-digit length for account matches."""

    if cfg is not None:
        return _resolve_tolerance_int(cfg, "MERGE_ACCOUNTNUMBER_MATCH_MINLEN", 0)

    # Fall back to the global merge configuration when no explicit cfg is passed.
    try:
        active_cfg = get_merge_cfg()
    except Exception:
        return 0

    try:
        return _resolve_tolerance_int(active_cfg, "MERGE_ACCOUNTNUMBER_MATCH_MINLEN", 0)
    except Exception:
        return 0


def _match_field_values(
    field: str,
    norm_a: Any,
    norm_b: Any,
    raw_a: Any,
    raw_b: Any,
    cfg: MergeCfg,
) -> Tuple[float, Dict[str, Any]]:
    """Apply the appropriate predicate for a normalized pair of values."""

    aux: Dict[str, Any] = {}

    def _finalize(match: bool, score: float) -> Tuple[float, Dict[str, Any]]:
        matched_flag = bool(match)
        aux["matched_bool"] = matched_flag
        aux["matched"] = matched_flag
        return max(0.0, min(1.0, float(score))), aux

    if field == "balance_owed":
        points_mode_active = bool(getattr(cfg, "points_mode", False))
        if points_mode_active:
            left_cents = normalize_balance_owed_cents(raw_a)
            right_cents = normalize_balance_owed_cents(raw_b)
            matched = (
                left_cents is not None
                and right_cents is not None
                and left_cents == right_cents
            )
            # Hard guard: points mode never uses configured tolerances for balances.
            aux["tolerance"] = {"abs": 0.0, "ratio": 0.0}
            aux["points_mode_balance_cents"] = (
                left_cents,
                right_cents,
            )
        else:
            tol_abs = _resolve_tolerance_float(
                cfg,
                "MERGE_TOL_BALANCE_ABS",
                0.0,
            )
            tol_ratio = _resolve_tolerance_float(
                cfg,
                "MERGE_TOL_BALANCE_RATIO",
                0.0,
            )
            matched = match_balance_owed(
                norm_a,
                norm_b,
                tol_abs=tol_abs,
                tol_ratio=tol_ratio,
            )
            aux["tolerance"] = {"abs": tol_abs, "ratio": tol_ratio}
        if matched and (_is_zero_amount(norm_a) or _is_zero_amount(norm_b)):
            return _finalize(False, 0.0)
        return _finalize(matched, 1.0 if matched else 0.0)

    if field == "payment_amount":
        tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
        tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))
        count_zero = int(cfg.tolerances.get("COUNT_ZERO_PAYMENT_MATCH", 0))
        matched = match_payment_amount(
            norm_a,
            norm_b,
            tol_abs=tol_abs,
            tol_ratio=tol_ratio,
            count_zero_payment_match=count_zero,
        )
        return _finalize(matched, 1.0 if matched else 0.0)

    if field in _AMOUNT_FIELDS:
        tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
        tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))
        matched = match_amount_field(norm_a, norm_b, tol_abs=tol_abs, tol_ratio=tol_ratio)
        if field in _ZERO_AMOUNT_FIELDS and (
            _is_zero_amount(norm_a) or _is_zero_amount(norm_b)
        ):
            return _finalize(False, 0.0)
        return _finalize(matched, 1.0 if matched else 0.0)

    if field == "last_payment":
        day_tol = int(cfg.tolerances.get("LAST_PAYMENT_DAY_TOL", 0))
        matched = date_within(norm_a, norm_b, day_tol)
        return _finalize(matched, 1.0 if matched else 0.0)

    if field in _DATE_FIELDS_DET:
        if field == "date_opened":
            tol_days = _resolve_tolerance_int(
                cfg,
                "MERGE_TOL_DATE_DAYS",
                0,
            )
            matched = date_within(norm_a, norm_b, tol_days)
            aux["tolerance_days"] = tol_days
            return _finalize(matched, 1.0 if matched else 0.0)
        matched = date_equal(norm_a, norm_b)
        return _finalize(matched, 1.0 if matched else 0.0)

    if field in {"history_2y", "history_7y"}:
        threshold = _resolve_tolerance_float(
            cfg,
            "MERGE_HISTORY_SIMILARITY_THRESHOLD",
            1.0,
        )
        matched, similarity = match_history_field(
            norm_a,
            norm_b,
            threshold=threshold,
        )
        similarity = max(0.0, min(1.0, float(similarity)))
        aux["similarity"] = similarity
        aux["threshold"] = threshold
        return _finalize(matched, similarity)

    if field in _TYPE_FIELDS:
        matched = norm_a == norm_b and norm_a is not None and norm_b is not None
        return _finalize(matched, 1.0 if matched else 0.0)

    raise KeyError(f"Unsupported merge field: {field}")


def match_field_best_of_9(
    field_name: str,
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    """Check all cross-bureau pairs for a field and return best match metadata."""

    if not isinstance(A, Mapping):
        A = {}
    if not isinstance(B, Mapping):
        B = {}

    bureaus = ("transunion", "experian", "equifax")
    bureau_positions = {name: idx for idx, name in enumerate(bureaus)}
    field_key = str(field_name)

    if field_key == "account_number":
        return _match_account_number_best_pair(A, B, cfg)

    best_aux: Dict[str, Any] | None = None
    best_score = -1
    best_matched_aux: Dict[str, Any] | None = None
    best_matched_score = -1
    first_candidate_aux: Dict[str, Any] | None = None
    best_pair_rank: tuple[int, int] | None = None

    for left in bureaus:
        left_branch = A.get(left)
        if not isinstance(left_branch, Mapping):
            continue
        raw_left = left_branch.get(field_key)
        if field_key == "account_number" and is_missing(raw_left):
            raw_left = left_branch.get("account_number_display")
        if is_missing(raw_left):
            continue
        norm_left = _normalize_field_value(field_key, raw_left)
        if norm_left is None:
            continue

        for right in bureaus:
            right_branch = B.get(right)
            if not isinstance(right_branch, Mapping):
                continue
            raw_right = right_branch.get(field_key)
            if is_missing(raw_right):
                continue
            norm_right = _normalize_field_value(field_key, raw_right)
            if norm_right is None:
                continue

            match_score, aux = _match_field_values(
                field_key, norm_left, norm_right, raw_left, raw_right, cfg
            )
            matched = bool(aux.get("matched_bool", match_score >= 1.0))

            result_aux = {
                "best_pair": (left, right),
                "normalized_values": _serialize_normalized_pair(norm_left, norm_right),
            }
            result_aux["match_score"] = match_score
            result_aux.update(aux)

            if first_candidate_aux is None:
                first_candidate_aux = dict(result_aux)

            if field_key == "account_number":
                level = _sanitize_acct_level(aux.get("acctnum_level"))
                level_score = _ACCOUNT_LEVEL_ORDER.get(level, 0)
                if level_score > best_score:
                    best_score = level_score
                    best_aux = dict(result_aux)
                if matched and level_score >= best_matched_score:
                    best_matched_score = level_score
                    best_matched_aux = dict(result_aux)
                # Continue searching for a better match even after finding one.
                continue

            if matched:
                pair_rank = (bureau_positions[left], bureau_positions[right])
                if best_matched_aux is None or (
                    best_pair_rank is None or pair_rank < best_pair_rank
                ):
                    best_matched_aux = dict(result_aux)
                    best_pair_rank = pair_rank

    if best_matched_aux is not None:
        return True, best_matched_aux
    if first_candidate_aux is not None:
        return False, first_candidate_aux

    return False, {}


def _collect_normalized_field_values(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> List[Any]:
    values: List[Any] = []
    if not isinstance(bureaus, Mapping):
        return values

    for bureau_key in ("transunion", "experian", "equifax"):
        branch = bureaus.get(bureau_key)
        if not isinstance(branch, Mapping):
            continue
        raw_value = branch.get(field)
        if is_missing(raw_value):
            continue
        norm_value = _normalize_field_value(field, raw_value)
        if norm_value is None:
            continue
        values.append(norm_value)
    return values


def _iter_bureau_field_pairs(
    left_bureaus: Mapping[str, Mapping[str, Any]] | None,
    right_bureaus: Mapping[str, Mapping[str, Any]] | None,
    field: str,
) -> Iterator[Tuple[str, str, Any, Any, Any, Any]]:
    """Yield normalized cross-bureau value pairs for a merge field."""

    if not isinstance(left_bureaus, Mapping) or not isinstance(right_bureaus, Mapping):
        return

    bureaus = ("transunion", "experian", "equifax")

    for left_key in bureaus:
        left_branch = left_bureaus.get(left_key)
        if not isinstance(left_branch, Mapping):
            continue

        left_raw = left_branch.get(field)
        if is_missing(left_raw):
            continue

        left_norm = _normalize_field_value(field, left_raw)
        if left_norm is None:
            continue

        for right_key in bureaus:
            right_branch = right_bureaus.get(right_key)
            if not isinstance(right_branch, Mapping):
                continue

            right_raw = right_branch.get(field)
            if is_missing(right_raw):
                continue

            right_norm = _normalize_field_value(field, right_raw)
            if right_norm is None:
                continue

            yield (left_key, right_key, left_norm, right_norm, left_raw, right_raw)


def _points_mode_balance_has_exact_match(
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return True when any bureau pair has an exact balance match."""

    left_pairs = _points_mode_collect_balance_cents(left_bureaus)
    right_pairs = _points_mode_collect_balance_cents(right_bureaus)

    for _, left_cents in left_pairs:
        if left_cents is None or left_cents == 0:
            continue
        for _, right_cents in right_pairs:
            if right_cents is None or right_cents == 0:
                continue
            if left_cents == right_cents:
                return True

    return False


def _points_mode_collect_balance_cents(
    bureaus: Mapping[str, Mapping[str, Any]] | None,
) -> Tuple[Tuple[str, Optional[int]], ...]:
    """Return (bureau, cents) tuples with strict normalization for balances."""

    if not isinstance(bureaus, Mapping):
        return tuple()

    pairs: List[Tuple[str, Optional[int]]] = []
    for bureau in ("transunion", "experian", "equifax"):
        cents: Optional[int] = None
        branch = bureaus.get(bureau)
        if isinstance(branch, Mapping):
            raw_value = branch.get("balance_owed")
            if not is_missing(raw_value):
                cents = normalize_balance_owed_cents(raw_value)
        pairs.append((bureau, cents))

    return tuple(pairs)


_BUREAU_ABBREVIATIONS = {
    "transunion": "TU",
    "experian": "EX",
    "equifax": "EQ",
}


def _points_mode_diagnostics_enabled(cfg: MergeCfg) -> bool:
    """Return True when verbose points diagnostics are enabled."""

    return bool(getattr(cfg, "points_diagnostics", False))


def _format_bureau_pair_label(left: str, right: str) -> str:
    left_code = _BUREAU_ABBREVIATIONS.get(str(left).lower(), str(left).upper())
    right_code = _BUREAU_ABBREVIATIONS.get(str(right).lower(), str(right).upper())
    return f"{left_code}~{right_code}"


def _sanitize_account_number_value(value: Any) -> str:
    if value is None:
        return ""
    return str(_redact_account_number_raw(str(value)))


def _sanitize_field_value_for_log(field: str, value: Any) -> str:
    if value is None:
        return ""
    if field == "account_number":
        return _sanitize_account_number_value(value)
    return str(value)


def _sanitize_normalized_value_for_log(field: str, value: Any) -> Any:
    sanitized = _serialize_normalized_value(value)
    if sanitized is None:
        return ""
    if field == "account_number":
        return _sanitize_account_number_value(sanitized)
    return sanitized


def _sanitize_candidate_entries(
    field: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    preferred_pair: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return sanitized candidate diagnostics focused on cross-bureau pairs."""

    pair_lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for candidate in candidates or []:
        pair_raw = candidate.get("pair") if isinstance(candidate, Mapping) else None
        if isinstance(pair_raw, (list, tuple)) and len(pair_raw) == 2:
            left = str(pair_raw[0])
            right = str(pair_raw[1])
        else:
            left = str(candidate.get("left_bureau", ""))
            right = str(candidate.get("right_bureau", ""))
        if not left or not right:
            continue
        pair_lookup[(left, right)] = candidate

    ordered_pairs = [
        ("transunion", "experian"),
        ("transunion", "equifax"),
        ("experian", "equifax"),
    ]

    sanitized: List[Dict[str, Any]] = []
    for pair in ordered_pairs:
        candidate = pair_lookup.get(pair)
        if candidate is None:
            # fall back to reverse orientation if available
            candidate = pair_lookup.get((pair[1], pair[0]))
        if candidate is None:
            continue

        raw_values = candidate.get("raw_values") if isinstance(candidate, Mapping) else {}
        normalized_values = candidate.get("normalized_values") if isinstance(candidate, Mapping) else {}

        entry: Dict[str, Any] = {
            "pair": _format_bureau_pair_label(pair[0], pair[1]),
            "evaluated": bool(candidate.get("evaluated")),
            "matched": bool(candidate.get("matched")),
            "pass": bool(candidate.get("evaluated") and candidate.get("matched")),
            "match_score": float(candidate.get("match_score", 0.0) or 0.0),
            "reasons": list(candidate.get("reasons", [])) if isinstance(candidate.get("reasons"), (list, tuple)) else [],
        }

        raw_left = raw_values.get("a") if isinstance(raw_values, Mapping) else None
        raw_right = raw_values.get("b") if isinstance(raw_values, Mapping) else None
        norm_left = normalized_values.get("a") if isinstance(normalized_values, Mapping) else None
        norm_right = normalized_values.get("b") if isinstance(normalized_values, Mapping) else None

        entry["raw_left"] = _sanitize_field_value_for_log(field, raw_left)
        entry["raw_right"] = _sanitize_field_value_for_log(field, raw_right)
        entry["norm_left"] = _sanitize_normalized_value_for_log(field, norm_left)
        entry["norm_right"] = _sanitize_normalized_value_for_log(field, norm_right)

        tolerance = candidate.get("tolerance") if isinstance(candidate, Mapping) else None
        if isinstance(tolerance, Mapping):
            entry["tolerance"] = {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in tolerance.items()
            }

        if "similarity" in candidate:
            try:
                entry["similarity"] = float(candidate.get("similarity"))
            except (TypeError, ValueError):
                entry["similarity"] = candidate.get("similarity")
        if "threshold" in candidate:
            try:
                entry["threshold"] = float(candidate.get("threshold"))
            except (TypeError, ValueError):
                entry["threshold"] = candidate.get("threshold")

        visible_digits = candidate.get("visible_digits") if isinstance(candidate, Mapping) else None
        if isinstance(visible_digits, Mapping):
            entry["visible_digits"] = {
                side: int(value) if isinstance(value, (int, float)) else 0
                for side, value in visible_digits.items()
            }
        if "hard_match_enabled" in candidate:
            entry["hard_match"] = bool(candidate.get("hard_match_enabled"))
        if "acctnum_level" in candidate:
            entry["acctnum_level"] = str(candidate.get("acctnum_level"))
        if "min_length_met" in candidate:
            entry["min_length_met"] = bool(candidate.get("min_length_met"))
        entry["selected"] = bool(candidate.get("selected"))

        sanitized.append(entry)

    if not sanitized:
        return []

    # Ensure the preferred pair, when supplied, preserves the 'selected' flag even if orientation differs.
    if preferred_pair is not None:
        preferred_label = _format_bureau_pair_label(preferred_pair[0], preferred_pair[1])
        for entry in sanitized:
            if entry["pair"] == preferred_label:
                entry["selected"] = True
                break

    return sanitized


def _format_points_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_points_pair_snippet(pair_entry: Mapping[str, Any]) -> str:
    label = str(pair_entry.get("bureaus") or "")
    if not label:
        label = "?"
    if pair_entry.get("pass"):
        status = "pass"
    elif not pair_entry.get("evaluated"):
        status = "skip"
    else:
        status = "fail"

    segments: List[str] = [f"{label}:{status}"]

    raw_left = str(pair_entry.get("raw_left") or "")
    raw_right = str(pair_entry.get("raw_right") or "")
    if raw_left or raw_right:
        segments.append(f"raw={raw_left}~{raw_right}")

    norm_left = str(pair_entry.get("normalized_left") or "")
    norm_right = str(pair_entry.get("normalized_right") or "")
    if norm_left or norm_right:
        segments.append(f"norm={norm_left}~{norm_right}")

    tolerance = pair_entry.get("tolerance")
    tol_bits: List[str] = []
    if isinstance(tolerance, Mapping):
        abs_tol = tolerance.get("abs")
        ratio_tol = tolerance.get("ratio")
        days_tol = tolerance.get("days")
        if abs_tol not in (None, ""):
            tol_bits.append(f"abs<={_format_points_float(abs_tol)}")
        if ratio_tol not in (None, ""):
            tol_bits.append(f"ratio<={_format_points_float(ratio_tol)}")
        if days_tol not in (None, ""):
            tol_bits.append(f"days<={_format_points_float(days_tol)}")
    if tol_bits:
        segments.append("tol=" + " ".join(tol_bits))

    similarity = pair_entry.get("similarity")
    threshold = pair_entry.get("threshold")
    if similarity not in (None, "") and threshold not in (None, ""):
        try:
            sim_val = float(similarity)
            thr_val = float(threshold)
        except (TypeError, ValueError):
            segments.append(f"sim={similarity} thr={threshold}")
        else:
            comparator = "≥" if sim_val >= thr_val else "<"
            segments.append(f"sim={_format_points_float(sim_val)}{comparator}{_format_points_float(thr_val)}")

    reasons = pair_entry.get("reasons")
    if isinstance(reasons, (list, tuple)) and reasons:
        segments.append("reasons=" + ",".join(str(r) for r in reasons))

    return " ".join(segments)


def _format_points_field_line(
    field: str,
    pair_label: str,
    field_payload: Mapping[str, Any],
) -> str:
    matched = "true" if field_payload.get("matched") else "false"
    points_value = _format_points_float(
        field_payload.get("points_awarded", field_payload.get("awarded", 0.0))
    )
    reason = str(field_payload.get("reason") or "n/a")
    pairs_data = field_payload.get("pairs")
    if isinstance(pairs_data, (list, tuple)):
        pairs_iter = [str(item) for item in pairs_data if item]
    else:
        pairs_iter = []
    pairs_text = ", ".join(pairs_iter)
    line = (
        f"FIELDS {pair_label} {field.upper()} matched={matched} "
        f"points={points_value} reason={reason} "
        f"pairs={ '[' + pairs_text + ']' if pairs_text else '[]' }"
    )
    skip_reasons = field_payload.get("skip_reasons")
    if isinstance(skip_reasons, (list, tuple)) and skip_reasons:
        skip_text = ", ".join(str(item) for item in skip_reasons if item)
        if skip_text:
            line += f" skip_reasons=[{skip_text}]"
    return line


def _format_points_oc_gate_line(pair_label: str, oc_gate: Mapping[str, Any]) -> str:
    required = "true" if oc_gate.get("required") else "false"
    present_a = "true" if oc_gate.get("present_a") else "false"
    present_b = "true" if oc_gate.get("present_b") else "false"
    action = str(oc_gate.get("action") or "pass")
    reason = oc_gate.get("reason")
    parts = [
        "OC_GATE" if not pair_label else f"OC_GATE {pair_label}",
        f"required={required}",
        f"present_a={present_a}",
        f"present_b={present_b}",
        f"action={action}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    return " ".join(parts)


def _candidate_pair_identifier(entry: Mapping[str, Any]) -> str:
    pair = entry.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        left, right = pair
        label = f"{left}-{right}"
    elif isinstance(pair, str):
        label = pair
    else:
        label = "unknown"
    return str(label).replace("~", "-").lower()


def _points_pair_entry_from_candidate(field: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "pair": _candidate_pair_identifier(entry),
        "matched": bool(entry.get("matched")),
        "pass": bool(entry.get("pass")),
        "evaluated": bool(entry.get("evaluated")),
        "match_score": float(entry.get("match_score", 0.0) or 0.0),
        "selected": bool(entry.get("selected")),
        "reasons": list(entry.get("reasons", []))
        if isinstance(entry.get("reasons"), (list, tuple))
        else [],
        "raw_values": {
            "a": entry.get("raw_left", ""),
            "b": entry.get("raw_right", ""),
        },
        "normalized_values": {
            "a": entry.get("norm_left", ""),
            "b": entry.get("norm_right", ""),
        },
    }

    tolerance = entry.get("tolerance")
    if isinstance(tolerance, Mapping):
        payload["tolerance"] = {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in tolerance.items()
        }

    for optional_key in ("similarity", "threshold", "acctnum_level", "min_length_met"):
        if optional_key in entry:
            payload[optional_key] = entry[optional_key]
    if "visible_digits" in entry and isinstance(entry["visible_digits"], Mapping):
        payload["visible_digits"] = {
            key: int(value) if isinstance(value, (int, float)) else 0
            for key, value in entry["visible_digits"].items()
        }
    if "hard_match" in entry:
        payload["hard_match"] = bool(entry.get("hard_match"))

    return payload


def _build_points_field_reason(
    field: str,
    *,
    matched: bool,
    aux_entry: Mapping[str, Any],
    skip_reasons: Sequence[str],
    cfg: MergeConfig,
) -> str:
    if matched:
        if field == "account_number":
            acct_level = aux_entry.get("points_mode_acctnum_level")
            if acct_level:
                return f"digits_match(level={acct_level})"
            min_len = _account_number_min_length(cfg)
            return f"digits_match(minlen>={min_len})" if min_len else "digits_match"
        if field in {"account_type", "account_status"}:
            return "equal"
        if field == "account_label":
            return "normalized_match"
        if field == "date_opened":
            tolerance_days = aux_entry.get("tolerance_days")
            if isinstance(tolerance_days, (int, float)):
                return f"within_days({int(tolerance_days)})"
            return "equal"
        if field == "balance_owed":
            if aux_entry.get("points_mode_exact_match"):
                return "exact_match"
            tolerance = aux_entry.get("tolerance")
            fragments: List[str] = []
            if isinstance(tolerance, Mapping):
                abs_tol = tolerance.get("abs")
                if isinstance(abs_tol, (int, float)):
                    fragments.append(f"abs<={_format_points_float(abs_tol)}")
                ratio_tol = tolerance.get("ratio")
                if isinstance(ratio_tol, (int, float)):
                    fragments.append(f"ratio<={_format_points_float(ratio_tol)}")
            return "within_tolerance" + (f"({', '.join(fragments)})" if fragments else "")
        if field in {"history_2y", "history_7y"}:
            threshold = aux_entry.get("threshold")
            if isinstance(threshold, (int, float)):
                return f"similarity>={_format_points_float(threshold)}"
            return "similarity_match"
        return "match"

    skip_set = set(skip_reasons or [])
    if "outside_tolerance_days" in skip_set:
        tolerance_days = aux_entry.get("tolerance_days")
        if isinstance(tolerance_days, (int, float)):
            return f"diff_days>{int(tolerance_days)}"
        return "diff_days>tolerance"
    if "outside_tolerance" in skip_set:
        tolerance = aux_entry.get("tolerance")
        descriptors: List[str] = []
        if isinstance(tolerance, Mapping):
            abs_tol = tolerance.get("abs")
            if isinstance(abs_tol, (int, float)):
                descriptors.append(f"abs>{_format_points_float(abs_tol)}")
            ratio_tol = tolerance.get("ratio")
            if isinstance(ratio_tol, (int, float)):
                descriptors.append(f"ratio>{_format_points_float(ratio_tol)}")
        return "delta>tolerance" + (f"({', '.join(descriptors)})" if descriptors else "")
    if "below_threshold" in skip_set:
        threshold = aux_entry.get("threshold")
        if isinstance(threshold, (int, float)):
            return f"similarity<{_format_points_float(threshold)}"
        return "similarity<threshold"
    if "no_matching_pair" in skip_set:
        return "no_pair_available"
    return "no_match"


_DUPLICATE_THRESHOLD_FALLBACK = 7.0


def _resolve_duplicate_points_threshold(cfg: MergeConfig) -> float:
    thresholds = getattr(cfg, "thresholds", None)
    candidate: Any = None
    if isinstance(thresholds, Mapping):
        for key in (
            "MERGE_DUPLICATE_POINTS_THRESHOLD",
            "duplicate_points_threshold",
            "MERGE_POINTS_DUPLICATE_THRESHOLD",
        ):
            value = thresholds.get(key)
            if value not in (None, ""):
                candidate = value
                break
    if candidate is None:
        candidate = getattr(cfg, "duplicate_points_threshold", None)
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return float(_DUPLICATE_THRESHOLD_FALLBACK)


_DEFAULT_HISTORY_BUREAU_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("transunion", "experian"),
    ("transunion", "equifax"),
    ("experian", "equifax"),
)

_ACCOUNT_LABEL_SYNONYMS: Dict[str, str] = {
    "NELNET": "NELNET",
    "NELNETSERVICING": "NELNET",
    "NELNETINC": "NELNET",
}


def _apply_account_label_synonym(value: str) -> str:
    return _ACCOUNT_LABEL_SYNONYMS.get(value, value)


def _normalize_account_label_value(value: Any, *, normalize: bool = True) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if not normalize:
        return text.upper()
    upper = text.upper()
    collapsed = re.sub(r"[\s\-_]+", " ", upper)
    stripped = re.sub(r"[^A-Z0-9]", "", collapsed)
    if not stripped:
        return None
    return _apply_account_label_synonym(stripped)


def _load_points_mode_artifacts(case_root: Optional[Path]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "heading_guess": None,
        "history_2y": {},
        "history_7y": {},
        "meta": {},
    }
    if case_root is None:
        return artifacts

    case_path = Path(case_root)
    artifacts["case_root"] = str(case_path)

    meta_path = case_path / "meta.json"
    artifacts["meta_path"] = str(meta_path)
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta_payload = json.load(handle)
            if isinstance(meta_payload, Mapping):
                artifacts["meta"] = dict(meta_payload)
                heading_guess = meta_payload.get("heading_guess")
                if not is_missing(heading_guess):
                    artifacts["heading_guess"] = heading_guess
        except Exception:
            logger.debug(
                "MERGE_POINTS_ARTIFACTS_META_LOAD_FAILED case_root=%s",
                case_path,
                exc_info=True,
            )

    bureaus_path = case_path / "bureaus.json"
    artifacts["bureaus_path"] = str(bureaus_path)
    if bureaus_path.exists():
        try:
            with bureaus_path.open("r", encoding="utf-8") as handle:
                bureaus_payload = json.load(handle)
            if isinstance(bureaus_payload, Mapping):
                history_2y_raw = bureaus_payload.get("two_year_payment_history")
                if isinstance(history_2y_raw, Mapping):
                    artifacts["history_2y"] = {
                        str(bureau): list(values)
                        for bureau, values in history_2y_raw.items()
                        if isinstance(values, (list, tuple))
                    }
                history_7y_raw = bureaus_payload.get("seven_year_history")
                if isinstance(history_7y_raw, Mapping):
                    artifacts["history_7y"] = {
                        str(bureau): {
                            "late30": _safe_int(branch.get("late30")),
                            "late60": _safe_int(branch.get("late60")),
                            "late90": _safe_int(branch.get("late90")),
                        }
                        for bureau, branch in history_7y_raw.items()
                        if isinstance(branch, Mapping)
                    }
        except Exception:
            logger.debug(
                "MERGE_POINTS_ARTIFACTS_BUREAUS_LOAD_FAILED case_root=%s",
                case_path,
                exc_info=True,
            )

    return artifacts


def _resolve_account_artifacts(
    artifacts: Optional[Mapping[str, Any]],
    case_root: Optional[Path],
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    if isinstance(artifacts, Mapping):
        resolved.update(artifacts)

    needs_load = False
    if case_root is not None:
        for key in ("heading_guess", "history_2y", "history_7y", "meta"):
            if key not in resolved:
                needs_load = True
                break
    if needs_load:
        disk_artifacts = _load_points_mode_artifacts(case_root)
        for key, value in disk_artifacts.items():
            resolved.setdefault(key, value)

    meta_payload = resolved.get("meta")
    if not isinstance(meta_payload, Mapping):
        meta_payload = {}
        resolved["meta"] = meta_payload

    if "heading_guess" not in resolved or is_missing(resolved.get("heading_guess")):
        heading_guess = meta_payload.get("heading_guess") if isinstance(meta_payload, Mapping) else None
        if not is_missing(heading_guess):
            resolved["heading_guess"] = heading_guess
        else:
            resolved.setdefault("heading_guess", None)

    history_2y_payload = resolved.get("history_2y")
    if not isinstance(history_2y_payload, Mapping):
        resolved["history_2y"] = {}

    history_7y_payload = resolved.get("history_7y")
    if not isinstance(history_7y_payload, Mapping):
        resolved["history_7y"] = {}

    if case_root is not None:
        resolved.setdefault("case_root", str(Path(case_root)))

    return resolved


def _build_inline_points_mode_context(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract points-mode context data embedded in a bureaus payload."""

    context: Dict[str, Any] = {}
    if not isinstance(payload, Mapping):
        return context

    meta_payload = payload.get("meta")
    if isinstance(meta_payload, Mapping):
        context["meta"] = dict(meta_payload)
        heading_guess = meta_payload.get("heading_guess")
        if not is_missing(heading_guess):
            context["heading_guess"] = heading_guess

    heading_inline = payload.get("heading_guess")
    if not is_missing(heading_inline):
        context.setdefault("heading_guess", heading_inline)

    def _coerce_history_2y(raw_map: Mapping[str, Any]) -> Dict[str, List[Any]]:
        coerced: Dict[str, List[Any]] = {}
        for bureau, entries in raw_map.items():
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                coerced[str(bureau)] = [item for item in entries]
        return coerced

    def _coerce_history_2y_from_monthly_tsv_v2(raw_map: Mapping[str, Any]) -> Dict[str, List[str]]:
        coerced: Dict[str, List[str]] = {}
        for bureau, entries in raw_map.items():
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                statuses: List[str] = []
                for entry in entries:
                    if isinstance(entry, Mapping):
                        status = entry.get("status")
                        if status is None:
                            statuses.append("--")
                        else:
                            status_text = str(status)
                            statuses.append(status_text if status_text != "" else "--")
                    else:
                        statuses.append(str(entry))
                coerced[str(bureau)] = statuses
        return coerced

    def _coerce_history_7y(raw_map: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
        coerced: Dict[str, Dict[str, int]] = {}
        for bureau, branch in raw_map.items():
            if not isinstance(branch, Mapping):
                continue
            coerced[str(bureau)] = {
                "late30": _safe_int(branch.get("late30")),
                "late60": _safe_int(branch.get("late60")),
                "late90": _safe_int(branch.get("late90")),
            }
        return coerced

    history_2y_payload = payload.get("history_2y")
    if isinstance(history_2y_payload, Mapping):
        context["history_2y"] = _coerce_history_2y(history_2y_payload)
    else:
        history_2y_monthly = payload.get("two_year_payment_history_monthly_tsv_v2")
        if isinstance(history_2y_monthly, Mapping):
            context["history_2y"] = _coerce_history_2y_from_monthly_tsv_v2(history_2y_monthly)
        else:
            history_2y_fallback = payload.get("two_year_payment_history")
            if isinstance(history_2y_fallback, Mapping):
                context["history_2y"] = _coerce_history_2y(history_2y_fallback)

    history_7y_payload = payload.get("history_7y")
    if isinstance(history_7y_payload, Mapping):
        context["history_7y"] = _coerce_history_7y(history_7y_payload)
    else:
        history_7y_fallback = payload.get("seven_year_history")
        if isinstance(history_7y_fallback, Mapping):
            context["history_7y"] = _coerce_history_7y(history_7y_fallback)

    meta_path_value = payload.get("meta_path")
    if isinstance(meta_path_value, str):
        context["meta_path"] = Path(meta_path_value)
    elif isinstance(meta_path_value, Path):
        context["meta_path"] = meta_path_value

    bureaus_path_value = payload.get("bureaus_path")
    if isinstance(bureaus_path_value, str):
        context["history_source_path"] = Path(bureaus_path_value)
    elif isinstance(bureaus_path_value, Path):
        context["history_source_path"] = bureaus_path_value

    inline_context = payload.get("points_mode_context")
    if isinstance(inline_context, Mapping):
        _merge_points_mode_context(context, inline_context)

    return context


def _merge_points_mode_context(
    target: Dict[str, Any],
    source: Optional[Mapping[str, Any]],
) -> None:
    """Merge ``source`` context data into ``target`` without losing detail."""

    if not isinstance(source, Mapping):
        return

    for key, value in source.items():
        if key in {"history_2y", "history_7y"} and isinstance(value, Mapping):
            existing = target.get(key)
            merged: Dict[str, Any] = {}
            if isinstance(existing, Mapping):
                merged.update({str(k): v for k, v in existing.items()})
            for bureau, branch in value.items():
                merged[str(bureau)] = branch
            target[key] = merged
            continue

        if key == "meta" and isinstance(value, Mapping):
            existing_meta = target.get("meta")
            merged_meta: Dict[str, Any] = {}
            if isinstance(existing_meta, Mapping):
                merged_meta.update(existing_meta)
            for meta_key, meta_value in value.items():
                if meta_key not in merged_meta or is_missing(merged_meta[meta_key]):
                    merged_meta[meta_key] = meta_value
            target["meta"] = merged_meta
            continue

        if key == "heading_guess":
            if key not in target or is_missing(target.get(key)):
                target[key] = value
            continue

        if key in {"meta_path", "history_source_path"} and isinstance(value, str):
            target[key] = Path(value)
            continue

        if key not in target or target.get(key) is None:
            target[key] = value

def _resolve_account_label_value(
    source: str,
    context: Mapping[str, Any],
    bureaus_map: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    source_key = str(source or "").strip().lower()
    if not source_key or source_key == "bureaus":
        for bureau in ("transunion", "experian", "equifax"):
            branch = bureaus_map.get(bureau)
            if not isinstance(branch, Mapping):
                continue
            candidate = branch.get("account_label")
            if not is_missing(candidate):
                return str(candidate)
        return None

    if source_key in {"heading_guess", "meta.heading_guess"}:
        heading_candidate = context.get("heading_guess")
        if not is_missing(heading_candidate):
            return str(heading_candidate)
        meta_payload = context.get("meta")
        if isinstance(meta_payload, Mapping):
            heading_candidate = meta_payload.get("heading_guess")
            if not is_missing(heading_candidate):
                return str(heading_candidate)
        return None

    if source_key.startswith("meta."):
        meta_key = source_key.split(".", 1)[1] or "heading_guess"
        meta_payload = context.get("meta")
        if isinstance(meta_payload, Mapping):
            candidate = meta_payload.get(meta_key)
            if not is_missing(candidate):
                return str(candidate)
        return None

    if source_key.startswith("bureaus."):
        branch_key = source_key.split(".", 1)[1] or "account_label"
        for bureau in ("transunion", "experian", "equifax"):
            branch = bureaus_map.get(bureau)
            if not isinstance(branch, Mapping):
                continue
            candidate = branch.get(branch_key)
            if not is_missing(candidate):
                return str(candidate)
        return None

    direct_candidate = context.get(source_key)
    if not is_missing(direct_candidate):
        return str(direct_candidate)
    meta_payload = context.get("meta")
    if isinstance(meta_payload, Mapping):
        direct_candidate = meta_payload.get(source_key)
        if not is_missing(direct_candidate):
            return str(direct_candidate)
    return None


def _points_mode_compare_account_label(
    left_context: Mapping[str, Any],
    right_context: Mapping[str, Any],
    *,
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    source = getattr(cfg, "account_label_source", "meta.heading_guess")
    normalize_flag = bool(getattr(cfg, "account_label_normalize", True))

    raw_left = _resolve_account_label_value(source, left_context, left_bureaus)
    raw_right = _resolve_account_label_value(source, right_context, right_bureaus)

    norm_left = _normalize_account_label_value(raw_left, normalize=normalize_flag)
    norm_right = _normalize_account_label_value(raw_right, normalize=normalize_flag)

    have_left = norm_left is not None
    have_right = norm_right is not None
    matched = bool(have_left and have_right and norm_left == norm_right)

    if matched:
        reason = "normalized_label_equal"
    elif not have_left or not have_right:
        reason = "label_missing"
    else:
        reason = "normalized_label_diff"

    candidate_entries: List[Dict[str, Any]] = []
    for left_bureau, right_bureau in _DEFAULT_HISTORY_BUREAU_PAIRS:
        entry: Dict[str, Any] = {
            "pair": (left_bureau, right_bureau),
            "raw_values": {"a": raw_left, "b": raw_right},
            "normalized_values": {"a": norm_left, "b": norm_right},
        }
        if not have_left or not have_right:
            entry["evaluated"] = False
            entry["matched"] = False
            entry["match_score"] = 0.0
            reasons: List[str] = []
            if not have_left:
                reasons.append("left_missing_value")
            if not have_right:
                reasons.append("right_missing_value")
            entry["reasons"] = reasons
        else:
            entry["evaluated"] = True
            entry["matched"] = matched
            entry["match_score"] = 1.0 if matched else 0.0
            entry["reasons"] = [] if matched else ["normalized_label_diff"]
        candidate_entries.append(entry)

    aux: Dict[str, Any] = {
        "matched": matched,
        "matched_bool": matched,
        "match_score": 1.0 if matched else 0.0,
        "reason": reason,
        "normalized_values": (norm_left, norm_right),
        "raw_values": {"a": raw_left, "b": raw_right},
        "candidate_pairs": candidate_entries,
    }
    if candidate_entries:
        best_pair = candidate_entries[0]["pair"]
        aux["best_pair"] = best_pair
        aux["best_pair_label"] = _format_bureau_pair_label(best_pair[0], best_pair[1])

    return matched, aux


def _coerce_history_sequence(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip().upper() for item in value if str(item).strip()]
        return tokens or None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        tokens = [token.strip().upper() for token in text.replace(",", " ").split() if token.strip()]
        return tokens or None
    return None


def _coerce_history_counts(value: Any) -> Optional[Dict[str, int]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            "late30": _safe_int(value.get("late30")),
            "late60": _safe_int(value.get("late60")),
            "late90": _safe_int(value.get("late90")),
        }

    if isinstance(value, (list, tuple, set)):
        tokens = [str(item) for item in value]
    else:
        tokens = str(value).replace(",", " ").split()

    counts = {"late30": 0, "late60": 0, "late90": 0}
    for raw_token in tokens:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        if "90" in token:
            counts["late90"] += 1
        elif "60" in token:
            counts["late60"] += 1
        elif "30" in token or token == "late":
            counts["late30"] += 1
    return counts


def _points_mode_compare_history_2y(
    left_context: Mapping[str, Any],
    right_context: Mapping[str, Any],
    *,
    threshold: float,
) -> Tuple[bool, Dict[str, Any]]:
    left_history_value = left_context.get("history_2y")
    right_history_value = right_context.get("history_2y")

    if not isinstance(left_history_value, Mapping):
        sequence = _coerce_history_sequence(left_history_value)
        left_history: Dict[str, Sequence[str] | None] = {}
        if sequence is not None:
            for bureau in ("transunion", "experian", "equifax"):
                left_history[bureau] = list(sequence)
        else:
            left_history = {}
    else:
        left_history = dict(left_history_value)

    if not isinstance(right_history_value, Mapping):
        sequence = _coerce_history_sequence(right_history_value)
        right_history: Dict[str, Sequence[str] | None] = {}
        if sequence is not None:
            for bureau in ("transunion", "experian", "equifax"):
                right_history[bureau] = list(sequence)
        else:
            right_history = {}
    else:
        right_history = dict(right_history_value)

    candidate_entries: List[Dict[str, Any]] = []
    best_info: Optional[Dict[str, Any]] = None
    best_similarity = -1.0
    best_match_info: Optional[Dict[str, Any]] = None
    best_match_similarity = -1.0

    for left_bureau, right_bureau in _DEFAULT_HISTORY_BUREAU_PAIRS:
        left_raw = left_history.get(left_bureau)
        right_raw = right_history.get(right_bureau)

        entry: Dict[str, Any] = {
            "pair": (left_bureau, right_bureau),
            "raw_values": {
                "a": list(left_raw) if isinstance(left_raw, (list, tuple)) else left_raw,
                "b": list(right_raw) if isinstance(right_raw, (list, tuple)) else right_raw,
            },
        }

        left_sequence = _coerce_history_sequence(left_raw)
        if not left_sequence:
            entry.update(
                {
                    "evaluated": False,
                    "matched": False,
                    "match_score": 0.0,
                    "reasons": ["left_missing_value"],
                }
            )
            candidate_entries.append(entry)
            continue

        right_sequence = _coerce_history_sequence(right_raw)
        if not right_sequence:
            entry.update(
                {
                    "evaluated": False,
                    "matched": False,
                    "match_score": 0.0,
                    "reasons": ["right_missing_value"],
                }
            )
            candidate_entries.append(entry)
            continue

        left_norm = list(left_sequence)
        right_norm = list(right_sequence)

        if not left_norm:
            entry.update(
                {
                    "evaluated": False,
                    "matched": False,
                    "match_score": 0.0,
                    "reasons": ["left_missing_value"],
                }
            )
            candidate_entries.append(entry)
            continue
        if not right_norm:
            entry.update(
                {
                    "evaluated": False,
                    "matched": False,
                    "match_score": 0.0,
                    "reasons": ["right_missing_value"],
                }
            )
            candidate_entries.append(entry)
            continue

        compared_len = min(len(left_norm), len(right_norm))
        matches = sum(
            1
            for a_val, b_val in zip(reversed(left_norm), reversed(right_norm))
            if a_val == b_val
        )
        similarity = matches / compared_len if compared_len else 0.0
        matched = similarity >= threshold and compared_len > 0

        reason = ""
        if matched:
            reason = "similarity_match"
        elif compared_len == 0:
            reason = "no_overlap"
        else:
            reason = "below_threshold"

        entry.update(
            {
                "evaluated": compared_len > 0,
                "matched": matched,
                "match_score": float(similarity),
                "similarity": float(similarity),
                "threshold": float(threshold),
                "compared": int(compared_len),
                "matches": int(matches),
                "normalized_values": {
                    "a": " ".join(left_norm),
                    "b": " ".join(right_norm),
                },
                "reasons": [] if matched else ([reason] if reason else []),
            }
        )

        candidate_entries.append(entry)

        info_payload = {
            "best_pair": (left_bureau, right_bureau),
            "best_pair_label": _format_bureau_pair_label(left_bureau, right_bureau),
            "matched": matched,
            "matched_bool": matched,
            "match_score": float(similarity),
            "similarity": float(similarity),
            "threshold": float(threshold),
            "compared": int(compared_len),
            "matches": int(matches),
            "normalized_values": (" ".join(left_norm), " ".join(right_norm)),
            "raw_values": {
                "a": " ".join(str(item) for item in left_raw),
                "b": " ".join(str(item) for item in right_raw),
            },
            "reason": reason,
        }

        if matched and similarity > best_match_similarity:
            best_match_similarity = similarity
            best_match_info = info_payload

        if similarity > best_similarity:
            best_similarity = similarity
            best_info = info_payload

    selected = best_match_info or best_info
    if not selected:
        return False, {"candidate_pairs": candidate_entries}

    aux = dict(selected)
    aux["candidate_pairs"] = candidate_entries
    return bool(aux.get("matched")), aux


def _points_mode_compare_history_7y(
    left_context: Mapping[str, Any],
    right_context: Mapping[str, Any],
    *,
    threshold: float,
) -> Tuple[bool, Dict[str, Any]]:
    left_history = (
        left_context.get("history_7y")
        if isinstance(left_context.get("history_7y"), Mapping)
        else {}
    )
    right_history = (
        right_context.get("history_7y")
        if isinstance(right_context.get("history_7y"), Mapping)
        else {}
    )

    def _coerce_branch(payload: Any) -> Optional[Dict[str, int]]:
        if not isinstance(payload, Mapping):
            return None
        return {
            "late30": _safe_int(payload.get("late30")),
            "late60": _safe_int(payload.get("late60")),
            "late90": _safe_int(payload.get("late90")),
        }

    candidate_entries: List[Dict[str, Any]] = []
    best_info: Optional[Dict[str, Any]] = None
    best_similarity = -1.0
    best_match_info: Optional[Dict[str, Any]] = None
    best_match_similarity = -1.0

    for left_bureau, right_bureau in _DEFAULT_HISTORY_BUREAU_PAIRS:
        left_branch = _coerce_branch(left_history.get(left_bureau))
        right_branch = _coerce_branch(right_history.get(right_bureau))

        entry: Dict[str, Any] = {
            "pair": (left_bureau, right_bureau),
        }

        if left_branch is None:
            counts = _coerce_history_counts(left_history.get(left_bureau))
            if counts is not None:
                left_branch = counts
            else:
                entry.update(
                    {
                        "evaluated": False,
                        "matched": False,
                        "match_score": 0.0,
                        "reasons": ["left_missing_value"],
                    }
                )
                candidate_entries.append(entry)
                continue
        if right_branch is None:
            counts = _coerce_history_counts(right_history.get(right_bureau))
            if counts is not None:
                right_branch = counts
            else:
                entry.update(
                    {
                        "evaluated": False,
                        "matched": False,
                        "match_score": 0.0,
                        "reasons": ["right_missing_value"],
                    }
                )
                candidate_entries.append(entry)
                continue

        sum_left = sum(max(0, _safe_int(left_branch.get(key))) for key in ("late30", "late60", "late90"))
        sum_right = sum(max(0, _safe_int(right_branch.get(key))) for key in ("late30", "late60", "late90"))
        total = sum_left + sum_right
        diff_sum = sum(
            abs(_safe_int(left_branch.get(key)) - _safe_int(right_branch.get(key)))
            for key in ("late30", "late60", "late90")
        )

        if total == 0:
            similarity = 1.0
        else:
            similarity = max(0.0, 1.0 - (diff_sum / float(total)))

        matched = similarity >= threshold

        entry.update(
            {
                "evaluated": True,
                "matched": matched,
                "match_score": float(similarity),
                "similarity": float(similarity),
                "threshold": float(threshold),
                "diff_sum": int(diff_sum),
                "total": int(total),
                "normalized_values": {
                    "a": {
                        "late30": _safe_int(left_branch.get("late30")),
                        "late60": _safe_int(left_branch.get("late60")),
                        "late90": _safe_int(left_branch.get("late90")),
                    },
                    "b": {
                        "late30": _safe_int(right_branch.get("late30")),
                        "late60": _safe_int(right_branch.get("late60")),
                        "late90": _safe_int(right_branch.get("late90")),
                    },
                },
                "raw_values": {
                    "a": left_branch,
                    "b": right_branch,
                },
                "reasons": [] if matched else ["below_threshold"],
            }
        )

        candidate_entries.append(entry)

        info_payload = {
            "best_pair": (left_bureau, right_bureau),
            "best_pair_label": _format_bureau_pair_label(left_bureau, right_bureau),
            "matched": matched,
            "matched_bool": matched,
            "match_score": float(similarity),
            "similarity": float(similarity),
            "threshold": float(threshold),
            "diff_sum": int(diff_sum),
            "total": int(total),
            "normalized_values": (
                {
                    "late30": _safe_int(left_branch.get("late30")),
                    "late60": _safe_int(left_branch.get("late60")),
                    "late90": _safe_int(left_branch.get("late90")),
                },
                {
                    "late30": _safe_int(right_branch.get("late30")),
                    "late60": _safe_int(right_branch.get("late60")),
                    "late90": _safe_int(right_branch.get("late90")),
                },
            ),
            "raw_values": {
                "a": left_branch,
                "b": right_branch,
            },
        }

        if matched and similarity > best_match_similarity:
            best_match_similarity = similarity
            best_match_info = dict(info_payload)

        if similarity > best_similarity:
            best_similarity = similarity
            best_info = dict(info_payload)

    selected = best_match_info or best_info
    if not selected:
        return False, {"candidate_pairs": candidate_entries}

    aux = dict(selected)
    aux["candidate_pairs"] = candidate_entries
    return bool(aux.get("matched")), aux

def _log_points_mode_field_failure(
    *,
    field: str,
    matched_flag: bool,
    part_value: float,
    parts_map: Mapping[str, float],
    matched_map: Mapping[str, bool],
    aux_entry: Any,
    reason: str,
) -> None:
    """Emit a diagnostic entry before raising an invariant assertion."""

    try:
        aux_payload = dict(aux_entry) if isinstance(aux_entry, Mapping) else aux_entry
    except Exception:  # pragma: no cover - defensive
        aux_payload = aux_entry

    logger.error(
        "[MERGE] Points-mode invariant failure %s",
        {
            "field": field,
            "matched": matched_flag,
            "part": float(part_value),
            "parts": {k: float(v) for k, v in dict(parts_map).items()},
            "matched_flags": dict(matched_map),
            "aux_entry": aux_payload,
            "reason": reason,
        },
    )


def _ensure_points_mode_field_invariants(
    *,
    field: str,
    matched_flag: bool,
    part_value: float,
    parts_map: Mapping[str, float],
    matched_map: Mapping[str, bool],
    aux_entry: Any,
) -> None:
    """Validate per-field invariants and log rich context on failure."""

    if not matched_flag and part_value > 0.0:
        _log_points_mode_field_failure(
            field=field,
            matched_flag=matched_flag,
            part_value=part_value,
            parts_map=parts_map,
            matched_map=matched_map,
            aux_entry=aux_entry,
            reason="positive_part_without_match",
        )
        assert matched_flag, f"points-mode mismatch: {field} part without matched flag"
    elif not matched_flag and abs(part_value) >= 1e-9:
        _log_points_mode_field_failure(
            field=field,
            matched_flag=matched_flag,
            part_value=part_value,
            parts_map=parts_map,
            matched_map=matched_map,
            aux_entry=aux_entry,
            reason="part_without_match",
        )
        assert abs(part_value) < 1e-9, (
            f"points-mode mismatch: {field} has part {part_value} without match"
        )

    if not isinstance(aux_entry, Mapping):
        _log_points_mode_field_failure(
            field=field,
            matched_flag=matched_flag,
            part_value=part_value,
            parts_map=parts_map,
            matched_map=matched_map,
            aux_entry=aux_entry,
            reason="aux_not_mapping",
        )
        assert isinstance(aux_entry, Mapping), (
            f"points-mode aux missing mapping for {field}"
        )

    if bool(aux_entry.get("matched")) != matched_flag:
        _log_points_mode_field_failure(
            field=field,
            matched_flag=matched_flag,
            part_value=part_value,
            parts_map=parts_map,
            matched_map=matched_map,
            aux_entry=aux_entry,
            reason="aux_mismatch",
        )
        assert bool(aux_entry.get("matched")) == matched_flag, (
            f"points-mode mismatch: {field} aux matched flag {aux_entry.get('matched')}"
            f" != {matched_flag}"
        )


def _points_mode_match_field_any_bureau(
    field: str,
    left_bureaus: Mapping[str, Mapping[str, Any]],
    right_bureaus: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> Tuple[bool, Dict[str, Any]]:
    """Return match metadata for a field using strict points-mode semantics."""

    if field == "account_number":
        matched, aux = _match_account_number_best_pair(left_bureaus, right_bureaus, cfg)
        return matched, dict(aux)

    bureaus = ("transunion", "experian", "equifax")
    bureau_positions = {name: idx for idx, name in enumerate(bureaus)}
    best_aux: Dict[str, Any] | None = None
    best_rank: Tuple[int, int] | None = None
    first_aux: Dict[str, Any] | None = None
    candidate_entries: List[Dict[str, Any]] = []

    def _normalize_branch(
        bureaus_map: Mapping[str, Mapping[str, Any]],
        bureau: str,
    ) -> Tuple[bool, Any, Any, str]:
        if not isinstance(bureaus_map, Mapping):
            return False, None, None, "missing_branch"
        branch = bureaus_map.get(bureau)
        if not isinstance(branch, Mapping):
            return False, None, None, "missing_branch"
        raw_value = branch.get(field)
        if is_missing(raw_value):
            return False, raw_value, None, "missing_value"
        normalized = _normalize_field_value(field, raw_value)
        if normalized is None:
            return False, raw_value, None, "normalize_failed"
        return True, raw_value, normalized, ""

    left_cache = {
        bureau: _normalize_branch(left_bureaus, bureau)
        for bureau in bureaus
    }
    right_cache = {
        bureau: _normalize_branch(right_bureaus, bureau)
        for bureau in bureaus
    }

    for left_key in bureaus:
        left_valid, left_raw, left_norm, left_reason = left_cache[left_key]
        for right_key in bureaus:
            right_valid, right_raw, right_norm, right_reason = right_cache[right_key]

            candidate_entry: Dict[str, Any] = {
                "left_bureau": left_key,
                "right_bureau": right_key,
                "raw_values": {
                    "a": left_raw,
                    "b": right_raw,
                },
                "normalized_values": {
                    "a": _serialize_normalized_value(left_norm),
                    "b": _serialize_normalized_value(right_norm),
                },
                "evaluated": False,
                "matched": False,
                "match_score": 0.0,
                "reasons": [],
            }

            evaluated = bool(left_valid and right_valid)
            if not evaluated:
                reasons: List[str] = []
                if not left_valid and left_reason:
                    reasons.append(f"left_{left_reason}")
                if not right_valid and right_reason:
                    reasons.append(f"right_{right_reason}")
                candidate_entry["reasons"] = reasons
                candidate_entries.append(candidate_entry)
                continue

            match_score, aux = _match_field_values(
                field,
                left_norm,
                right_norm,
                left_raw,
                right_raw,
                cfg,
            )
            matched = bool(aux.get("matched_bool", match_score >= 1.0))

            result_aux: Dict[str, Any] = {
                "best_pair": (left_key, right_key),
                "normalized_values": _serialize_normalized_pair(left_norm, right_norm),
                "match_score": match_score,
            }
            result_aux.update(aux)

            reasons_list: List[str] = []
            if not matched:
                reasons_list.append("no_match")
                if "threshold" in aux:
                    reasons_list.append("below_threshold")
                if "tolerance" in aux:
                    reasons_list.append("outside_tolerance")
                if "tolerance_days" in aux:
                    reasons_list.append("outside_tolerance_days")

            candidate_entry.update(
                {
                    "evaluated": True,
                    "matched": matched,
                    "match_score": float(match_score),
                    "reasons": reasons_list,
                }
            )

            if "similarity" in aux:
                try:
                    candidate_entry["similarity"] = float(aux["similarity"])
                except (TypeError, ValueError):
                    candidate_entry["similarity"] = aux["similarity"]
            if "threshold" in aux:
                try:
                    candidate_entry["threshold"] = float(aux["threshold"])
                except (TypeError, ValueError):
                    candidate_entry["threshold"] = aux["threshold"]
            if "tolerance" in aux and isinstance(aux.get("tolerance"), Mapping):
                tolerance_map = aux.get("tolerance")
                candidate_entry["tolerance"] = {
                    key: float(value)
                    if isinstance(value, (int, float))
                    else value
                    for key, value in dict(tolerance_map).items()
                }

            candidate_entries.append(candidate_entry)

            if first_aux is None:
                first_aux = dict(result_aux)

            if matched:
                pair_rank = (bureau_positions[left_key], bureau_positions[right_key])
                if best_aux is None or best_rank is None or pair_rank < best_rank:
                    best_aux = dict(result_aux)
                    best_rank = pair_rank

    if best_aux is not None:
        best_payload = dict(best_aux)
        best_payload["candidate_pairs"] = list(candidate_entries)
        return True, best_payload
    if first_aux is not None:
        first_payload = dict(first_aux)
        first_payload["candidate_pairs"] = list(candidate_entries)
        return False, first_payload
    return False, {"candidate_pairs": list(candidate_entries)}




def _detect_amount_conflicts(
    A: Mapping[str, Mapping[str, Any]],
    B: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
) -> List[str]:
    conflicts: List[str] = []
    points_mode_active = bool(getattr(cfg, "points_mode", False))
    allowed_fields = set(_resolve_points_mode_allowlist(cfg))
    tol_abs = float(cfg.tolerances.get("AMOUNT_TOL_ABS", 0.0))
    tol_ratio = float(cfg.tolerances.get("AMOUNT_TOL_RATIO", 0.0))

    if points_mode_active:
        # Hard guard: points mode never applies balance tolerances, even if
        # configured. Other amount fields (if any) still respect tol_abs/tol_ratio.
        balance_tol_abs = 0.0
        balance_tol_ratio = 0.0
    else:
        balance_tol_abs = _resolve_tolerance_float(
            cfg,
            "MERGE_TOL_BALANCE_ABS",
            tol_abs,
        )
        balance_tol_ratio = _resolve_tolerance_float(
            cfg,
            "MERGE_TOL_BALANCE_RATIO",
            tol_ratio,
        )
    amount_fields: Iterable[str] = (
        field for field in _AMOUNT_CONFLICT_FIELDS if field in allowed_fields
    )

    for field in amount_fields:
        if field == "balance_owed":
            if points_mode_active:
                if _points_mode_balance_has_exact_match(A, B):
                    continue
                conflicts.append("amount_conflict:balance_owed")
                continue

            values_a = _collect_normalized_field_values(A, field)
            values_b = _collect_normalized_field_values(B, field)
        else:
            values_a = _collect_normalized_field_values(A, field)
            values_b = _collect_normalized_field_values(B, field)

        if not values_a or not values_b:
            continue

        conflict = True
        if field == "balance_owed":
            for left in values_a:
                for right in values_b:
                    if match_balance_owed(
                        left,
                        right,
                        tol_abs=balance_tol_abs,
                        tol_ratio=balance_tol_ratio,
                    ):
                        conflict = False
                        break
                if not conflict:
                    break
        else:
            for left in values_a:
                for right in values_b:
                    if match_amount_field(left, right, tol_abs=tol_abs, tol_ratio=tol_ratio):
                        conflict = False
                        break
                if not conflict:
                    break

        if conflict:
            conflicts.append(f"amount_conflict:{field}")

    return conflicts


def _score_pair_points_mode(
    A_data: Mapping[str, Mapping[str, Any]],
    B_data: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
    *,
    field_sequence: Sequence[str],
    weights_map: Mapping[str, float],
    debug_enabled: bool,
    left_context: Optional[Mapping[str, Any]] = None,
    right_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the allow-listed weighted score for points mode."""

    allowlist_fields = _resolve_points_mode_allowlist(cfg)
    allowlist_set = set(allowlist_fields)
    evaluated_fields = [field for field in field_sequence if field in allowlist_set]
    field_matches: Dict[str, float] = {}
    field_contributions: Dict[str, float] = {}
    field_weights: Dict[str, float] = {}
    field_breakdown: Dict[str, Dict[str, Any]] = {}
    fields_aux: Dict[str, Dict[str, Any]] = {}
    parts: Dict[str, float] = {}
    matched_bools: Dict[str, bool] = {}
    total_points = 0.0
    diagnostics_entries: List[Tuple[str, float, float, float, float]] = []
    points_breakdown_fields: Dict[str, Dict[str, Any]] = {}
    points_breakdown: Optional[Dict[str, Any]] = None

    left_context_map: Dict[str, Any] = {}
    right_context_map: Dict[str, Any] = {}

    _merge_points_mode_context(left_context_map, left_context)
    _merge_points_mode_context(left_context_map, _build_inline_points_mode_context(A_data))
    _merge_points_mode_context(right_context_map, right_context)
    _merge_points_mode_context(right_context_map, _build_inline_points_mode_context(B_data))

    global _POINTS_MODE_DIAGNOSTICS_EMITTED
    diagnostics_pair_index = _POINTS_MODE_DIAGNOSTICS_EMITTED + 1

    balance_exact_match = False
    if "balance_owed" in allowlist_set:
        balance_exact_match = _points_mode_balance_has_exact_match(A_data, B_data)

    for field in evaluated_fields:
        parts[field] = 0.0
        if field == "account_label":
            matched, match_aux = _points_mode_compare_account_label(
                left_context_map,
                right_context_map,
                left_bureaus=A_data,
                right_bureaus=B_data,
                cfg=cfg,
            )
        elif field == "history_2y":
            history_threshold = _resolve_tolerance_float(
                cfg,
                "MERGE_HISTORY_SIMILARITY_THRESHOLD",
                1.0,
            )
            matched, match_aux = _points_mode_compare_history_2y(
                left_context_map,
                right_context_map,
                threshold=history_threshold,
            )
        elif field == "history_7y":
            history_threshold = _resolve_tolerance_float(
                cfg,
                "MERGE_HISTORY_SIMILARITY_THRESHOLD",
                1.0,
            )
            matched, match_aux = _points_mode_compare_history_7y(
                left_context_map,
                right_context_map,
                threshold=history_threshold,
            )
        else:
            matched, match_aux = _points_mode_match_field_any_bureau(
                field, A_data, B_data, cfg
            )
        aux = dict(match_aux) if isinstance(match_aux, Mapping) else {}

        if "match_score" in aux:
            try:
                aux["raw_match_score"] = float(aux["match_score"])
            except (TypeError, ValueError):
                aux.pop("match_score", None)

        weight = float(weights_map.get(field, 0.0))
        matched_flag = bool(aux.get("matched_bool", aux.get("matched", matched)))

        if field == "account_number":
            resolved_level = _normalize_points_mode_acct_level(aux.get("acctnum_level"))
            aux["points_mode_acctnum_level"] = resolved_level
            matched_flag = resolved_level in _POINTS_MODE_ACCOUNT_NUMBER_MATCH_LEVELS

        if field == "balance_owed":
            aux["points_mode_exact_match"] = bool(balance_exact_match)
            matched_flag = bool(balance_exact_match)

        match_score = 1.0 if matched_flag else 0.0
        contribution = weight if matched_flag else 0.0
        if matched_flag:
            parts[field] = weight
        else:
            parts[field] = 0.0
        matched_bools[field] = matched_flag

        aux["matched"] = matched_flag
        aux["matched_bool"] = matched_flag
        aux["points_mode_matched_bool"] = matched_flag
        aux["match_score"] = match_score
        aux["weight"] = weight
        aux["contribution"] = contribution

        field_matches[field] = match_score
        field_contributions[field] = contribution
        field_weights[field] = weight
        fields_aux[field] = aux
        field_breakdown[field] = {
            "match": match_score,
            "weight": weight,
            "contribution": contribution,
            "matched": matched_flag,
        }

        current_total = total_points + contribution

        diagnostics_entries.append(
            (
                field,
                match_score,
                weight,
                contribution,
                current_total,
            )
        )

        total_points = current_total

        if debug_enabled:
            logger.info(
                "[MERGE] Points-mode comparing %s: match=%.3f weight=%.3f contribution=%.3f",
                field,
                match_score,
                weight,
                contribution,
            )

    ignored_fields = [field for field in field_sequence if field not in allowlist_set]

    def _coerce_threshold(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    ai_threshold = _coerce_threshold(getattr(cfg, "ai_points_threshold", 3.0), 3.0)
    direct_threshold = _coerce_threshold(getattr(cfg, "direct_points_threshold", 5.0), 5.0)
    conflicts = list(_detect_amount_conflicts(A_data, B_data, cfg))
    has_conflict = bool(conflicts)

    triggers: List[str] = []
    trigger_events: List[Dict[str, Any]] = []
    decision = "different"

    if (
        total_points >= direct_threshold
        and not has_conflict
        and balance_exact_match
    ):
        decision = "auto"
        triggers.append("points:direct")
        trigger_events.append(
            {
                "kind": "points_direct",
                "details": {
                    "score_points": float(total_points),
                    "threshold": float(direct_threshold),
                },
            }
        )
    elif total_points >= ai_threshold:
        decision = "ai"
        triggers.append("points:ai")
        trigger_events.append(
            {
                "kind": "points_ai",
                "details": {
                    "score_points": float(total_points),
                    "threshold": float(ai_threshold),
                },
            }
        )

    diagnostics_entry_map: Dict[str, Tuple[float, float, float, float]] = {
        field_name: (
            float(match_score),
            float(weight_entry),
            float(contribution_entry),
            float(running_total),
        )
        for field_name, match_score, weight_entry, contribution_entry, running_total in diagnostics_entries
    }

    diagnostics_fields_verbose: List[Dict[str, Any]] = []
    for field in evaluated_fields:
        aux_entry = fields_aux.get(field, {})
        best_pair_raw = aux_entry.get("best_pair") if isinstance(aux_entry, Mapping) else None
        best_pair_tuple: Tuple[str, str] | None = None
        if isinstance(best_pair_raw, (list, tuple)) and len(best_pair_raw) == 2:
            best_pair_tuple = (str(best_pair_raw[0]), str(best_pair_raw[1]))
        best_pair_label = (
            _format_bureau_pair_label(best_pair_tuple[0], best_pair_tuple[1])
            if best_pair_tuple is not None
            else None
        )

        entry_info = diagnostics_entry_map.get(field)
        weight_value = float(field_weights.get(field, 0.0))
        match_score_value = float(entry_info[0]) if entry_info else float(field_matches.get(field, 0.0))
        running_total_value = float(entry_info[3]) if entry_info else float(total_points)
        matched_flag = bool(matched_bools.get(field, False))
        contribution_value = float(field_contributions.get(field, 0.0))

        candidate_payloads: List[Dict[str, Any]] = []
        raw_candidates = aux_entry.get("candidate_pairs") if isinstance(aux_entry, Mapping) else None
        if isinstance(raw_candidates, list):
            for candidate in raw_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                left_bureau = str(candidate.get("left_bureau") or "")
                right_bureau = str(candidate.get("right_bureau") or "")
                raw_values_map = candidate.get("raw_values") if isinstance(candidate.get("raw_values"), Mapping) else {}
                normalized_values_raw = candidate.get("normalized_values")
                if isinstance(normalized_values_raw, Mapping):
                    normalized_values_iter = normalized_values_raw.items()
                elif isinstance(normalized_values_raw, (list, tuple)) and len(normalized_values_raw) == 2:
                    normalized_values_iter = (("a", normalized_values_raw[0]), ("b", normalized_values_raw[1]))
                else:
                    normalized_values_iter = (())
                normalized_values_sanitized = {
                    key: _sanitize_normalized_value_for_log(field, value)
                    for key, value in normalized_values_iter
                }
                if "a" not in normalized_values_sanitized:
                    normalized_values_sanitized["a"] = _sanitize_normalized_value_for_log(field, None)
                if "b" not in normalized_values_sanitized:
                    normalized_values_sanitized["b"] = _sanitize_normalized_value_for_log(field, None)

                reasons_raw = candidate.get("reasons")
                if isinstance(reasons_raw, (list, tuple)):
                    reasons_list = [str(reason) for reason in reasons_raw if reason not in {None, ""}]
                elif isinstance(reasons_raw, str) and reasons_raw:
                    reasons_list = [reasons_raw]
                else:
                    reasons_list = []

                candidate_payload = {
                    "pair": [left_bureau, right_bureau],
                    "matched": bool(candidate.get("matched")),
                    "evaluated": bool(candidate.get("evaluated")),
                    "match_score": float(candidate.get("match_score", 0.0) or 0.0),
                    "raw_values": {
                        "a": _sanitize_field_value_for_log(field, raw_values_map.get("a")),
                        "b": _sanitize_field_value_for_log(field, raw_values_map.get("b")),
                    },
                    "normalized_values": normalized_values_sanitized,
                    "reasons": tuple(sorted(set(reasons_list))),
                }

                if "similarity" in candidate:
                    try:
                        candidate_payload["similarity"] = float(candidate.get("similarity"))
                    except (TypeError, ValueError):
                        candidate_payload["similarity"] = candidate.get("similarity")
                if "threshold" in candidate:
                    try:
                        candidate_payload["threshold"] = float(candidate.get("threshold"))
                    except (TypeError, ValueError):
                        candidate_payload["threshold"] = candidate.get("threshold")
                if "tolerance" in candidate and isinstance(candidate.get("tolerance"), Mapping):
                    tolerance_map = candidate.get("tolerance")
                    candidate_payload["tolerance"] = {
                        key: float(value)
                        if isinstance(value, (int, float))
                        else value
                        for key, value in dict(tolerance_map).items()
                    }
                if "min_length_met" in candidate:
                    candidate_payload["min_length_met"] = bool(candidate.get("min_length_met"))
                if "acctnum_level" in candidate:
                    candidate_payload["acctnum_level"] = str(candidate.get("acctnum_level"))
                if "visible_digits" in candidate and isinstance(candidate.get("visible_digits"), Mapping):
                    visible_map = candidate.get("visible_digits")
                    candidate_payload["visible_digits"] = {
                        key: int(value)
                        if isinstance(value, (int, float))
                        else 0
                        for key, value in dict(visible_map).items()
                    }
                if "hard_match_enabled" in candidate:
                    candidate_payload["hard_match_enabled"] = bool(candidate.get("hard_match_enabled"))

                candidate_payload["selected"] = bool(
                    matched_flag
                    and best_pair_tuple is not None
                    and (left_bureau, right_bureau) == best_pair_tuple
                    and candidate_payload["evaluated"]
                    and candidate_payload["matched"]
                )

                candidate_payloads.append(candidate_payload)

        skip_reasons: List[str] = []
        if not matched_flag:
            aggregated: set[str] = set()
            evaluated_candidates = [c for c in candidate_payloads if c.get("evaluated")]
            if evaluated_candidates:
                matched_candidates = [c for c in evaluated_candidates if c.get("matched")]
                if not matched_candidates:
                    for candidate in evaluated_candidates:
                        reasons = candidate.get("reasons")
                        if isinstance(reasons, (list, tuple)):
                            aggregated.update(str(reason) for reason in reasons if reason)
            else:
                for candidate in candidate_payloads:
                    reasons = candidate.get("reasons")
                    if isinstance(reasons, (list, tuple)):
                        aggregated.update(str(reason) for reason in reasons if reason)
            if not aggregated:
                aggregated.add("no_matching_pair")
            skip_reasons = sorted(aggregated)

        sanitized_candidates: List[Dict[str, Any]] = _sanitize_candidate_entries(
            field,
            candidate_payloads,
            preferred_pair=best_pair_tuple,
        )
        pairs_verbose: List[Dict[str, Any]] = [
            _points_pair_entry_from_candidate(field, candidate)
            for candidate in sanitized_candidates
        ]
        pair_labels: List[str] = [
            entry.get("pair", "unknown")
            for entry in pairs_verbose
        ]

        field_entry: Dict[str, Any] = {
            "field": field,
            "weight": weight_value,
            "match_score": match_score_value,
            "matched": matched_flag,
            "contribution": contribution_value,
            "weight_counted": matched_flag,
            "running_total": running_total_value,
            "candidates": sanitized_candidates,
        }
        if not matched_flag and skip_reasons:
            field_entry["skip_reasons"] = list(skip_reasons)

        diagnostics_fields_verbose.append(field_entry)

        reason_text = _build_points_field_reason(
            field,
            matched=matched_flag,
            aux_entry=aux_entry,
            skip_reasons=skip_reasons,
            cfg=cfg,
        )

        breakdown_field: Dict[str, Any] = {
            "matched": matched_flag,
            "points_awarded": float(contribution_value),
            "weight": float(weight_value),
            "weight_counted": matched_flag,
            "match_score": float(match_score_value),
            "reason": reason_text,
            "pairs": pair_labels,
            "pairs_verbose": pairs_verbose,
        }
        if best_pair_label:
            breakdown_field["best_pair"] = str(best_pair_label).replace("~", "-").lower()
        if skip_reasons:
            breakdown_field["skip_reasons"] = list(skip_reasons)

        tolerance = aux_entry.get("tolerance") if isinstance(aux_entry, Mapping) else None
        if isinstance(tolerance, Mapping):
            tol_payload: Dict[str, Any] = {}
            if "abs" in tolerance and isinstance(tolerance["abs"], (int, float)):
                tol_payload["abs"] = float(tolerance["abs"])
            if "ratio" in tolerance and isinstance(tolerance["ratio"], (int, float)):
                tol_payload["ratio"] = float(tolerance["ratio"])
            if tol_payload:
                breakdown_field["tolerance"] = tol_payload

        tol_days = aux_entry.get("tolerance_days") if isinstance(aux_entry, Mapping) else None
        if isinstance(tol_days, (int, float)):
            breakdown_field["tolerance_days"] = int(tol_days)

        if isinstance(aux_entry, Mapping) and "similarity" in aux_entry:
            try:
                breakdown_field["similarity"] = float(aux_entry.get("similarity"))
            except (TypeError, ValueError):
                breakdown_field["similarity"] = aux_entry.get("similarity")
        if isinstance(aux_entry, Mapping) and "threshold" in aux_entry:
            try:
                breakdown_field["threshold"] = float(aux_entry.get("threshold"))
            except (TypeError, ValueError):
                breakdown_field["threshold"] = aux_entry.get("threshold")

        if field == "account_label":
            breakdown_field["allowlist"] = bool(getattr(cfg, "use_account_label", False))
            breakdown_field["normalize"] = bool(getattr(cfg, "account_label_normalize", True))
            breakdown_field["source"] = str(getattr(cfg, "account_label_source", "")) or ""
            normalized_values_entry = aux_entry.get("normalized_values")
            if isinstance(normalized_values_entry, (list, tuple)):
                breakdown_field["values"] = [
                    _sanitize_normalized_value_for_log(field, value)
                    for value in normalized_values_entry
                ]
            elif isinstance(normalized_values_entry, Mapping):
                breakdown_field["values"] = {
                    key: _sanitize_normalized_value_for_log(field, value)
                    for key, value in normalized_values_entry.items()
                }

        if field == "account_number":
            acct_level = aux_entry.get("points_mode_acctnum_level")
            if acct_level:
                breakdown_field["acctnum_level"] = str(acct_level)

        points_breakdown_fields[field] = breakdown_field

    ordered_field_breakdown: Dict[str, Dict[str, Any]] = {}
    for field in evaluated_fields:
        field_payload = dict(points_breakdown_fields.get(field, {}))
        field_payload.setdefault("matched", bool(matched_bools.get(field, False)))
        field_payload.setdefault("weight", float(field_weights.get(field, 0.0)))
        field_payload.setdefault("points_awarded", float(field_contributions.get(field, 0.0)))
        field_payload.setdefault("weight_counted", bool(matched_bools.get(field, False)))
        field_payload.setdefault("pairs", [])
        field_payload.setdefault("pairs_verbose", [])
        ordered_field_breakdown[field] = field_payload

    thresholds_payload = {
        "ai": float(ai_threshold),
        "direct": float(direct_threshold),
        "duplicate": float(_resolve_duplicate_points_threshold(cfg)),
    }

    points_breakdown = {
        "pair_index": diagnostics_pair_index,
        "pair": {"lo": None, "hi": None},
        "total_points": float(total_points),
        "decision": decision,
        "thresholds": thresholds_payload,
        "allowlist_fields": list(allowlist_fields),
        "ignored_fields": list(ignored_fields),
        "triggers": list(triggers),
        "conflicts": list(conflicts),
        "balance_exact_match": bool(balance_exact_match),
        "field_order": list(evaluated_fields),
        "fields": ordered_field_breakdown,
        "weights": {
            field: float(field_weights.get(field, 0.0))
            for field in evaluated_fields
        },
        "verbose": diagnostics_fields_verbose,
    }

    diagnostics_block = {
        "pair_index": diagnostics_pair_index,
        "pair": points_breakdown["pair"],
        "total_points": float(total_points),
        "decision": decision,
        "thresholds": thresholds_payload,
        "fields": ordered_field_breakdown,
        "ignored_fields": list(ignored_fields),
        "triggers": list(triggers),
        "conflicts": list(conflicts),
        "balance_exact_match": bool(balance_exact_match),
    }
    logger.info(
        "[MERGE] Points-mode diagnostics %s",
        json.dumps(diagnostics_block, sort_keys=True),
    )
    _POINTS_MODE_DIAGNOSTICS_EMITTED += 1

    sum_parts = 0.0
    for field in evaluated_fields:
        try:
            part_value = float(parts.get(field, 0.0))
        except (TypeError, ValueError):
            part_value = 0.0
        matched_flag = bool(matched_bools.get(field, False))
        aux_entry = fields_aux.get(field, {})
        _ensure_points_mode_field_invariants(
            field=field,
            matched_flag=matched_flag,
            part_value=part_value,
            parts_map=parts,
            matched_map=matched_bools,
            aux_entry=aux_entry,
        )
        parts[field] = part_value
        sum_parts += part_value

    total_points = float(sum_parts)

    if abs(total_points - sum_parts) >= 1e-6:
        logger.error(
            "[MERGE] Points-mode invariant failure %s",
            {
                "reason": "total_mismatch",
                "total": float(total_points),
                "sum_parts": float(sum_parts),
                "parts": {k: float(v) for k, v in parts.items()},
            },
        )
    assert abs(total_points - sum_parts) < 1e-6

    diagnostics_payload: List[Dict[str, Any]] = []
    for (
        field_name,
        match_score_entry,
        weight_entry,
        contribution_entry,
        running_total,
    ) in diagnostics_entries:
        diagnostics_payload.append(
            {
                "field": str(field_name),
                "match_score": float(match_score_entry),
                "weight": float(weight_entry),
                "contribution": float(contribution_entry),
                "running_total": float(running_total),
                "matched": bool(
                    matched_bools.get(field_name, match_score_entry >= 1.0)
                ),
            }
        )

    return {
        "total": float(total_points),
        "score_points": float(total_points),
        "score_legacy": None,
        "points_mode": True,
        "fields_evaluated": tuple(evaluated_fields),
        "field_matches": field_matches,
        "field_contributions": field_contributions,
        "field_weights": {field: field_weights.get(field, 0.0) for field in evaluated_fields},
        "field_breakdown": field_breakdown,
        "field_aux": fields_aux,
        "aux": fields_aux,
        "allowlist_fields": allowlist_fields,
        "ignored_fields": tuple(ignored_fields),
        "parts": {field: parts.get(field, 0.0) for field in evaluated_fields},
        "conflicts": conflicts,
        "triggers": triggers,
        "decision": decision,
        "trigger_events": trigger_events,
        "dates_all": False,
        "mid_sum": 0.0,
        "identity_sum": 0.0,
        "identity_score": 0.0,
        "points_diagnostics": tuple(diagnostics_payload),
        "points_breakdown": points_breakdown,
    }


def score_pair_0_100(
    A_bureaus: Mapping[str, Mapping[str, Any]],
    B_bureaus: Mapping[str, Mapping[str, Any]],
    cfg: MergeCfg,
    *,
    debug_pair: Optional[bool] = None,
) -> Dict[str, Any]:
    if not isinstance(A_bureaus, Mapping):
        A_data: Mapping[str, Mapping[str, Any]] = {}
    else:
        A_data = A_bureaus
    if not isinstance(B_bureaus, Mapping):
        B_data: Mapping[str, Mapping[str, Any]] = {}
    else:
        B_data = B_bureaus

    if debug_pair is None:
        debug_enabled = bool(cfg.MERGE_DEBUG)
    else:
        debug_enabled = bool(debug_pair)

    field_sequence = tuple(_field_sequence_from_cfg(cfg))
    weights_attr = getattr(cfg, "MERGE_WEIGHTS", None)
    if isinstance(weights_attr, Mapping):
        weights_map = dict(weights_attr)
    else:
        weights_map = dict(getattr(cfg, "weights", {}) or {})

    if bool(getattr(cfg, "points_mode", False)):
        logger.debug(
            "[MERGE] Points-mode scoring config %s",
            {
                "points_mode": True,
                "fields": list(field_sequence),
                "weights_map": dict(weights_map),
                "ai_points_threshold": float(
                    getattr(cfg, "ai_points_threshold", 3.0) or 0.0
                ),
                "direct_points_threshold": float(
                    getattr(cfg, "direct_points_threshold", 5.0) or 0.0
                ),
            },
        )

    return _score_pair_points_mode(
        A_data,
        B_data,
        cfg,
        field_sequence=field_sequence,
        weights_map=weights_map,
        debug_enabled=debug_enabled,
    )

def _strong_priority(triggers: Iterable[str]) -> int:
    """Return a numeric priority for strong triggers."""

    trigger_set = set(triggers or [])
    if "strong:balance_owed" in trigger_set:
        return 2
    if "strong:account_number" in trigger_set:
        return 1
    return 0


def score_all_pairs_0_100(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Score all unordered account pairs for a case run."""

    merge_env_cfg = get_merge_config()
    explicit_enabled, merge_enabled = _merge_env_state(merge_env_cfg)
    if explicit_enabled and not merge_enabled:
        # When MERGE_ENABLED=0 we honour the guardrail and skip scoring entirely.
        logger.info("Merge disabled via ENV sid=%s", sid)
        return {}

    runs_root = Path(runs_root)
    try:
        run_base_dir = (runs_root / sid).resolve(strict=False)
    except TypeError:  # pragma: no cover - fallback for older Python
        run_base_dir = (runs_root / sid)
    merge_paths = get_merge_paths(runs_root, sid, create=True)
    packs_dir = merge_paths.packs_dir
    log_file = merge_paths.log_file
    pairs_index_path = merge_paths.index_file
    cfg = get_merge_cfg()
    field_sequence = tuple(_field_sequence_from_cfg(cfg))
    weights_map = getattr(cfg, "MERGE_WEIGHTS", {})
    sum_weights = 0.0
    if isinstance(weights_map, Mapping):
        for field in field_sequence:
            value = weights_map.get(field, 0.0)
            try:
                sum_weights += float(value)
            except (TypeError, ValueError):
                continue
    points_mode_flag = bool(getattr(cfg, "points_mode", False))
    persist_breakdown_enabled = bool(getattr(cfg, "points_persist_breakdown", False))
    if not MERGE_POINTS_DIAGNOSTICS_EMIT_JSON:
        persist_breakdown_enabled = False
    diag_dir_setting_raw = getattr(cfg, "points_diag_dir", "ai_packs/merge/diagnostics")
    diag_dir_setting = str(diag_dir_setting_raw or "ai_packs/merge/diagnostics")
    if points_mode_flag:
        try:
            ai_points_threshold_value = float(
                getattr(cfg, "ai_points_threshold", 3.0) or 0.0
            )
        except (TypeError, ValueError):
            ai_points_threshold_value = 0.0
        ai_threshold = 0
    else:
        ai_points_threshold_value = 0.0
        ai_threshold = int(
            cfg.thresholds.get("AI_THRESHOLD", AI_PACK_SCORE_THRESHOLD)
        )
    logger.info(
        "[MERGE] Points configuration resolved points_mode=%s field_sequence=%s sum_weights=%.3f",
        points_mode_flag,
        field_sequence,
        sum_weights,
    )

    snapshot_logger = logger.info if getattr(cfg, "debug", False) else logger.debug
    snapshot_logger(
        "[MERGE] Runtime snapshot %s",
        {
            "points_mode": points_mode_flag,
            "fields": list(field_sequence),
            "allowlist_fields": list(_resolve_points_mode_allowlist(cfg)),
            "weights_map": dict(weights_map) if isinstance(weights_map, Mapping) else {},
            "ai_points_threshold": float(getattr(cfg, "ai_points_threshold", 3.0) or 0.0),
            "direct_points_threshold": float(
                getattr(cfg, "direct_points_threshold", 5.0) or 0.0
            ),
            "MERGE_PACKS_DIR": str(app_config.MERGE_PACKS_DIR),
            "MERGE_INDEX_PATH": str(app_config.MERGE_INDEX_PATH),
            "MERGE_PACK_GLOB": str(app_config.MERGE_PACK_GLOB),
            "allowlist_enforce": bool(getattr(cfg, "allowlist_enforce", False)),
        },
    )
    requested_raw = list(idx_list) if idx_list is not None else []
    requested_indices: List[int] = []
    for raw_idx in requested_raw:
        if isinstance(raw_idx, bool):
            continue
        try:
            idx_val = int(raw_idx)
        except (TypeError, ValueError):
            logger.warning("MERGE_V2_SCORE sid=<%s> invalid_index=%r", sid, raw_idx)
            continue
        requested_indices.append(idx_val)

    requested_set = set(requested_indices)

    _configure_candidate_logger(log_file)

    accounts_root = runs_root / sid / "cases" / "accounts"
    discovered_indices: List[int] = []
    if accounts_root.exists():
        for entry in accounts_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                idx_val = int(entry.name)
            except (TypeError, ValueError):
                logger.debug(
                    "MERGE_V2_SCORE sid=<%s> skip_account_dir=%r", sid, entry.name
                )
                continue
            discovered_indices.append(idx_val)
    else:
        logger.warning(
            "MERGE_V2_SCORE sid=<%s> accounts_dir_missing path=%s",
            sid,
            accounts_root,
        )

    if requested_raw:
        if requested_set:
            indices = sorted(idx for idx in set(discovered_indices) if idx in requested_set)
            missing = requested_set - set(indices)
            if missing:
                logger.debug(
                    "MERGE_V2_SCORE sid=<%s> missing_requested_indices=%s",
                    sid,
                    sorted(missing),
                )
        else:
            indices = []
    else:
        indices = sorted(set(discovered_indices))

    total_accounts = len(indices)
    expected_pairs = total_accounts * (total_accounts - 1) // 2

    merge_scoring_span = start_span(
        sid,
        "merge",
        "merge_scoring",
        ctx={"accounts": total_accounts},
    )

    overview_log = {
        "sid": sid,
        "indices": indices,
        "total_accounts": total_accounts,
        "expected_pairs": expected_pairs,
    }
    logger.debug("MERGE_PAIR_OVERVIEW %s", json.dumps(overview_log, sort_keys=True))

    bureaus_by_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
    points_context_by_idx: Dict[int, Dict[str, Any]] = {}
    normalized_accounts = 0
    for idx in indices:
        try:
            bureaus = load_bureaus(sid, idx, runs_root=runs_root)
        except FileNotFoundError:
            logger.warning(
                "MERGE_V2_SCORE sid=<%s> idx=<%s> bureaus_missing", sid, idx
            )
            bureaus = {}
        except Exception:
            logger.exception(
                "MERGE_V2_SCORE sid=<%s> idx=<%s> bureaus_load_failed", sid, idx
            )
            bureaus = {}
        bureaus_by_idx[idx] = bureaus
        case_root = runs_root / sid / "cases" / "accounts" / str(idx)
        points_context_by_idx[idx] = _resolve_account_artifacts(None, case_root)
        for branch in bureaus.values():
            normalized = _normalize_account_display(branch)
            if normalized.has_digits or normalized.canon_mask:
                normalized_accounts += 1

    scores: Dict[int, Dict[int, Dict[str, Any]]] = {idx: {} for idx in indices}

    logger.info("CANDIDATE_LOOP_START sid=%s total_accounts=%s", sid, total_accounts)

    span_step(
        sid,
        "merge",
        "acctnum_normalize",
        parent_span_id=merge_scoring_span,
        metrics={"normalized": normalized_accounts},
    )

    pair_counter = 0
    created_packs = 0
    matches_strong = 0
    matches_weak = 0
    conflict_pairs = 0
    skipped_pairs = 0
    pair_summaries: List[Dict[str, Any]] = []
    pair_topn_limit = steps_pair_topn()
    skip_counts: Dict[str, int] = defaultdict(int)

    def score_and_maybe_build_pack(left_pos: int, right_pos: int) -> None:
        nonlocal pair_counter, created_packs, matches_strong, matches_weak, conflict_pairs, skipped_pairs

        left = indices[left_pos]
        right = indices[right_pos]

        pair_counter += 1
        should_debug_pair = False
        if cfg.MERGE_DEBUG:
            log_every = cfg.MERGE_LOG_EVERY
            # ``log_every`` <= 1 means log every pair when debugging is enabled.
            if log_every <= 1:
                should_debug_pair = True
            elif log_every > 1 and pair_counter % log_every == 0:
                should_debug_pair = True
            if should_debug_pair:
                logger.info(
                    "[MERGE] Debug logging for pair sid=%s i=%s j=%s pair_index=%s log_every=%s",
                    sid,
                    left,
                    right,
                    pair_counter,
                    log_every,
                )

        step_log = {
            "sid": sid,
            "i": left,
            "j": right,
            "pair_index": pair_counter,
            "expected_pairs": expected_pairs,
        }
        logger.debug("MERGE_PAIR_STEP %s", json.dumps(step_log, sort_keys=True))

        left_bureaus = bureaus_by_idx.get(left, {})
        right_bureaus = bureaus_by_idx.get(right, {})
        points_mode_active = bool(points_mode_flag)
        if not isinstance(left_bureaus, Mapping):
            left_data: Mapping[str, Mapping[str, Any]] = {}
        else:
            left_data = left_bureaus
        if not isinstance(right_bureaus, Mapping):
            right_data: Mapping[str, Mapping[str, Any]] = {}
        else:
            right_data = right_bureaus

        if points_mode_active:
            debug_enabled = bool(cfg.MERGE_DEBUG) or should_debug_pair
            result = _score_pair_points_mode(
                left_data,
                right_data,
                cfg,
                field_sequence=field_sequence,
                weights_map=weights_map,
                debug_enabled=debug_enabled,
                left_context=points_context_by_idx.get(left),
                right_context=points_context_by_idx.get(right),
            )
        else:
            if should_debug_pair:
                result = score_pair_0_100(
                    left_data,
                    right_data,
                    cfg,
                    debug_pair=True,
                )
            else:
                result = score_pair_0_100(
                    left_data,
                    right_data,
                    cfg,
                )

        if not isinstance(result, dict):
            result = dict(result)

        try:
            score_points = float(result.get("score_points", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_points = 0.0
        if points_mode_active:
            total_score_value = score_points
        else:
            try:
                total_score_value = float(result.get("total", 0) or 0.0)
            except (TypeError, ValueError):
                total_score_value = 0.0

        try:
            total_score_int = int(total_score_value)
        except (TypeError, ValueError):
            total_score_int = 0

        sanitized_parts = _sanitize_parts(
            result.get("parts"),
            cfg,
            points_mode=points_mode_active,
        )
        aux_payload = _build_aux_payload(
            result.get("aux", {}), cfg=cfg, points_mode=points_mode_active
        )

        acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))
        raw_aux = result.get("aux") if isinstance(result.get("aux"), Mapping) else {}
        acct_aux = raw_aux.get("account_number") if isinstance(raw_aux, Mapping) else None
        if not isinstance(acct_aux, Mapping):
            acct_aux = {}
        raw_values: Dict[str, Any] = {}
        if isinstance(acct_aux, Mapping):
            raw_values_candidate = acct_aux.get("raw_values")
            if isinstance(raw_values_candidate, Mapping):
                raw_values = raw_values_candidate
        a_acct_str = str(raw_values.get("a") or "")
        b_acct_str = str(raw_values.get("b") or "")
        if not a_acct_str:
            a_acct_str = _extract_account_number_string(left_bureaus)
        if not b_acct_str:
            b_acct_str = _extract_account_number_string(right_bureaus)
        gate_level, gate_detail = acctnum_match_level(a_acct_str, b_acct_str)
        level_candidate = (
            acct_aux.get("acctnum_level")
            or acct_level
            or gate_level
            or "none"
        )
        level_value = _sanitize_acct_level(level_candidate)
        debug_payload = {}
        if isinstance(acct_aux, Mapping):
            debug_candidate = acct_aux.get("acctnum_debug")
            if isinstance(debug_candidate, Mapping):
                debug_payload = debug_candidate
        short_debug = ""
        long_debug = ""
        why_debug = ""
        if isinstance(debug_payload, Mapping):
            short_debug = str(debug_payload.get("short", ""))
            long_debug = str(debug_payload.get("long", ""))
            why_debug = str(debug_payload.get("why", ""))
        if not short_debug and isinstance(gate_detail, Mapping):
            short_debug = str(gate_detail.get("short", ""))
        if not long_debug and isinstance(gate_detail, Mapping):
            long_debug = str(gate_detail.get("long", ""))
        if not why_debug and isinstance(gate_detail, Mapping):
            why_debug = str(gate_detail.get("why", ""))

        pair_snapshot = _build_pair_logging_snapshot(
            sid=sid,
            left_index=left,
            right_index=right,
            total=float(total_score_value),
            result=result,
            parts=sanitized_parts,
            aux_payload=aux_payload,
            acctnum_level=level_value,
            points_mode=points_mode_active,
        )
        logger.debug("MERGE_PAIR_SCORE %s", json.dumps(pair_snapshot, sort_keys=True))

        logger.info(
            "MERGE_V2_ACCTNUM_MATCH sid=%s i=%s j=%s level=%s short=%s long=%s why=%s",
            sid,
            left,
            right,
            level_value,
            short_debug,
            long_debug,
            why_debug,
        )

        debug_reason = why_debug
        if not debug_reason and isinstance(gate_detail, Mapping):
            debug_reason = str(gate_detail.get("why", ""))
        digit_conflict = 1 if debug_reason in {"digit_conflict", "visible_digits_conflict"} else 0
        alnum_conflict = 1 if debug_reason == "alnum_conflict" else 0

        runflow_event(
            sid,
            "merge",
            "acctnum_match_level",
            account=str(left),
            metrics={
                "level": level_value,
                "digit_conflicts": digit_conflict,
                "alnum_conflicts": alnum_conflict,
            },
            out={"other": str(right)},
        )
        runflow_event(
            sid,
            "merge",
            "acctnum_match_level",
            account=str(right),
            metrics={
                "level": level_value,
                "digit_conflicts": digit_conflict,
                "alnum_conflicts": alnum_conflict,
            },
            out={"other": str(left)},
        )

        score_log = dict(pair_snapshot)
        score_log["score_points"] = float(total_score_value)
        score_log["score_int"] = int(total_score_int)
        logger.info("MERGE_V2_SCORE %s", json.dumps(score_log, sort_keys=True))

        if points_mode_active:
            score_message = f"SCORE {left}-{right} = {total_score_value:.3f}"
        else:
            score_message = f"SCORE {left}-{right} = {total_score_int}"
        logger.info(score_message)
        _candidate_logger.info(score_message)

        for event in result.get("trigger_events", []) or []:
            if not isinstance(event, Mapping):
                continue
            kind = event.get("kind")
            trigger_log = {
                "sid": sid,
                "i": left,
                "j": right,
                "kind": kind,
                "details": event.get("details", {}),
            }
            logger.info(
                "MERGE_V2_TRIGGER %s",
                json.dumps(trigger_log, sort_keys=True),
            )

        decision_log = {
            "sid": sid,
            "i": left,
            "j": right,
            "decision": str(result.get("decision", "different")),
            "total": total_score_value,
            "triggers": list(result.get("triggers", [])),
            "conflicts": list(result.get("conflicts", [])),
        }
        logger.info(
            "MERGE_V2_DECISION %s", json.dumps(decision_log, sort_keys=True)
        )

        dates_all_equal = bool(result.get("dates_all"))

        level_value = _sanitize_acct_level(level_value)

        if points_mode_active:
            allowed = total_score_value >= ai_points_threshold_value
        else:
            allowed = total_score_int >= ai_threshold

        oc_required = bool(getattr(cfg, "require_original_creditor_for_ai", False))
        oc_present_left = _has_original_creditor(left_bureaus)
        oc_present_right = _has_original_creditor(right_bureaus)

        pack_eligible = bool(allowed)
        pack_skip_reason = "no_candidates"
        pack_gate_reason = ""
        if allowed:
            gate_allowed, gate_reason = _ai_pack_gate_allows(
                cfg,
                left_bureaus,
                right_bureaus,
            )
            if not gate_allowed:
                allowed = False
                pack_gate_reason = gate_reason or "missing_original_creditor"
                pack_skip_reason = pack_gate_reason
                logger.info(
                    "MERGE_V2_PACK_GATE sid=%s i=%s j=%s reason=%s",
                    sid,
                    left,
                    right,
                    pack_gate_reason,
                )

        breakdown_payload = result.get("points_breakdown")
        diagnostics_path: Optional[Path] = None
        if isinstance(breakdown_payload, Mapping):
            breakdown_copy = dict(breakdown_payload)
            oc_gate_payload: Dict[str, Any] = {
                "required": oc_required,
                "present_left": oc_present_left,
                "present_right": oc_present_right,
                "score_allowed": bool(pack_eligible),
                "final_allowed": bool(allowed),
            }
            if pack_eligible:
                oc_gate_payload["gate_passed"] = bool(allowed)
            if not oc_required:
                action = "pass"
            elif oc_present_left or oc_present_right:
                action = "pass"
            else:
                action = "skip"
            oc_gate_payload["action"] = action
            if pack_gate_reason:
                oc_gate_payload["reason"] = str(pack_gate_reason)
            elif pack_skip_reason and pack_skip_reason != "no_candidates" and action != "pass":
                oc_gate_payload["reason"] = str(pack_skip_reason)
            if "reason" in oc_gate_payload:
                oc_gate_payload["gate_reason"] = oc_gate_payload["reason"]

            def _safe_index(value: Any) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    try:
                        return int(str(value))
                    except (TypeError, ValueError):
                        return 0

            left_idx_int = _safe_index(left)
            right_idx_int = _safe_index(right)
            lo_idx, hi_idx = sorted((left_idx_int, right_idx_int))

            pair_meta = {
                "sid": sid,
                "a": left_idx_int,
                "b": right_idx_int,
                "left": left_idx_int,
                "right": right_idx_int,
                "lo": lo_idx,
                "hi": hi_idx,
            }
            breakdown_copy["pair"] = pair_meta
            breakdown_copy.setdefault("total_score", float(total_score_value))
            breakdown_copy.setdefault("score_points", float(total_score_value))
            breakdown_copy.setdefault("decision", str(result.get("decision", "different")))
            breakdown_copy["oc_gate"] = oc_gate_payload

            result["points_breakdown"] = breakdown_copy

            if points_mode_active:
                if persist_breakdown_enabled:
                    diagnostics_dir = Path(diag_dir_setting)
                    if not diagnostics_dir.is_absolute():
                        diagnostics_dir = (run_base_dir / diagnostics_dir).resolve()
                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    diag_filename = f"points_breakdown_{lo_idx}_{hi_idx}.json"
                    diagnostics_path = diagnostics_dir / diag_filename
                    try:
                        persisted_payload = json.loads(
                            json.dumps(breakdown_copy, ensure_ascii=False)
                        )
                    except TypeError:
                        persisted_payload = breakdown_copy
                    try:
                        _atomic_write_json(diagnostics_path, persisted_payload)
                    except Exception:
                        logger.exception(
                            "MERGE_V2_POINTS_DIAGNOSTICS_WRITE_FAILED sid=%s i=%s j=%s path=%s",
                            sid,
                            left,
                            right,
                            diagnostics_path,
                        )
                    else:
                        result["points_breakdown_path"] = str(diagnostics_path)
                else:
                    result.pop("points_breakdown_path", None)

        if points_mode_active and isinstance(result.get("points_breakdown"), Mapping):
            breakdown_for_log = result.get("points_breakdown", {})
            pair_meta = breakdown_for_log.get("pair") if isinstance(breakdown_for_log, Mapping) else None
            if isinstance(pair_meta, Mapping):
                lo_idx = pair_meta.get("lo", left)
                hi_idx = pair_meta.get("hi", right)
            else:
                lo_idx, hi_idx = left, right
            pair_label = f"{lo_idx}-{hi_idx}"
            fields_map = breakdown_for_log.get("fields")
            field_order = breakdown_for_log.get("field_order")
            if not isinstance(field_order, (list, tuple)) and isinstance(fields_map, Mapping):
                field_order = list(fields_map.keys())
            if isinstance(fields_map, Mapping):
                for field_name in field_order or []:
                    entry = fields_map.get(field_name)
                    if not isinstance(entry, Mapping):
                        continue
                    formatted_line = _format_points_field_line(field_name, pair_label, entry)
                    logger.info(formatted_line)
                    _candidate_logger.info(formatted_line)
            oc_gate_line = _format_points_oc_gate_line(
                pair_label,
                result["points_breakdown"].get("oc_gate", {}),
            )
            logger.info(oc_gate_line)
            _candidate_logger.info(oc_gate_line)

        if level_value in _STRONG_MATCH_LEVELS:
            matches_strong += 1
        elif level_value in _WEAK_MATCH_LEVELS:
            matches_weak += 1
        if digit_conflict or alnum_conflict:
            conflict_pairs += 1
        if not allowed:
            skipped_pairs += 1

        result["pack_allowed"] = bool(allowed)
        if allowed:
            result.pop("pack_skip_reason", None)
        else:
            normalized_skip_reason = pack_skip_reason or "unspecified"
            result["pack_skip_reason"] = normalized_skip_reason
            pack_skip_reason = normalized_skip_reason

        pair_summaries.append(
            {
                "left": int(left),
                "right": int(right),
                "account": f"{left}-{right}",
                "level": level_value,
                "score": total_score_value,
                "digit_conflicts": digit_conflict,
                "alnum_conflicts": alnum_conflict,
                "allowed": allowed,
                "decision": str(result.get("decision", "different")),
                "debug": {
                    "short": short_debug,
                    "long": long_debug,
                    "why": why_debug,
                    **({"gate_reason": pack_gate_reason} if pack_gate_reason else {}),
                },
            }
        )

        scores[left][right] = deepcopy(result)
        scores[right][left] = deepcopy(result)

        if allowed:
            pack_built_message = f"PACK_BUILT {left}-{right}"
            logger.info(pack_built_message)
            _candidate_logger.info(pack_built_message)
            created_packs += 1

            highlights = _build_ai_highlights(result)
            highlights.update(
                {
                    "acctnum_level": level_value,
                    "total": total_score_value,
                    "dates_all": dates_all_equal,
                }
            )

            pack_path: Optional[Path] = None
            try:
                left_idx = int(left)
                right_idx = int(right)
            except (TypeError, ValueError):
                pack_path = None
            else:
                pack_path = pair_pack_path(merge_paths, left_idx, right_idx)

            if pack_path is not None and pack_path.exists():
                logger.debug(
                    "PACK_ALREADY_EXISTS sid=%s i=%s j=%s path=%s",
                    sid,
                    left,
                    right,
                    pack_path,
                )

            build_ai_pack_for_pair(
                sid,
                runs_root,
                left,
                right,
                highlights,
            )
            pack_account = f"{left}-{right}"
            pack_out: dict[str, object] | None = None
            if pack_path is not None:
                pack_out = {"path": str(pack_path)}
            span_step(
                sid,
                "merge",
                "pack_create",
                parent_span_id=merge_scoring_span,
                account=pack_account,
                out=pack_out,
            )
        else:
            reason_suffix = f" reason={pack_skip_reason}" if pack_skip_reason else ""
            pack_skipped_message = f"PACK_SKIPPED {left}-{right}{reason_suffix}"
            logger.info(pack_skipped_message)
            _candidate_logger.info(pack_skipped_message)

            runflow_event(
                sid,
                "merge",
                "pack_skip",
                parent_span_id=merge_scoring_span,
                account=f"{left}-{right}",
                out={"reason": pack_skip_reason},
            )
            if MERGE_SKIP_COUNTS_ENABLED:
                reason_key = _normalize_skip_reason(pack_skip_reason)
                skip_counts[reason_key] += 1
            if result.get("conflicts"):
                scores[left].pop(right, None)
                scores[right].pop(left, None)

    for left_pos in range(total_accounts - 1):
        for right_pos in range(left_pos + 1, total_accounts):
            score_and_maybe_build_pack(left_pos, right_pos)

    def _pair_sort_key(item: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        conflict_flag = 1 if item.get("digit_conflicts") or item.get("alnum_conflicts") else 0
        allowed_flag = 1 if item.get("allowed") else 0
        level_rank = _ACCT_LEVEL_PRIORITY.get(str(item.get("level")), 0)
        score_value = int(item.get("score", 0) or 0)
        left_idx = int(item.get("left", 0) or 0)
        right_idx = int(item.get("right", 0) or 0)
        return (conflict_flag, allowed_flag, level_rank, score_value, left_idx * -1, right_idx * -1)

    ranked_pairs = sorted(pair_summaries, key=_pair_sort_key, reverse=True)
    topn_pairs = ranked_pairs[:pair_topn_limit] if pair_topn_limit > 0 else []

    for rank, entry in enumerate(topn_pairs, start=1):
        account_label = entry.get("account", f"{entry.get('left')}-{entry.get('right')}")
        metrics = {
            "level": entry.get("level", "none"),
            "digit_conflicts": int(entry.get("digit_conflicts", 0) or 0),
            "alnum_conflicts": int(entry.get("alnum_conflicts", 0) or 0),
            "score": int(entry.get("score", 0) or 0),
            "rank": rank,
            "allowed": 1 if entry.get("allowed") else 0,
        }
        out_payload: Dict[str, Any] = {
            "left": str(entry.get("left")),
            "right": str(entry.get("right")),
            "decision": entry.get("decision"),
        }
        debug_payload = entry.get("debug")
        if isinstance(debug_payload, Mapping):
            out_payload["debug"] = {
                "short": str(debug_payload.get("short", "")),
                "long": str(debug_payload.get("long", "")),
                "why": str(debug_payload.get("why", "")),
            }

        span_step(
            sid,
            "merge",
            "acctnum_match_level",
            parent_span_id=merge_scoring_span,
            account=str(account_label),
            metrics=metrics,
            out=out_payload,
        )

    skip_counts_payload: Dict[str, int] = {}
    skip_reason_top: Optional[str] = None
    if MERGE_SKIP_COUNTS_ENABLED:
        skip_counts_payload = {
            key: int(value)
            for key, value in skip_counts.items()
            if int(value) > 0
        }
        if skip_counts_payload:
            selected = _select_skip_reason_top(skip_counts_payload)
            if selected is not None:
                skip_reason_top = selected

    merge_zero_packs = created_packs == 0 and pair_counter > 0
    human_reason: Optional[str] = None
    if merge_zero_packs:
        if skip_reason_top:
            human_reason = f"All pairs gated: {skip_reason_top.replace('_', ' ')}"
        else:
            human_reason = "All pairs below scoring threshold"
            skip_reason_top = "unspecified"
            if MERGE_SKIP_COUNTS_ENABLED and not skip_counts_payload:
                skip_counts_payload = {"unspecified": int(pair_counter)}

    totals_metrics = {
        "scored_pairs": pair_counter,
        "matches_strong": matches_strong,
        "matches_weak": matches_weak,
        "conflicts": conflict_pairs,
        "skipped": skipped_pairs,
        "packs_built": created_packs,
        "created_packs": created_packs,
        "topn_limit": pair_topn_limit,
    }

    if MERGE_SKIP_COUNTS_ENABLED:
        totals_metrics["skip_counts"] = skip_counts_payload
        if skip_reason_top is not None:
            totals_metrics["skip_reason_top"] = skip_reason_top
    if MERGE_ZERO_PACKS_SIGNAL:
        totals_metrics["merge_zero_packs"] = bool(merge_zero_packs)
        if merge_zero_packs:
            emit_counter("merge.zero_packs.total")
            if skip_counts_payload:
                for reason_key, reason_count in skip_counts_payload.items():
                    emit_counter(
                        f"merge.zero_packs.reason.{reason_key}",
                        int(reason_count),
                    )
            else:
                fallback_reason = (
                    _normalize_skip_reason(skip_reason_top)
                    if skip_reason_top
                    else "unspecified"
                )
                emit_counter(f"merge.zero_packs.reason.{fallback_reason}")
    if human_reason:
        totals_metrics["reason"] = human_reason

    ranked_for_index: List[Dict[str, Any]] = []
    for rank, entry in enumerate(ranked_pairs, start=1):
        record = {
            "rank": rank,
            "left": entry.get("left"),
            "right": entry.get("right"),
            "score": entry.get("score"),
            "level": entry.get("level"),
            "digit_conflicts": entry.get("digit_conflicts"),
            "alnum_conflicts": entry.get("alnum_conflicts"),
            "allowed": bool(entry.get("allowed")),
            "decision": entry.get("decision"),
        }
        if entry.get("debug"):
            record["debug"] = entry.get("debug")
        ranked_for_index.append(record)

    index_payload = {
        "sid": sid,
        "totals": totals_metrics,
        "pairs": ranked_for_index,
    }

    try:
        _atomic_write_json(pairs_index_path, index_payload)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception(
            "MERGE_PAIR_INDEX_WRITE_FAILED sid=%s path=%s", sid, pairs_index_path
        )

    span_step(
        sid,
        "merge",
        "acctnum_pairs_summary",
        parent_span_id=merge_scoring_span,
        metrics=totals_metrics,
        out={"pairs_index": str(pairs_index_path)},
    )

    if created_packs == 0:
        span_step(
            sid,
            "merge",
            "no_merge_candidates",
            parent_span_id=merge_scoring_span,
            metrics={"scored_pairs": pair_counter},
        )

    end_span(
        merge_scoring_span,
        metrics={
            **totals_metrics,
            "normalized_accounts": normalized_accounts,
        },
    )

    summary_log = {
        "sid": sid,
        "total_accounts": total_accounts,
        "expected_pairs": expected_pairs,
        "pairs_scored": pair_counter,
        "pairs_allowed": created_packs,
        "pairs_built": created_packs,
    }
    logger.debug("MERGE_PAIR_SUMMARY %s", json.dumps(summary_log, sort_keys=True))

    logger.info("CANDIDATE_LOOP_END sid=%s built_pairs=%s", sid, created_packs)

    try:
        from backend.ai.merge.sender import trigger_autosend_after_build

        trigger_autosend_after_build(
            sid,
            runs_root=runs_root,
            created=created_packs,
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "MERGE_AUTOSEND_TRIGGER_FAILED sid=%s", sid, exc_info=True
        )

    return scores


def choose_best_partner(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    """Select the best partner for each account using deterministic tie-breakers."""

    best_map: Dict[int, Dict[str, Any]] = {}
    for idx in sorted(scores_by_idx.keys()):
        partner_map = scores_by_idx.get(idx) or {}
        best_partner: Optional[int] = None
        best_priority = -1
        best_score = -1
        tiebreaker_reason = "none"
        best_result: Optional[Dict[str, Any]] = None

        for partner_idx in sorted(partner_map.keys()):
            if partner_idx == idx:
                continue
            result = partner_map.get(partner_idx)
            if not isinstance(result, Mapping):
                continue
            triggers = result.get("triggers") or []
            strong_rank = _strong_priority(triggers)
            total_score, points_mode_active = _extract_total_from_result(result)

            choose = False
            reason = tiebreaker_reason
            if best_partner is None:
                choose = True
                if strong_rank > 0:
                    reason = "strong"
                elif total_score > 0:
                    reason = "score"
                else:
                    reason = "index"
            elif strong_rank > best_priority:
                choose = True
                reason = "strong"
            elif strong_rank == best_priority:
                if total_score > best_score:
                    choose = True
                    reason = "score"
                elif total_score == best_score and partner_idx < best_partner:
                    choose = True
                    reason = "index"

            if not choose:
                continue

            best_partner = partner_idx
            best_priority = strong_rank
            best_score = total_score
            tiebreaker_reason = reason
            best_result = deepcopy(result)
            if isinstance(best_result, Mapping):
                best_result["points_mode"] = points_mode_active

        best_map[idx] = {
            "partner_index": best_partner,
            "result": best_result,
            "tiebreaker": tiebreaker_reason,
            "strong_rank": best_priority,
            "score_total": best_score if best_score >= 0 else 0,
        }

    return best_map


def _build_aux_payload(
    aux: Mapping[str, Any], *, cfg: Optional[MergeCfg] = None, points_mode: Optional[bool] = None
) -> Dict[str, Any]:
    acct_level = _sanitize_acct_level(None)
    by_field_pairs: Dict[str, List[str]] = {}
    resolved_points_mode = points_mode
    if resolved_points_mode is None and cfg is not None:
        resolved_points_mode = bool(getattr(cfg, "points_mode", False))
    if resolved_points_mode is None:
        resolved_points_mode = False

    if resolved_points_mode:
        field_sequence = _resolve_points_mode_allowlist(cfg)
        matched_fields: Dict[str, bool] = {field: False for field in field_sequence}
        if "account_number" not in matched_fields:
            matched_fields["account_number"] = False
    else:
        field_sequence = _field_sequence_from_cfg(cfg)
        matched_fields = {"account_number": False}
    account_number_matched = False
    acct_digits_len_a: Optional[int] = None
    acct_digits_len_b: Optional[int] = None

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if level is not None:
                acct_level = _sanitize_acct_level(level)
            if "matched" in acct_aux:
                account_number_matched = bool(acct_aux.get("matched"))
            elif acct_level != "none":
                account_number_matched = True
            len_a = acct_aux.get("acctnum_digits_len_a")
            len_b = acct_aux.get("acctnum_digits_len_b")
            try:
                acct_digits_len_a = int(len_a) if len_a is not None else acct_digits_len_a
            except (TypeError, ValueError):
                acct_digits_len_a = acct_digits_len_a
            try:
                acct_digits_len_b = int(len_b) if len_b is not None else acct_digits_len_b
            except (TypeError, ValueError):
                acct_digits_len_b = acct_digits_len_b

        for field in field_sequence:
            field_aux = aux.get(field) if isinstance(aux, Mapping) else None
            if not isinstance(field_aux, Mapping):
                continue
            if field == "account_number":
                level_value = field_aux.get("acctnum_level")
                level = (
                    acct_level if level_value is None else _sanitize_acct_level(level_value)
                )
                if level != "none":
                    acct_level = level
                if "matched" in field_aux:
                    account_number_matched = bool(field_aux.get("matched"))
                elif level != "none":
                    account_number_matched = True
            elif "matched" in field_aux:
                matched_fields[field] = bool(field_aux.get("matched"))
            best_pair = field_aux.get("best_pair")
            if best_pair and isinstance(best_pair, (list, tuple)) and len(best_pair) == 2:
                by_field_pairs[field] = [str(best_pair[0]), str(best_pair[1])]

    matched_fields["account_number"] = account_number_matched
    if "account_number" not in by_field_pairs:
        by_field_pairs["account_number"] = []

    payload: Dict[str, Any] = {
        "acctnum_level": acct_level,
        "by_field_pairs": by_field_pairs,
        "matched_fields": matched_fields,
    }

    if acct_digits_len_a is not None:
        payload["acctnum_digits_len_a"] = acct_digits_len_a
    if acct_digits_len_b is not None:
        payload["acctnum_digits_len_b"] = acct_digits_len_b

    return payload


def _build_pair_logging_snapshot(
    *,
    sid: str,
    left_index: int,
    right_index: int,
    total: float,
    result: Mapping[str, Any] | None,
    parts: Mapping[str, Union[int, float]],
    aux_payload: Mapping[str, Any],
    acctnum_level: str,
    points_mode: bool,
) -> Dict[str, Any]:
    """Return a structured payload describing a scored merge pair."""

    result_payload: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    by_field_pairs: Mapping[str, Any] = (
        aux_payload.get("by_field_pairs") if isinstance(aux_payload, Mapping) else {}
    )
    matched_fields_payload: Mapping[str, Any] = (
        aux_payload.get("matched_fields") if isinstance(aux_payload, Mapping) else {}
    )

    def _coerce_pair(field_name: str) -> List[str]:
        pair = by_field_pairs.get(field_name) if isinstance(by_field_pairs, Mapping) else []
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return [str(pair[0]), str(pair[1])]
        return []

    parts_payload = {
        str(field): float(value)
        for field, value in (parts.items() if isinstance(parts, Mapping) else [])
    }

    matched_fields = {
        str(field): bool(value)
        for field, value in (matched_fields_payload.items() if isinstance(matched_fields_payload, Mapping) else [])
    }

    aux_pairs = {
        "balance_owed": _coerce_pair("balance_owed"),
        "date_opened": _coerce_pair("date_opened"),
    }

    return {
        "sid": sid,
        "i": int(left_index),
        "j": int(right_index),
        "points_mode": bool(points_mode),
        "acctnum_level": str(acctnum_level or "none"),
        "total": float(total),
        "decision": result_payload.get("decision"),
        "parts": parts_payload,
        "matched_fields": matched_fields,
        "aux_pairs": aux_pairs,
        "conflicts": list(result_payload.get("conflicts", []) or []),
        "triggers": list(result_payload.get("triggers", []) or []),
    }


def _build_ai_highlights(result: Mapping[str, Any] | None) -> Dict[str, Any]:
    result_payload: Mapping[str, Any] = result if isinstance(result, Mapping) else {}

    total, points_mode_active = _extract_total_from_result(result_payload)

    triggers_raw = result_payload.get("triggers", []) or []
    if isinstance(triggers_raw, Iterable) and not isinstance(triggers_raw, (str, bytes)):
        triggers = [str(item) for item in triggers_raw if item is not None]
    else:
        triggers = []

    conflicts_raw = result_payload.get("conflicts", []) or []
    if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
        conflicts = [str(item) for item in conflicts_raw if item is not None]
    else:
        conflicts = []

    parts = _sanitize_parts(
        result_payload.get("parts"),
        points_mode=bool(result_payload.get("points_mode")),
    )
    aux_payload = _build_aux_payload(
        result_payload.get("aux", {}), points_mode=points_mode_active
    )

    matched_fields_raw = aux_payload.get("matched_fields", {})
    if isinstance(matched_fields_raw, Mapping):
        matched_fields = {
            str(field): bool(flag) for field, flag in matched_fields_raw.items()
        }
    else:
        matched_fields = {}

    acctnum_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))

    return {
        "total": total,
        "triggers": triggers,
        "parts": parts,
        "matched_fields": matched_fields,
        "conflicts": conflicts,
        "acctnum_level": acctnum_level,
        "points_mode": points_mode_active,
    }


def _coerce_points_mode_flag(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _detect_points_mode_from_result(result: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(result, Mapping):
        return False

    flag = _coerce_points_mode_flag(result.get("points_mode"))
    if flag is not None:
        return flag

    if "score_points" in result:
        return True

    parts_candidate = result.get("parts")
    if isinstance(parts_candidate, Mapping):
        for candidate in parts_candidate.values():
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if not float(value).is_integer():
                return True

    return False


def detect_points_mode_from_payload(result: Optional[Mapping[str, Any]]) -> bool:
    """Public helper exposing points-mode detection for merge payloads."""

    return _detect_points_mode_from_result(result)


def _coerce_score_value(value: Any, *, points_mode: bool) -> Union[int, float]:
    if points_mode:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def coerce_score_value(value: Any, *, points_mode: bool) -> Union[int, float]:
    """Public wrapper that preserves score scale for downstream consumers."""

    return _coerce_score_value(value, points_mode=points_mode)


def _extract_total_from_result(
    result: Optional[Mapping[str, Any]]
) -> Tuple[Union[int, float], bool]:
    points_mode_active = _detect_points_mode_from_result(result)
    raw_total: Any = 0.0 if points_mode_active else 0
    if isinstance(result, Mapping):
        if points_mode_active:
            raw_total = result.get("score_points", result.get("total", 0.0))
        else:
            raw_total = result.get("total", 0)
    total_value = _coerce_score_value(raw_total, points_mode=points_mode_active)
    return total_value, points_mode_active


def _extract_mid_from_result(
    result: Optional[Mapping[str, Any]], *, points_mode: bool
) -> Union[int, float]:
    raw_mid: Any = 0.0 if points_mode else 0
    if isinstance(result, Mapping):
        raw_mid = result.get("mid_sum")
        if raw_mid is None:
            raw_mid = result.get("mid", raw_mid)
    return _coerce_score_value(raw_mid, points_mode=points_mode)


def _sanitize_parts(
    parts: Optional[Mapping[str, Any]],
    cfg: Optional[MergeCfg] = None,
    *,
    points_mode: Optional[bool] = None,
) -> Dict[str, Union[int, float]]:
    resolved_points_mode = points_mode
    if resolved_points_mode is None and cfg is not None:
        resolved_points_mode = bool(getattr(cfg, "points_mode", False))

    if resolved_points_mode is None and isinstance(parts, Mapping):
        for candidate in parts.values():
            if isinstance(candidate, float) and not float(candidate).is_integer():
                resolved_points_mode = True
                break
    if resolved_points_mode is None:
        resolved_points_mode = False

    if resolved_points_mode and isinstance(parts, Mapping):
        for field_name, raw_value in parts.items():
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            assert 0.0 <= numeric_value <= 8.0, (
                "points_mode parts contained out-of-range value"
                f" field={field_name!s} value={numeric_value!r}"
            )

    if resolved_points_mode:
        field_iterable: Sequence[str] = _resolve_points_mode_allowlist(cfg)
    else:
        field_iterable = _field_sequence_from_cfg(cfg)

    values: Dict[str, Union[int, float]] = {}
    for field in field_iterable:
        if isinstance(parts, Mapping):
            raw_value = parts.get(field, 0)
        else:
            raw_value = 0

        if resolved_points_mode:
            try:
                value = float(raw_value or 0.0)
            except (TypeError, ValueError):
                value = 0.0
        else:
            try:
                value = int(raw_value or 0)
            except (TypeError, ValueError):
                value = 0
        values[field] = value
    return values


def normalize_parts_for_serialization(
    parts: Optional[Mapping[str, Any]],
    cfg: Optional[MergeCfg] = None,
    *,
    points_mode: Optional[bool] = None,
) -> Dict[str, Union[int, float]]:
    """Expose sanitized parts for downstream serialization helpers."""

    return _sanitize_parts(parts, cfg=cfg, points_mode=points_mode)


def _build_pair_entry(partner_idx: int, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a stable payload describing the merge relationship with a partner."""

    total_value, points_mode_active = _extract_total_from_result(result)
    mid_value = _extract_mid_from_result(result, points_mode=points_mode_active)
    dates_all = bool(result.get("dates_all"))
    decision = str(result.get("decision", "different"))
    aux_candidate = result.get("aux")
    if isinstance(aux_candidate, Mapping):
        aux_payload: Mapping[str, Any] = aux_candidate
    else:
        aux_payload = {}
    parts_candidate = result.get("parts")
    if isinstance(parts_candidate, Mapping):
        parts_payload: Mapping[str, Any] = parts_candidate
    else:
        parts_payload = {}
    triggers_candidate = result.get("triggers", []) or []
    if isinstance(triggers_candidate, Iterable):
        triggers_raw: Iterable[Any] = triggers_candidate
    else:
        triggers_raw = []

    triggers = [str(trigger) for trigger in triggers_raw if trigger is not None]
    strong = any(trigger.startswith("strong:") for trigger in triggers)
    sanitized_parts = _sanitize_parts(
        parts_payload,
        points_mode=points_mode_active,
    )
    aux_slim = _build_aux_payload(
        aux_payload, points_mode=points_mode_active
    )
    acct_level = _sanitize_acct_level(aux_slim.get("acctnum_level"))

    entry: Dict[str, Any] = {
        "with": int(partner_idx),
        "total": total_value,
        "points_mode": points_mode_active,
        "decision": decision,
        "strong": strong,
        "mid": mid_value,
        "dates_all": dates_all,
        "acctnum_level": acct_level,
        "reasons": triggers,
        "parts": sanitized_parts,
    }

    return entry


def _build_best_match_entry(
    best_info: Optional[Mapping[str, Any]],
    pair_entries: Mapping[int, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Construct the best-match payload leveraging computed pair entries."""

    if not isinstance(best_info, Mapping):
        return None

    partner = best_info.get("partner_index")
    if not isinstance(partner, int):
        return None

    entry = pair_entries.get(partner)
    acct_level = _sanitize_acct_level(None)
    points_mode_active = False
    decision = "different"
    total_value: Union[int, float]

    if isinstance(entry, Mapping):
        total_value, points_mode_active = _extract_total_from_result(entry)
        decision = str(entry.get("decision", "different"))
        acct_level = _sanitize_acct_level(entry.get("acctnum_level", acct_level))
    else:
        result_payload = best_info.get("result")
        if isinstance(result_payload, Mapping):
            total_value, points_mode_active = _extract_total_from_result(result_payload)
            decision = str(result_payload.get("decision", "different"))
            aux_payload = _build_aux_payload(
                result_payload.get("aux", {}), points_mode=points_mode_active
            )
            acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level", acct_level))
        else:
            total_value = 0.0
        fallback_total = best_info.get("score_total")
        if fallback_total is not None:
            total_value = _coerce_score_value(
                fallback_total, points_mode=points_mode_active
            )

    tiebreaker = str(best_info.get("tiebreaker", "none"))

    return {
        "with": int(partner),
        "total": total_value,
        "decision": decision,
        "tiebreaker": tiebreaker,
        "acctnum_level": acct_level,
        "points_mode": points_mode_active,
    }


def _tag_safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tag_safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _tag_normalize_str_list(values: Any) -> List[str]:
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [str(item) for item in values if item is not None]
    return []


def _tag_normalize_merge_parts(
    parts: Any,
    *,
    points_mode: bool,
    cfg: Optional[MergeCfg] = None,
) -> Dict[str, Union[int, float]]:
    normalized: Dict[str, Union[int, float]] = {}
    if not isinstance(parts, Mapping):
        return normalized

    if points_mode:
        allowlist_sequence = _resolve_points_mode_allowlist(cfg)
        for field in allowlist_sequence:
            if field not in parts:
                continue
            normalized[field] = _tag_safe_float(parts.get(field))
        return normalized

    for key in sorted(parts.keys(), key=str):
        value = parts.get(key)
        normalized[str(key)] = _tag_safe_int(value)
    return normalized


def _tag_normalize_merge_aux(
    aux: Any,
    *,
    points_mode: bool,
    cfg: Optional[MergeCfg] = None,
) -> Dict[str, Any]:
    acct_level = _sanitize_acct_level(None)
    by_field_pairs: Dict[str, List[str]] = {}
    matched_fields: Dict[str, bool] = {"account_number": False}
    account_number_matched = False
    acct_digits_len_a: Optional[int] = None
    acct_digits_len_b: Optional[int] = None

    allowed_fields: Tuple[str, ...] = ()
    allowed_field_set: Set[str] = set()
    if points_mode:
        allowed_fields = _resolve_points_mode_allowlist(cfg)
        allowed_field_set = set(allowed_fields)
        matched_fields = {field: False for field in allowed_fields}
        if "account_number" not in matched_fields:
            matched_fields["account_number"] = False

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if level is not None:
                acct_level = _sanitize_acct_level(level)
            if "matched" in acct_aux:
                account_number_matched = bool(acct_aux.get("matched"))
            elif acct_level != "none":
                account_number_matched = True
            len_a = acct_aux.get("acctnum_digits_len_a")
            len_b = acct_aux.get("acctnum_digits_len_b")
            try:
                acct_digits_len_a = int(len_a) if len_a is not None else acct_digits_len_a
            except (TypeError, ValueError):
                acct_digits_len_a = acct_digits_len_a
            try:
                acct_digits_len_b = int(len_b) if len_b is not None else acct_digits_len_b
            except (TypeError, ValueError):
                acct_digits_len_b = acct_digits_len_b

        for field, field_aux in aux.items():
            if not isinstance(field_aux, Mapping):
                continue
            field_name = str(field)
            if points_mode and field_name not in allowed_field_set:
                continue
            if field_name == "account_number":
                level_value = field_aux.get("acctnum_level")
                if level_value is not None:
                    level = _sanitize_acct_level(level_value)
                    if level != "none":
                        acct_level = level
                if "matched" in field_aux:
                    account_number_matched = bool(field_aux.get("matched"))
                elif account_number_matched:
                    account_number_matched = True
            elif "matched" in field_aux:
                matched_fields[field_name] = bool(field_aux.get("matched"))
            best_pair = field_aux.get("best_pair")
            if (
                isinstance(best_pair, (list, tuple))
                and len(best_pair) == 2
                and all(part is not None for part in best_pair)
            ):
                by_field_pairs[field_name] = [
                    str(best_pair[0]),
                    str(best_pair[1]),
                ]

    matched_fields["account_number"] = account_number_matched
    if "account_number" not in by_field_pairs:
        by_field_pairs["account_number"] = []

    if points_mode:
        filtered_pairs = {
            field: pair
            for field, pair in by_field_pairs.items()
            if field in allowed_field_set or field == "account_number"
        }
        ordered_pairs: Dict[str, List[str]] = {}
        for field in allowed_fields:
            if field in filtered_pairs:
                ordered_pairs[field] = filtered_pairs[field]
        if "account_number" not in ordered_pairs:
            ordered_pairs["account_number"] = filtered_pairs.get("account_number", [])
        by_field_pairs = ordered_pairs

        ordered_matched: Dict[str, bool] = {}
        for field in allowed_fields:
            ordered_matched[field] = matched_fields.get(field, False)
        if "account_number" not in ordered_matched:
            ordered_matched["account_number"] = account_number_matched
        matched_fields = ordered_matched

    payload: Dict[str, Any] = {
        "acctnum_level": acct_level,
        "by_field_pairs": by_field_pairs,
        "matched_fields": matched_fields,
    }

    if acct_digits_len_a is not None:
        payload["acctnum_digits_len_a"] = acct_digits_len_a
    if acct_digits_len_b is not None:
        payload["acctnum_digits_len_b"] = acct_digits_len_b

    return payload


def _normalize_merge_payload_for_tag(
    result: Mapping[str, Any] | None,
    *,
    cfg: Optional[MergeCfg] = None,
) -> Dict[str, Any]:
    payload = {
        "decision": "different",
        "total": 0,
        "mid": 0,
        "dates_all": False,
        "parts": {},
        "aux": {
            "acctnum_level": "none",
            "by_field_pairs": {},
            "matched_fields": {},
        },
        "reasons": [],
        "conflicts": [],
        "strong": False,
        "matched_pairs": {},
    }

    if isinstance(result, Mapping):
        points_mode_active = _detect_points_mode_from_result(result)
        cfg_for_points = cfg
        if points_mode_active and cfg_for_points is None:
            cfg_for_points = get_merge_cfg()
        payload["points_mode"] = points_mode_active
        payload["decision"] = str(result.get("decision", "different"))
        total_value, _ = _extract_total_from_result(result)
        payload["total"] = total_value
        mid_value = _extract_mid_from_result(result, points_mode=points_mode_active)
        payload["mid"] = mid_value
        payload["dates_all"] = bool(result.get("dates_all"))
        payload["parts"] = _tag_normalize_merge_parts(
            result.get("parts"),
            points_mode=points_mode_active,
            cfg=cfg_for_points,
        )
        payload["aux"] = _tag_normalize_merge_aux(
            result.get("aux"),
            points_mode=points_mode_active,
            cfg=cfg_for_points,
        )
        triggers_source = result.get("triggers")
        if not triggers_source:
            triggers_source = result.get("reasons")
        payload["reasons"] = _tag_normalize_str_list(triggers_source)
        payload["conflicts"] = _tag_normalize_str_list(result.get("conflicts"))

        pack_allowed_value = result.get("pack_allowed")
        if pack_allowed_value is None:
            pack_allowed_value = payload["decision"].lower() in {"ai", "auto"}
        payload["pack_allowed"] = bool(pack_allowed_value)

        pack_skip_reason_value = result.get("pack_skip_reason")
        if pack_skip_reason_value is not None:
            payload["pack_skip_reason"] = str(pack_skip_reason_value)

    payload["strong"] = any(
        isinstance(reason, str) and reason.startswith("strong:")
        for reason in payload["reasons"]
    )

    aux_payload = payload.get("aux")
    acct_level = _sanitize_acct_level(None)
    if isinstance(aux_payload, Mapping):
        acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))
        matched_pairs = aux_payload.get("by_field_pairs", {})
        if isinstance(matched_pairs, Mapping):
            payload["matched_pairs"] = {
                str(field): [str(pair[0]), str(pair[1])]
                for field, pair in matched_pairs.items()
                if isinstance(pair, (list, tuple))
                and len(pair) == 2
            }
        for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
            value = aux_payload.get(key)
            if value is None:
                continue
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                continue
    payload.setdefault("matched_pairs", {})
    payload["matched_pairs"].setdefault("account_number", [])
    payload["acctnum_level"] = acct_level
    payload.setdefault("pack_allowed", payload["decision"].lower() in {"ai", "auto"})

    if "pack_allowed" in payload and not payload["pack_allowed"]:
        if "pack_skip_reason" not in payload:
            payload["pack_skip_reason"] = "unspecified"

    return payload


def _copy_summary_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_summary_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy_summary_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_summary_value(item) for item in value]
    return value


def _is_empty_summary_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _merge_summary_mapping(
    existing: Mapping[str, Any], new_data: Mapping[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    merged = {key: _copy_summary_value(val) for key, val in existing.items()}
    changed = False
    for key, value in new_data.items():
        existing_value = merged.get(key)
        if key not in merged or _is_empty_summary_value(existing_value) or (
            isinstance(value, bool) and value and existing_value is False
        ):
            merged[key] = _copy_summary_value(value)
            changed = True
            continue
        if isinstance(existing_value, Mapping) and isinstance(value, Mapping):
            nested_merged, nested_changed = _merge_summary_mapping(existing_value, value)
            if nested_changed:
                merged[key] = nested_merged
                changed = True
    return merged, changed


def _coerce_partner_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        partner = int(value)
    except (TypeError, ValueError):
        return None
    return partner


def build_summary_merge_entry(
    kind: str,
    partner_idx: Any,
    payload: Mapping[str, Any] | None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    partner = _coerce_partner_index(partner_idx)
    if partner is None:
        return None

    normalized = _normalize_merge_payload_for_tag(payload)
    normalized["kind"] = str(kind)
    normalized["with"] = partner

    aux_payload = normalized.get("aux")
    if isinstance(aux_payload, Mapping):
        matched_fields = aux_payload.get("matched_fields")
        if isinstance(matched_fields, Mapping) and matched_fields:
            existing_fields = normalized.get("matched_fields")
            if isinstance(existing_fields, Mapping):
                merged_fields = dict(existing_fields)
                for field, flag in matched_fields.items():
                    merged_fields[str(field)] = bool(flag)
            else:
                merged_fields = {
                    str(field): bool(flag) for field, flag in matched_fields.items()
                }
            normalized["matched_fields"] = merged_fields

    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if key in {"strong_rank", "score_total"}:
                points_mode_active = bool(normalized.get("points_mode"))
                normalized[key] = _coerce_score_value(
                    value, points_mode=points_mode_active
                )
            else:
                normalized[key] = value

    return normalized


def _normalize_flag_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def build_summary_ai_entries(
    partner_idx: Any,
    decision: Any,
    reason: Any,
    flags: Mapping[str, Any] | None,
    *,
    normalized: bool = False,
) -> List[Dict[str, Any]]:
    partner = _coerce_partner_index(partner_idx)
    if partner is None:
        return []

    entries: List[Dict[str, Any]] = []
    normalized_flag = bool(normalized)
    decision_entry: Dict[str, Any] = {
        "kind": "ai_decision",
        "with": partner,
        "normalized": normalized_flag,
    }
    normalized_decision, normalized_from_input = _normalize_merge_decision(
        decision,
        partner=partner,
        log_missing=True,
        log_unknown=True,
    )
    normalized_flag = normalized_flag or normalized_from_input

    raw_flags: Mapping[str, Any]
    if isinstance(flags, Mapping):
        raw_flags = {str(key): value for key, value in flags.items()}
    else:
        raw_flags = {}

    ai_payload: Dict[str, Any] = {
        "decision": normalized_decision,
        "flags": raw_flags,
    }
    if reason is not None:
        ai_payload["reason"] = reason

    try:
        normalized_payload = validate_ai_payload(ai_payload)
    except AdjudicatorError as exc:
        logger.warning(
            "AI_DECISION_PAYLOAD_INVALID partner=%s decision=%r error=%s; defaulting to 'different'",
            partner,
            decision,
            exc,
        )
        normalized_flag = True
        normalized_payload = {
            "decision": normalized_decision or "different",
            "flags": {"account_match": "unknown", "debt_match": "unknown"},
        }
        if reason is not None:
            normalized_payload["reason"] = reason

    normalized_decision = str(normalized_payload.get("decision") or "different")
    normalized_flags_payload = normalized_payload.get("flags", {})
    if not isinstance(normalized_flags_payload, Mapping):
        normalized_flags_payload = {
            "account_match": "unknown",
            "debt_match": "unknown",
        }

    def _flag_as_string(value: Any) -> Optional[str]:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false", "unknown"}:
                return lowered
        return None

    default_flags: Dict[str, str] = {}
    lowered_decision = normalized_decision.strip().lower()
    if lowered_decision.startswith("same_account_"):
        default_flags["account_match"] = "true"
        if lowered_decision.endswith("_same_debt"):
            default_flags["debt_match"] = "true"
    if lowered_decision.startswith("same_debt_"):
        default_flags["debt_match"] = "true"
    if lowered_decision == "duplicate":
        default_flags["account_match"] = "true"
        default_flags["debt_match"] = "true"
    elif lowered_decision == "not_duplicate":
        default_flags["account_match"] = "false"
        default_flags["debt_match"] = "false"

    final_flags: Dict[str, str] = {}
    for key in ("account_match", "debt_match"):
        normalized_value = normalized_flags_payload.get(key)
        normalized_value_str = _flag_as_string(normalized_value)
        if normalized_value_str is not None:
            final_flags[key] = normalized_value_str
        elif key in default_flags:
            final_flags[key] = default_flags[key]
        else:
            final_flags[key] = "unknown"

        original_value = _flag_as_string(raw_flags.get(key)) if raw_flags else None
        if original_value is None and raw_flags and key in raw_flags:
            normalized_flag = True
        elif (
            original_value is not None
            and normalized_value_str is not None
            and original_value != normalized_value_str
        ):
            normalized_flag = True

    duplicate_flag_str: Optional[str] = None
    if isinstance(normalized_flags_payload, Mapping):
        duplicate_flag_str = _flag_as_string(normalized_flags_payload.get("duplicate"))
    if duplicate_flag_str is None:
        if lowered_decision == "duplicate":
            duplicate_flag_str = "true"
        elif lowered_decision == "not_duplicate":
            duplicate_flag_str = "false"
    if duplicate_flag_str is not None:
        final_flags["duplicate"] = duplicate_flag_str
        if raw_flags is not None and "duplicate" in raw_flags:
            original_duplicate = _flag_as_string(raw_flags.get("duplicate"))
            if original_duplicate is None or original_duplicate != duplicate_flag_str:
                normalized_flag = True

    reason_value = normalized_payload.get("reason")
    if isinstance(reason_value, str):
        reason_text = reason_value.strip()
    elif reason_value is not None:
        reason_text = str(reason_value).strip()
    else:
        reason_text = ""

    decision_entry["decision"] = normalized_decision or "different"
    decision_entry["normalized"] = normalized_flag
    if reason_text:
        decision_entry["reason"] = reason_text

    ai_result_payload: Dict[str, Any] = {
        "decision": normalized_decision or "different",
        "flags": {key: final_flags[key] for key in ("account_match", "debt_match")},
    }
    if "duplicate" in final_flags:
        ai_result_payload["flags"]["duplicate"] = final_flags["duplicate"]
    if reason_text:
        ai_result_payload["reason"] = reason_text

    decision_entry["flags"] = dict(final_flags)
    decision_entry["ai_result"] = dict(ai_result_payload)
    entries.append(decision_entry)

    resolution_entry: Dict[str, Any] = {
        "kind": "ai_resolution",
        "with": partner,
        "normalized": normalized_flag,
        "decision": normalized_decision or "different",
        "flags": dict(final_flags),
        "ai_result": dict(ai_result_payload),
    }
    if reason_text:
        resolution_entry["reason"] = reason_text
    entries.append(resolution_entry)

    pair_kind = AI_PAIR_KIND_BY_DECISION.get(normalized_decision or "")
    if pair_kind:
        pair_entry: Dict[str, Any] = {"kind": pair_kind, "with": partner, "ai_result": dict(ai_result_payload)}
        if reason_text:
            pair_entry["reason"] = reason_text
        entries.append(pair_entry)

        if pair_kind == "same_account_pair" and normalized_decision == "same_account_same_debt":
            pair_entry.setdefault("notes", []).append("accounts_and_debt_match")
        elif pair_kind == "same_account_pair" and normalized_decision == "same_account_diff_debt":
            pair_entry.setdefault("notes", []).append("debt_differs")
        elif normalized_decision == "same_account_debt_unknown":
            pair_entry.setdefault("notes", []).append("debt_information_missing")
        elif normalized_decision == "same_debt_diff_account":
            pair_entry.setdefault("notes", []).append("same_debt_different_accounts")
        elif normalized_decision == "same_debt_account_unknown":
            pair_entry.setdefault("notes", []).append("same_debt_account_unclear")
        elif normalized_decision == "different":
            pair_entry.setdefault("notes", []).append("no_match")

    return entries


def _append_summary_entries(
    summary_data: Dict[str, Any],
    key: str,
    entries: Iterable[Mapping[str, Any]],
    unique_keys: Tuple[str, ...],
) -> bool:
    normalized_entries: List[Dict[str, Any]] = []
    existing_raw = summary_data.get(key)
    if isinstance(existing_raw, Iterable) and not isinstance(existing_raw, (str, bytes)):
        for entry in existing_raw:
            if isinstance(entry, Mapping):
                normalized_entries.append({key_: _copy_summary_value(val) for key_, val in entry.items()})

    changed = False
    index: Dict[Tuple[Any, ...], int] = {}
    for position, entry in enumerate(normalized_entries):
        index[tuple(entry.get(field) for field in unique_keys)] = position

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        lookup_key = tuple(entry.get(field) for field in unique_keys)
        candidate = {key_: _copy_summary_value(val) for key_, val in entry.items()}
        if lookup_key in index:
            existing_entry = normalized_entries[index[lookup_key]]
            merged, entry_changed = _merge_summary_mapping(existing_entry, candidate)
            if entry_changed:
                normalized_entries[index[lookup_key]] = merged
                changed = True
        else:
            normalized_entries.append(candidate)
            index[lookup_key] = len(normalized_entries) - 1
            changed = True

    if normalized_entries:
        original_len = len(normalized_entries)
        normalized_entries = _dedupe_summary_list(
            normalized_entries, unique_keys=unique_keys
        )
        if len(normalized_entries) != original_len:
            changed = True
    if changed or (key not in summary_data and normalized_entries):
        summary_data[key] = normalized_entries
    return changed


def _summary_entry_score(entry: Mapping[str, Any]) -> int:
    score = 0
    parts = entry.get("parts")
    if isinstance(parts, Mapping):
        score += len(parts)
    matched_fields = entry.get("matched_fields")
    if isinstance(matched_fields, Mapping):
        score += sum(1 for flag in matched_fields.values() if bool(flag))
    aux_payload = entry.get("aux")
    if isinstance(aux_payload, Mapping):
        acct_level = aux_payload.get("acctnum_level")
        if isinstance(acct_level, str) and acct_level and acct_level != "none":
            score += 1
        matched_aux = aux_payload.get("matched_fields")
        if isinstance(matched_aux, Mapping):
            score += sum(1 for flag in matched_aux.values() if bool(flag))
    if entry.get("strong"):
        score += 1
    points_mode_active = detect_points_mode_from_payload(entry)
    total_value = coerce_score_value(
        entry.get("total"), points_mode=points_mode_active
    )
    if total_value:
        score += 1
    return score


def _dedupe_summary_list(
    entries: Iterable[Mapping[str, Any]], *, unique_keys: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[Any, ...], Tuple[int, Dict[str, Any]]] = {}
    order: List[Tuple[Any, ...]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        candidate = {key: _copy_summary_value(value) for key, value in entry.items()}
        key = tuple(candidate.get(field) for field in unique_keys)
        score = _summary_entry_score(candidate)
        if key not in best_by_key:
            best_by_key[key] = (score, candidate)
            order.append(key)
        else:
            existing_score, _ = best_by_key[key]
            if score >= existing_score:
                best_by_key[key] = (score, candidate)
    result: List[Dict[str, Any]] = []
    for key in order:
        best = best_by_key.get(key)
        if best is None:
            continue
        result.append(best[1])
    return result


def merge_summary_sections(
    summary_data: Dict[str, Any],
    *,
    merge_entries: Iterable[Mapping[str, Any]] | None = None,
    ai_entries: Iterable[Mapping[str, Any]] | None = None,
) -> bool:
    changed = False
    if merge_entries:
        changed |= _append_summary_entries(
            summary_data,
            "merge_explanations",
            merge_entries,
            ("kind", "with"),
        )
    elif isinstance(summary_data.get("merge_explanations"), list):
        existing_merge = summary_data.get("merge_explanations") or []
        deduped_merge = _dedupe_summary_list(
            [entry for entry in existing_merge if isinstance(entry, Mapping)],
            unique_keys=("kind", "with"),
        )
        if len(deduped_merge) != len(existing_merge):
            summary_data["merge_explanations"] = deduped_merge
            changed = True
    if ai_entries:
        changed |= _append_summary_entries(
            summary_data,
            "ai_explanations",
            ai_entries,
            ("kind", "with"),
        )
    elif isinstance(summary_data.get("ai_explanations"), list):
        existing_ai = summary_data.get("ai_explanations") or []
        deduped_ai = _dedupe_summary_list(
            [entry for entry in existing_ai if isinstance(entry, Mapping)],
            unique_keys=("kind", "with"),
        )
        if len(deduped_ai) != len(existing_ai):
            summary_data["ai_explanations"] = deduped_ai
            changed = True
    return changed


def apply_summary_updates(
    summary_path: Path,
    *,
    merge_entries: Iterable[Mapping[str, Any]] | None = None,
    ai_entries: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    try:
        raw = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        summary_data: Dict[str, Any] = {}
    except OSError:
        summary_data = {}
    else:
        try:
            loaded = json.loads(raw)
        except Exception:
            summary_data = {}
        else:
            summary_data = dict(loaded) if isinstance(loaded, Mapping) else {}

    changed = merge_summary_sections(
        summary_data, merge_entries=merge_entries, ai_entries=ai_entries
    )

    if changed:
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary_data)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(summary_path, summary_data)

    return summary_data


def build_merge_pair_tag(partner_idx: int, result: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _normalize_merge_payload_for_tag(result)
    payload.update(
        {
            "tag": "merge_pair",
            "kind": "merge_pair",
            "source": "merge_scorer",
            "with": int(partner_idx),
        }
    )
    return payload


def build_merge_best_tag(best_info: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(best_info, Mapping):
        return None

    partner = best_info.get("partner_index")
    result = best_info.get("result")
    if not isinstance(partner, int) or not isinstance(result, Mapping):
        return None

    payload = _normalize_merge_payload_for_tag(result)
    points_mode_active = bool(payload.get("points_mode"))
    payload.update(
        {
            "tag": "merge_best",
            "kind": "merge_best",
            "source": "merge_scorer",
            "with": int(partner),
            "tiebreaker": str(best_info.get("tiebreaker", "none")),
            "strong_rank": _tag_safe_int(best_info.get("strong_rank")),
            "score_total": _coerce_score_value(
                best_info.get("score_total", payload["total"]),
                points_mode=points_mode_active,
            ),
        }
    )
    return payload


def _build_score_entries(
    partner_scores: Mapping[int, Mapping[str, Any]],
    best_partner: Optional[int],
) -> List[Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for partner_idx, result in partner_scores.items():
        if partner_idx == best_partner:
            # handle separately to ensure copy from best result later
            continue
        if not isinstance(result, Mapping):
            continue
        score_value, points_mode_active = _extract_total_from_result(result)
        entry = {
            "account_index": partner_idx,
            "score": score_value,
            "decision": str(result.get("decision", "different")),
            "triggers": list(result.get("triggers", [])),
            "conflicts": list(result.get("conflicts", [])),
            "points_mode": points_mode_active,
        }
        entries[partner_idx] = entry

    sorted_entries = sorted(
        entries.values(), key=lambda item: (-item["score"], item["account_index"])
    )
    return sorted_entries


def _merge_tag_from_best(
    idx: int,
    partner_scores: Mapping[int, Mapping[str, Any]],
    best_info: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = get_merge_cfg()
    field_sequence = _field_sequence_from_cfg(cfg)
    best_partner = best_info.get("partner_index")
    best_result = best_info.get("result")
    tiebreaker = str(best_info.get("tiebreaker") or "none")

    points_mode_active = bool(getattr(cfg, "points_mode", False))

    if not isinstance(best_partner, int) or not isinstance(best_result, Mapping):
        parts = {field: 0.0 if points_mode_active else 0 for field in field_sequence}
        merge_tag = {
            "group_id": f"g{idx}",
            "decision": "different",
            "score_total": 0.0 if points_mode_active else 0,
            "score_to": _build_score_entries(partner_scores, None),
            "parts": parts,
            "aux": {"acctnum_level": "none", "by_field_pairs": {}, "matched_fields": {}},
            "reasons": [],
            "tiebreaker": "none",
        }
        merge_tag["acctnum_level"] = "none"
        merge_tag["matched_pairs"] = {"account_number": []}
        merge_tag["points_mode"] = points_mode_active
        return merge_tag

    score_total, points_mode_active = _extract_total_from_result(best_result)
    decision = str(best_result.get("decision", "different"))
    triggers = list(best_result.get("triggers", []))
    conflicts = list(best_result.get("conflicts", []))
    parts = _sanitize_parts(
        best_result.get("parts"),
        cfg,
        points_mode=points_mode_active,
    )
    aux_payload = _build_aux_payload(
        best_result.get("aux", {}), cfg=cfg, points_mode=points_mode_active
    )
    acct_level = _sanitize_acct_level(aux_payload.get("acctnum_level"))

    score_entries = _build_score_entries(partner_scores, best_partner)
    best_entry = {
        "account_index": best_partner,
        "score": score_total,
        "decision": decision,
        "triggers": triggers,
        "conflicts": conflicts,
    }
    score_to = [best_entry] + score_entries

    reasons = list(triggers)
    if conflicts:
        reasons.extend([f"conflict:{name}" for name in conflicts])

    merge_tag = {
        "group_id": f"g{idx}",
        "decision": decision,
        "score_total": score_total,
        "score_to": score_to,
        "parts": parts,
        "aux": aux_payload,
        "reasons": reasons,
        "tiebreaker": tiebreaker,
        "points_mode": points_mode_active,
    }
    merge_tag["acctnum_level"] = acct_level
    matched_pairs_raw = aux_payload.get("by_field_pairs", {})
    pairs_payload: Dict[str, List[str]] = {}
    if isinstance(matched_pairs_raw, Mapping):
        for field, pair in matched_pairs_raw.items():
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(part is not None for part in pair)
            ):
                pairs_payload[str(field)] = [str(pair[0]), str(pair[1])]
    pairs_payload.setdefault("account_number", [])
    merge_tag["matched_pairs"] = pairs_payload

    for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
        value = aux_payload.get(key)
        if value is None:
            continue
        try:
            merge_tag[key] = int(value)
        except (TypeError, ValueError):
            continue
    return merge_tag


def persist_merge_tags(
    sid: str,
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]],
    best_by_idx: Mapping[int, Mapping[str, Any]],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[str, Any]]:
    """Persist merge tags for each account based on best-partner selection."""

    locks_dir = runs_root / sid / _LOCKS_DIRNAME
    merge_lock_path = locks_dir / _MERGE_INFLIGHT_LOCK_FILENAME
    lock_written = False
    try:
        try:
            locks_dir.mkdir(parents=True, exist_ok=True)
            merge_lock_path.write_text("1", encoding="utf-8")
            lock_written = True
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "MERGE_INFLIGHT_LOCK_CREATE_FAILED sid=%s path=%s",
                sid,
                merge_lock_path,
                exc_info=True,
            )

        cfg = get_merge_cfg()
        points_mode_active = bool(getattr(cfg, "points_mode", False))
        ai_pack_threshold = (
            float(getattr(cfg, "ai_points_threshold", 0.0) or 0.0)
            if points_mode_active
            else AI_PACK_SCORE_THRESHOLD
        )

        merge_tags: Dict[int, Dict[str, Any]] = {}
        all_indices = sorted(set(scores_by_idx.keys()) | set(best_by_idx.keys()))

        tags_root = runs_root / sid / "cases" / "accounts"
        tag_paths: Dict[int, Path] = {
            idx: tags_root / str(idx) / "tags.json" for idx in all_indices
        }
        summary_paths: Dict[int, Path] = {
            idx: tags_root / str(idx) / "summary.json" for idx in all_indices
        }

        merge_kinds = {"merge_pair", "merge_best"}
        bureaus_cache: Dict[int, Mapping[str, Mapping[str, Any]]] = {}
        for idx, path in tag_paths.items():
            existing_tags = read_tags(path)
            filtered = [tag for tag in existing_tags if tag.get("kind") not in merge_kinds]
            if filtered != existing_tags:
                write_tags_atomic(path, filtered)

        valid_decisions = {"ai", "auto"}
        processed_pairs: Set[Tuple[int, int]] = set()
        summary_updates: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: {"merge": [], "ai": []}
        )
        for left in sorted(scores_by_idx.keys()):
            partner_map = scores_by_idx.get(left) or {}
            for right in sorted(partner_map.keys()):
                if right == left or not isinstance(right, int):
                    continue
                ordered = (min(left, right), max(left, right))
                if ordered in processed_pairs:
                    continue
                processed_pairs.add(ordered)

                result = partner_map.get(right)
                if not isinstance(result, Mapping):
                    continue

                total_value, result_points_mode = _extract_total_from_result(result)
                threshold = ai_pack_threshold if result_points_mode else AI_PACK_SCORE_THRESHOLD
                if total_value >= threshold:
                    left_merge_entry = build_summary_merge_entry("merge_pair", right, result)
                    if left_merge_entry is not None:
                        summary_updates[left]["merge"].append(left_merge_entry)
                    right_merge_entry = build_summary_merge_entry("merge_pair", left, result)
                    if right_merge_entry is not None:
                        summary_updates[right]["merge"].append(right_merge_entry)

                left_tag = build_merge_pair_tag(right, result)
                if left_tag.get("decision") in valid_decisions:
                    left_path = tag_paths.get(left)
                    if left_path is not None:
                        upsert_tag(left_path, left_tag, unique_keys=("kind", "with"))

                    right_tag = build_merge_pair_tag(left, result)
                    right_path = tag_paths.get(right)
                    if right_path is not None:
                        upsert_tag(right_path, right_tag, unique_keys=("kind", "with"))

                if left_tag.get("decision") == "ai":
                    gate_allowed = True
                    pack_skip_reason = ""
                    if bool(getattr(cfg, "require_original_creditor_for_ai", False)):
                        try:
                            left_bureaus = bureaus_cache[left]
                        except KeyError:
                            try:
                                left_bureaus = load_bureaus(sid, left, runs_root=runs_root)
                            except FileNotFoundError:
                                left_bureaus = {}
                            except Exception:
                                logger.exception(
                                    "MERGE_V2_BUREAUS_LOAD_FAILED sid=%s idx=%s", sid, left
                                )
                                left_bureaus = {}
                            bureaus_cache[left] = left_bureaus
                        try:
                            right_bureaus = bureaus_cache[right]
                        except KeyError:
                            try:
                                right_bureaus = load_bureaus(sid, right, runs_root=runs_root)
                            except FileNotFoundError:
                                right_bureaus = {}
                            except Exception:
                                logger.exception(
                                    "MERGE_V2_BUREAUS_LOAD_FAILED sid=%s idx=%s", sid, right
                                )
                                right_bureaus = {}
                            bureaus_cache[right] = right_bureaus

                        gate_allowed, pack_skip_reason = _ai_pack_gate_allows(
                            cfg,
                            left_bureaus,
                            right_bureaus,
                        )
                        if not pack_skip_reason:
                            pack_skip_reason = "missing_original_creditor"

                    if not gate_allowed:
                        reason_suffix = f" reason={pack_skip_reason}" if pack_skip_reason else ""
                        skip_message = f"PACK_SKIPPED {left}-{right}{reason_suffix}"
                        logger.info(skip_message)
                        _candidate_logger.info(skip_message)
                        runflow_event(
                            sid,
                            "merge",
                            "pack_skip",
                            account=f"{left}-{right}",
                            out={"reason": pack_skip_reason or "missing_original_creditor"},
                        )
                        continue

                    highlights_from_pair = _build_ai_highlights(result)
                    pack_payload = build_ai_pack_for_pair(
                        sid,
                        runs_root,
                        left,
                        right,
                        highlights_from_pair,
                    )
                    try:
                        total_value = None
                        if isinstance(pack_payload, Mapping):
                            total_value = pack_payload.get("score_total")
                        if total_value is None:
                            try:
                                total_value = (
                                    float(total_score_value)
                                    if points_mode_active
                                    else int(total_score_int)
                                )
                            except NameError:
                                total_value = 0.0 if points_mode_active else 0

                        pack_log = {
                            "sid": sid,
                            "pair": {"a": left, "b": right},
                            "schema": pack_payload.get("schema") if isinstance(pack_payload, Mapping) else None,
                            "lines_a": 0,
                            "lines_b": 0,
                            "total": total_value,
                        }
                        logger.info("MERGE_V2_PACK %s", json.dumps(pack_log, sort_keys=True))
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "MERGE_V2_PACK_FAILED sid=%s pair=(%s,%s)", sid, left, right
                        )

        for idx in all_indices:
            best_info = best_by_idx.get(idx, {})
            best_tag = build_merge_best_tag(best_info)
            if not best_tag or best_tag.get("decision") not in valid_decisions:
                continue
            path = tag_paths.get(idx)
            if path is not None:
                upsert_tag(path, best_tag, unique_keys=("kind",))

            best_partner = best_info.get("partner_index") if isinstance(best_info, Mapping) else None
            best_result = best_info.get("result") if isinstance(best_info, Mapping) else None
            if best_partner is not None and isinstance(best_result, Mapping):
                score_total_value, _ = _extract_total_from_result(best_result)
                extra_fields = {
                    "tiebreaker": str(best_info.get("tiebreaker", "none")),
                    "strong_rank": best_info.get("strong_rank"),
                    "score_total": best_info.get("score_total", score_total_value),
                }
                best_entry = build_summary_merge_entry(
                    "merge_best", best_partner, best_result, extra=extra_fields
                )
                if best_entry is not None:
                    summary_updates[idx]["merge"].append(best_entry)

                    partner_idx = best_entry.get("with")
                    if isinstance(partner_idx, int):
                        partner_scores = scores_by_idx.get(partner_idx, {})
                        reverse_result = partner_scores.get(idx)
                        if not isinstance(reverse_result, Mapping):
                            reverse_result = best_result

                        partner_info = best_by_idx.get(partner_idx)
                        if isinstance(partner_info, Mapping):
                            partner_extra = {
                                "tiebreaker": str(partner_info.get("tiebreaker", "none")),
                                "strong_rank": partner_info.get("strong_rank"),
                                "score_total": partner_info.get("score_total"),
                            }
                        else:
                            partner_extra = {
                                "tiebreaker": "peer_best",
                                "strong_rank": None,
                                "score_total": score_total_value,
                            }

                        reverse_entry = build_summary_merge_entry(
                            "merge_best", idx, reverse_result, extra=partner_extra
                        )
                        if reverse_entry is not None:
                            summary_updates[partner_idx]["merge"].append(reverse_entry)

        for idx in all_indices:
            partner_scores = scores_by_idx.get(idx, {})
            best_info = best_by_idx.get(idx, {})
            merge_tag = _merge_tag_from_best(idx, partner_scores, best_info)
            merge_tags[idx] = merge_tag

        for idx, updates in summary_updates.items():
            summary_path = summary_paths.get(idx)
            if summary_path is None:
                continue
            merge_entries = updates.get("merge") or []
            ai_entries = updates.get("ai") or []
            if merge_entries or ai_entries:
                apply_summary_updates(
                    summary_path,
                    merge_entries=merge_entries,
                    ai_entries=ai_entries,
                )

        emit_validation_tag = _read_env_flag(
            os.environ, "VALIDATION_REQUIREMENTS_TAGS", False
        )

        for idx in all_indices:
            summary_path = summary_paths.get(idx)
            if summary_path is None:
                continue
            try:
                bureaus_payload = load_bureaus(sid, idx, runs_root=runs_root)
            except FileNotFoundError:
                logger.warning(
                    "VALIDATION_BUREAUS_MISSING sid=%s idx=%s runs_root=%s",
                    sid,
                    idx,
                    runs_root,
                )
                continue
            except Exception:
                logger.exception(
                    "VALIDATION_BUREAUS_LOAD_FAILED sid=%s idx=%s runs_root=%s",
                    sid,
                    idx,
                    runs_root,
                )
                continue

            try:
                (
                    requirements,
                    inconsistencies,
                    field_consistency,
                ) = build_validation_requirements(bureaus_payload)
            except Exception:
                logger.exception(
                    "VALIDATION_REQUIREMENTS_COMPUTE_FAILED sid=%s idx=%s runs_root=%s",
                    sid,
                    idx,
                    runs_root,
                )
                continue

            try:
                raw_provider = _raw_value_provider_for_account_factory(bureaus_payload)
                summary_payload = build_validation_summary_payload(
                    requirements,
                    field_consistency=field_consistency,
                    raw_value_provider=raw_provider,
                    sid=sid,
                    runs_root=runs_root,
                )
                summary_after = apply_validation_summary(summary_path, summary_payload)
            except Exception:
                logger.exception(
                    "VALIDATION_SUMMARY_WRITE_FAILED sid=%s idx=%s summary=%s",
                    sid,
                    idx,
                    summary_path,
                )
                continue

            tag_path = tag_paths.get(idx)
            if tag_path is not None:
                try:
                    findings_block = summary_payload.get("findings")
                    fields_for_tag = [
                        str(entry.get("field"))
                        for entry in findings_block
                        if isinstance(entry, Mapping) and entry.get("field")
                    ]
                    sync_validation_tag(tag_path, fields_for_tag, emit=emit_validation_tag)
                except Exception:
                    logger.exception(
                        "VALIDATION_TAG_SYNC_FAILED sid=%s idx=%s path=%s",
                        sid,
                        idx,
                        tag_path,
                    )

            bureaus_path = summary_path.parent / "bureaus.json"
            try:
                # Lazy-import here to avoid circular import during module load.
                from backend.ai.validation_builder import build_validation_pack_for_account

                build_validation_pack_for_account(
                    sid,
                    idx,
                    summary_path,
                    bureaus_path,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "VALIDATION_PACK_BUILD_FAILED sid=%s account=%s summary=%s",
                    sid,
                    idx,
                    summary_path,
                )

        return merge_tags
    finally:
        if lock_written:
            try:
                merge_lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "MERGE_INFLIGHT_LOCK_REMOVE_FAILED sid=%s path=%s",
                    sid,
                    merge_lock_path,
                    exc_info=True,
                )


def score_and_tag_best_partners(
    sid: str,
    idx_list: Iterable[int],
    runs_root: Path = Path("runs"),
) -> Dict[int, Dict[str, Any]]:
    """Convenience wrapper to score accounts, pick best partners, and persist tags."""

    scores = score_all_pairs_0_100(sid, idx_list, runs_root=runs_root)
    best = choose_best_partner(scores)
    return persist_merge_tags(sid, scores, best, runs_root=runs_root)


def to_amount(value: Any) -> Optional[float]:
    """Normalize free-form amount text to a float."""

    if is_missing(value):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    cleaned = _AMOUNT_SANITIZE_RE.sub("", text)
    if not cleaned:
        return None

    match = _AMOUNT_RE.search(cleaned)
    if not match:
        return None

    number_text = match.group()
    try:
        number = float(number_text)
    except ValueError:
        return None

    if negative and number >= 0:
        number = -number

    return number


def to_amount_cents(value: Any) -> Optional[int]:
    """Normalize free-form amount text to cents represented as an integer."""

    if is_missing(value):
        return None

    decimal_value: Decimal

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float):
                decimal_value = Decimal(str(value))
            else:
                decimal_value = Decimal(int(value))
        except (ValueError, InvalidOperation):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None

        negative = False
        if text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1]

        cleaned = _AMOUNT_SANITIZE_RE.sub("", text)
        if not cleaned:
            return None

        match = _AMOUNT_RE.search(cleaned)
        if not match:
            return None

        number_text = match.group()
        try:
            decimal_value = Decimal(number_text)
        except InvalidOperation:
            return None

        if negative and decimal_value >= 0:
            decimal_value = -decimal_value

    try:
        scaled = (decimal_value * _DECIMAL_HUNDRED).quantize(
            _DECIMAL_ONE, rounding=ROUND_HALF_UP
        )
    except InvalidOperation:
        return None

    return int(scaled)


def amounts_match(a: Optional[float], b: Optional[float], tol_abs: float, tol_ratio: float) -> bool:
    """Return True when two normalized amounts match within tolerance."""

    if a is None or b is None:
        return False

    tol_abs = max(float(tol_abs), 0.0)
    tol_ratio = max(float(tol_ratio), 0.0)
    base = min(abs(a), abs(b))
    allowed = max(tol_abs, base * tol_ratio)
    return abs(a - b) <= allowed


def normalize_balance_owed(value: Any) -> Optional[float]:
    return to_amount(value)


def normalize_balance_owed_cents(value: Any) -> Optional[int]:
    return to_amount_cents(value)


def match_balance_owed(
    a: Optional[float],
    b: Optional[float],
    *,
    tol_abs: float = 0.0,
    tol_ratio: float = 0.0,
) -> bool:
    """Return True when balances match within configured tolerances."""

    if a is None or b is None:
        return False
    return amounts_match(a, b, tol_abs, tol_ratio)


def normalize_amount_field(value: Any) -> Optional[float]:
    return to_amount(value)


def match_amount_field(
    a: Optional[float],
    b: Optional[float],
    *,
    tol_abs: float,
    tol_ratio: float,
) -> bool:
    return amounts_match(a, b, tol_abs, tol_ratio)


def match_payment_amount(
    a: Optional[float],
    b: Optional[float],
    *,
    tol_abs: float,
    tol_ratio: float,
    count_zero_payment_match: int,
) -> bool:
    if a is None or b is None:
        return False
    if not count_zero_payment_match and a == 0 and b == 0:
        return False
    return amounts_match(a, b, tol_abs, tol_ratio)


def normalize_history_field(value: Any) -> Optional[str]:
    """Return a normalized history blob for similarity comparisons."""

    if is_missing(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.lower()


def history_similarity_score(a: Optional[str], b: Optional[str]) -> float:
    """Return ``SequenceMatcher`` similarity for normalized history pairs."""

    if a is None or b is None:
        return 0.0

    # ``SequenceMatcher`` is tolerant to whitespace differences which mirrors the
    # historic fuzzy comparison behaviour for payment histories.
    return SequenceMatcher(None, a, b).ratio()


def match_history_field(
    a: Optional[str],
    b: Optional[str],
    *,
    threshold: float,
) -> Tuple[bool, float]:
    """Return whether histories match and the computed similarity score."""

    threshold = max(float(threshold), 0.0)
    similarity = history_similarity_score(a, b)
    return similarity >= threshold, similarity


def digits_only(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def normalize_account_number(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    normalized = _normalize_acctnum_basic(str(value))
    return normalized


def account_number_level(a: Any, b: Any) -> str:
    if is_missing(a) or is_missing(b):
        return "none"

    level, _ = acctnum_match_level(str(a), str(b))
    return level


def account_numbers_match(
    a: Any, b: Any, min_level: str = "exact_or_known_match"
) -> Tuple[bool, str]:
    if is_missing(a) or is_missing(b):
        return False, "none"

    level, _ = acctnum_match_level(str(a), str(b))
    threshold = _ACCOUNT_LEVEL_ORDER.get(min_level, 0)
    match = level != "none" and _ACCOUNT_LEVEL_ORDER.get(level, 0) >= threshold
    return match, level


def to_date(value: Any) -> Optional[date]:
    if is_missing(value):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None

    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text and len(text.split()) > 1:
        text = text.split()[0]

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    alt = re.sub(r"[.\\-]", "/", text)
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(alt, fmt).date()
        except ValueError:
            continue

    return None


def date_equal(a: Optional[date], b: Optional[date]) -> bool:
    if a is None or b is None:
        return False
    return a == b


def date_within(a: Optional[date], b: Optional[date], days: int) -> bool:
    if a is None or b is None:
        return False
    days = max(int(days), 0)
    return abs((a - b).days) <= days


def normalize_type(value: Any) -> Optional[str]:
    if is_missing(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[._]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip("- ")
    if not text:
        return None

    alias_key = text.replace("-", " ")
    alias_key = re.sub(r"\s+", " ", alias_key)
    alias = _TYPE_ALIAS_MAP.get(alias_key)
    if alias:
        return alias

    normalized = alias_key
    normalized = normalized.replace("creditcard", "credit card")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if "credit card" in normalized:
        return "credit card"

    return normalized or None

