"""Build and persist validation requirements derived from bureau mismatches."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
)

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    conint,
    field_validator,
)

from backend import config as backend_config
# NOTE: do not import validation_builder at module import-time.
# We'll lazy-import inside the function to avoid circular imports.
from backend.core.io.json_io import _atomic_write_json
from backend.core.io.tags import read_tags, write_tags_atomic
from backend.core.logic import summary_writer
from backend.core.runflow import runflow_account_steps_enabled, runflow_step
from backend.core.logic.context import set_validation_context
from backend.core.logic.consistency import _get_bureau_value, compute_field_consistency
from backend.core.logic.reason_classifier import classify_reason, decide_send_to_ai
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.telemetry import metrics
from backend.validation.decision_matrix import decide_default
from backend.validation.seed_arguments import build_seed_argument
from backend.core.merge.acctnum import acctnum_level, acctnum_match_level
from backend.prevalidation import read_date_convention
from backend.validation.utils_dates import business_to_calendar

tolerance_logger = logging.getLogger("backend.validation.tolerance")
logger = logging.getLogger(__name__)

__all__ = [
    "ValidationRule",
    "ValidationConfig",
    "ValidationConfigError",
    "load_validation_config",
    "build_validation_requirements",
    "build_findings",
    "build_summary_payload",
    "apply_validation_summary",
    "apply_missing_only_downgrade",
    "sync_validation_tag",
    "build_validation_requirements_for_account",
]


_VALIDATION_TAG_KIND = "validation_required"
_CONFIG_PATH = Path(__file__).with_name("validation_config.yml")
_SUMMARY_SCHEMA_VERSION = 3
_SUMMARY_BUREAUS: tuple[str, ...] = ("equifax", "experian", "transunion")
_DEFAULT_SUMMARY_POINTERS = {
    "raw": "raw_lines.json",
    "bureaus": "bureaus.json",
    "flat": "fields_flat.json",
    "tags": "tags.json",
    "summary": "summary.json",
}

_FALLBACK_DECISIONS: Mapping[str, str] = {
    "C1_TWO_PRESENT_ONE_MISSING": "supportive_needs_companion",
    "C2_ONE_MISSING": "supportive_needs_companion",
    "C3_TWO_PRESENT_CONFLICT": "supportive_needs_companion",
    "C4_TWO_MATCH_ONE_DIFF": "strong_actionable",
}

build_validation_pack_for_account: Callable[..., Sequence[str]] | None = None

_ACCTCHECK_ENV_FLAG = "VALIDATION_ACCTCHECK_LOG"
_ACCOUNT_NUMBER_PAIR_ORDER: tuple[tuple[str, str, str], ...] = (
    ("Eq-Ex", "equifax", "experian"),
    ("Eq-Tu", "equifax", "transunion"),
    ("Ex-Tu", "experian", "transunion"),
)
_ACCOUNT_NUMBER_BUREAU_LABELS: Mapping[str, str] = {
    "equifax": "Eq",
    "experian": "Ex",
    "transunion": "Tu",
}
_ACCOUNT_NUMBER_DETERMINISTIC_FLAG = "VALIDATION_ACCOUNT_NUMBER_DISPLAY_DETERMINISTIC"
_BUSINESS_DAY_SLA_FLAG = "VALIDATION_USE_BUSINESS_DAYS"
_BUSINESS_ONLY_MODE_FLAG = "VALIDATION_BUSINESS_ONLY_MODE"
_EMIT_BUSINESS_FIELDS_FLAG = "VALIDATION_REQUIREMENTS_EMIT_BUSINESS_FIELDS"
_HIDE_CALENDAR_FIELDS_FLAG = "VALIDATION_REQUIREMENTS_HIDE_CALENDAR_FIELDS"

_FALSE_FLAG_VALUES = {"0", "false", "off", "no"}
_TRUE_FLAG_VALUES = {"1", "true", "on", "yes"}


def _use_deterministic_account_number_policy() -> bool:
    override = os.getenv(_ACCOUNT_NUMBER_DETERMINISTIC_FLAG)
    if override is None:
        return True

    lowered = override.strip().lower()
    if lowered in _FALSE_FLAG_VALUES:
        return False

    return True


def _business_day_sla_enabled() -> bool:
    override = os.getenv(_BUSINESS_DAY_SLA_FLAG)
    if override is None:
        return False

    lowered = override.strip().lower()
    if lowered in _TRUE_FLAG_VALUES:
        return True
    if lowered in _FALSE_FLAG_VALUES:
        return False

    return False


def _flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(flag_name)
    if raw_value is None:
        return default

    lowered = raw_value.strip().lower()
    if lowered in _TRUE_FLAG_VALUES:
        return True
    if lowered in _FALSE_FLAG_VALUES:
        return False
    return default


def _business_only_mode_enabled() -> bool:
    return _flag_enabled(_BUSINESS_ONLY_MODE_FLAG, default=False)


def _emit_business_fields_enabled() -> bool:
    return _flag_enabled(_EMIT_BUSINESS_FIELDS_FLAG, default=False)


def _hide_calendar_fields_enabled() -> bool:
    return _flag_enabled(_HIDE_CALENDAR_FIELDS_FLAG, default=False)


def _is_dry_run_enabled() -> bool:
    return bool(getattr(backend_config, "VALIDATION_DRY_RUN", False))


@lru_cache(maxsize=1)
def _load_tolerance_evaluator():
    from backend.validation.tolerance import evaluate_field_with_tolerance

    return evaluate_field_with_tolerance


def _emit_tolerance_log(payload: Mapping[str, Any]) -> None:
    kind = str(payload.get("kind"))
    if kind == "date":
        tolerance_logger.info(
            "TOLCHECK date sid=%s field=%s conv=%s tol_days=%s span=%s within=%s",
            payload.get("sid", ""),
            payload.get("field", ""),
            payload.get("conv"),
            payload.get("tol_days"),
            payload.get("span"),
            payload.get("within"),
        )
    elif kind == "amount":
        tolerance_logger.info(
            "TOLCHECK amount sid=%s field=%s abs=%s ratio=%s diff=%s maxv=%s within=%s",
            payload.get("sid", ""),
            payload.get("field", ""),
            payload.get("abs"),
            payload.get("ratio"),
            payload.get("diff"),
            payload.get("maxv"),
            payload.get("within"),
        )

    if os.getenv("VALIDATION_DEBUG") == "1":
        sid_value = str(payload.get("sid") or "")
        if sid_value:
            step_name = f"TOLCHECK_{kind}" if kind else "TOLCHECK_unknown"
            metrics: dict[str, Any] = {}
            if kind == "date":
                span = payload.get("span")
                tol_days = payload.get("tol_days")
                try:
                    metrics["span"] = int(span)
                except (TypeError, ValueError):
                    pass
                try:
                    metrics["tol_days"] = int(tol_days)
                except (TypeError, ValueError):
                    pass
            elif kind == "amount":
                for key, target in (("diff", "diff"), ("abs", "abs_tol"), ("ratio", "ratio_tol")):
                    value = payload.get(key)
                    try:
                        metrics[target] = float(value)
                    except (TypeError, ValueError):
                        continue
            within = payload.get("within")
            if isinstance(within, bool):
                metrics["within"] = within

            if runflow_account_steps_enabled():
                runflow_step(
                    sid_value,
                    "validation",
                    step_name,
                    metrics=metrics or None,
                    out={"field": str(payload.get("field") or "")},
                )


def _evaluate_with_tolerance(
    sid: str | None,
    runs_root: str | os.PathLike[str] | None,
    field: str,
    bureau_values: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    try:
        evaluator = _load_tolerance_evaluator()
    except Exception:  # pragma: no cover - defensive import guard
        logger.debug(
            "TOLERANCE_EVALUATOR_IMPORT_FAILED field=%s", field, exc_info=True
        )
        return None

    try:
        result = evaluator(sid, runs_root, field, bureau_values)
    except Exception:  # pragma: no cover - defensive evaluation guard
        logger.exception("TOLERANCE_EVALUATION_FAILED field=%s", field)
        return None

    if not isinstance(result, Mapping):
        return result

    log_payload = result.get("log_payload")
    if isinstance(log_payload, Mapping) and log_payload:
        try:
            _emit_tolerance_log(log_payload)
        except Exception:  # pragma: no cover - defensive logging guard
            logger.debug("TOLERANCE_LOGGING_FAILED field=%s", field, exc_info=True)
        sanitized = dict(result)
        sanitized.pop("log_payload", None)
        return sanitized

    return result


def _coerce_tolerance_note(
    field_name: Any, note_payload: Any
) -> dict[str, Any] | None:
    """Normalize tolerance note payload for debug summaries."""

    if not isinstance(note_payload, Mapping):
        return None

    normalized: dict[str, Any] = {
        "field": str(field_name),
        "status": str(note_payload.get("status") or "within"),
    }

    for key in ("span_days", "tol_days", "diff", "ceil"):
        if key in note_payload:
            value = note_payload.get(key)
            if value is not None:
                normalized[key] = value

    return normalized


def _clear_tolerance_state() -> None:
    _load_tolerance_evaluator.cache_clear()
    try:
        from backend.validation.tolerance import clear_cached_conventions
    except Exception:  # pragma: no cover - defensive import guard
        logger.debug("TOLERANCE_CLEAR_FAILED", exc_info=True)
        return

    try:
        clear_cached_conventions()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("TOLERANCE_CLEAR_EXEC_FAILED", exc_info=True)


def _get_canary_percent() -> int:
    try:
        percent = int(getattr(backend_config, "VALIDATION_CANARY_PERCENT", 100))
    except Exception:
        return 100
    if percent < 0:
        return 0
    if percent > 100:
        return 100
    return percent


def _derive_canary_identifier(account_path: Path) -> str:
    account_id = account_path.name
    sid: str | None = None
    try:
        sid_candidate = account_path.parents[2].name
    except IndexError:
        sid_candidate = None
    if sid_candidate and sid_candidate not in {"", "."}:
        sid = sid_candidate

    if sid:
        return f"{sid}:{account_id}"
    return str(account_path)


def _account_selected_for_canary(account_path: Path, percent: int) -> bool:
    if percent >= 100:
        return True
    if percent <= 0:
        return False

    identifier = _derive_canary_identifier(account_path)
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:2], "big") % 100
    return bucket < percent


def _is_validation_reason_enabled() -> bool:
    """Return ``True`` when reason enrichment should be applied."""

    raw_value = os.getenv("VALIDATION_REASON_ENABLED")
    if raw_value is None:
        return True

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return False


@lru_cache(maxsize=1)
def _should_redact_pii() -> bool:
    """Return ``True`` when optional PII redaction should be applied."""

    raw_value = os.getenv("VALIDATION_REDACT_PII")
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _redact_account_number_raw(raw_value: Any) -> Any:
    """Mask account numbers so only the last four digits remain visible."""

    if not isinstance(raw_value, str):
        return raw_value

    masked_chars: list[str] = []
    digits_seen = 0

    for char in reversed(raw_value):
        if char.isdigit():
            digits_seen += 1
            if digits_seen > 4:
                masked_chars.append("*")
            else:
                masked_chars.append(char)
        else:
            masked_chars.append(char)

    return "".join(reversed(masked_chars))


@lru_cache(maxsize=1)
def _include_creditor_remarks() -> bool:
    """Return ``True`` when ``creditor_remarks`` validation is enabled."""

    raw_value = os.getenv("VALIDATION_INCLUDE_CREDITOR_REMARKS")
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


DurationUnit = Literal["calendar_days", "business_days"]


@dataclass(frozen=True)
class ValidationRule:
    """Metadata describing the validation needed for a field."""

    category: str
    min_days: int
    documents: tuple[str, ...]
    points: int
    strength: str
    ai_needed: bool
    min_corroboration: int
    conditional_gate: bool
    duration_unit: DurationUnit


@dataclass(frozen=True)
class CategoryRule:
    """Fallback configuration scoped to a category."""

    min_days: int
    documents: tuple[str, ...]


@dataclass(frozen=True)
class ValidationConfig:
    defaults: ValidationRule
    fields: Mapping[str, ValidationRule]
    category_defaults: Mapping[str, CategoryRule]
    schema_version: int
    mode: str
    broadcast_disputes: bool
    threshold_points: int


class ValidationConfigError(RuntimeError):
    """Raised when validation requirements configuration is invalid."""


_VALID_BOOL_STRINGS: frozenset[str] = frozenset(
    {"1", "0", "true", "false", "yes", "no", "on", "off", "y", "n"}
)


def _normalize_documents(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, (str, bytes, bytearray)):
        text = str(raw).strip()
        return (text,) if text else tuple()
    if isinstance(raw, Iterable):
        collected: List[str] = []
        for entry in raw:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                collected.append(text)
        return tuple(collected)
    raise TypeError("documents must be an iterable of strings")


def _normalize_strength(value: str) -> str:
    lowered = value.strip().lower()
    if lowered not in {"strong", "medium", "soft"}:
        raise ValueError("strength must be one of: strong, medium, soft")
    return lowered


def _normalize_duration_unit(value: Any, *, allow_none: bool = False) -> DurationUnit | None:
    if value is None:
        if allow_none:
            return None
        return "calendar_days"

    text = str(value).strip().lower()
    if text not in {"calendar_days", "business_days"}:
        raise ValueError("duration_unit must be 'calendar_days' or 'business_days'")

    return cast(DurationUnit, text)


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationDefaultsSchema(_BaseSchema):
    category: str
    min_days: conint(ge=0)  # type: ignore[call-overload]
    points: conint(ge=0)  # type: ignore[call-overload]
    documents: tuple[str, ...] = Field(default_factory=tuple)
    strength: str
    ai_needed: bool
    min_corroboration: conint(ge=1) = 1  # type: ignore[call-overload]
    conditional_gate: bool = False
    duration_unit: DurationUnit = "calendar_days"

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("category must not be empty")
        return text

    @field_validator("documents", mode="before")
    @classmethod
    def _normalize_documents_validator(cls, value: Any) -> tuple[str, ...]:
        return _normalize_documents(value)

    @field_validator("strength", mode="before")
    @classmethod
    def _normalize_strength_validator(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("strength must be a string")
        return _normalize_strength(value)

    @field_validator("duration_unit", mode="before")
    @classmethod
    def _normalize_duration_unit_validator(
        cls, value: Any
    ) -> DurationUnit:
        normalized = _normalize_duration_unit(value)
        assert normalized is not None
        return normalized


class CategoryRuleSchema(_BaseSchema):
    min_days: conint(ge=0)  # type: ignore[call-overload]
    documents: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("documents", mode="before")
    @classmethod
    def _normalize_documents_validator(cls, value: Any) -> tuple[str, ...]:
        return _normalize_documents(value)


class ValidationFieldSchema(_BaseSchema):
    category: str
    min_days: conint(ge=0)  # type: ignore[call-overload]
    points: conint(ge=0) | None = None  # type: ignore[call-overload]
    documents: tuple[str, ...] | None = None
    strength: str
    ai_needed: bool
    min_corroboration: conint(ge=1) | None = None  # type: ignore[call-overload]
    conditional_gate: bool | None = None
    duration_unit: DurationUnit | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("category must not be empty")
        return text

    @field_validator("documents", mode="before")
    @classmethod
    def _normalize_documents_validator(
        cls, value: Any
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        return _normalize_documents(value)

    @field_validator("strength", mode="before")
    @classmethod
    def _normalize_strength_validator(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("strength must be a string")
        return _normalize_strength(value)

    @field_validator("duration_unit", mode="before")
    @classmethod
    def _normalize_duration_unit_validator(
        cls, value: Any
    ) -> DurationUnit | None:
        return _normalize_duration_unit(value, allow_none=True)


class ValidationConfigSchema(_BaseSchema):
    schema_version: conint(ge=1) = 1  # type: ignore[call-overload]
    mode: str = "broad"
    broadcast_disputes: bool | None = None
    threshold_points: conint(ge=0) = 45  # type: ignore[call-overload]
    defaults: ValidationDefaultsSchema
    fields: Dict[str, ValidationFieldSchema]
    category_defaults: Dict[str, CategoryRuleSchema] = Field(default_factory=dict)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> str:
        if value is None:
            return "broad"
        text = str(value).strip().lower()
        if text not in {"broad", "strict"}:
            raise ValueError("mode must be 'broad' or 'strict'")
        return text

    @field_validator("fields")
    @classmethod
    def _ensure_fields_present(
        cls, value: Dict[str, ValidationFieldSchema]
    ) -> Dict[str, ValidationFieldSchema]:
        if not value:
            raise ValueError("fields section must not be empty")
        return value


def _build_validation_defaults(schema: ValidationDefaultsSchema) -> ValidationRule:
    return ValidationRule(
        schema.category,
        int(schema.min_days),
        tuple(schema.documents),
        int(schema.points),
        schema.strength,
        bool(schema.ai_needed),
        int(schema.min_corroboration),
        bool(schema.conditional_gate),
        schema.duration_unit,
    )


def _build_category_defaults(
    schema: Mapping[str, CategoryRuleSchema]
) -> Dict[str, CategoryRule]:
    category_defaults: Dict[str, CategoryRule] = {}
    for name, entry in schema.items():
        category_defaults[str(name)] = CategoryRule(
            min_days=int(entry.min_days),
            documents=tuple(entry.documents),
        )
    return category_defaults


def _select_documents(
    requested: tuple[str, ...] | None,
    category_fallback: CategoryRule | None,
    defaults: ValidationRule,
) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    if category_fallback and category_fallback.documents:
        return tuple(category_fallback.documents)
    return tuple(defaults.documents)


def _build_field_rule(
    schema: ValidationFieldSchema,
    defaults: ValidationRule,
    category_defaults: Mapping[str, CategoryRule],
) -> ValidationRule:
    category = schema.category or defaults.category
    category_fallback = category_defaults.get(category)
    min_days = int(schema.min_days)
    documents = _select_documents(schema.documents, category_fallback, defaults)
    points = int(schema.points) if schema.points is not None else defaults.points
    strength = schema.strength
    ai_needed = bool(schema.ai_needed)
    min_corroboration = (
        int(schema.min_corroboration)
        if schema.min_corroboration is not None
        else defaults.min_corroboration
    )
    conditional_gate = (
        bool(schema.conditional_gate)
        if schema.conditional_gate is not None
        else defaults.conditional_gate
    )
    duration_unit = schema.duration_unit or defaults.duration_unit
    return ValidationRule(
        category,
        min_days,
        documents,
        points,
        strength,
        ai_needed,
        min_corroboration,
        conditional_gate,
        duration_unit,
    )


def _build_validation_config_from_schema(
    schema: ValidationConfigSchema,
) -> ValidationConfig:
    defaults = _build_validation_defaults(schema.defaults)
    category_defaults = _build_category_defaults(schema.category_defaults)
    fields: Dict[str, ValidationRule] = {}
    for field_name, rule_schema in schema.fields.items():
        fields[str(field_name)] = _build_field_rule(
            rule_schema, defaults, category_defaults
        )

    broadcast = schema.broadcast_disputes
    if broadcast is None:
        broadcast = True if schema.mode == "broad" else False

    return ValidationConfig(
        defaults=defaults,
        fields=fields,
        category_defaults=category_defaults,
        schema_version=int(schema.schema_version),
        mode=schema.mode,
        broadcast_disputes=bool(broadcast),
        threshold_points=int(schema.threshold_points),
    )


def _validate_environment_settings() -> None:
    errors: List[str] = []

    percent_raw = os.getenv("VALIDATION_CANARY_PERCENT")
    if percent_raw:
        try:
            percent = int(percent_raw)
        except ValueError:
            errors.append(
                "VALIDATION_CANARY_PERCENT must be an integer between 0 and 100"
            )
        else:
            if not 0 <= percent <= 100:
                errors.append(
                    "VALIDATION_CANARY_PERCENT must be between 0 and 100"
                )

    mode_raw = os.getenv("VALIDATION_MODE")
    if mode_raw:
        normalized_mode = mode_raw.strip().lower()
        if normalized_mode not in {"broad", "strict"}:
            errors.append("VALIDATION_MODE must be either 'broad' or 'strict'")

    broadcast_raw = os.getenv("BROADCAST_DISPUTES")
    if broadcast_raw is not None and broadcast_raw.strip():
        normalized_broadcast = broadcast_raw.strip().lower()
        if normalized_broadcast not in _VALID_BOOL_STRINGS:
            errors.append(
                "BROADCAST_DISPUTES must be a boolean flag (use 1/0 or true/false)"
            )

    if errors:
        raise ValidationConfigError("; ".join(errors))


@lru_cache(maxsize=1)
def load_validation_config(path: str | Path = _CONFIG_PATH) -> ValidationConfig:
    """Load validation metadata from YAML configuration."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        message = f"validation configuration not found at {config_path}"
        raise ValidationConfigError(message) from exc

    try:
        loaded = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        message = f"validation configuration could not be parsed: {config_path}"
        raise ValidationConfigError(message) from exc

    if not isinstance(loaded, Mapping):
        raise ValidationConfigError(
            f"validation configuration must be a mapping: {config_path}"
        )

    try:
        schema = ValidationConfigSchema.model_validate(loaded)
    except ValidationError as exc:
        message = f"validation configuration invalid: {config_path}\n{exc}"
        raise ValidationConfigError(message) from exc

    config = _build_validation_config_from_schema(schema)
    _validate_environment_settings()
    return config


def _clone_field_consistency(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Any]:
    """Deep-copy ``field_consistency`` ensuring plain ``dict``/``list`` containers."""

    def _clone(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _clone(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_clone(item) for item in value]
        return value

    return {str(field): _clone(details) for field, details in field_consistency.items()}


def _strip_raw_from_field_consistency(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return a deep copy of ``field_consistency`` without bureau raw values."""

    cloned = _clone_field_consistency(field_consistency)

    for field, details in list(cloned.items()):
        if isinstance(details, dict):
            details.pop("raw", None)

    return cloned


def _filter_inconsistent_fields(
    field_consistency: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Return only the inconsistent fields from a field consistency payload."""

    result: Dict[str, Dict[str, Any]] = {}
    for raw_field, raw_details in field_consistency.items():
        if not isinstance(raw_details, Mapping):
            continue

        consensus = str(raw_details.get("consensus", "")).lower()
        if consensus == "unanimous":
            continue

        field = str(raw_field)
        normalized = raw_details.get("normalized")
        if isinstance(normalized, Mapping):
            normalized_payload = dict(normalized)
        else:
            normalized_payload = normalized

        raw_values = raw_details.get("raw")
        if isinstance(raw_values, Mapping):
            raw_payload = dict(raw_values)
        else:
            raw_payload = raw_values

        disagreeing = raw_details.get("disagreeing_bureaus") or []
        if isinstance(disagreeing, Sequence) and not isinstance(
            disagreeing, (str, bytes, bytearray)
        ):
            disagreeing_list = sorted(str(item) for item in disagreeing)
        else:
            disagreeing_list = []

        missing = raw_details.get("missing_bureaus") or []
        if isinstance(missing, Sequence) and not isinstance(
            missing, (str, bytes, bytearray)
        ):
            missing_list = sorted(str(item) for item in missing)
        else:
            missing_list = []

        result[field] = {
            "normalized": normalized_payload,
            "raw": raw_payload,
            "consensus": raw_details.get("consensus"),
            "disagreeing_bureaus": disagreeing_list,
            "missing_bureaus": missing_list,
        }

    return result


_HISTORY_FIELDS = {"two_year_payment_history", "seven_year_history"}

MISSING_MATERIAL_FIELDS: frozenset[str] = frozenset(
    {
        "date_opened",
        "closed_date",
        "last_payment",
        "date_of_last_activity",
        "last_verified",
        "date_reported",
        "account_status",
        "payment_status",
        "account_rating",
        "balance_owed",
        "past_due_amount",
        "two_year_payment_history",
        "seven_year_history",
        "high_balance",
        "credit_limit",
        "payment_amount",
    }
)

_LEGACY_CALENDAR_MINIMUMS: Mapping[str, int] = {
    "account_number_display": 2,
    "date_opened": 3,
    "closed_date": 6,
    "account_type": 2,
    "creditor_type": 6,
    "high_balance": 8,
    "credit_limit": 8,
    "term_length": 3,
    "payment_amount": 5,
    "payment_frequency": 3,
    "balance_owed": 8,
    "last_payment": 3,
    "past_due_amount": 8,
    "date_of_last_activity": 12,
    "account_status": 12,
    "payment_status": 25,
    "account_rating": 18,
    "last_verified": 6,
    "date_reported": 3,
    "two_year_payment_history": 18,
    "seven_year_history": 25,
}

_HISTORY_REQUIREMENT_OVERRIDES: Mapping[str, ValidationRule] = {
    "two_year_payment_history": ValidationRule(
        category="history",
        min_days=14,
        documents=("monthly_statements_2y", "internal_payment_history"),
        points=10,
        strength="strong",
        ai_needed=False,
        min_corroboration=1,
        conditional_gate=False,
        duration_unit="business_days",
    ),
    "seven_year_history": ValidationRule(
        category="history",
        min_days=19,
        documents=(
            "cra_report_7y",
            "cra_audit_logs",
            "collection_history",
        ),
        points=12,
        strength="strong",
        ai_needed=False,
        min_corroboration=1,
        conditional_gate=False,
        duration_unit="business_days",
    ),
}

_DELINQUENCY_MARKERS = {
    "30",
    "60",
    "90",
    "120",
    "150",
    "180",
    "CO",
    "LATE30",
    "LATE60",
    "LATE90",
}


def _coerce_history_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_structured_history_value(field: str, value: Any) -> bool:
    if value is None:
        return False
    if field == "two_year_payment_history":
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return True
        return False
    if field == "seven_year_history":
        return isinstance(value, Mapping)
    return True


def _history_counts_signature(field: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        return ()
    if field == "two_year_payment_history":
        counts = value.get("counts")
        if not isinstance(counts, Mapping):
            return ()
        return (
            _coerce_history_int(counts.get("CO")),
            _coerce_history_int(counts.get("late30")),
            _coerce_history_int(counts.get("late60")),
            _coerce_history_int(counts.get("late90")),
        )
    if field == "seven_year_history":
        return (
            _coerce_history_int(value.get("late30")),
            _coerce_history_int(value.get("late60")),
            _coerce_history_int(value.get("late90")),
        )
    return ()


def _history_tokens_signature(field: str, value: Any) -> tuple[str, ...]:
    if field != "two_year_payment_history":
        return ()
    if not isinstance(value, Mapping):
        return ()
    tokens = value.get("tokens")
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes, bytearray)):
        return ()
    signature: list[str] = []
    for token in tokens:
        if token is None:
            continue
        if isinstance(token, Mapping):
            status = token.get("status")
            if status is not None:
                text = str(status).strip().upper()
                if text:
                    signature.append(text)
                continue
            serialized = json.dumps(token, sort_keys=True)
            if serialized:
                signature.append(serialized.upper())
            continue
        text = str(token).strip()
        if text:
            signature.append(text.upper())
    return tuple(signature)


def _history_signature_has_delinquency(signature: Sequence[str]) -> bool:
    for token in signature:
        normalized = str(token).strip().upper()
        if not normalized:
            continue
        condensed = normalized.replace(" ", "")
        if condensed in _DELINQUENCY_MARKERS:
            return True
        if any(marker in normalized for marker in {"CHARGE", "COLLECT", "DEROG"}):
            return True
    return False


def _determine_history_strength(
    field: str, details: Mapping[str, Any]
) -> tuple[str, bool]:
    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        return "strong", False

    missing_raw = details.get("missing_bureaus") or []
    if isinstance(missing_raw, Sequence) and not isinstance(
        missing_raw, (str, bytes, bytearray)
    ):
        missing = {str(bureau) for bureau in missing_raw}
    else:
        missing = set()

    present_bureaus = [
        str(bureau)
        for bureau in normalized.keys()
        if str(bureau) not in missing
    ]
    if missing and present_bureaus:
        return "soft", True

    if not present_bureaus:
        return "strong", False

    raw_values = details.get("raw")
    raw_map = raw_values if isinstance(raw_values, Mapping) else {}

    for bureau in present_bureaus:
        if not _is_structured_history_value(field, raw_map.get(bureau)):
            return "soft", True

    counts_signatures = [
        _history_counts_signature(field, normalized.get(bureau))
        for bureau in present_bureaus
    ]
    if field == "two_year_payment_history":
        unique_counts = {signature for signature in counts_signatures}
        if len(unique_counts) <= 1:
            token_signatures = [
                _history_tokens_signature(field, normalized.get(bureau))
                for bureau in present_bureaus
            ]
            unique_tokens = {signature for signature in token_signatures}
            if len(unique_tokens) > 1:
                if not any(
                    _history_signature_has_delinquency(signature)
                    for signature in unique_tokens
                ):
                    return "soft", True

    return "strong", False
_SEMANTIC_FIELDS = {"account_type", "creditor_type", "account_rating"}


def _looks_like_date_field(field: str) -> bool:
    return "date" in field.lower()


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_field(field: str, normalized: Mapping[str, Any]) -> bool:
    if any(_is_numeric_value(value) for value in normalized.values() if value is not None):
        return True
    lowered = field.lower()
    if any(
        keyword in lowered
        for keyword in (
            "amount",
            "balance",
            "limit",
            "payment",
            "value",
            "due",
            "credit",
            "loan",
            "debt",
        )
    ):
        return True
    return False


def _has_missing_mismatch(normalized: Mapping[str, Any]) -> bool:
    values = list(normalized.values())
    return any(value is None for value in values) and any(
        value is not None for value in values
    )


def _determine_account_number_strength(normalized: Mapping[str, Any]) -> tuple[str, bool]:
    last4_values = set()
    for value in normalized.values():
        if isinstance(value, Mapping):
            last4 = value.get("last4")
        else:
            last4 = None
        if last4:
            last4_values.add(str(last4))
    if len(last4_values) > 1:
        return "strong", False
    if _use_deterministic_account_number_policy():
        return "soft", False
    return "soft", True


def _apply_strength_policy(
    field: str, details: Mapping[str, Any], rule: ValidationRule
) -> ValidationRule:
    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        normalized = {}

    strength = rule.strength
    ai_needed = rule.ai_needed

    if field in _HISTORY_FIELDS:
        strength, ai_needed = _determine_history_strength(field, details)
    elif field == "account_number_display":
        strength, ai_needed = _determine_account_number_strength(normalized)
    elif field in _SEMANTIC_FIELDS:
        strength, ai_needed = "soft", True
    elif _looks_like_date_field(field):
        strength, ai_needed = "strong", False
    elif _is_numeric_field(field, normalized):
        strength, ai_needed = "strong", False
    elif _has_missing_mismatch(normalized) and strength != "strong":
        strength = "medium"

    if strength == rule.strength and ai_needed == rule.ai_needed:
        return rule

    return ValidationRule(
        rule.category,
        rule.min_days,
        rule.documents,
        rule.points,
        strength,
        ai_needed,
        rule.min_corroboration,
        rule.conditional_gate,
        rule.duration_unit,
    )


def _resolve_validation_mode(config: ValidationConfig) -> str:
    override = os.getenv("VALIDATION_MODE")
    if override:
        lowered = override.strip().lower()
        if lowered in {"broad", "strict"}:
            return lowered
    return config.mode


def _should_broadcast(config: ValidationConfig) -> bool:
    override = os.getenv("BROADCAST_DISPUTES")
    if override is not None:
        return override.strip() == "1"

    mode = _resolve_validation_mode(config)
    if mode == "broad":
        return True

    return config.broadcast_disputes


def _select_requirement_bureaus(
    details: Mapping[str, Any], *, broadcast_all: bool
) -> List[str]:
    disagreeing = details.get("disagreeing_bureaus")
    if not isinstance(disagreeing, Sequence) or isinstance(disagreeing, (str, bytes, bytearray)):
        disagreeing_list: List[str] = []
    else:
        disagreeing_list = [str(item) for item in disagreeing]

    normalized = details.get("normalized")
    if not isinstance(normalized, Mapping):
        normalized = {}

    if broadcast_all:
        bureaus = sorted(str(key) for key in normalized.keys())
        if bureaus:
            return bureaus

    missing = details.get("missing_bureaus")
    if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)):
        missing_list = [str(item) for item in missing]
    else:
        missing_list = []

    participants = set(disagreeing_list)
    participants.update(missing_list)

    if missing_list:
        present_bureaus = [
            str(bureau)
            for bureau, value in normalized.items()
            if value is not None and str(bureau) not in missing_list
        ]
        participants.update(present_bureaus)

    if not participants:
        participants.update(str(key) for key in normalized.keys())

    return sorted(participants)


def _build_requirement_entries(
    fields: Mapping[str, Any],
    config: ValidationConfig,
    *,
    broadcast_all: bool,
) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []
    business_only_mode = _business_only_mode_enabled()
    emit_business_fields = _emit_business_fields_enabled() and not business_only_mode
    hide_calendar_fields = _hide_calendar_fields_enabled() or business_only_mode
    business_sla_enabled = _business_day_sla_enabled()

    for field in sorted(fields.keys()):
        details = fields[field]
        if field == "creditor_remarks" and not _include_creditor_remarks():
            continue
        base_rule = config.fields.get(field)
        if base_rule is None and field not in _HISTORY_REQUIREMENT_OVERRIDES:
            continue
        rule = base_rule or config.defaults

        if field in _HISTORY_REQUIREMENT_OVERRIDES:
            override = _HISTORY_REQUIREMENT_OVERRIDES[field]
            rule = ValidationRule(
                category=override.category,
                min_days=override.min_days,
                documents=override.documents,
                points=override.points,
                strength=override.strength,
                ai_needed=override.ai_needed,
                min_corroboration=override.min_corroboration,
                conditional_gate=override.conditional_gate,
                duration_unit=override.duration_unit,
            )

        rule = _apply_strength_policy(field, details, rule)
        bureaus = _select_requirement_bureaus(details, broadcast_all=broadcast_all)
        business_min_days: int | None = None
        min_days_value = rule.min_days
        duration_unit = rule.duration_unit

        if rule.duration_unit == "business_days":
            business_min_days = rule.min_days
            if business_only_mode:
                min_days_value = business_min_days
                duration_unit = "business_days"
            else:
                if business_sla_enabled:
                    min_days_value = business_to_calendar(0, rule.min_days)
                else:
                    legacy_value = _LEGACY_CALENDAR_MINIMUMS.get(field)
                    if legacy_value is not None:
                        min_days_value = legacy_value
                    else:
                        min_days_value = business_to_calendar(0, rule.min_days)
                duration_unit = "calendar_days"

        payload: Dict[str, Any] = {
            "field": field,
            "category": rule.category,
            "min_days": min_days_value,
            "documents": list(rule.documents),
            "strength": rule.strength,
            "ai_needed": rule.ai_needed,
            "min_corroboration": rule.min_corroboration,
            "conditional_gate": rule.conditional_gate,
            "bureaus": bureaus,
            "duration_unit": duration_unit,
        }

        if emit_business_fields and business_min_days is not None:
            payload["min_days_business"] = business_min_days

        if hide_calendar_fields:
            for key in list(payload.keys()):
                if key.endswith("_calendar"):
                    payload.pop(key, None)

        if business_only_mode and rule.duration_unit == "business_days":
            if payload.get("duration_unit") != "business_days":
                logger.warning(
                    "VALIDATION_BUSINESS_ONLY_DURATION field=%s unit=%s",
                    field,
                    payload.get("duration_unit"),
                )
            if "min_days_business" in payload:
                logger.warning(
                    "VALIDATION_BUSINESS_ONLY_EXTRA_FIELD field=%s key=min_days_business",
                    field,
                )

        requirements.append(payload)
    return requirements


def build_validation_requirements(
    bureaus: Mapping[str, Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Return validation requirements for fields with cross-bureau inconsistencies."""

    config = load_validation_config()
    if not isinstance(field_consistency, Mapping):
        field_consistency_full = compute_field_consistency(dict(bureaus))
    else:
        field_consistency_full = _clone_field_consistency(field_consistency)

    inconsistencies = _filter_inconsistent_fields(field_consistency_full)
    _emit_field_debug(field_consistency_full, inconsistencies)
    broadcast_all = _should_broadcast(config)
    requirements = _build_requirement_entries(
        inconsistencies, config, broadcast_all=broadcast_all
    )

    return requirements, inconsistencies, field_consistency_full


def _coerce_normalized_map(details: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Extract a mapping of bureau values from ``details`` safely."""

    if not isinstance(details, Mapping):
        return {}

    normalized = details.get("normalized")
    if isinstance(normalized, Mapping):
        return {str(key): value for key, value in normalized.items()}

    return {}


def _coerce_raw_map(details: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Extract raw bureau values from ``details`` safely."""

    if not isinstance(details, Mapping):
        return {}

    raw = details.get("raw")
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}

    return {}


def _sanitize_for_log(value: Any) -> Any:
    """Convert ``value`` into JSON-serializable primitives for logging."""

    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_log(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_for_log(item) for item in value]

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive guard
            return repr(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _json_default(value: Any) -> str:
    """Fallback JSON serializer for structured debug output."""

    return repr(value)


def _fallback_decision_for_reason(reason_code: str) -> str | None:
    normalized = reason_code.strip().upper()
    return _FALLBACK_DECISIONS.get(normalized)


def _compute_mismatch_rate(
    details: Mapping[str, Any] | None,
    reason_details: Mapping[str, Any],
) -> float:
    """Compute mismatch rate for telemetry reporting."""

    present_count = reason_details.get("present_count")
    try:
        present_total = float(present_count)
    except (TypeError, ValueError):
        present_total = 0.0

    if present_total <= 0.0:
        return 0.0

    disagreeing: Sequence[Any] | None = None
    if isinstance(details, Mapping):
        candidate = details.get("disagreeing_bureaus")
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            disagreeing = candidate

    disagreeing_count = float(len(disagreeing or ()))

    if disagreeing_count <= 0.0 and bool(reason_details.get("is_mismatch")):
        distinct_values = reason_details.get("distinct_values")
        try:
            distinct_total = float(distinct_values)
        except (TypeError, ValueError):
            distinct_total = 0.0
        if distinct_total > 1.0:
            disagreeing_count = max(distinct_total - 1.0, 0.0)

    mismatch_rate = disagreeing_count / present_total

    if mismatch_rate < 0.0:
        return 0.0
    if mismatch_rate > 1.0:
        return 1.0
    return mismatch_rate


def _emit_field_debug(
    field_consistency: Mapping[str, Any],
    inconsistencies: Mapping[str, Any],
) -> None:
    """Emit structured debug logs and telemetry for every field."""

    debug_enabled = os.getenv("VALIDATION_DEBUG") == "1"
    reasons_enabled = _is_validation_reason_enabled()

    for field, details in sorted(field_consistency.items(), key=lambda item: str(item[0])):
        if not isinstance(details, Mapping):
            continue

        normalized_map = _coerce_normalized_map(details)
        raw_map = _coerce_raw_map(details)

        try:
            reason_details = classify_reason(normalized_map)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("VALIDATION_FIELD_CLASSIFY_FAILED field=%s", field)
            reason_details = {
                "reason_code": None,
                "reason_label": None,
                "is_missing": False,
                "is_mismatch": False,
                "missing_count": None,
                "present_count": None,
                "distinct_values": None,
            }

        send_to_ai = False
        if reasons_enabled and field in inconsistencies:
            try:
                send_to_ai = bool(decide_send_to_ai(field, reason_details))
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("VALIDATION_FIELD_AI_DECISION_FAILED field=%s", field)

        mismatch_rate = _compute_mismatch_rate(details, reason_details)

        metrics.gauge(
            "validation.field_mismatch_rate",
            mismatch_rate,
            tags={
                "field": str(field),
                "reason_code": str(reason_details.get("reason_code") or ""),
            },
        )

        if not debug_enabled:
            continue

        log_payload = {
            "field": str(field),
            "consensus": details.get("consensus"),
            "raw": _sanitize_for_log(raw_map),
            "normalized": _sanitize_for_log(normalized_map),
            "missing_bureaus": _sanitize_for_log(
                details.get("missing_bureaus") or []
            ),
            "disagreeing_bureaus": _sanitize_for_log(
                details.get("disagreeing_bureaus") or []
            ),
            "reason_code": reason_details.get("reason_code"),
            "reason_label": reason_details.get("reason_label"),
            "is_missing": reason_details.get("is_missing"),
            "is_mismatch": reason_details.get("is_mismatch"),
            "missing_count": reason_details.get("missing_count"),
            "present_count": reason_details.get("present_count"),
            "distinct_values": reason_details.get("distinct_values"),
            "has_finding": field in inconsistencies,
            "send_to_ai": send_to_ai,
            "mismatch_rate": mismatch_rate,
        }

        try:
            logger.debug(
                "VALIDATION_FIELD_TRACE %s",
                json.dumps(log_payload, default=_json_default, sort_keys=True),
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("VALIDATION_FIELD_TRACE_FAILED field=%s", field)


def _extract_field_details(
    field_consistency: Mapping[str, Any] | None, field_name: Any
) -> Mapping[str, Any] | None:
    if not isinstance(field_consistency, Mapping) or not isinstance(field_name, str):
        return None

    candidate = field_consistency.get(field_name)
    if isinstance(candidate, Mapping):
        return candidate
    return None


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        if text == "--":
            return True
    return False


def _account_number_letters(value: Any) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value) if ch.isalpha()).upper()


def _account_number_pair_metrics(a_raw: Any, b_raw: Any) -> tuple[str, str]:
    left = "" if _value_is_missing(a_raw) else str(a_raw or "")
    right = "" if _value_is_missing(b_raw) else str(b_raw or "")

    visible_level, visible_debug = acctnum_match_level(left, right)
    if visible_level == "exact_or_known_match":
        return visible_level, "match"

    if (visible_debug or {}).get("why") == "visible_digits_conflict":
        strict_level, strict_debug = acctnum_level(left, right)
        strict_reason = (strict_debug or {}).get("why", "")
        if strict_reason:
            if strict_reason in {"digit_conflict", "alnum_conflict"}:
                return strict_level, "conflict"
        return strict_level, "conflict"

    strict_level, strict_debug = acctnum_level(left, right)
    if strict_level == "exact_or_known_match":
        return strict_level, "match"

    strict_reason = (strict_debug or {}).get("why", "")
    if strict_reason in {"digit_conflict", "alnum_conflict"}:
        return strict_level, "conflict"

    if strict_reason == "empty":
        letters_left = _account_number_letters(left)
        letters_right = _account_number_letters(right)
        if letters_left and letters_right and letters_left != letters_right:
            return strict_level, "conflict"

    return strict_level, "insufficient"


def _sanitize_account_number_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        mask_class = value.get("mask_class")
        last4 = value.get("last4")
        parts: list[str] = []
        if mask_class:
            parts.append(str(mask_class))
        if last4:
            digits = str(last4)
            if digits:
                parts.append(digits[-4:])
        if parts:
            return "/".join(parts)
    return None


def _account_number_log_enabled() -> bool:
    return os.getenv(_ACCTCHECK_ENV_FLAG, "1") != "0"


def _emit_account_number_check_log(
    sid: str | None,
    normalized_map: Mapping[str, Any],
    raw_map: Mapping[str, Any],
) -> None:
    sid_value = sid or ""
    pairs: list[str] = []
    relations: list[str] = []
    decisions: list[str] = []

    for label, left_key, right_key in _ACCOUNT_NUMBER_PAIR_ORDER:
        level, relation = _account_number_pair_metrics(
            raw_map.get(left_key), raw_map.get(right_key)
        )
        pairs.append(f"{label}:{level}")
        relations.append(f"{label}:{relation}")
        decisions.append(relation)

    if any(state == "conflict" for state in decisions):
        decision = "different"
    elif any(state == "match" for state in decisions):
        decision = "same"
    else:
        decision = "unknown"

    fragments: list[str] = []
    for bureau in _SUMMARY_BUREAUS:
        normalized_value = normalized_map.get(bureau)
        sanitized = _sanitize_account_number_value(normalized_value)
        if sanitized:
            label = _ACCOUNT_NUMBER_BUREAU_LABELS.get(bureau, bureau[:2].title())
            fragments.append(f"{label}:{sanitized}")

    message_parts = [
        "ACCTCHECK",
        f"sid={sid_value}",
        "field=account_number_display",
        f"pairs=<{', '.join(pairs)}>",
        f"decision={decision}",
    ]

    if relations:
        message_parts.append(f"relations=<{', '.join(relations)}>")

    if fragments:
        message_parts.append(f"fragments=<{', '.join(fragments)}>")

    logger.info(" ".join(message_parts))


def _raw_value_provider_for_account_factory(
    bureaus_dict: Mapping[str, Mapping[str, Any]] | None,
):
    """Return a callable that yields raw bureau values for the active account."""

    if not isinstance(bureaus_dict, Mapping):
        sanitized: dict[str, Any] = {}
    else:
        sanitized = {str(key): value for key, value in bureaus_dict.items()}

    def _provider(field_name: Any, bureau_name: Any) -> Any:
        bureau_key = str(bureau_name)
        field_key = str(field_name)
        value = _get_bureau_value(sanitized, field_key, bureau_key)
        if value in (None, "", "--"):
            return None
        return value

    return _provider


def _build_bureau_value_snapshot(
    field_name: Any,
    details: Mapping[str, Any] | None,
    *,
    normalized_map: Mapping[str, Any] | None = None,
    raw_value_provider: Any | None = None,
) -> Dict[str, Dict[str, Any]]:
    if isinstance(normalized_map, Mapping):
        normalized_values = {str(key): value for key, value in normalized_map.items()}
    else:
        normalized_values = _coerce_normalized_map(details)
    raw_map = _coerce_raw_map(details)

    missing_set: set[str] = set()
    if isinstance(details, Mapping):
        missing = details.get("missing_bureaus")
        if isinstance(missing, Sequence) and not isinstance(
            missing, (str, bytes, bytearray)
        ):
            missing_set = {str(entry) for entry in missing}

    snapshot: Dict[str, Dict[str, Any]] = {}
    redact_pii = _should_redact_pii() and str(field_name) == "account_number_display"

    for bureau in _SUMMARY_BUREAUS:
        raw_value = None
        if raw_value_provider is not None:
            try:
                raw_value = raw_value_provider(field_name, bureau)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "VALIDATION_BUREAU_RAW_PROVIDER_FAILED field=%s bureau=%s",
                    field_name,
                    bureau,
                )
                raw_value = None
        if raw_value is None and raw_map:
            raw_value = raw_map.get(bureau)
        normalized_value = normalized_values.get(bureau)

        present = bureau not in missing_set
        if present:
            present = not _value_is_missing(raw_value) or not _value_is_missing(
                normalized_value
            )

        if not present:
            raw_value = None
            normalized_value = None
        elif redact_pii and isinstance(raw_value, str):
            raw_value = _redact_account_number_raw(raw_value)

        snapshot[bureau] = {
            "present": present,
            "raw": raw_value,
            "normalized": normalized_value,
        }

    return snapshot


def _freeze_bureau_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_bureau_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_bureau_value(item) for item in value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)
    return value


def classify_bureaus_for_finding(finding: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(finding, Mapping):
        return {}

    bureau_snapshot = finding.get("bureau_values")
    if not isinstance(bureau_snapshot, Mapping):
        return {}

    states: Dict[str, str] = {}
    present_entries: list[tuple[str, Any]] = []

    for bureau in _SUMMARY_BUREAUS:
        entry = bureau_snapshot.get(bureau)
        if not isinstance(entry, Mapping):
            states[bureau] = "missing"
            continue

        present = bool(entry.get("present"))
        if not present:
            states[bureau] = "missing"
            continue

        frozen = _freeze_bureau_value(entry.get("normalized"))
        present_entries.append((bureau, frozen))

    if not present_entries:
        for bureau in _SUMMARY_BUREAUS:
            states.setdefault(bureau, "missing")
        return states

    if len(present_entries) == 1:
        solo_bureau, _ = present_entries[0]
        states[solo_bureau] = "solo"
        for bureau in _SUMMARY_BUREAUS:
            states.setdefault(bureau, "missing")
        return states

    counts: Counter[Any] = Counter(value for _, value in present_entries)
    max_count = max(counts.values()) if counts else 0
    consensus_values = {value for value, count in counts.items() if count == max_count and max_count > 1}

    for bureau, frozen in present_entries:
        if frozen in consensus_values:
            states[bureau] = "aligned"
        else:
            states[bureau] = "conflict"

    for bureau in _SUMMARY_BUREAUS:
        states.setdefault(bureau, "missing")

    return states


def apply_missing_only_downgrade(finding: MutableMapping[str, Any]) -> None:
    if not isinstance(finding, MutableMapping):
        return

    field = finding.get("field")
    if not isinstance(field, str) or not field:
        return

    if field in MISSING_MATERIAL_FIELDS:
        return

    if not finding.get("is_missing") or finding.get("is_mismatch"):
        return

    downgrade_target = "neutral_context_only"
    changed = False

    for key in ("decision", "default_decision"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip() == "supportive_needs_companion":
            finding[key] = downgrade_target
            changed = True

    hint_value = finding.get("default_decision_hint")
    if isinstance(hint_value, str) and hint_value.strip() == "supportive_needs_companion":
        finding["default_decision_hint"] = downgrade_target
        changed = True

    if changed:
        finding.setdefault("decision_source", "rules")


def _build_finding(
    entry: Mapping[str, Any],
    field_consistency: Mapping[str, Any] | None,
    *,
    details: Mapping[str, Any] | None = None,
    normalized_map: Mapping[str, Any] | None = None,
    raw_value_provider: Any | None = None,
) -> Dict[str, Any]:
    """Return a finding enriched with reason metadata and AI routing."""

    finding = dict(entry)

    field_name = finding.get("field")
    if details is None:
        details = _extract_field_details(field_consistency, field_name)

    if normalized_map is None:
        normalized_map = _coerce_normalized_map(details)

    reason_details = classify_reason(dict(normalized_map))

    finding.update(reason_details)

    default_decision = decide_default(
        str(field_name) if field_name is not None else "",
        str(reason_details.get("reason_code") or ""),
    )
    if not default_decision:
        fallback = _fallback_decision_for_reason(
            str(reason_details.get("reason_code") or "")
        )
        if fallback:
            default_decision = fallback
    ai_needed = bool(finding.get("ai_needed"))
    if default_decision:
        if ai_needed:
            finding["default_decision_hint"] = default_decision
        else:
            finding["default_decision"] = default_decision
            finding["decision"] = default_decision
            finding.setdefault("decision_source", "rules")

    finding["send_to_ai"] = decide_send_to_ai(field_name, reason_details)
    finding["bureau_values"] = _build_bureau_value_snapshot(
        field_name,
        details,
        normalized_map=normalized_map,
        raw_value_provider=raw_value_provider,
    )

    if backend_config.SEED_ARGUMENTS_ENABLE:
        decision_value = finding.get("decision")
        if decision_value in {"strong_actionable", "strong"}:
            field_value = str(finding.get("field") or "")
            reason_code_value = str(finding.get("reason_code") or "")
            if field_value and reason_code_value:
                argument_block = finding.get("argument")
                if isinstance(argument_block, Mapping):
                    if "seed" not in argument_block:
                        seed_payload = build_seed_argument(
                            field_value, reason_code_value
                        )
                        if seed_payload:
                            merged_argument = dict(argument_block)
                            merged_argument.update(seed_payload)
                            finding["argument"] = merged_argument
                else:
                    seed_payload = build_seed_argument(field_value, reason_code_value)
                    if seed_payload:
                        finding["argument"] = dict(seed_payload)

    return finding


def _collect_ai_overrides(
    existing_block: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    if not isinstance(existing_block, Mapping):
        return overrides

    findings = existing_block.get("findings")
    if not isinstance(findings, Sequence):
        return overrides

    for entry in findings:
        if not isinstance(entry, Mapping):
            continue

        field_value = entry.get("field")
        if not isinstance(field_value, str):
            continue

        field_key = field_value.strip().lower()
        if not field_key:
            continue

        preserved: dict[str, Any] = {}
        for key, value in entry.items():
            if key.startswith("ai_"):
                preserved[key] = value
            elif key == "validation_ai" and isinstance(value, Mapping):
                preserved[key] = dict(value)

        if "pre_ai_decision" in entry:
            preserved.setdefault("pre_ai_decision", entry["pre_ai_decision"])

        if "ai_decision" in preserved or "validation_ai" in preserved:
            if "decision" in entry:
                preserved["decision"] = entry["decision"]
            if "decision_source" in entry:
                preserved["decision_source"] = entry["decision_source"]
            if "default_decision" in entry:
                preserved["default_decision"] = entry["default_decision"]

        if preserved:
            overrides[field_key] = preserved

    return overrides


def _apply_ai_overrides_to_findings(
    findings: list[MutableMapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> None:
    if not overrides or not findings:
        return

    for finding in findings:
        if not isinstance(finding, MutableMapping):
            continue

        field_value = finding.get("field")
        if not isinstance(field_value, str):
            continue

        field_key = field_value.strip().lower()
        if not field_key:
            continue

        override = overrides.get(field_key)
        if not override:
            continue

        if not finding.get("send_to_ai"):
            continue

        for key, value in override.items():
            if value is None:
                finding.pop(key, None)
                continue
            finding[key] = value


def _collect_seed_arguments(findings: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    seeds_by_id: dict[str, dict[str, Any]] = {}
    if not findings:
        return []

    for entry in findings:
        if not isinstance(entry, Mapping):
            continue
        argument_block = entry.get("argument")
        if not isinstance(argument_block, Mapping):
            continue
        seed_entry = argument_block.get("seed")
        if not isinstance(seed_entry, Mapping):
            continue
        seed_id = str(seed_entry.get("id") or "").strip()
        if not seed_id or seed_id in seeds_by_id:
            continue

        tone_value = seed_entry.get("tone")
        tone = str(tone_value).strip() if tone_value is not None else "firm_courteous"
        if not tone:
            tone = "firm_courteous"

        text_value = seed_entry.get("text")
        text = str(text_value or "").strip()

        seeds_by_id[seed_id] = {
            "id": seed_id,
            "tone": tone,
            "text": text,
        }

    return list(seeds_by_id.values())


def _normalize_arguments_block(
    arguments_block: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    seeds: list[dict[str, Any]] = []
    composites: list[Any] = []

    if not isinstance(arguments_block, Mapping):
        return seeds, composites

    seeds_raw = arguments_block.get("seeds")
    if isinstance(seeds_raw, Sequence) and not isinstance(
        seeds_raw, (str, bytes, bytearray)
    ):
        seen: set[str] = set()
        for seed_entry in seeds_raw:
            if not isinstance(seed_entry, Mapping):
                continue
            seed_id = str(seed_entry.get("id") or "").strip()
            if not seed_id or seed_id in seen:
                continue
            tone_value = seed_entry.get("tone")
            tone = (
                str(tone_value).strip() if tone_value is not None else "firm_courteous"
            )
            if not tone:
                tone = "firm_courteous"
            text_value = seed_entry.get("text")
            text = str(text_value or "").strip()
            seeds.append({"id": seed_id, "tone": tone, "text": text})
            seen.add(seed_id)

    composites_raw = arguments_block.get("composites")
    if isinstance(composites_raw, list):
        composites = composites_raw
    elif isinstance(composites_raw, Sequence) and not isinstance(
        composites_raw, (str, bytes, bytearray)
    ):
        composites = list(composites_raw)

    return seeds, composites


def _merge_summary_arguments(
    summary_data: MutableMapping[str, Any],
    normalized_payload: MutableMapping[str, Any],
) -> bool:
    payload_arguments = normalized_payload.get("arguments")
    seeds, payload_composites = _normalize_arguments_block(
        payload_arguments if isinstance(payload_arguments, Mapping) else None
    )

    normalized_payload["arguments"] = {
        "seeds": seeds,
        "composites": payload_composites,
    }

    existing_arguments = summary_data.get("arguments")
    _, existing_composites = _normalize_arguments_block(
        existing_arguments if isinstance(existing_arguments, Mapping) else None
    )

    composites = existing_composites or payload_composites
    normalized_payload["arguments"]["composites"] = composites

    target_arguments = {
        "seeds": seeds,
        "composites": composites,
    }

    if summary_data.get("arguments") != target_arguments:
        summary_data["arguments"] = target_arguments
        return True

    return False


def build_findings(
    requirements: Sequence[Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
    raw_value_provider: Any | None = None,
) -> List[Dict[str, Any]]:
    """Return normalized findings enriched with metadata when enabled."""

    findings: List[Dict[str, Any]] = []
    reasons_enabled = _is_validation_reason_enabled()

    for entry in requirements:
        if not isinstance(entry, Mapping):
            continue

        try:
            normalized_entry = dict(entry)
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception(
                "VALIDATION_FINDING_NORMALIZE_FAILED field=%s", entry
            )
            continue

        field_name = normalized_entry.get("field")

        if not field_name:
            findings.append(normalized_entry)
            continue

        details = _extract_field_details(field_consistency, field_name)
        normalized_map = _coerce_normalized_map(details)

        if reasons_enabled:
            try:
                finding = _build_finding(
                    normalized_entry,
                    field_consistency,
                    details=details,
                    normalized_map=normalized_map,
                    raw_value_provider=raw_value_provider,
                )
            except Exception:  # pragma: no cover - defensive enrichment
                logger.exception(
                    "VALIDATION_FINDING_ENRICH_FAILED field=%s",
                    normalized_entry.get("field"),
                )
                finding = dict(normalized_entry)
        else:
            finding = dict(normalized_entry)

        if "bureau_values" not in finding:
            finding["bureau_values"] = _build_bureau_value_snapshot(
                field_name,
                details,
                normalized_map=normalized_map,
                raw_value_provider=raw_value_provider,
            )

        finding["bureau_dispute_state"] = classify_bureaus_for_finding(finding)
        apply_missing_only_downgrade(finding)

        findings.append(finding)

    return findings


def build_summary_payload(
    requirements: Sequence[Mapping[str, Any]],
    *,
    field_consistency: Mapping[str, Any] | None = None,
    raw_value_provider: Any | None = None,
    sid: str | None = None,
    runs_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Build the summary.json payload for validation requirements."""

    normalized_requirements: list[dict[str, Any]] = []
    suppressed_fields: list[str] = []
    tolerance_notes: list[dict[str, Any]] = []
    sid_value = str(sid) if sid is not None else None
    runs_root_value = os.fspath(runs_root) if runs_root is not None else None
    debug_enabled = os.getenv("VALIDATION_DEBUG") == "1"

    for entry in requirements:
        if not isinstance(entry, Mapping):
            continue

        normalized_entry = dict(entry)
        field_name = normalized_entry.get("field")

        details = _extract_field_details(field_consistency, field_name)
        normalized_map = _coerce_normalized_map(details)
        raw_map = _coerce_raw_map(details)

        if (
            debug_enabled
            and _account_number_log_enabled()
            and field_name == "account_number_display"
            and details is not None
        ):
            _emit_account_number_check_log(
                sid_value,
                normalized_map,
                raw_map,
            )

        if field_name == "account_number_display" and sid_value:
            digit_conflicts = 0
            alnum_conflicts = 0
            aggregate_level = "none"
            for _label, left_key, right_key in _ACCOUNT_NUMBER_PAIR_ORDER:
                left_value = raw_map.get(left_key)
                right_value = raw_map.get(right_key)
                level, debug = acctnum_match_level(
                    str(left_value or ""), str(right_value or "")
                )
                if aggregate_level == "none" and level:
                    aggregate_level = level
                reason = ""
                if isinstance(debug, Mapping):
                    reason = str(debug.get("why") or "")
                if reason in {"digit_conflict", "visible_digits_conflict"}:
                    digit_conflicts += 1
                elif reason == "alnum_conflict":
                    alnum_conflicts += 1

            if runflow_account_steps_enabled():
                runflow_step(
                    sid_value,
                    "validation",
                    "acctnum_compare_merge_semantics",
                    metrics={
                        "level": aggregate_level,
                        "digit_conflicts": digit_conflicts,
                        "alnum_conflicts": alnum_conflicts,
                    },
                    out={"field": "account_number_display"},
                )

        if (
            field_name
            and isinstance(field_consistency, Mapping)
            and sid_value
            and runs_root_value
        ):
            bureau_values = {
                bureau: normalized_map.get(bureau)
                for bureau in _SUMMARY_BUREAUS
            }

            if any(value is not None for value in bureau_values.values()):
                tolerance_result = _evaluate_with_tolerance(
                    sid_value,
                    runs_root_value,
                    str(field_name),
                    bureau_values,
                )

                if (
                    isinstance(tolerance_result, Mapping)
                    and tolerance_result.get("tolerance_applied")
                    and not tolerance_result.get("is_mismatch", True)
                ):
                    suppressed_fields.append(str(field_name))
                    if debug_enabled:
                        note = _coerce_tolerance_note(
                            field_name,
                            tolerance_result.get("note"),
                        )
                        if note is not None:
                            tolerance_notes.append(note)
                    logger.debug(
                        "VALIDATION_TOLERANCE_SUPPRESS field=%s metric=%s",
                        field_name,
                        tolerance_result.get("metric"),
                    )
                    continue

        normalized_requirements.append(normalized_entry)
    reasons_enabled = _is_validation_reason_enabled()
    findings = build_findings(
        normalized_requirements,
        field_consistency=field_consistency,
        raw_value_provider=raw_value_provider,
    )

    payload: Dict[str, Any] = {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "findings": findings,
    }

    seeds = _collect_seed_arguments(findings)
    existing_arguments = payload.get("arguments")
    composites: list[Any] = []
    if isinstance(existing_arguments, Mapping):
        raw_composites = existing_arguments.get("composites")
        if isinstance(raw_composites, list):
            composites = raw_composites
        elif isinstance(raw_composites, Sequence) and not isinstance(
            raw_composites, (str, bytes, bytearray)
        ):
            composites = list(raw_composites)

    payload["arguments"] = {
        "seeds": seeds,
        "composites": composites,
    }

    if suppressed_fields:
        payload.setdefault("tolerance", {})["suppressed_fields"] = suppressed_fields

    if tolerance_notes:
        payload["tolerance_notes"] = tolerance_notes

    if summary_writer.include_legacy_requirements():
        payload["requirements"] = normalized_requirements

    if field_consistency and summary_writer.include_field_consistency():
        if reasons_enabled:
            sanitized_consistency = _strip_raw_from_field_consistency(field_consistency)
            if sanitized_consistency:
                payload["field_consistency"] = sanitized_consistency
        else:
            cloned_consistency = _clone_field_consistency(field_consistency)
            if cloned_consistency:
                payload["field_consistency"] = cloned_consistency

    if sid_value:
        runflow_step(
            sid_value,
            "validation",
            "build_summary_payload",
            metrics={
                "requirements": len(normalized_requirements),
                "suppressed": len(suppressed_fields),
            },
        )

    return summary_writer.sanitize_validation_payload(payload)


def _load_summary(summary_path: Path) -> MutableMapping[str, Any]:
    try:
        raw = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        loaded = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(loaded, Mapping):
        return {}
    return dict(loaded)


def _load_summary_meta(summary_path: Path) -> Mapping[str, Any]:
    meta_path = summary_path.parent / "meta.json"
    try:
        raw = meta_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    try:
        loaded = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(loaded, Mapping):
        return {}

    return dict(loaded)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None


def _coerce_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)):
        text = str(raw).strip()
        return [text] if text else []
    if isinstance(raw, Sequence):
        result: list[str] = []
        for entry in raw:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                result.append(text)
        return result
    text = str(raw).strip()
    return [text] if text else []


def _ensure_list_field(
    summary_data: MutableMapping[str, Any], key: str, fallback: Any
) -> bool:
    if key in summary_data:
        normalized = _coerce_string_list(summary_data.get(key))
        if summary_data.get(key) != normalized:
            summary_data[key] = normalized
            return True
        return False

    normalized_fallback = _coerce_string_list(fallback)
    summary_data[key] = normalized_fallback
    return True


def _ensure_summary_scaffold(
    summary_path: Path, summary_data: MutableMapping[str, Any]
) -> bool:
    changed = False
    meta = _load_summary_meta(summary_path)

    idx_value = summary_data.get("account_index")
    if idx_value is None:
        idx_value = meta.get("account_index")
        if idx_value is None:
            idx_value = summary_path.parent.name
    idx_int = _coerce_int(idx_value)
    if idx_int is not None:
        if summary_data.get("account_index") != idx_int:
            summary_data["account_index"] = idx_int
            changed = True

    pointers_map: dict[str, Any] = dict(_DEFAULT_SUMMARY_POINTERS)
    meta_pointers = meta.get("pointers")
    if isinstance(meta_pointers, Mapping):
        for key, value in meta_pointers.items():
            pointers_map[str(key)] = str(value)

    existing_pointers = summary_data.get("pointers")
    if isinstance(existing_pointers, Mapping):
        for key, value in existing_pointers.items():
            pointers_map[str(key)] = str(value)
    if summary_data.get("pointers") != pointers_map:
        summary_data["pointers"] = pointers_map
        changed = True

    if "account_id" in summary_data:
        account_id = summary_data.get("account_id")
        if account_id is not None:
            normalized_account_id = str(account_id)
            if summary_data.get("account_id") != normalized_account_id:
                summary_data["account_id"] = normalized_account_id
                changed = True
    else:
        account_id_meta = meta.get("account_id")
        if account_id_meta is None and idx_int is not None:
            account_id_meta = f"idx-{idx_int:03d}"
        if account_id_meta is not None:
            summary_data["account_id"] = str(account_id_meta)
            changed = True

    if _ensure_list_field(summary_data, "problem_reasons", meta.get("problem_reasons")):
        changed = True
    if _ensure_list_field(summary_data, "problem_tags", meta.get("problem_tags")):
        changed = True

    return changed


def apply_validation_summary(
    summary_path: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Update ``summary.json`` with validation requirements when they changed."""

    summary_data = _load_summary(summary_path)
    scaffold_changed = _ensure_summary_scaffold(summary_path, summary_data)
    existing_block = summary_data.get("validation_requirements")
    ai_overrides = _collect_ai_overrides(existing_block)

    normalized_payload = dict(payload)

    findings_payload = normalized_payload.get("findings")
    if isinstance(findings_payload, Sequence) and not isinstance(
        findings_payload, (str, bytes, bytearray)
    ):
        findings_list = [
            dict(entry)
            for entry in findings_payload
            if isinstance(entry, Mapping)
        ]
    else:
        findings_list = []
    findings_count = len(findings_list)
    _apply_ai_overrides_to_findings(findings_list, ai_overrides)
    normalized_payload["findings"] = findings_list
    normalized_payload = summary_writer.sanitize_validation_payload(normalized_payload)
    tolerance_notes_payload = normalized_payload.pop("tolerance_notes", None)

    arguments_changed = _merge_summary_arguments(summary_data, normalized_payload)
    existing_normalized = (
        dict(existing_block) if isinstance(existing_block, Mapping) else None
    )
    if isinstance(existing_normalized, dict):
        existing_normalized = summary_writer.sanitize_validation_payload(
            existing_normalized
        )
        existing_normalized.pop("tolerance_notes", None)
        seeds_existing, composites_existing = _normalize_arguments_block(
            existing_normalized.get("arguments")
        )
        existing_normalized["arguments"] = {
            "seeds": seeds_existing,
            "composites": composites_existing,
        }

    needs_update = existing_normalized != normalized_payload

    tolerance_notes_changed = False
    normalized_notes: list[Mapping[str, Any]] | None = None
    if isinstance(tolerance_notes_payload, Sequence) and not isinstance(
        tolerance_notes_payload, (str, bytes, bytearray)
    ):
        normalized_notes = [
            dict(note)
            for note in tolerance_notes_payload
            if isinstance(note, Mapping)
        ]

    if normalized_notes:
        existing_notes = summary_data.get("tolerance_notes")
        if existing_notes != normalized_notes:
            summary_data["tolerance_notes"] = normalized_notes
            tolerance_notes_changed = True
    else:
        if summary_data.pop("tolerance_notes", None) is not None:
            tolerance_notes_changed = True

    if findings_count == 0 and not summary_writer.should_write_empty_requirements():
        write_required = scaffold_changed or arguments_changed
        if existing_block is not None:
            summary_data.pop("validation_requirements", None)
            write_required = True
        if write_required:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_data)
            logger.debug(
                "summary: findings=%s, requirements_written=%s, schema_version=%s",
                findings_count,
                False,
                normalized_payload.get("schema_version"),
            )
            _atomic_write_json(summary_path, summary_data)
        return summary_data

    if needs_update:
        summary_data["validation_requirements"] = dict(normalized_payload)

    write_required = (
        scaffold_changed
        or needs_update
        or tolerance_notes_changed
        or arguments_changed
    )

    if summary_writer.strip_disallowed_sections(summary_data):
        write_required = True

    if write_required:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary_data)
        logger.debug(
            "summary: findings=%s, requirements_written=%s, schema_version=%s",
            findings_count,
            "requirements" in normalized_payload,
            normalized_payload.get("schema_version"),
        )
        _atomic_write_json(summary_path, summary_data)

    return summary_data


def _apply_dry_run_summary(
    summary_path: Path, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Persist dry-run validation payload alongside existing results."""

    summary_data = _load_summary(summary_path)
    scaffold_changed = _ensure_summary_scaffold(summary_path, summary_data)

    normalized_payload = dict(payload)

    findings_payload = normalized_payload.get("findings")
    if isinstance(findings_payload, Sequence) and not isinstance(
        findings_payload, (str, bytes, bytearray)
    ):
        findings_list = [
            entry for entry in findings_payload if isinstance(entry, Mapping)
        ]
    else:
        findings_list = []

    findings_count = len(findings_list)
    normalized_payload["findings"] = findings_list
    normalized_payload = summary_writer.sanitize_validation_payload(normalized_payload)

    existing_block = summary_data.get("validation_requirements_dry_run")
    existing_normalized = (
        dict(existing_block) if isinstance(existing_block, Mapping) else None
    )
    if isinstance(existing_normalized, dict):
        existing_normalized = summary_writer.sanitize_validation_payload(
            existing_normalized
        )

    arguments_changed = _merge_summary_arguments(summary_data, normalized_payload)

    if isinstance(existing_normalized, dict):
        seeds_existing, composites_existing = _normalize_arguments_block(
            existing_normalized.get("arguments")
        )
        existing_normalized["arguments"] = {
            "seeds": seeds_existing,
            "composites": composites_existing,
        }

    write_required = scaffold_changed

    if findings_count == 0 and not summary_writer.should_write_empty_requirements():
        if existing_block is not None:
            summary_data.pop("validation_requirements_dry_run", None)
            write_required = True
    else:
        if existing_normalized != normalized_payload:
            summary_data["validation_requirements_dry_run"] = dict(normalized_payload)
            write_required = True

    if summary_writer.strip_disallowed_sections(summary_data):
        write_required = True

    if write_required or arguments_changed:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary_data)
        _atomic_write_json(summary_path, summary_data)

    return summary_data


def sync_validation_tag(
    tag_path: Path,
    fields: Sequence[str],
    *,
    emit: bool,
) -> None:
    """Ensure ``tags.json`` reflects the validation requirements state."""

    try:
        tags = read_tags(tag_path)
    except ValueError:
        logger.exception("VALIDATION_TAG_READ_FAILED path=%s", tag_path)
        return

    filtered = [tag for tag in tags if tag.get("kind") != _VALIDATION_TAG_KIND]
    changed = len(filtered) != len(tags)

    if emit and fields:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {
            "kind": _VALIDATION_TAG_KIND,
            "fields": list(fields),
            "at": timestamp,
        }
        filtered.append(entry)
        changed = True

    if changed:
        try:
            write_tags_atomic(tag_path, filtered)
        except Exception:  # pragma: no cover - defensive file IO
            logger.exception("VALIDATION_TAG_WRITE_FAILED path=%s", tag_path)


def _should_emit_tags() -> bool:
    return os.environ.get("WRITE_VALIDATION_TAGS") == "1"


def build_validation_requirements_for_account(
    account_dir: str | Path,
    *,
    build_pack: bool = True,
    sid: str | None = None,
    runs_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Compute and persist validation requirements for ``account_dir``.

    Parameters
    ----------
    account_dir:
        Filesystem path pointing at ``runs/<sid>/cases/accounts/<idx>``.
    build_pack:
        When ``True`` (the default) a validation pack is built for the account
        after the summary has been written.  The pipeline orchestrator can pass
        ``False`` to defer pack generation until it explicitly decides the
        account should be queued for AI review.
    sid:
        Optional run identifier. When provided alongside ``runs_root`` the
        tolerance layer can resolve native trace files without inferring from
        ``account_dir``.
    runs_root:
        Optional base directory containing run artifacts. Used in combination
        with ``sid`` to locate tolerance dependencies such as date
        conventions.
    """

    account_path = Path(account_dir)
    account_label = account_path.name
    dry_run_enabled = _is_dry_run_enabled()
    canary_percent = _get_canary_percent()
    bureaus_path = account_path / "bureaus.json"
    summary_path = account_path / "summary.json"
    tags_path = account_path / "tags.json"

    if not bureaus_path.exists():
        logger.debug(
            "VALIDATION_REQUIREMENTS_SKIP_NO_BUREAUS path=%s", bureaus_path
        )
        return {"status": "no_bureaus_json"}

    try:
        raw_text = bureaus_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "VALIDATION_REQUIREMENTS_READ_FAILED path=%s", bureaus_path, exc_info=True
        )
        return {"status": "invalid_bureaus_json"}

    try:
        bureaus_raw = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(
            "VALIDATION_REQUIREMENTS_INVALID_JSON path=%s", bureaus_path, exc_info=True
        )
        return {"status": "invalid_bureaus_json"}

    if not isinstance(bureaus_raw, Mapping):
        logger.warning(
            "VALIDATION_REQUIREMENTS_INVALID_TYPE path=%s type=%s",
            bureaus_path,
            type(bureaus_raw).__name__,
        )
        return {"status": "invalid_bureaus_json"}

    run_dir: Path | None = None
    sid_for_tolerance = str(sid) if sid is not None else None
    runs_root_for_tolerance = Path(runs_root) if runs_root is not None else None

    if sid_for_tolerance and runs_root_for_tolerance is not None:
        run_dir = runs_root_for_tolerance / sid_for_tolerance

    if run_dir is None:
        try:
            if (
                account_path.parent.name == "accounts"
                and account_path.parent.parent.name == "cases"
            ):
                run_dir = account_path.parents[2]
        except IndexError:
            run_dir = None

        if run_dir is not None:
            if not sid_for_tolerance:
                sid_for_tolerance = run_dir.name
            if runs_root_for_tolerance is None:
                runs_root_for_tolerance = run_dir.parent

    convention_block = None
    if run_dir is not None:
        convention_block = read_date_convention(run_dir)
        if convention_block is None:
            logger.warning(
                "DATE_DETECT_MISSING run_dir=%s fallback=MDY/en",
                run_dir,
            )
    set_validation_context(convention_block)

    if not _account_selected_for_canary(account_path, canary_percent):
        logger.info(
            "VALIDATION_REQUIREMENTS_CANARY_SKIP account=%s percent=%s",
            account_label,
            canary_percent,
        )
        return {
            "status": "canary_skipped",
            "count": 0,
            "fields": [],
            "validation_requirements": None,
            "dry_run": dry_run_enabled,
        }

    summary_snapshot = _load_summary(summary_path)
    summary_consistency = summary_snapshot.get("field_consistency")
    consistency_override = (
        _clone_field_consistency(summary_consistency)
        if isinstance(summary_consistency, Mapping)
        else None
    )

    requirements, _, field_consistency_full = build_validation_requirements(
        bureaus_raw, field_consistency=consistency_override
    )
    raw_provider = _raw_value_provider_for_account_factory(bureaus_raw)
    payload = build_summary_payload(
        requirements,
        field_consistency=field_consistency_full,
        raw_value_provider=raw_provider,
        sid=sid_for_tolerance,
        runs_root=runs_root_for_tolerance,
    )
    if dry_run_enabled:
        summary_after = _apply_dry_run_summary(summary_path, payload)
    else:
        summary_after = apply_validation_summary(summary_path, payload)

    debug_enabled = os.getenv("VALIDATION_DEBUG") == "1"
    debug_key = "validation_debug"

    if debug_enabled:
        field_to_bureau: Dict[str, Any] = {}
        for field, details in field_consistency_full.items():
            if not isinstance(details, Mapping):
                continue
            raw_map = details.get("raw")
            if isinstance(raw_map, Mapping):
                entries = sorted(raw_map.items(), key=lambda item: str(item[0]))
                field_to_bureau[str(field)] = {
                    str(bureau): value for bureau, value in entries
                }
        debug_payload = {"raw_snapshot": {"field_to_bureau": field_to_bureau}}
        if summary_after.get(debug_key) != debug_payload:
            summary_after[debug_key] = debug_payload
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_after)
            _atomic_write_json(summary_path, summary_after)
    else:
        if debug_key in summary_after:
            summary_after.pop(debug_key, None)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_after)
            _atomic_write_json(summary_path, summary_after)

    logger.info("SUMMARY_WRITTEN account_id=%s dry_run=%s", account_label, dry_run_enabled)

    findings_payload = payload.get("findings")
    if isinstance(findings_payload, Sequence):
        fields = [
            str(entry.get("field"))
            for entry in findings_payload
            if isinstance(entry, Mapping) and entry.get("field")
        ]
        findings_count = len(findings_payload)
    else:
        fields = []
        findings_count = 0
    if not dry_run_enabled:
        sync_validation_tag(tags_path, fields, emit=_should_emit_tags())

    sid: str | None = None
    account_id: str | None = None
    try:
        sid = summary_path.parents[3].name
        account_id = summary_path.parent.name
    except IndexError:
        sid = None
        account_id = None

    # Orchestrator mode: disable legacy per-account pack building/sending
    try:
        _orchestrator_mode = str(os.getenv("VALIDATION_ORCHESTRATOR_MODE", "1")).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        _orchestrator_mode = True
    if _orchestrator_mode and build_pack:
        logger.info(
            "VALIDATION_LEGACY_PACKS_DISABLED sid=%s account_id=%s reason=orchestrator_mode",
            sid,
            account_id or account_label,
        )
        build_pack = False

    if build_pack and not dry_run_enabled and sid and account_id:
        builder_func = build_validation_pack_for_account
        if builder_func is None:
            try:  # pragma: no cover - defensive import guard
                from backend.ai.validation_builder import (
                    build_validation_pack_for_account as builder_imported,
                )
            except Exception:
                logger.exception(
                    "ERROR account_id=%s sid=%s summary=%s event=VALIDATION_PACK_BUILD_FAILED",
                    account_id,
                    sid,
                    summary_path,
                )
                builder_func = None
            else:
                globals()["build_validation_pack_for_account"] = builder_imported
                builder_func = builder_imported

        if builder_func is not None:
            try:
                pack_lines = builder_func(
                    sid,
                    account_id,
                    summary_path,
                    bureaus_path,
                )
                pack_count = len(pack_lines)
                logger.info(
                    "PACKS_BUILT account_id=%s count=%d",
                    account_id,
                    pack_count,
                )
                logger.info("PACKS_SENT account_id=%s", account_id)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "ERROR account_id=%s sid=%s summary=%s event=VALIDATION_PACK_BUILD_FAILED",
                    account_id,
                    sid,
                    summary_path,
                )

    result = {
        "status": "ok",
        "count": findings_count,
        "fields": fields,
        "validation_requirements": payload,
    }

    result["dry_run"] = dry_run_enabled

    if __debug__ and not summary_writer.include_legacy_requirements():
        validation_payload = result.get("validation_requirements")
        if isinstance(validation_payload, Mapping):
            assert (
                "requirements" not in validation_payload
            ), "Legacy requirements array must not be written when VALIDATION_SUMMARY_INCLUDE_REQUIREMENTS=0"

    return result

