import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

from environs import Env

env = Env()
env.read_env()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIAdjudicatorConfig:
    """Environment-backed configuration for the merge AI adjudicator."""

    enabled: bool
    base_url: str
    api_key: str | None
    model: str
    request_timeout: int
    pack_max_lines_per_side: int


_WARNED_DEFAULT_KEYS: set[str] = set()


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def env_str(name: str, default: str) -> str:
    """Fetch a string environment variable."""
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    """Parse a float environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Parse an int environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def env_list(name: str, default: list[str]) -> list[str]:
    """Parse a comma-separated list environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts or default


# Backwards compatibility for older imports
_env_bool = env_bool


def _warn_default(key: str, raw: object, default: object, reason: str) -> None:
    """Emit a structured warning when falling back to a default value."""

    if key in _WARNED_DEFAULT_KEYS:
        return

    _WARNED_DEFAULT_KEYS.add(key)
    payload = {
        "key": key,
        "value": "" if raw is None else str(raw),
        "default": default,
        "reason": reason,
    }
    logger.warning("MERGE_V2_CONFIG_DEFAULT %s", json.dumps(payload, sort_keys=True))


def _coerce_non_empty_str(key: str, default: str, *, fallback_keys: tuple[str, ...] = ()) -> str:
    """Return a sanitized string value for ``key`` or ``default`` when empty."""

    raw = os.getenv(key)
    if raw is not None:
        value = str(raw).strip()
        if value:
            return value
        _warn_default(key, raw, default, "empty")
        return default

    for fallback in fallback_keys:
        fallback_raw = os.getenv(fallback)
        if fallback_raw is None:
            continue
        value = str(fallback_raw).strip()
        if value:
            return value
        _warn_default(fallback, fallback_raw, default, "empty")
        return default

    return default


def _coerce_positive_int(key: str, default: int, *, min_value: int) -> int:
    """Return a positive integer parsed from the environment."""

    raw = os.getenv(key)
    if raw is None:
        return default

    try:
        value = int(str(raw).strip())
    except Exception:
        _warn_default(key, raw, default, "invalid_int")
        return default

    if value < min_value:
        _warn_default(key, raw, default, f"min_{min_value}")
        return default

    return value


def _warn_ai_disabled(reason: str) -> None:
    """Emit a structured warning when the AI adjudicator is disabled."""

    payload = {"reason": reason}
    logger.warning("MERGE_V2_AI_DISABLED %s", json.dumps(payload, sort_keys=True))


def _load_ai_adjudicator_config() -> AIAdjudicatorConfig:
    """Parse adjudicator-specific environment configuration."""

    enabled = env_bool("ENABLE_AI_ADJUDICATOR", False)
    base_url_default = "https://api.openai.com/v1"
    base_url = _coerce_non_empty_str("OPENAI_BASE_URL", base_url_default)
    base_url = base_url.rstrip("/") or base_url_default

    model = _coerce_non_empty_str("AI_MODEL", "gpt-4o-mini", fallback_keys=("AI_MODEL_ID",))
    request_timeout = _coerce_positive_int("AI_REQUEST_TIMEOUT", 30, min_value=1)
    pack_max_lines = _coerce_positive_int(
        "AI_PACK_MAX_LINES_PER_SIDE", 20, min_value=5
    )

    api_key_raw = os.getenv("OPENAI_API_KEY")
    api_key = api_key_raw.strip() if isinstance(api_key_raw, str) else None
    if api_key == "":
        api_key = None

    if enabled and not api_key:
        _warn_ai_disabled("missing_api_key")
        enabled = False

    return AIAdjudicatorConfig(
        enabled=enabled,
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_timeout=request_timeout,
        pack_max_lines_per_side=pack_max_lines,
    )


def _load_keyword_lists() -> Tuple[dict, dict, dict]:
    """Load tiered keyword dictionaries from JSON/YAML config.

    Returns empty dictionaries when the config file is missing or malformed.
    """

    path = os.getenv(
        "KEYWORDS_CONFIG_PATH",
        Path(__file__).resolve().parent.parent / "keywords.json",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except Exception:
        data = {}
    t1 = data.get("tier1") or data.get("TIER1_KEYWORDS") or {}
    t2 = data.get("tier2") or data.get("TIER2_KEYWORDS") or {}
    t3 = data.get("tier3") or data.get("TIER3_KEYWORDS") or {}
    return t1, t2, t3


UTILIZATION_PROBLEM_THRESHOLD = float(
    os.getenv("UTILIZATION_PROBLEM_THRESHOLD", "0.90")
)
SERIOUS_DELINQUENCY_MIN_DPD = int(os.getenv("SERIOUS_DELINQUENCY_MIN_DPD", "60"))

os.environ.setdefault("VALIDATION_MAX_RETRIES", "2")
os.environ.setdefault("VALIDATION_WRITE_JSON_ENVELOPE", "0")
os.environ.setdefault("VALIDATION_REQUEST_GROUP_SIZE", "1")
os.environ.setdefault("PREVALIDATION_DETECT_DATES", "1")
os.environ.setdefault("DATE_CONVENTION_PATH", "traces/date_convention.json")
os.environ.setdefault("DATE_CONVENTION_SCOPE", "global")

VALIDATION_ROLLBACK = _env_bool("VALIDATION_ROLLBACK", False)
ENABLE_VALIDATION_REQUIREMENTS = (
    not VALIDATION_ROLLBACK
    and os.getenv("ENABLE_VALIDATION_REQUIREMENTS", "1") == "1"
)
PREVALIDATION_DETECT_DATES = _env_bool("PREVALIDATION_DETECT_DATES", True)
DATE_CONVENTION_PATH = env_str("DATE_CONVENTION_PATH", "traces/date_convention.json")
DATE_CONVENTION_SCOPE = env_str("DATE_CONVENTION_SCOPE", "global")
ENABLE_VALIDATION_AI = _env_bool("ENABLE_VALIDATION_AI", False) and not VALIDATION_ROLLBACK
VALIDATION_DEBUG = os.getenv("VALIDATION_DEBUG", "0") == "1"
VALIDATION_MAX_RETRIES = env_int("VALIDATION_MAX_RETRIES", 2)
VALIDATION_WRITE_JSON_ENVELOPE = env_bool("VALIDATION_WRITE_JSON_ENVELOPE", False)
SEED_ARGUMENTS_ENABLE = env_bool("SEED_ARGUMENTS_ENABLE", True)


def _clamp(value: int, *, lower: int, upper: int) -> int:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


VALIDATION_DRY_RUN = _env_bool("VALIDATION_DRY_RUN", False)
VALIDATION_CANARY_PERCENT = _clamp(
    env_int("VALIDATION_CANARY_PERCENT", 100), lower=0, upper=100
)

ACCTNUM_EXACT_WEIGHT = env_int("ACCTNUM_EXACT_WEIGHT", 40)
ACCTNUM_MASKED_WEIGHT = env_int("ACCTNUM_MASKED_WEIGHT", 15)

ENABLE_TIER1_KEYWORDS = _env_bool("ENABLE_TIER1_KEYWORDS", False)
ENABLE_TIER2_KEYWORDS = _env_bool("ENABLE_TIER2_KEYWORDS", False)
ENABLE_TIER3_KEYWORDS = _env_bool("ENABLE_TIER3_KEYWORDS", False)
ENABLE_TIER2_NUMERIC = _env_bool("ENABLE_TIER2_NUMERIC", True)

_AI_ADJUDICATOR_CONFIG = _load_ai_adjudicator_config()
ENABLE_AI_ADJUDICATOR = _AI_ADJUDICATOR_CONFIG.enabled
AI_PACK_MAX_LINES_PER_SIDE = _AI_ADJUDICATOR_CONFIG.pack_max_lines_per_side
AI_MODEL = _AI_ADJUDICATOR_CONFIG.model
AI_MODEL_ID = AI_MODEL
AI_REQUEST_TIMEOUT = _AI_ADJUDICATOR_CONFIG.request_timeout
OPENAI_BASE_URL = _AI_ADJUDICATOR_CONFIG.base_url
OPENAI_API_KEY = _AI_ADJUDICATOR_CONFIG.api_key
AI_MIN_CONFIDENCE = env_float("AI_MIN_CONFIDENCE", 0.70)
AI_TEMPERATURE_DEFAULT = env_float("AI_TEMPERATURE_DEFAULT", 0.0)
AI_REQUEST_TIMEOUT_S = env_int("AI_REQUEST_TIMEOUT_S", 8)
AI_MAX_TOKENS = env_int("AI_MAX_TOKENS", 600)
AI_MAX_RETRIES = env_int("AI_MAX_RETRIES", 1)
AI_HIERARCHY_VERSION = env_str("AI_HIERARCHY_VERSION", "v1")
AI_REDACT_STRATEGY = os.getenv("AI_REDACT_STRATEGY", "hash_last4")

MERGE_AUTOSEND = env_bool("MERGE_AUTOSEND", False)
MERGE_SEND_ON_BUILD = env_bool("MERGE_SEND_ON_BUILD", False)
MERGE_STAGE_AUTORUN = env_bool("MERGE_STAGE_AUTORUN", True)
MERGE_STAGE_DIR = env_str("MERGE_STAGE_DIR", "ai_packs/merge")
MERGE_PACKS_DIR = env_str("MERGE_PACKS_DIR", f"{MERGE_STAGE_DIR}/packs")
MERGE_RESULTS_DIR = env_str("MERGE_RESULTS_DIR", f"{MERGE_STAGE_DIR}/results")
MERGE_PACK_GLOB = env_str("MERGE_PACK_GLOB", "pair_*.jsonl")
MERGE_INDEX_PATH = env_str("MERGE_INDEX_PATH", f"{MERGE_STAGE_DIR}/pairs_index.json")
MERGE_RESULTS_BASENAME = env_str(
    "MERGE_RESULTS_BASENAME", "pair_{lo:03d}_{hi:03d}.result.json"
)
MERGE_MODEL = env_str("MERGE_MODEL", AI_MODEL)
MERGE_REQUEST_TIMEOUT = env_int("MERGE_REQUEST_TIMEOUT", AI_REQUEST_TIMEOUT)
MERGE_MAX_RETRIES = env_int("MERGE_MAX_RETRIES", AI_MAX_RETRIES)
MERGE_BACKOFF_SCHEDULE = os.getenv("MERGE_BACKOFF_SCHEDULE")
MERGE_CELERY_RATE_LIMIT = os.getenv("MERGE_CELERY_RATE_LIMIT")

NOTE_STYLE_ENABLED = env_bool("NOTE_STYLE_ENABLED", True)
NOTE_STYLE_AUTOSEND = os.getenv("NOTE_STYLE_AUTOSEND", "1") == "1"
NOTE_STYLE_STAGE_DIR = env_str("NOTE_STYLE_STAGE_DIR", "ai_packs/note_style")
NOTE_STYLE_PACKS_DIR = env_str(
    "NOTE_STYLE_PACKS_DIR", f"{NOTE_STYLE_STAGE_DIR}/packs"
)
NOTE_STYLE_RESULTS_DIR = env_str(
    "NOTE_STYLE_RESULTS_DIR", f"{NOTE_STYLE_STAGE_DIR}/results"
)
NOTE_STYLE_USE_MANIFEST_PATHS = env_bool(
    "NOTE_STYLE_USE_MANIFEST_PATHS", True
)
NOTE_STYLE_WAIT_FOR_INDEX = env_bool("NOTE_STYLE_WAIT_FOR_INDEX", False)
NOTE_STYLE_RESULTS_BASENAME = env_str(
    "NOTE_STYLE_RESULTS_BASENAME", "acc_{account}.result.jsonl"
)
NOTE_STYLE_DEFAULT_PACK_GLOB = "acc_*.jsonl"
NOTE_STYLE_PACK_GLOB = env_str("NOTE_STYLE_PACK_GLOB", NOTE_STYLE_DEFAULT_PACK_GLOB)
NOTE_STYLE_SEND_ON_RESPONSE_WRITE = env_bool(
    "NOTE_STYLE_SEND_ON_RESPONSE_WRITE", False
)
NOTE_STYLE_DISABLE_SALT = env_bool("NOTE_STYLE_DISABLE_SALT", False)
NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH = env_bool(
    "NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH", True
)
NOTE_STYLE_SKIP_IF_RESULT_EXISTS = env_bool(
    "NOTE_STYLE_SKIP_IF_RESULT_EXISTS", False
)
NOTE_STYLE_JSON_ONLY_ENFORCE = (
    os.getenv("NOTE_STYLE_JSON_ONLY_ENFORCE", "1").strip().lower() == "1"
)
NOTE_STYLE_MASK_PII = env_bool("NOTE_STYLE_MASK_PII", True)
NOTE_STYLE_REQUIRE_NOTE_TEXT = env_bool("NOTE_STYLE_REQUIRE_NOTE_TEXT", True)
NOTE_STYLE_ALLOW_EMPTY_NOTE = env_bool("NOTE_STYLE_ALLOW_EMPTY_NOTE", False)
NOTE_STYLE_PACK_MAX_SIZE_KB = _coerce_positive_int(
    "NOTE_STYLE_PACK_MAX_SIZE_KB", 64, min_value=1
)
NOTE_STYLE_MAX_NOTE_CHARS = _coerce_positive_int(
    "NOTE_STYLE_MAX_NOTE_CHARS", 2000, min_value=1
)
NOTE_STYLE_MODEL = _coerce_non_empty_str("NOTE_STYLE_MODEL", "gpt-4o-mini")
NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS = _coerce_positive_int(
    "NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS",
    2,
    min_value=0,
)
NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL = env_bool(
    "NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL", False
)

RUNFLOW_NEW_SID_ON_UPLOAD_ONLY = env_bool(
    "RUNFLOW_NEW_SID_ON_UPLOAD_ONLY", False
)
RUNFLOW_RECONCILE_ON_BOOT = env_bool("RUNFLOW_RECONCILE_ON_BOOT", False)
RUNFLOW_IGNORE_EMPTY_SIDS = env_bool("RUNFLOW_IGNORE_EMPTY_SIDS", True)
FRONTEND_STAGE_AUTORUN = env_bool("FRONTEND_STAGE_AUTORUN", True)
REVIEW_STAGE_AUTORUN = env_bool("REVIEW_STAGE_AUTORUN", True)
ANALYZER_USE_MANIFEST_PATHS = env_bool("ANALYZER_USE_MANIFEST_PATHS", True)
UMBRELLA_BARRIERS_STRICT_VALIDATION = env_bool(
    "UMBRELLA_BARRIERS_STRICT_VALIDATION", False
)

_raw_t1, _raw_t2, _raw_t3 = _load_keyword_lists()

TIER1_KEYWORDS = _raw_t1 if ENABLE_TIER1_KEYWORDS else {}
TIER2_KEYWORDS = _raw_t2 if ENABLE_TIER2_KEYWORDS else {}
TIER3_KEYWORDS = _raw_t3 if ENABLE_TIER3_KEYWORDS else {}


def get_ai_adjudicator_config() -> AIAdjudicatorConfig:
    """Return the parsed AI adjudicator configuration."""

    return _AI_ADJUDICATOR_CONFIG


def _default_casestore_dir() -> str:
    """Determine the Case Store directory.

    Preference is given to explicit environment variables ``CASESTORE_DIR`` or
    ``CASE_STORE_DIR``.  When neither is provided, we fall back to a
    project-relative ``.cases`` directory to ensure cross-platform portability.
    The directory is created on demand.
    """

    env_dir = os.getenv("CASESTORE_DIR") or os.getenv("CASE_STORE_DIR")
    if env_dir:
        base = Path(env_dir)
    else:
        # ``config.py`` lives under ``backend/``; its grandparent is the
        # repository root.
        base = Path(__file__).resolve().parent.parent.parent / ".cases"
    base.mkdir(parents=True, exist_ok=True)
    return base.as_posix()


# Case Store configuration
CASESTORE_DIR = _default_casestore_dir()
CASESTORE_REDACT_BEFORE_STORE = env_bool("CASESTORE_REDACT_BEFORE_STORE", True)
CASESTORE_ATOMIC_WRITES = env_bool("CASESTORE_ATOMIC_WRITES", True)
CASESTORE_VALIDATE_ON_LOAD = env_bool("CASESTORE_VALIDATE_ON_LOAD", True)

# Parser dual-write flags
ENABLE_CASESTORE_WRITE = env_bool("ENABLE_CASESTORE_WRITE", False)
CASESTORE_PARSER_LOG_PARITY = env_bool("CASESTORE_PARSER_LOG_PARITY", True)

# Stage A Case Store migration flags
ENABLE_CASESTORE_STAGEA = env_bool("ENABLE_CASESTORE_STAGEA", True)
STAGEA_LABEL_PREFIX_MATCH = env_bool("STAGEA_LABEL_PREFIX_MATCH", True)
STAGEA_COLONLESS_TU_SPLIT = env_bool("STAGEA_COLONLESS_TU_SPLIT", True)
STAGEA_COLONLESS_TU_BOUNDARY = env_bool("STAGEA_COLONLESS_TU_BOUNDARY", True)
STAGEA_COLONLESS_TU_TEXT_FALLBACK = env_bool("STAGEA_COLONLESS_TU_TEXT_FALLBACK", True)
STAGEA_ORIGCRED_PREFIX_RESCUE = env_bool("STAGEA_ORIGCRED_PREFIX_RESCUE", True)
STAGEA_ORIGCRED_TU_BOUNDARY = env_bool("STAGEA_ORIGCRED_TU_BOUNDARY", True)
STAGEA_ORIGCRED_COLONLESS_SPLIT = env_bool("STAGEA_ORIGCRED_COLONLESS_SPLIT", True)
STAGEA_ORIGCRED_POST_EXTRACT = env_bool("STAGEA_ORIGCRED_POST_EXTRACT", False)

logger.info(
    (
        "StageA flags: prefix_match=%s colonless_tu_split=%s "
        "colonless_tu_boundary=%s colonless_tu_text_fallback=%s "
        "origcred_prefix_rescue=%s origcred_tu_boundary=%s "
        "origcred_colonless_split=%s origcred_post_extract=%s"
    ),
    STAGEA_LABEL_PREFIX_MATCH,
    STAGEA_COLONLESS_TU_SPLIT,
    STAGEA_COLONLESS_TU_BOUNDARY,
    STAGEA_COLONLESS_TU_TEXT_FALLBACK,
    STAGEA_ORIGCRED_PREFIX_RESCUE,
    STAGEA_ORIGCRED_TU_BOUNDARY,
    STAGEA_ORIGCRED_COLONLESS_SPLIT,
    STAGEA_ORIGCRED_POST_EXTRACT,
)

# Stage A detection mode
PROBLEM_DETECTION_ONLY = env_bool("PROBLEM_DETECTION_ONLY", True)

# Automatic trace cleanup after Stage-A export
PURGE_TRACE_AFTER_EXPORT = env_bool("PURGE_TRACE_AFTER_EXPORT", False)
PURGE_TRACE_KEEP_TEXTS = env_bool("PURGE_TRACE_KEEP_TEXTS", False)

# Candidate token logging
ENABLE_CANDIDATE_TOKEN_LOGGER = env_bool("ENABLE_CANDIDATE_TOKEN_LOGGER", True)
CANDIDATE_LOG_FORMAT = env_str("CANDIDATE_LOG_FORMAT", "jsonl")

# API decision metadata exposure
API_INCLUDE_DECISION_META = env_bool("API_INCLUDE_DECISION_META", True)
API_DECISION_META_MAX_FIELDS_USED = env_int("API_DECISION_META_MAX_FIELDS_USED", 6)

# Cross-bureau resolution
ENABLE_CROSS_BUREAU_RESOLUTION = env_bool("ENABLE_CROSS_BUREAU_RESOLUTION", False)
API_AGGREGATION_ID_STRATEGY = env_str("API_AGGREGATION_ID_STRATEGY", "winner")
API_INCLUDE_AGG_MEMBERS_META = env_bool("API_INCLUDE_AGG_MEMBERS_META", False)

# Parser audit instrumentation
PARSER_AUDIT_ENABLED = env_bool("PARSER_AUDIT_ENABLED", True)
PDF_TEXT_MIN_CHARS_PER_PAGE = env_int("PDF_TEXT_MIN_CHARS_PER_PAGE", 64)

# OCR fallback configuration (flag-gated; disabled by default for prod)
# Parsing is deterministic; OCR runs only when page text is too sparse.
OCR_ENABLED = env_bool("OCR_ENABLED", False)
OCR_PROVIDER = env_str("OCR_PROVIDER", "tesseract")
OCR_TIMEOUT_MS = env_int("OCR_TIMEOUT_MS", 8000)
OCR_LANGS = env_list("OCR_LANGS", ["eng"])

# Unified text normalization configuration (enabled by default)
TEXT_NORMALIZE_ENABLED = env_bool("TEXT_NORMALIZE_ENABLED", True)
TEXT_NORMALIZE_COLLAPSE_SPACES = env_bool("TEXT_NORMALIZE_COLLAPSE_SPACES", True)
TEXT_NORMALIZE_BIDI_STRIP = env_bool("TEXT_NORMALIZE_BIDI_STRIP", True)
TEXT_NORMALIZE_DATE_ISO = env_bool("TEXT_NORMALIZE_DATE_ISO", True)
TEXT_NORMALIZE_AMOUNT_CANONICAL = env_bool("TEXT_NORMALIZE_AMOUNT_CANONICAL", True)

# Deterministic SmartCredit extractors (final/primary path)
# Leave this flag enabled by default; downstream may hard-code in future.
DETERMINISTIC_EXTRACTORS_ENABLED = env_bool("DETERMINISTIC_EXTRACTORS_ENABLED", True)

EXTRACTOR_STRICT_MODE = env_bool("EXTRACTOR_STRICT_MODE", True)

# Template-first parsing experiment (disables Stage B when enabled)
# When set, orchestrators will attempt the template detector+parser and will not
# invoke Stage B RAW builder as a fallback. Intended for isolated testing.
TEMPLATE_FIRST = env_bool("TEMPLATE_FIRST", False)

# Template-first thresholds
# Confidence threshold to accept template parse path
TEMPLATE_CONFIDENCE_THRESHOLD = env_float("TEMPLATE_CONFIDENCE_THRESHOLD", 0.70)
# Minimum label fields mapped per block to count as "good"
try:
    TEMPLATE_LABEL_MIN_PER_BLOCK = int(os.getenv("TEMPLATE_LABEL_MIN_PER_BLOCK", "8"))
except Exception:
    TEMPLATE_LABEL_MIN_PER_BLOCK = 8
# Minimum bureaus detected (TU/EX/EQ) per block to count as "good"
try:
    TEMPLATE_MIN_BUREAUS = int(os.getenv("TEMPLATE_MIN_BUREAUS", "2"))
except Exception:
    TEMPLATE_MIN_BUREAUS = 2


RAW_TRIAD_FROM_X = env_bool("RAW_TRIAD_FROM_X", True)
RAW_JOIN_TOKENS_WITH_SPACE = env.bool("RAW_JOIN_TOKENS_WITH_SPACE", True)

# Two-Year Payment History row-aware + X-aligned matching (feature-gated)
# When enabled, the history extractor will pair month rows to status rows by Y,
# then align statuses to months using X anchors and boundary bucketing.
HISTORY_X_MATCH_ENABLED = env_bool("HISTORY_X_MATCH_ENABLED", False)
HISTORY_Y_CLUSTER_DY = env_float("HISTORY_Y_CLUSTER_DY", 2.5)
HISTORY_Y_PAIR_MAX_DY = env_float("HISTORY_Y_PAIR_MAX_DY", 10.0)
HISTORY_X_SEAM_GUARD = env_float("HISTORY_X_SEAM_GUARD", 2.0)
HISTORY_ROW_PAIR_MAX_DY = env_float("HISTORY_ROW_PAIR_MAX_DY", 120.0)
HISTORY_DEBUG = env_bool("HISTORY_DEBUG", False)

# Wire the new Two-Year history logic into MAIN Stage A output (accounts_from_full.json)
# Requires both HISTORY_MAIN_WIRING_ENABLED=1 AND HISTORY_X_MATCH_ENABLED=1
HISTORY_MAIN_WIRING_ENABLED = env_bool("HISTORY_MAIN_WIRING_ENABLED", False)

# TSV-first Two-Year Payment History month extraction (per-bureau months only)
# When enabled, Stage-A will extract month labels directly from account-scoped
# TSV token rows in-memory and populate monthly fields accordingly.
# DISABLED: Per-account token extraction doesn't include the full 2Y month grid;
# the main history_extractor already correctly extracts months via HIST_MATCH.
HISTORY_TSV_2Y_ENABLED = env_bool("HISTORY_TSV_2Y_ENABLED", False)

# Disable legacy window/block_windows 2Y monthly/months generation
# When enabled (default 0), skip extract_two_year_payment_history() entirely
HISTORY_DISABLE_LEGACY_2Y_MONTHLY = env_bool("HISTORY_DISABLE_LEGACY_2Y_MONTHLY", False)

# TSV v2: Simple Y-segmentation 2Y months extraction (new clean pipeline)
# When enabled (default 0), extract Hebrew month abbreviations by bureau from 2Y region
HISTORY_TSV_2Y_V2_ENABLED = env_bool("HISTORY_TSV_2Y_V2_ENABLED", False)

# TSV v2: Monthly status pairing (status ABOVE month, no translation of month text)
# When enabled (default 0), pair month tokens with status tokens per bureau
# Status row must be above month row (status_y < month_y) within tolerance
HISTORY_TSV_2Y_V2_MONTHLY_ENABLED = env_bool("HISTORY_TSV_2Y_V2_MONTHLY_ENABLED", False)

from .note_style import (
    NOTE_STYLE_ALLOW_TOOL_CALLS,
    NOTE_STYLE_ALLOW_TOOLS,
    NOTE_STYLE_RESPONSE_MODE,
    NOTE_STYLE_RETRY_COUNT,
    NOTE_STYLE_STRICT_SCHEMA,
)
