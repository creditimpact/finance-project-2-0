import logging
import os
from dataclasses import dataclass, field

from backend.core.services.ai_config import AIConfig

_logger = logging.getLogger("config")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback."""

    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a floating point environment variable with fallback."""

    value = os.getenv(name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


CLASSIFY_CACHE_ENABLED = env_bool("CLASSIFY_CACHE_ENABLED", True)
CLASSIFY_CACHE_MAXSIZE = env_int("CLASSIFY_CACHE_MAXSIZE", 5000)
CLASSIFY_CACHE_TTL_SEC = env_int("CLASSIFY_CACHE_TTL_SEC", 0)
ANALYSIS_DEBUG_STORE_RAW = env_bool("ANALYSIS_DEBUG_STORE_RAW", False)
STAGE4_POLICY_ENFORCEMENT = env_bool("STAGE4_POLICY_ENFORCEMENT", False)
STAGE4_POLICY_CANARY = env_float("STAGE4_POLICY_CANARY", 0.0)
ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY = env_bool(
    "ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY", False
)
ENABLE_PLANNER = env_bool("ENABLE_PLANNER", False)
PLANNER_CANARY_PERCENT = env_float("PLANNER_CANARY_PERCENT", 100.0)
ENABLE_PLANNER_PIPELINE = env_bool("ENABLE_PLANNER_PIPELINE", False)
PLANNER_PIPELINE_CANARY_PERCENT = env_float("PLANNER_PIPELINE_CANARY_PERCENT", 100.0)
ENABLE_FIELD_POPULATION = env_bool("ENABLE_FIELD_POPULATION", True)
FIELD_POPULATION_CANARY_PERCENT = env_float("FIELD_POPULATION_CANARY_PERCENT", 100.0)
ENABLE_OBSERVABILITY_H = env_bool("ENABLE_OBSERVABILITY_H", True)
ENABLE_BATCH_RUNNER = env_bool("ENABLE_BATCH_RUNNER", True)
EXCLUDE_PARSER_AGGREGATED_ACCOUNTS = env_bool(
    "EXCLUDE_PARSER_AGGREGATED_ACCOUNTS", False
)


def env_list(name: str) -> list[str]:
    """Read a comma-separated environment variable into a list."""

    value = os.getenv(name)
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class AppConfig:
    """Application configuration loaded from the environment."""

    ai: AIConfig
    wkhtmltopdf_path: str
    rulebook_fallback_enabled: bool
    export_trace_file: bool
    smtp_server: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    celery_broker_url: str
    admin_password: str | None = None
    secret_key: str = "change-me"
    auth_tokens: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60


def get_app_config() -> AppConfig:
    """Load and validate application configuration from environment variables."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    os.environ.setdefault("OPENAI_BASE_URL", base_url)

    wkhtmltopdf_path = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")
    rulebook_fallback_enabled = os.getenv("RULEBOOK_FALLBACK_ENABLED", "1") != "0"
    export_trace_file = os.getenv("EXPORT_TRACE_FILE", "1") != "0"
    smtp_server = os.getenv("SMTP_SERVER", "localhost")
    smtp_port = int(os.getenv("SMTP_PORT", "1025"))
    smtp_username = os.getenv("SMTP_USERNAME", "noreply@example.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    admin_password = os.getenv("ADMIN_PASSWORD")
    secret_key = os.getenv("SECRET_KEY", "change-me")
    auth_tokens = env_list("API_AUTH_TOKENS")
    rate_limit = env_int("API_RATE_LIMIT_PER_MINUTE", 60)

    _logger.info("OPENAI_BASE_URL=%s", base_url)
    _logger.info("OPENAI_API_KEY present=%s", bool(api_key))
    _logger.info("RULEBOOK_FALLBACK_ENABLED=%s", rulebook_fallback_enabled)
    _logger.info("EXPORT_TRACE_FILE=%s", export_trace_file)

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if "localhost" in base_url:
        raise EnvironmentError("OPENAI_BASE_URL points to localhost")

    ai_conf = AIConfig(
        api_key=api_key,
        base_url=base_url,
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        response_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )

    return AppConfig(
        ai=ai_conf,
        wkhtmltopdf_path=wkhtmltopdf_path,
        rulebook_fallback_enabled=rulebook_fallback_enabled,
        export_trace_file=export_trace_file,
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        celery_broker_url=celery_broker_url,
        admin_password=admin_password,
        secret_key=secret_key,
        auth_tokens=auth_tokens,
        rate_limit_per_minute=rate_limit,
    )


def get_ai_config() -> AIConfig:
    """Backward compatible helper returning the AI sub-config."""

    return get_app_config().ai
