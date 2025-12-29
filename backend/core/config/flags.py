import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY

SAFE_MERGE_ENABLED = env_bool("SAFE_MERGE_ENABLED", False)
NORMALIZED_OVERLAY_ENABLED = env_bool("NORMALIZED_OVERLAY_ENABLED", False)
CASE_FIRST_BUILD_ENABLED = env_bool("CASE_FIRST_BUILD_ENABLED", False)
ONE_CASE_PER_ACCOUNT_ENABLED = env_bool("ONE_CASE_PER_ACCOUNT_ENABLED", False)
CASE_FIRST_BUILD_REQUIRED = env_bool("CASE_FIRST_BUILD_REQUIRED", True)
DISABLE_PARSER_UI_SUMMARY = env_bool("DISABLE_PARSER_UI_SUMMARY", True)
METRICS_ENABLED = env_bool("METRICS_ENABLED", True)
CASEBUILDER_DEBUG = env_bool("CASEBUILDER_DEBUG", True)


@dataclass(frozen=True)
class Flags:
    safe_merge_enabled: bool = SAFE_MERGE_ENABLED
    normalized_overlay_enabled: bool = NORMALIZED_OVERLAY_ENABLED
    case_first_build_enabled: bool = CASE_FIRST_BUILD_ENABLED
    one_case_per_account_enabled: bool = ONE_CASE_PER_ACCOUNT_ENABLED
    case_first_build_required: bool = CASE_FIRST_BUILD_REQUIRED
    disable_parser_ui_summary: bool = DISABLE_PARSER_UI_SUMMARY
    metrics_enabled: bool = METRICS_ENABLED
    casebuilder_debug: bool = CASEBUILDER_DEBUG


FLAGS = Flags()
