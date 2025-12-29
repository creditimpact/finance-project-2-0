import logging
import os

log = logging.getLogger(__name__)


def sanitize_openai_env() -> None:
    """
    Normalize OpenAI env vars for all processes (Flask & Celery) before any client is created:
    - Trim whitespace
    - Fill default OPENAI_BASE_URL
    """
    key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    proj = (os.getenv("OPENAI_PROJECT_ID", "") or "").strip()
    base = (os.getenv("OPENAI_BASE_URL", "") or "").strip() or "https://api.openai.com/v1"
    send_project_header = os.getenv("OPENAI_SEND_PROJECT_HEADER", "0") == "1"

    # Re-write sanitized values back to the environment
    if key:
        os.environ["OPENAI_API_KEY"] = key
    if proj:
        os.environ["OPENAI_PROJECT_ID"] = proj
    os.environ["OPENAI_BASE_URL"] = base

    if send_project_header and not proj:
        log.warning(
            "OPENAI_SEND_PROJECT_HEADER is enabled but OPENAI_PROJECT_ID is not set"
        )
