"""OpenAI authentication helpers and diagnostics."""

from __future__ import annotations

import logging
import os
import requests

log = logging.getLogger(__name__)

_OPENAI_API_BASE = "https://api.openai.com/v1"
_PROBE_URL = f"{_OPENAI_API_BASE}/models"
_PROJECT_HEADER_FLAG = "OPENAI_SEND_PROJECT_HEADER"
PROJECT_HEADER_NAME = "OpenAI-Project"


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


def _resolved_api_key(api_key: str | None) -> str:
    key = _clean_env(api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return key


def _project_header_enabled() -> bool:
    return os.getenv(_PROJECT_HEADER_FLAG, "0") == "1"


def build_openai_headers(
    *, api_key: str | None = None, project_id: str | None = None
) -> dict[str, str]:
    """Construct the required headers for OpenAI API requests."""

    key = _resolved_api_key(api_key)

    headers: dict[str, str] = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    project_clean = _clean_env(project_id)
    if _project_header_enabled() and project_clean:
        headers[PROJECT_HEADER_NAME] = project_clean
    return headers


def _truncate(text: str, limit: int = 200) -> str:
    snippet = text.strip()
    if len(snippet) > limit:
        return snippet[:limit]
    return snippet


def auth_probe(session: requests.Session | None = None) -> None:
    """Validate OpenAI credentials by querying the models endpoint."""

    headers = build_openai_headers()
    owns_session = session is None
    sess = session or requests.Session()
    try:
        try:
            log.debug(
                "OPENAI_CALL endpoint=%s has_project_header=%s",
                "/v1/models",
                PROJECT_HEADER_NAME in headers,
            )
            response = sess.get(_PROBE_URL, headers=headers)
        except requests.RequestException as exc:
            log.error(
                "OPENAI_AUTH probe failed: status=error url=/v1/models request_id=<none> error=%s",
                exc,
            )
            raise RuntimeError("OpenAI auth probe failed") from exc

        if response.status_code == 200:
            return

        request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
        snippet = _truncate(getattr(response, "text", ""))
        log.error(
            'OPENAI_AUTH probe failed: status=%s url=/v1/models request_id=%s body_snippet="%s"',
            response.status_code,
            request_id or "<none>",
            snippet,
        )
        raise RuntimeError(f"OpenAI auth probe failed with {response.status_code}")
    finally:
        if owns_session:
            sess.close()
