"""Lightweight authentication helpers for API endpoints.

This module exposes decorators to enforce that a request provides either a
valid API key or a required RBAC role via the ``X-Roles`` header. API keys are
validated against the configured list in :func:`backend.api.config.get_app_config`.
"""

from __future__ import annotations

from functools import wraps
from typing import Iterable

from flask import Request, jsonify, request

from backend.api.config import get_app_config


def _has_api_key(req: Request) -> bool:
    """Return ``True`` if ``req`` provides a valid API key."""

    tokens = get_app_config().auth_tokens
    if not tokens:
        return False
    auth_header = req.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else None
    return token in tokens


def _has_role(req: Request, roles: Iterable[str] | None) -> bool:
    if not roles:
        return False
    provided = {
        r.strip() for r in req.headers.get("X-Roles", "").split(",") if r.strip()
    }
    return any(role in provided for role in roles)


def require_api_key_or_role(*, roles: Iterable[str] | None = None):
    """Decorator enforcing API key or RBAC role authorization."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _has_api_key(request) or _has_role(request, roles):
                return func(*args, **kwargs)
            return (
                jsonify({"status": "error", "message": "Unauthorized"}),
                401,
            )

        return wrapper

    return decorator


__all__ = ["require_api_key_or_role"]

