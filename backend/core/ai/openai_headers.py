"""Compatibility wrapper for legacy imports."""

from __future__ import annotations

from .openai_auth import PROJECT_HEADER_NAME, build_openai_headers

__all__ = ["PROJECT_HEADER_NAME", "build_openai_headers"]
