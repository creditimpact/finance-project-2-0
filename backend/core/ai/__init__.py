"""Core AI helpers and shared OpenAI utilities."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Final

from .openai_auth import PROJECT_HEADER_NAME, auth_probe, build_openai_headers

_SUBMODULES: Final[frozenset[str]] = frozenset(
    {
        "adjudicator",
        "adjudicator_client",
        "eligibility_policy",
        "merge_validation",
        "models",
        "paraphrase",
        "paths",
        "report_compare",
        "service",
        "validators",
    }
)

__all__ = ["PROJECT_HEADER_NAME", "auth_probe", "build_openai_headers", *sorted(_SUBMODULES)]


def __getattr__(name: str) -> ModuleType:
    """Lazily import submodules so package consumers can access them directly."""

    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(__all__))
