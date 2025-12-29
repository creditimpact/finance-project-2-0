from __future__ import annotations

from typing import Any, Mapping, Sequence

# Increment whenever the precedence resolution algorithm changes.
precedence_version = "2"


def get_precedence(rulebook: Mapping[str, Any] | Any) -> list[str]:
    """Return rule precedence list from ``rulebook``.

    The rulebook may expose ``precedence`` either as an attribute or key.
    Missing values default to an empty list.
    """
    precedence: Sequence[str] | None = getattr(rulebook, "precedence", None)
    if precedence is None and isinstance(rulebook, Mapping):
        precedence = rulebook.get("precedence")
    return list(precedence or [])


__all__ = ["get_precedence", "precedence_version"]
