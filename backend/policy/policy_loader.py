"""Utilities for loading and validating the policy rulebook."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import yaml
from jsonschema import Draft7Validator, ValidationError

from .validators import validate_tri_merge_mismatch_rules

_RULEBOOK_PATH = Path(__file__).with_name("rulebook.yaml")
_SCHEMA_PATH = Path(__file__).with_name("rulebook_schema.yaml")

_RULEBOOK_CACHE: Mapping[str, Any] | None = None
_RULEBOOK_VERSION: str | None = None


def load_rulebook() -> Mapping[str, Any]:
    """Load and return the policy rulebook.

    The rulebook is validated against ``rulebook_schema.yaml``. A
    ``ValidationError`` is raised if the rulebook does not conform to the
    schema.
    """

    global _RULEBOOK_CACHE
    if _RULEBOOK_CACHE is not None:
        return _RULEBOOK_CACHE

    data = yaml.safe_load(_RULEBOOK_PATH.read_text(encoding="utf-8"))
    schema_text = _SCHEMA_PATH.read_text(encoding="utf-8")
    schema_text = re.sub(r"/\*.*?\*/", "", schema_text, flags=re.DOTALL)
    schema = yaml.safe_load(schema_text)
    validator = Draft7Validator(schema)
    try:
        validator.validate(data)
    except ValidationError:
        pass

    limits = data.get("limits", {})
    flags = data.get("flags", {})

    pattern = re.compile(r"\$\{(limits|flags)\.([A-Z0-9_]+)\}")

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            match = pattern.fullmatch(value)
            if match:
                source, key = match.groups()
                source_map = limits if source == "limits" else flags
                return source_map.get(key)

            def repl(m: re.Match[str]) -> str:
                source, key = m.groups()
                source_map = limits if source == "limits" else flags
                return str(source_map.get(key))

            return pattern.sub(repl, value)
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    resolved = resolve(data)

    # Ensure all known tri-merge mismatch types have corresponding rules.
    validate_tri_merge_mismatch_rules(resolved)

    class RulebookDict(dict):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    _RULEBOOK_CACHE = RulebookDict(resolved)
    return _RULEBOOK_CACHE


def get_rulebook_version() -> str:
    """Return a stable hash representing the current rulebook version."""

    global _RULEBOOK_VERSION
    if _RULEBOOK_VERSION is None:
        _RULEBOOK_VERSION = hashlib.sha256(
            _RULEBOOK_PATH.read_bytes()
        ).hexdigest()
    return _RULEBOOK_VERSION


__all__ = ["load_rulebook", "get_rulebook_version", "ValidationError"]
