"""Validation AI helper entry points."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ManifestPaths",
    "ValidationPackBuilder",
    "ValidationPackError",
    "ValidationPackSender",
    "build_validation_packs",
    "check_index",
    "load_index_for_sid",
    "rewrite_index_to_v2",
    "resolve_manifest_paths",
    "run_case",
    "send_validation_packs",
]

_EXPORTS = {
    "ManifestPaths": ("backend.validation.build_packs", "ManifestPaths"),
    "ValidationPackBuilder": ("backend.validation.build_packs", "ValidationPackBuilder"),
    "build_validation_packs": ("backend.validation.build_packs", "build_validation_packs"),
    "resolve_manifest_paths": ("backend.validation.build_packs", "resolve_manifest_paths"),
    "check_index": ("backend.validation.manifest", "check_index"),
    "load_index_for_sid": ("backend.validation.manifest", "load_index_for_sid"),
    "rewrite_index_to_v2": ("backend.validation.manifest", "rewrite_index_to_v2"),
    "run_case": ("backend.validation.run_case", "run_case"),
    "ValidationPackError": ("backend.validation.send_packs", "ValidationPackError"),
    "ValidationPackSender": ("backend.validation.send_packs", "ValidationPackSender"),
    "send_validation_packs": ("backend.validation.send_packs", "send_validation_packs"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'backend.validation' has no attribute {name!r}")
