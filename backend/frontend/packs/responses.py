"""Helpers for storing frontend review responses."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, MutableSequence

from backend.frontend.packs.config import load_frontend_stage_config


def _sanitize_account_id(account_id: str) -> str:
    trimmed = account_id.strip() if isinstance(account_id, str) else ""
    if not trimmed:
        return "account"

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed)
    return sanitized or "account"


def _resolve_stage_responses_dir(run_dir: Path) -> Path:
    config = load_frontend_stage_config(run_dir)
    return config.responses_dir


def _resolve_legacy_responses_dir(run_dir: Path) -> Path:
    legacy_dir_env = os.getenv("FRONTEND_RESPONSES_DIR")
    if legacy_dir_env:
        return run_dir / Path(legacy_dir_env)

    return run_dir / "frontend" / "responses"


def append_frontend_response(
    run_dir: Path | str, account_id: str, payload_dict: Mapping[str, object]
) -> None:
    """Append a JSON payload for ``account_id`` to the frontend responses log."""

    base_dir = Path(run_dir)
    sanitized_id = _sanitize_account_id(account_id)
    record = json.dumps(payload_dict, ensure_ascii=False)

    target_dirs: MutableSequence[Path] = []
    for candidate in (
        _resolve_stage_responses_dir(base_dir),
        _resolve_legacy_responses_dir(base_dir),
    ):
        if candidate not in target_dirs:
            target_dirs.append(candidate)

    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        target_path = directory / f"{sanitized_id}.jsonl"
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(record)
            handle.write("\n")
