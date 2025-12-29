"""Manifest integration helpers for the strategy planner stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from backend.pipeline.runs import RunManifest


def update_manifest_for_account(
    manifest_path: Path,
    account_key: str,
    strategy_dir: Path,
    master_path: Path,
    weekday_paths: Mapping[str, Path],
    *,
    log_path: Path | None = None,
) -> None:
    """Merge planner outputs into the per-account manifest block."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    _ = (master_path, weekday_paths, log_path)

    try:
        manifest = RunManifest.load_or_create(
            manifest_path,
            manifest_path.parent.name or None,
            allow_create=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest not found at {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest at {manifest_path} is not valid JSON") from exc

    account_key = str(account_key)

    artifacts = manifest.data.setdefault("artifacts", {})
    cases = artifacts.setdefault("cases", {})
    accounts = cases.setdefault("accounts", {})
    account_block = accounts.setdefault(account_key, {})

    try:
        resolved_strategy_dir = strategy_dir.resolve()
    except (OSError, RuntimeError):
        resolved_strategy_dir = strategy_dir

    account_dir = resolved_strategy_dir.parent
    account_block.setdefault("dir", str(account_dir.resolve()))

    runs_root_override: Path | None = None
    try:
        runs_root_override = resolved_strategy_dir.parents[4]
    except IndexError:
        runs_root_override = None

    manifest.register_strategy_artifacts_for_account(
        account_key,
        runs_root=runs_root_override,
    )

    manifest.save()
