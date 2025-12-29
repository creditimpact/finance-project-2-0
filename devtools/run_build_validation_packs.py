"""CLI helper to manually build validation validation AI packs for a SID."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.core.ai.paths import ensure_validation_paths
from backend.core.logic import validation_ai_packs
from backend.pipeline.runs import RUNS_ROOT_ENV


def _resolve_runs_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env_value = os.getenv(RUNS_ROOT_ENV)
    if env_value:
        return Path(env_value)
    return Path("runs")


def _iter_account_indices(accounts_root: Path) -> list[int]:
    indices: list[int] = []
    if not accounts_root.is_dir():
        return indices

    for entry in accounts_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue
        indices.append(idx)

    return sorted(indices)


def _load_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


class _InferenceOverride:
    """Temporarily adjust validation pack config to disable inference."""

    def __init__(self, disabled: bool):
        self._disabled = disabled
        self._original: Any | None = None

    def __enter__(self) -> None:
        if not self._disabled:
            return None

        self._original = validation_ai_packs.load_validation_packs_config

        def _patched(base_dir: Path) -> validation_ai_packs.ValidationPacksConfig:
            config = self._original(base_dir)
            return replace(config, enable_infer=False)

        validation_ai_packs.load_validation_packs_config = _patched  # type: ignore[assignment]
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._disabled and self._original is not None:
            validation_ai_packs.load_validation_packs_config = self._original  # type: ignore[assignment]


def _collect_account_summary(
    accounts: Iterable[int],
    validation_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    entries: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    weak_accounts = 0
    weak_total = 0

    for idx in accounts:
        account_dir = validation_root / f"{idx}"
        pack_path = account_dir / "pack.json"
        prompt_path = account_dir / "prompt.txt"
        result_path = account_dir / "results" / "model.json"

        pack_payload = _load_json(pack_path)
        weak_items = []
        if isinstance(pack_payload, dict):
            weak_items = pack_payload.get("weak_items") or []
            if not isinstance(weak_items, list):
                weak_items = []

        weak_count = len(weak_items)
        if weak_count:
            weak_accounts += 1
            weak_total += weak_count

        model_payload = _load_json(result_path)
        status = "unknown"
        reason = None
        attempts = None
        duration_ms = None
        if isinstance(model_payload, dict):
            status = str(model_payload.get("status") or "unknown")
            reason_val = model_payload.get("reason")
            reason = str(reason_val) if reason_val is not None else None
            try:
                attempts = int(model_payload.get("attempts"))
            except (TypeError, ValueError):
                attempts = None
            try:
                duration_ms = int(model_payload.get("duration_ms"))
            except (TypeError, ValueError):
                duration_ms = None

        status_counts[status] = status_counts.get(status, 0) + 1

        entries.append(
            {
                "account_index": idx,
                "pack_path": str(pack_path),
                "prompt_path": str(prompt_path),
                "model_results_path": str(result_path),
                "weak_count": weak_count,
                "inference_status": status,
                "inference_reason": reason,
                "attempts": attempts,
                "duration_ms": duration_ms,
            }
        )

    return entries, status_counts, weak_accounts, weak_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually build validation AI packs for a SID",
    )
    parser.add_argument("sid", help="Case SID to rebuild")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs root (defaults to $RUNS_ROOT or ./runs)",
    )
    parser.add_argument(
        "--no-infer",
        action="store_true",
        help="Only build packs and prompts; skip LLM inference",
    )

    args = parser.parse_args()

    sid = args.sid
    runs_root = _resolve_runs_root(args.runs_root).resolve()
    accounts_root = runs_root / sid / "cases" / "accounts"

    account_indices = _iter_account_indices(accounts_root)

    with _InferenceOverride(args.no_infer):
        pack_stats = validation_ai_packs.build_validation_ai_packs_for_accounts(
            sid,
            account_indices=account_indices,
            runs_root=runs_root,
        )
        validation_paths = ensure_validation_paths(runs_root, sid, create=False)
        effective_config = validation_ai_packs.load_validation_packs_config(
            validation_paths.base
        )

    (
        account_entries,
        status_counts,
        weak_accounts,
        weak_total,
    ) = _collect_account_summary(
        account_indices,
        validation_paths.base,
    )

    summary = {
        "sid": sid,
        "runs_root": str(runs_root),
        "accounts_root": str(accounts_root),
        "validation_base": str(validation_paths.base),
        "log_path": str(validation_paths.log_file),
        "config": {
            "enable_write": effective_config.enable_write,
            "enable_infer": effective_config.enable_infer,
            "model": effective_config.model,
            "weak_limit": effective_config.weak_limit,
            "max_attempts": effective_config.max_attempts,
            "backoff_seconds": list(effective_config.backoff_seconds),
        },
        "accounts_processed": len(account_indices),
        "pack_stats": pack_stats,
        "status_counts": status_counts,
        "weak_accounts": weak_accounts,
        "weak_item_total": weak_total,
        "accounts": account_entries,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
