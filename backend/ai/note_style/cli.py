"""Diagnostic utilities for inspecting note_style runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from backend.ai.note_style.io import note_style_snapshot, note_style_stage_view


def _resolve_runs_root(value: str | None) -> Path:
    if value:
        return Path(value)

    env_value = os.getenv("RUNS_ROOT")
    if env_value:
        return Path(env_value)

    return Path("runs")


def _load_runflow_stage(run_dir: Path) -> tuple[Mapping[str, Any] | None, Path]:
    path = run_dir / "runflow.json"
    if not path.is_file():
        return None, path

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None, path

    raw_text = raw_text.strip()
    if not raw_text:
        return None, path

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, path

    if not isinstance(payload, Mapping):
        return None, path

    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        return None, path

    stage_payload = stages.get("note_style")
    if isinstance(stage_payload, Mapping):
        return stage_payload, path
    return None, path


def _determine_account_status(
    account: str,
    *,
    built: set[str],
    completed: set[str],
    failed: set[str],
    ready: set[str],
    pending: set[str],
    missing: set[str],
) -> str:
    if account in completed:
        return "completed"
    if account in failed:
        return "failed"
    if account in missing:
        return "missing_pack"
    if account in ready:
        return "ready_to_send"
    if account in built:
        if account in pending:
            return "awaiting_result"
        return "built"
    return "pending"


def _recommended_action(view) -> dict[str, Any]:
    if not view.has_expected:
        return {"action": "empty", "reason": "no_expected_packs"}

    if view.is_terminal:
        return {
            "action": "complete",
            "total": view.total_expected,
            "completed": view.completed_total,
            "failed": view.failed_total,
        }

    missing = sorted(view.missing_builds)
    if missing:
        return {
            "action": "await_builds",
            "missing": missing,
            "count": len(missing),
        }

    ready = sorted(view.ready_to_send)
    if ready:
        return {
            "action": "send",
            "accounts": ready,
            "count": len(ready),
        }

    pending = sorted(view.pending_results)
    return {
        "action": "await_results",
        "pending": pending,
        "count": len(pending),
    }


def diagnose_note_style_stage(
    sid: str,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a structured diagnostic payload for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        raise ValueError("sid is required")

    runs_root_path = _resolve_runs_root(os.fspath(runs_root) if runs_root is not None else None)
    run_dir = runs_root_path / sid_text
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    snapshot = note_style_snapshot(sid_text, runs_root=runs_root_path)
    view = note_style_stage_view(sid_text, runs_root=runs_root_path, snapshot=snapshot)

    runflow_stage, runflow_path = _load_runflow_stage(run_dir)
    runflow_status: str | None = None
    runflow_sent: bool | None = None
    runflow_empty_ok: bool | None = None
    if isinstance(runflow_stage, Mapping):
        status_value = runflow_stage.get("status")
        if isinstance(status_value, str):
            runflow_status = status_value
        if "sent" in runflow_stage:
            runflow_sent = bool(runflow_stage.get("sent"))
        if "empty_ok" in runflow_stage:
            runflow_empty_ok = bool(runflow_stage.get("empty_ok"))

    built_accounts = set(view.packs_built)
    completed_accounts = set(view.packs_completed)
    failed_accounts = set(view.packs_failed)
    ready_accounts = set(view.ready_to_send)
    pending_accounts = set(view.pending_results)
    missing_accounts = set(view.missing_builds)

    accounts_summary: list[dict[str, Any]] = []
    for account in sorted(view.packs_expected):
        accounts_summary.append(
            {
                "account_id": account,
                "built": account in built_accounts,
                "completed": account in completed_accounts,
                "failed": account in failed_accounts,
                "ready_to_send": account in ready_accounts,
                "pending_result": account in pending_accounts,
                "missing_pack": account in missing_accounts,
                "status": _determine_account_status(
                    account,
                    built=built_accounts,
                    completed=completed_accounts,
                    failed=failed_accounts,
                    ready=ready_accounts,
                    pending=pending_accounts,
                    missing=missing_accounts,
                ),
            }
        )

    runflow_normalized = (runflow_status or "").strip().lower()
    view_status = view.state
    if view.is_terminal:
        expected_statuses = {"success", "empty"}
    else:
        expected_statuses = {view_status}
    runflow_consistent = not runflow_normalized or runflow_normalized in expected_statuses

    payload = {
        "sid": sid_text,
        "runs_root": str(runs_root_path),
        "state": {
            "status": view_status,
            "is_terminal": view.is_terminal,
            "built_complete": view.built_complete,
        },
        "counts": {
            "expected": view.total_expected,
            "built": view.built_total,
            "completed": view.completed_total,
            "failed": view.failed_total,
            "terminal": view.terminal_total,
            "missing_builds": len(view.missing_builds),
            "pending_results": len(view.pending_results),
            "ready_to_send": len(view.ready_to_send),
        },
        "snapshot": {
            "packs_expected": sorted(snapshot.packs_expected),
            "packs_built": sorted(snapshot.packs_built),
            "packs_completed": sorted(snapshot.packs_completed),
            "packs_failed": sorted(snapshot.packs_failed),
        },
        "accounts": accounts_summary,
        "recommended_action": _recommended_action(view),
        "runflow": {
            "path": str(runflow_path),
            "status": runflow_status,
            "sent": runflow_sent,
            "empty_ok": runflow_empty_ok,
            "mismatch": not runflow_consistent,
        },
    }

    if runflow_stage is not None:
        payload["runflow"]["raw"] = dict(runflow_stage)

    return payload


def _cmd_diagnose(args: argparse.Namespace) -> int:
    payload = diagnose_note_style_stage(args.sid, runs_root=args.runs_root)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="note-style",
        description="Diagnostics for note_style AI stage state.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Inspect snapshot/runflow state for a single SID"
    )
    diagnose_parser.add_argument("--sid", required=True, help="Run SID to inspect")
    diagnose_parser.add_argument(
        "--runs-root",
        help="Root directory containing run folders (default: RUNS_ROOT env or ./runs)",
    )
    diagnose_parser.set_defaults(func=_cmd_diagnose)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    return 1


__all__ = ["diagnose_note_style_stage", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
