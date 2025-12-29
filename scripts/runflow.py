try:  # pragma: no cover - import shim to support direct execution
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback path setup
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from backend.ai.note_style_stage import (
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.ai.validation_index import ValidationPackIndexWriter
from backend.validation.index_schema import ValidationIndex, ValidationPackRecord, load_validation_index

from backend.runflow.decider import (
    refresh_validation_stage_from_index,
    refresh_note_style_stage_from_results,
    reconcile_umbrella_barriers,
    runflow_refresh_umbrella_barriers,
    _validation_record_has_results,
    _validation_record_result_paths,
)
from backend.runflow.counters import note_style_stage_counts


def _resolve_runs_root(value: str | None) -> Path:
    if not value:
        return Path("runs").resolve()
    return Path(value).resolve()


def _load_validation_index(index_path: Path) -> ValidationIndex:
    try:
        return load_validation_index(index_path)
    except FileNotFoundError:
        raise SystemExit(f"validation index not found: {index_path}")


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def _select_result_path(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    for candidate in paths:
        if candidate.suffix.lower() == ".json":
            return candidate
    return paths[0]


def _parse_account_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _load_note_style_stage_payload(runs_root: Path, sid: str) -> Mapping[str, object] | None:
    run_dir = runs_root / sid
    runflow_path = run_dir / "runflow.json"
    try:
        raw = runflow_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    stages_payload = payload.get("stages")
    if not isinstance(stages_payload, Mapping):
        return None
    stage_payload = stages_payload.get("note_style")
    return stage_payload if isinstance(stage_payload, Mapping) else None


def _record_completed_result(
    writer: ValidationPackIndexWriter,
    index: ValidationIndex,
    record: ValidationPackRecord,
    *,
    result_paths: Sequence[Path],
) -> bool:
    summary_path = _select_result_path(result_paths)
    if summary_path is None:
        return False

    pack_path = index.resolve_pack_path(record)
    request_lines = None
    model = None
    completed_at = None

    extra = getattr(record, "extra", None)
    if isinstance(extra, Mapping):
        request_lines = extra.get("request_lines")
        model = extra.get("model") or extra.get("ai_model")
        completed_at = extra.get("completed_at")

    line_count = record.lines if getattr(record, "lines", 0) > 0 else None
    if line_count is None and summary_path.suffix.lower() == ".jsonl":
        line_count = _count_lines(summary_path)

    updated = writer.record_result(
        pack_path,
        status="completed",
        error=None,
        request_lines=request_lines,
        model=model,
        completed_at=completed_at,
        result_path=summary_path,
        line_count=line_count,
    )

    return updated is not None


def _cmd_note_style_build(args: argparse.Namespace) -> int:
    sid = (args.sid or "").strip()
    if not sid:
        print("error: SID is required", file=sys.stderr)
        return 2

    runs_root = _resolve_runs_root(getattr(args, "runs_root", None))
    only_accounts = _parse_account_list(getattr(args, "only", None))

    if only_accounts:
        accounts = [account for account in only_accounts]
    else:
        discovered = discover_note_style_response_accounts(sid, runs_root=runs_root)
        accounts = [entry.account_id for entry in discovered]

    results: dict[str, Mapping[str, object]] = {}
    status_counts: dict[str, int] = {}

    for account_id in accounts:
        result = build_note_style_pack_for_account(
            sid, account_id, runs_root=runs_root
        )
        normalized_status = str(result.get("status") or "").lower()
        status_counts[normalized_status] = status_counts.get(normalized_status, 0) + 1
        results[account_id] = dict(result)

    counts_payload = note_style_stage_counts(runs_root / sid) or {
        "packs_total": 0,
        "packs_completed": 0,
        "packs_failed": 0,
    }
    stage_payload = _load_note_style_stage_payload(runs_root, sid)
    stage_status = None
    stage_results: Mapping[str, object] | None = None
    if isinstance(stage_payload, Mapping):
        stage_status = stage_payload.get("status")
        results_payload = stage_payload.get("results")
        if isinstance(results_payload, Mapping):
            stage_results = dict(results_payload)

    payload = {
        "sid": sid,
        "processed_accounts": accounts,
        "status_counts": status_counts,
        "stage_status": stage_status,
        "stage_results": stage_results,
        "counts": counts_payload,
        "results": results,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_note_style_refresh(args: argparse.Namespace) -> int:
    sid = (args.sid or "").strip()
    if not sid:
        print("error: SID is required", file=sys.stderr)
        return 2

    runs_root = _resolve_runs_root(getattr(args, "runs_root", None))

    refresh_note_style_stage_from_results(sid, runs_root=runs_root)
    runflow_refresh_umbrella_barriers(sid)
    reconcile_umbrella_barriers(sid, runs_root=runs_root)

    counts_payload = note_style_stage_counts(runs_root / sid) or {
        "packs_total": 0,
        "packs_completed": 0,
        "packs_failed": 0,
    }
    stage_payload = _load_note_style_stage_payload(runs_root, sid)

    stage_dict = dict(stage_payload) if isinstance(stage_payload, Mapping) else None

    payload = {
        "sid": sid,
        "counts": counts_payload,
        "stage": stage_dict,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_backfill_validation(args: argparse.Namespace) -> int:
    sid = (args.sid or "").strip()
    if not sid:
        print("error: SID is required", file=sys.stderr)
        return 2

    runs_root = _resolve_runs_root(getattr(args, "runs_root", None))
    run_dir = runs_root / sid
    index_path = run_dir / "ai_packs" / "validation" / "index.json"

    index = _load_validation_index(index_path)
    writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=index_path,
        packs_dir=index.packs_dir_path,
        results_dir=index.results_dir_path,
    )

    completed = 0
    missing_results: list[int] = []

    for record in index.packs:
        normalized_status = (record.status or "").strip().lower()
        has_results = _validation_record_has_results(index, record)
        if not has_results:
            if normalized_status == "completed":
                missing_results.append(record.account_id)
            continue

        if normalized_status == "completed":
            continue

        result_paths = _validation_record_result_paths(index, record)
        if not result_paths:
            continue

        if _record_completed_result(writer, index, record, result_paths=result_paths):
            completed += 1

    refresh_validation_stage_from_index(sid, runs_root=runs_root)
    runflow_refresh_umbrella_barriers(sid)
    reconcile_umbrella_barriers(sid, runs_root=runs_root)

    payload = {
        "sid": sid,
        "updated": completed,
        "missing_results": missing_results,
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runflow management commands")
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser(
        "backfill-validation",
        help="Backfill validation index entries based on on-disk results",
    )
    backfill.add_argument("sid", help="Run identifier")
    backfill.add_argument(
        "--runs-root",
        dest="runs_root",
        help="Override the runs root directory",
    )
    backfill.set_defaults(func=_cmd_backfill_validation)

    note_style = sub.add_parser(
        "note-style",
        help="Manage note_style AI stage artifacts",
    )
    note_style_sub = note_style.add_subparsers(dest="note_style_command", required=True)

    note_style_build = note_style_sub.add_parser(
        "build",
        help="Build note_style packs for response notes",
    )
    note_style_build.add_argument("--sid", required=True, help="Run identifier")
    note_style_build.add_argument(
        "--only",
        help="Comma separated list of account IDs to rebuild",
    )
    note_style_build.add_argument(
        "--runs-root",
        dest="runs_root",
        help="Override the runs root directory",
    )
    note_style_build.set_defaults(func=_cmd_note_style_build)

    note_style_refresh = note_style_sub.add_parser(
        "refresh",
        help="Refresh note_style stage status from on-disk results",
    )
    note_style_refresh.add_argument("--sid", required=True, help="Run identifier")
    note_style_refresh.add_argument(
        "--runs-root",
        dest="runs_root",
        help="Override the runs root directory",
    )
    note_style_refresh.set_defaults(func=_cmd_note_style_refresh)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.error("a subcommand is required")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
