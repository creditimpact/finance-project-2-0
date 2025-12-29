"""Utility to migrate legacy problem cases into the lean layout.

The historic case builder persisted a full Stage-A ``account.json`` blob for
each flagged account which included ``triad_rows``.  The lean builder introduced
in this release stores a handful of compact JSON artefacts instead.  This script
converts older case folders to the lean layout so that existing runs match the
new structure and, critically, so that ``triad_rows`` never land on disk.

Usage::

    $ python -m scripts.migrate_cases_to_lean runs/<sid>

The command accepts one or more paths.  Each path can point to an individual run
directory, a ``cases`` directory, or the ``cases/accounts`` directory itself.
All matching account folders are migrated in-place.  A ``--dry-run`` flag is
available to preview the operations without touching the filesystem.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from backend.core.logic.report_analysis.account_merge import merge_v2_only_enabled
from backend.core.logic.report_analysis.problem_case_builder import (
    _build_bureaus_payload_from_stagea,
)
from backend.core.logic.report_analysis.problem_extractor import (
    build_rule_fields_from_triad,
)

logger = logging.getLogger(__name__)

POINTERS = {
    "raw": "raw_lines.json",
    "bureaus": "bureaus.json",
    "flat": "fields_flat.json",
    "tags": "tags.json",
    "summary": "summary.json",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to parse JSON file: path=%s exc=%s", path, exc)
        return None


def _write_json(path: Path, obj: Any, dry_run: bool) -> None:
    if dry_run:
        logger.info("DRY_RUN write %s", path)
        return
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_reason_fields(
    summary_data: Mapping[str, Any] | None,
    account_data: Mapping[str, Any] | None,
) -> tuple[Any, List[Any], List[Any]]:
    primary_issue = None
    problem_reasons: List[Any] | None = None
    problem_tags: List[Any] | None = None

    candidates: List[Mapping[str, Any]] = []
    if isinstance(summary_data, Mapping):
        candidates.append(summary_data)
        reason = summary_data.get("reason")
        if isinstance(reason, Mapping):
            candidates.append(reason)
    if isinstance(account_data, Mapping):
        reason = account_data.get("reason")
        if isinstance(reason, Mapping):
            candidates.append(reason)

    for cand in candidates:
        if primary_issue is None:
            primary_issue = cand.get("primary_issue")
        if problem_reasons is None:
            values = cand.get("problem_reasons") or cand.get("issue_types")
            if values is not None:
                problem_reasons = _ensure_list(values)
        if problem_tags is None:
            values = cand.get("problem_tags")
            if values is not None:
                problem_tags = _ensure_list(values)

    return (
        primary_issue,
        problem_reasons if problem_reasons is not None else [],
        problem_tags if problem_tags is not None else [],
    )


def _sanitize_merge_tag(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return json.loads(json.dumps(value, ensure_ascii=False))
        except TypeError:
            return dict(value)
    return value


def _account_dirs_from_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []

    def _collect(base: Path) -> None:
        if not base.exists():
            return
        if base.is_file():
            return
        accounts_dir = None
        if (base / "cases" / "accounts").is_dir():
            accounts_dir = (base / "cases" / "accounts").resolve()
        elif base.name == "cases" and (base / "accounts").is_dir():
            accounts_dir = (base / "accounts").resolve()
        elif base.name == "accounts" and base.is_dir():
            accounts_dir = base.resolve()

        if accounts_dir is not None:
            if accounts_dir not in out:
                out.append(accounts_dir)
            return

        for child in base.iterdir():
            if child.is_dir():
                _collect(child)

    for p in paths:
        _collect(p.resolve())

    return out


def migrate_account_dir(account_dir: Path, dry_run: bool = False) -> bool:
    account_path = account_dir / "account.json"
    if not account_path.exists():
        # Already lean (or unexpected layout) – nothing to do.
        return False

    account_data = _load_json(account_path)
    if not isinstance(account_data, MutableMapping):
        logger.warning("Skipping account dir (invalid account.json): %s", account_dir)
        return False

    summary_path = account_dir / POINTERS["summary"]
    summary_data = _load_json(summary_path)
    if summary_data is not None and not isinstance(summary_data, Mapping):
        logger.warning("Discarding malformed summary.json in %s", account_dir)
        summary_data = None

    account_index = account_data.get("account_index")
    if account_index is None:
        try:
            account_index = int(account_dir.name)
        except ValueError:
            logger.warning("Unable to infer account index for %s", account_dir)
            return False

    try:
        account_index = int(account_index)
    except Exception:
        logger.warning("Invalid account_index for %s: %r", account_dir, account_index)
        return False

    account_id = account_data.get("account_id")

    raw_lines = list(account_data.get("lines") or [])
    bureaus = _build_bureaus_payload_from_stagea(account_data)
    flat_fields, _prov = build_rule_fields_from_triad(dict(account_data))

    meta: Dict[str, Any] = {
        "account_index": account_index,
        "heading_guess": account_data.get("heading_guess"),
        "page_start": account_data.get("page_start"),
        "line_start": account_data.get("line_start"),
        "page_end": account_data.get("page_end"),
        "line_end": account_data.get("line_end"),
        "pointers": POINTERS,
    }
    if account_id is not None:
        meta["account_id"] = account_id

    primary_issue, fallback_reasons, fallback_tags = _extract_reason_fields(
        summary_data if isinstance(summary_data, Mapping) else None,
        account_data,
    )

    merge_v2_only = merge_v2_only_enabled()

    summary_obj: Dict[str, Any] = {
        "account_index": account_index,
        "pointers": POINTERS,
        "problem_reasons": fallback_reasons,
        "problem_tags": fallback_tags,
    }
    if not merge_v2_only:
        summary_obj["merge_tag"] = None
    if account_id is not None:
        summary_obj["account_id"] = account_id
    if primary_issue is not None:
        summary_obj["primary_issue"] = primary_issue

    if isinstance(summary_data, Mapping):
        if "problem_reasons" in summary_data:
            summary_obj["problem_reasons"] = _ensure_list(
                summary_data.get("problem_reasons")
            )
        if "problem_tags" in summary_data:
            summary_obj["problem_tags"] = _ensure_list(summary_data.get("problem_tags"))
        if "primary_issue" in summary_data and summary_data.get("primary_issue") is not None:
            summary_obj["primary_issue"] = summary_data.get("primary_issue")
        if not merge_v2_only and "merge_tag" in summary_data:
            summary_obj["merge_tag"] = _sanitize_merge_tag(summary_data.get("merge_tag"))

    summary_obj["problem_tags"] = summary_obj.get("problem_tags") or []
    summary_obj["problem_reasons"] = summary_obj.get("problem_reasons") or []

    if not merge_v2_only:
        summary_obj["merge_tag"] = _sanitize_merge_tag(summary_obj.get("merge_tag"))
    else:
        summary_obj.pop("merge_tag", None)

    tags_path = account_dir / POINTERS["tags"]
    if tags_path.exists():
        tags_data = _load_json(tags_path)
        if not isinstance(tags_data, list):
            tags_data = []
    else:
        tags_data = []

    logger.info("Migrating account folder: %s", account_dir)

    _write_json(account_dir / POINTERS["raw"], raw_lines, dry_run=dry_run)
    _write_json(account_dir / POINTERS["bureaus"], bureaus, dry_run=dry_run)
    _write_json(account_dir / POINTERS["flat"], flat_fields, dry_run=dry_run)
    _write_json(account_dir / "meta.json", meta, dry_run=dry_run)
    _write_json(account_dir / POINTERS["summary"], summary_obj, dry_run=dry_run)
    _write_json(tags_path, tags_data, dry_run=dry_run)

    if not dry_run:
        try:
            account_path.unlink()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to remove legacy account.json %s: %s", account_path, exc)

    return True


def migrate(paths: Sequence[Path | str], dry_run: bool = False) -> Dict[str, int]:
    path_objs = [Path(p) for p in paths]
    account_dirs = _account_dirs_from_paths(path_objs)
    migrated = 0
    processed = 0

    for accounts_dir in account_dirs:
        for account_dir in sorted(p for p in accounts_dir.iterdir() if p.is_dir()):
            processed += 1
            try:
                if migrate_account_dir(account_dir, dry_run=dry_run):
                    migrated += 1
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to migrate %s: %s", account_dir, exc)

    return {"processed": processed, "migrated": migrated}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, cases directories or accounts directories to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the work to be performed without modifying any files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    stats = migrate(args.paths, dry_run=args.dry_run)
    logger.info(
        "Migration complete processed=%d migrated=%d dry_run=%s",
        stats["processed"],
        stats["migrated"],
        args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
