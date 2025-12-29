#!/usr/bin/env python3
"""Migrate legacy frontend packs into the review stage layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - best effort helper
        LOGGER.warning("Failed to read JSON from %%s: %%s", path, exc)
        return None


def _looks_like_pack(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    # Packs always include at least one of these core fields.
    return any(key in payload for key in ("display", "holder_name", "primary_issue"))


def _account_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    numeric = int(digits) if digits else 0
    return (numeric, stem)


def _collect_legacy_packs(frontend_dir: Path) -> list[tuple[str, Path, Mapping[str, Any] | None]]:
    legacy: list[tuple[str, Path, Mapping[str, Any] | None]] = []
    if not frontend_dir.is_dir():
        return legacy

    # Root-level packs: runs/<sid>/frontend/<account_id>.json
    for candidate in frontend_dir.glob("*.json"):
        if candidate.name == "index.json":
            continue
        payload = _load_json(candidate)
        if not _looks_like_pack(payload):
            continue
        account_id = _optional_str(payload.get("account_id")) if isinstance(payload, Mapping) else None
        if not account_id:
            account_id = candidate.stem
        legacy.append((account_id, candidate, payload if isinstance(payload, Mapping) else None))

    accounts_dir = frontend_dir / "accounts"
    if accounts_dir.is_dir():
        for account_path in accounts_dir.iterdir():
            if not account_path.is_dir():
                continue
            pack_path = account_path / "pack.json"
            if not pack_path.is_file():
                continue
            payload = _load_json(pack_path)
            if not _looks_like_pack(payload):
                continue
            account_id = _optional_str(payload.get("account_id")) if isinstance(payload, Mapping) else None
            if not account_id:
                account_id = account_path.name
            legacy.append((account_id, pack_path, payload if isinstance(payload, Mapping) else None))

    return legacy


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _sha1(path: Path) -> str | None:
    try:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:  # pragma: no cover - filesystem edge
        LOGGER.warning("Failed to hash %%s: %%s", path, exc)
        return None


def _count_responses(responses_dir: Path) -> int:
    if not responses_dir.is_dir():
        return 0
    return sum(1 for path in responses_dir.iterdir() if path.is_file())


def _collect_questions(payloads: Iterable[Mapping[str, Any]]) -> list[Any]:
    questions: list[Any] = []
    seen = set()
    for payload in payloads:
        raw_questions = payload.get("questions")
        if isinstance(raw_questions, Sequence) and not isinstance(raw_questions, (str, bytes, bytearray)):
            for question in raw_questions:
                marker = json.dumps(question, sort_keys=True) if isinstance(question, Mapping) else repr(question)
                if marker in seen:
                    continue
                seen.add(marker)
                questions.append(question)
    return questions


def _build_manifest(
    run_dir: Path,
    packs_dir: Path,
    responses_dir: Path,
    *,
    sid: str,
) -> dict[str, Any]:
    pack_payloads: list[tuple[Path, Mapping[str, Any]]] = []
    if packs_dir.is_dir():
        for pack_path in sorted(packs_dir.glob("*.json"), key=_account_sort_key):
            payload = _load_json(pack_path)
            if not isinstance(payload, Mapping):
                continue
            pack_payloads.append((pack_path, payload))

    pack_entries: list[dict[str, Any]] = []
    for pack_path, payload in pack_payloads:
        account_id = _optional_str(payload.get("account_id")) or pack_path.stem
        holder_name = _optional_str(payload.get("holder_name"))
        primary_issue = _optional_str(payload.get("primary_issue"))
        questions = payload.get("questions")
        has_questions = False
        if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes, bytearray)):
            has_questions = len(questions) > 0
        display_payload = payload.get("display") if isinstance(payload, Mapping) else None
        entry: dict[str, Any] = {
            "account_id": account_id,
            "holder_name": holder_name,
            "primary_issue": primary_issue,
            "path": str(pack_path.relative_to(run_dir)),
            "pack_path": str(pack_path.relative_to(run_dir)),
            "bytes": pack_path.stat().st_size,
            "has_questions": has_questions,
        }
        if isinstance(display_payload, Mapping):
            entry["display"] = display_payload
        sha1_digest = _sha1(pack_path)
        if sha1_digest:
            entry["sha1"] = sha1_digest
        pack_entries.append(entry)

    responses_count = _count_responses(responses_dir)
    responses_rel = str(responses_dir.relative_to(run_dir))
    question_bank = _collect_questions(payload for _, payload in pack_payloads)

    manifest_core: dict[str, Any] = {
        "sid": sid,
        "stage": "review",
        "schema_version": "1.0",
        "counts": {"packs": len(pack_entries), "responses": responses_count},
        "packs": pack_entries,
        "responses_dir": responses_rel,
        "packs_count": len(pack_entries),
        "questions": question_bank,
    }

    manifest_path = run_dir / "frontend" / "review" / "index.json"
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    existing = _load_json(manifest_path)
    if isinstance(existing, Mapping):
        previous = dict(existing)
        previous_generated = previous.pop("generated_at", None)
        if previous == manifest_core and isinstance(previous_generated, str):
            generated_at = previous_generated

    payload = {**manifest_core, "generated_at": generated_at}
    _ensure_dirs(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def migrate_run(run_dir: Path) -> dict[str, Any] | None:
    frontend_dir = run_dir / "frontend"
    if not frontend_dir.is_dir():
        LOGGER.info("Skipping %s (no frontend directory)", run_dir)
        return None

    review_dir = frontend_dir / "review"
    packs_dir = review_dir / "packs"
    responses_dir = review_dir / "responses"
    _ensure_dirs(packs_dir, responses_dir)

    legacy_packs = _collect_legacy_packs(frontend_dir)
    if legacy_packs:
        LOGGER.info("Migrating %d pack(s) in %s", len(legacy_packs), run_dir)
    for account_id, source_path, _payload in legacy_packs:
        target_path = packs_dir / f"{account_id}.json"
        if target_path.exists():
            LOGGER.warning(
                "Target pack already exists for %s in %s; leaving legacy file in place",
                account_id,
                run_dir,
            )
            continue
        _ensure_dirs(target_path.parent)
        LOGGER.info("Moving %s -> %s", source_path.relative_to(run_dir), target_path.relative_to(run_dir))
        target_path.write_bytes(source_path.read_bytes())
        source_path.unlink()

        legacy_parent = source_path.parent
        if legacy_parent.is_dir() and legacy_parent != frontend_dir and not any(legacy_parent.iterdir()):
            legacy_parent.rmdir()

    sid = run_dir.name
    manifest = _build_manifest(run_dir, packs_dir, responses_dir, sid=sid)
    LOGGER.info(
        "Updated manifest for %s with %d pack(s)",
        sid,
        manifest.get("packs_count", 0),
    )
    return manifest


def _iter_run_dirs(targets: Sequence[str], runs_root: Path, all_runs: bool) -> list[Path]:
    run_dirs: list[Path] = []
    if all_runs:
        for path in sorted(runs_root.iterdir()):
            if path.is_dir():
                run_dirs.append(path)
        return run_dirs

    if not targets:
        raise SystemExit("No runs specified")

    for target in targets:
        candidate = Path(target)
        if candidate.is_dir():
            run_dirs.append(candidate)
            continue
        run_dir = runs_root / target
        if run_dir.is_dir():
            run_dirs.append(run_dir)
            continue
        raise SystemExit(f"Run directory not found: {target}")
    return run_dirs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run directories or IDs to migrate. When providing IDs, --runs-root controls the base path.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Base directory containing run folders (default: runs)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate every run directory under --runs-root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(message)s")

    run_dirs = _iter_run_dirs(args.runs, args.runs_root, args.all)
    if not run_dirs:
        LOGGER.info("No runs to migrate")
        return

    for run_dir in run_dirs:
        migrate_run(run_dir)


if __name__ == "__main__":
    main()
