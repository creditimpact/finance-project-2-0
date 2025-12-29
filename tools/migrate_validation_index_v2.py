"""One-time migration helper for validation manifest index files."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

backend_module = sys.modules.get("backend")
backend_path = str(REPO_ROOT / "backend")
if backend_module is None:
    backend_module = ModuleType("backend")
    backend_module.__path__ = [backend_path]
    sys.modules["backend"] = backend_module
else:
    existing_backend_path = list(getattr(backend_module, "__path__", []))
    if backend_path not in existing_backend_path:
        existing_backend_path.append(backend_path)
    backend_module.__path__ = existing_backend_path

validation_pkg = sys.modules.get("backend.validation")
validation_path = str(REPO_ROOT / "backend" / "validation")
if validation_pkg is None:
    validation_pkg = ModuleType("backend.validation")
    validation_pkg.__path__ = [validation_path]
    sys.modules["backend.validation"] = validation_pkg
else:
    existing_validation_path = list(getattr(validation_pkg, "__path__", []))
    if validation_path not in existing_validation_path:
        existing_validation_path.append(validation_path)
    validation_pkg.__path__ = existing_validation_path

index_schema = importlib.import_module("backend.validation.index_schema")
manifest_module = importlib.import_module("backend.validation.manifest")
paths_module = importlib.import_module("backend.core.ai.paths")

ValidationIndex = index_schema.ValidationIndex
load_validation_index = index_schema.load_validation_index
rewrite_index_to_v2 = manifest_module.rewrite_index_to_v2
validation_index_path = paths_module.validation_index_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite legacy validation manifest index.json files to schema v2 and verify packs"
        )
    )
    parser.add_argument(
        "--sid",
        help="Migrate only the manifest for the provided SID",
    )
    parser.add_argument(
        "--root",
        help="Runs root directory to scan (defaults to RUNS_ROOT env or ./runs)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _schema_version_from_document(document: Mapping[str, object]) -> int:
    value = document.get("schema_version")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _load_document(path: Path) -> tuple[str, Mapping[str, object]]:
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if not isinstance(document, Mapping):  # pragma: no cover - defensive
        raise TypeError("Validation index root must be a mapping")
    return text, document


def _iter_index_paths(root: Path) -> Iterable[Path]:
    pattern = ("ai_packs", "validation", "index.json")
    for candidate in root.rglob("index.json"):
        parts = candidate.parts
        if len(parts) >= 3 and parts[-3:] == pattern:
            yield candidate


def _resolve_sid_index_path(sid: str, runs_root: str | Path | None) -> Path:
    index_path = validation_index_path(sid, runs_root=runs_root, create=False)
    if not index_path.exists():
        raise FileNotFoundError(f"Validation index for SID {sid!r} not found at {index_path}")
    return index_path


def _verify_index(index: ValidationIndex) -> tuple[int, int]:
    verified = 0
    missing = 0
    for record in index.packs:
        pack_path = index.resolve_pack_path(record)
        if pack_path.is_file():
            verified += 1
        else:
            missing += 1
            print(
                f"  MISSING pack for account {record.account_id:03d}: {pack_path}",
                file=sys.stderr,
            )
    return verified, missing


def migrate(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.sid:
        runs_root = Path(args.root) if args.root else None
        try:
            index_paths = [_resolve_sid_index_path(args.sid, runs_root=runs_root)]
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        runs_root = Path(args.root) if args.root else Path("runs")
        if not runs_root.exists():
            print(f"Runs root {runs_root} does not exist", file=sys.stderr)
            return 2
        index_paths = sorted(_iter_index_paths(runs_root))

    if not index_paths:
        print("No validation index files found.")
        return 0

    converted_count = 0
    verified_count = 0
    missing_count = 0

    for index_path in index_paths:
        print(f"Processing {index_path}")
        try:
            original_text, document = _load_document(index_path)
        except FileNotFoundError:
            print(f"  Skipped: index file missing", file=sys.stderr)
            continue
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"  Failed to load manifest: {exc}", file=sys.stderr)
            continue

        index: ValidationIndex
        if _schema_version_from_document(document) < 2:
            index, converted = rewrite_index_to_v2(
                index_path,
                document=document,
                original_text=original_text,
                stream=sys.stdout,
            )
            if converted:
                converted_count += 1
        else:
            index = load_validation_index(index_path)

        verified, missing = _verify_index(index)
        verified_count += verified
        missing_count += missing
        print(f"  Packs verified: {verified}, missing: {missing}")

    print(f"converted: {converted_count}, verified: {verified_count}, missing: {missing_count}")
    return 0 if missing_count == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(migrate())
