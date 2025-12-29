"""Utilities for migrating legacy note_style .result files to JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


LEGACY_FAILURE_PAYLOAD = {
    "status": "failed",
    "error": "legacy_result_text",
}


def _iter_result_files(root: Path) -> Iterable[Path]:
    """Yield legacy ``*.result`` files below ``root``."""

    pattern = "runs/*/ai_packs/note_style/results/*.result"
    yield from root.glob(pattern)


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def migrate_file(path: Path) -> Path | None:
    """Migrate a single ``.result`` file to ``.jsonl``.

    Returns the path of the created JSONL file, or ``None`` if no work was
    performed (e.g. because the JSONL file already exists).
    """

    target = path.with_suffix(".jsonl")
    if target.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()

    json_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Skip empty lines entirely – they never carried information.
            continue

        if _is_json(stripped):
            json_lines.append(stripped)
        else:
            payload = dict(LEGACY_FAILURE_PAYLOAD)
            payload["raw"] = line
            json_lines.append(json.dumps(payload, ensure_ascii=False))

    if not json_lines:
        # Do not create empty files – but keep idempotency by creating an
        # empty file so we do not try again.
        target.write_text("", encoding="utf-8")
    else:
        target.write_text("\n".join(json_lines) + "\n", encoding="utf-8")

    return target


def migrate_legacy_note_style_results(root: Path | None = None) -> List[Path]:
    """Migrate all legacy ``.result`` files under ``root`` (default: cwd)."""

    base = root or Path.cwd()
    migrated: List[Path] = []

    for result_file in _iter_result_files(base):
        created = migrate_file(result_file)
        if created is not None:
            migrated.append(created)

    return migrated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Root directory that contains the `runs/` folder (default: cwd)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    created = migrate_legacy_note_style_results(args.root)

    if created:
        print(f"Migrated {len(created)} file(s):")
        for path in created:
            print(f" - {path}")
    else:
        print("No legacy .result files required migration.")


if __name__ == "__main__":
    main()

