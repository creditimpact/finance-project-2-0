from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from backend.core.logic.summary_compact import compact_merge_sections

__all__ = ["main"]


def _iter_summary_paths(root: Path) -> Iterable[Path]:
    """Yield summary.json files under ``root``."""

    if not root.exists():
        return

    yield from (path for path in root.glob("*/summary.json") if path.is_file())


def _compact_summary(path: Path) -> None:
    """Compact the merge sections for ``path`` in place."""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    compact_merge_sections(data)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Compact merge sections for an existing SID."""

    parser = argparse.ArgumentParser(
        description=(
            "Compact merge scoring/explanations across all account summaries for the"
            " provided SID."
        )
    )
    parser.add_argument("sid", help="Run identifier under runs/<SID>/cases/accounts/")

    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path("runs") / args.sid / "cases" / "accounts"
    if not root.exists():
        parser.error(f"no such run directory: {root}")

    count = 0
    for summary_path in sorted(_iter_summary_paths(root)):
        _compact_summary(summary_path)
        count += 1

    print(f"Compacted {count} summary file(s) under {root}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
