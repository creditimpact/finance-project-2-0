"""Sanity checks for validation AI pack runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def load_index(index_path: Path) -> Mapping[str, Any]:
    """Load and return the validation index JSON."""

    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if not isinstance(data, Mapping):
        raise TypeError(f"Unexpected index payload type: {type(data)!r}")

    return data


def count_jsonl_lines(path: Path) -> int:
    """Return the number of non-empty lines in a JSONL file."""

    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def _packs_dir(run_dir: Path) -> Path:
    return run_dir / "ai_packs" / "validation" / "packs"


def _index_path(run_dir: Path) -> Path:
    return run_dir / "ai_packs" / "validation" / "index.json"


def _results_dir(run_dir: Path) -> Path:
    return run_dir / "ai_packs" / "validation" / "results"


def main(run_dir: Path) -> None:
    validation_dir = run_dir / "ai_packs" / "validation"

    index = load_index(_index_path(run_dir))
    packs = index.get("packs")
    if not isinstance(packs, list):
        raise TypeError("index['packs'] must be a list")

    packs_dir = _packs_dir(run_dir)
    actual_pack_files = []
    if packs_dir.exists():
        actual_pack_files = [path for path in packs_dir.rglob("*.jsonl") if path.is_file()]

    print(
        f"packs_in_index={len(packs)} "
        f"actual_pack_files={len(actual_pack_files)}"
    )

    for record in packs:
        if not isinstance(record, Mapping):
            continue
        account_id = record.get("account_id")
        pack_name = record.get("pack")
        if not isinstance(account_id, int):
            continue
        pack_path = validation_dir
        if isinstance(pack_name, str):
            pack_candidate = pack_path / pack_name
            pack_path = pack_candidate if not pack_candidate.is_dir() else pack_candidate / f"acc_{account_id:03d}.jsonl"
        else:
            pack_path = pack_path / "packs" / f"acc_{account_id:03d}.jsonl"
        result_path = _results_dir(run_dir) / f"acc_{account_id:03d}.result.jsonl"

        pack_exists = pack_path.is_file()
        result_exists = result_path.is_file()
        pack_lines = count_jsonl_lines(pack_path) if pack_exists else 0
        result_lines = count_jsonl_lines(result_path) if result_exists else 0
        result_nonempty = result_lines > 0
        lines_match = pack_lines == result_lines and pack_exists and result_exists

        print(
            f"acc={account_id:03d} "
            f"pack_exists={pack_exists} "
            f"pack_lines={pack_lines} "
            f"result_exists={result_exists} "
            f"result_nonempty={result_nonempty} "
            f"result_lines={result_lines} "
            f"lines_match={lines_match}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a completed run directory")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.run_dir)
