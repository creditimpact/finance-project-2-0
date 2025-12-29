"""CLI utilities for inspecting AI adjudication packs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from backend.core.ai.paths import get_merge_paths, pair_pack_path
from backend.core.io.tags import read_tags
from backend.core.merge.acctnum import normalize_level
from backend.core.logic.report_analysis.ai_pack import build_ai_pack_for_pair
from backend.pipeline.runs import RUNS_ROOT_ENV


DEFAULT_LINES = 5


def _resolve_runs_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env_value = os.getenv(RUNS_ROOT_ENV)
    if env_value:
        return Path(env_value)
    return Path("runs")


def _coerce_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _extract_highlights_from_tag(tag: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tag, Mapping):
        return {}

    aux_value = tag.get("aux")
    aux = aux_value if isinstance(aux_value, Mapping) else {}

    triggers_raw = tag.get("reasons")
    if isinstance(triggers_raw, (list, tuple, set)):
        triggers = [str(item) for item in triggers_raw if item is not None]
    else:
        triggers = []

    conflicts_raw = tag.get("conflicts")
    if isinstance(conflicts_raw, (list, tuple, set)):
        conflicts = [str(item) for item in conflicts_raw if item is not None]
    else:
        conflicts = []

    parts_raw = tag.get("parts")
    parts: dict[str, int] = {}
    if isinstance(parts_raw, Mapping):
        for key, value in parts_raw.items():
            try:
                parts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    matched_raw = aux.get("matched_fields")
    matched: dict[str, bool] = {}
    if isinstance(matched_raw, Mapping):
        for field, flag in matched_raw.items():
            matched[str(field)] = bool(flag)

    try:
        total = int(tag.get("total"))
    except (TypeError, ValueError):
        total = None

    acctnum_level = normalize_level(aux.get("acctnum_level"))

    return {
        "total": total,
        "triggers": triggers,
        "parts": parts,
        "matched_fields": matched,
        "conflicts": conflicts,
        "acctnum_level": acctnum_level,
    }


def _load_merge_pair_tag(tags_path: Path, partner_idx: int) -> Mapping[str, Any]:
    tags = read_tags(tags_path)
    for tag in tags:
        if tag.get("kind") != "merge_pair":
            continue
        partner_value = _coerce_int(tag.get("with"))
        if partner_value == partner_idx:
            return tag
    raise ValueError(
        f"merge_pair tag targeting account {partner_idx} not found in {tags_path}"
    )


def _print_context(label: str, lines: list[str], limit: int) -> None:
    to_show = lines if limit <= 0 else lines[:limit]
    count = len(to_show)
    total = len(lines)
    header = f"Context {label} (showing {count} of {total} lines)"
    print(header)
    for idx, line in enumerate(to_show, 1):
        print(f"  {idx:>2}: {line}")
    if total > count:
        print("  ...")


def preview_pair_pack(
    sid: str,
    a_idx: int,
    b_idx: int,
    *,
    runs_root: Path,
    lines_limit: int,
) -> dict[str, Any]:
    tags_path = runs_root / sid / "cases" / "accounts" / str(a_idx) / "tags.json"
    tag = _load_merge_pair_tag(tags_path, b_idx)
    highlights = _extract_highlights_from_tag(tag)

    pack = build_ai_pack_for_pair(
        sid,
        runs_root,
        a_idx,
        b_idx,
        highlights,
    )

    pair = pack.get("pair", {})
    ids = pack.get("ids", {})
    highlights_payload = pack.get("highlights", {})

    print(f"SID: {sid}")
    print(f"Pair: {pair.get('a')} ↔ {pair.get('b')}")
    print(
        "Account Numbers: A={0} | B={1}".format(
            ids.get("account_number_a", "--"),
            ids.get("account_number_b", "--"),
        )
    )

    merge_paths = get_merge_paths(runs_root, sid, create=True)
    pack_path = pair_pack_path(merge_paths, a_idx, b_idx)
    print(f"Pack path: {pack_path}")

    print("Highlights:")
    print(json.dumps(highlights_payload, ensure_ascii=False, indent=2, sort_keys=True))

    context = pack.get("context", {})
    context_a = list(context.get("a", []))
    context_b = list(context.get("b", []))

    _print_context("A", context_a, lines_limit)
    _print_context("B", context_b, lines_limit)

    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview AI adjudication pack for a pair")
    parser.add_argument("--sid", required=True, help="Case SID to inspect")
    parser.add_argument("--a", type=int, required=True, help="Primary account index")
    parser.add_argument("--b", type=int, required=True, help="Partner account index")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs root directory (defaults to $RUNS_ROOT or ./runs)",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=DEFAULT_LINES,
        help="Maximum context lines per side to display (<=0 to show all)",
    )

    args = parser.parse_args()

    runs_root = _resolve_runs_root(args.runs_root)
    try:
        preview_pair_pack(
            args.sid,
            int(args.a),
            int(args.b),
            runs_root=runs_root,
            lines_limit=int(args.lines),
        )
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        raise SystemExit(f"Failed to preview AI pack: {exc}") from exc


if __name__ == "__main__":
    main()
