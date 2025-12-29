"""Build AI adjudication packs for merge V2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - convenience bootstrap
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.ai.paths import (
    MergePaths,
    ensure_merge_paths,
    merge_paths_from_any,
    pair_pack_filename,
    pair_pack_path,
)
from backend.core.logic.report_analysis.account_merge import (
    coerce_score_value,
)
from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs
from backend.pipeline.runs import RunManifest, persist_manifest


log = logging.getLogger(__name__)


def _write_pack(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _write_index(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def _merge_paths_with_override(base: MergePaths, override: Path | None) -> MergePaths:
    if override is None:
        return base

    override_paths = merge_paths_from_any(override, create=True)
    return override_paths


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--packs-dir",
        help="Destination directory for generated AI packs (defaults to runs/<SID>/ai_packs)",
    )
    parser.add_argument(
        "--max-lines-per-side",
        type=int,
        default=20,
        help="Maximum number of context lines per account",
    )
    parser.add_argument(
        "--only-merge-best",
        dest="only_merge_best",
        action="store_true",
        help="Include only pairs marked as merge_best (default)",
    )
    parser.add_argument(
        "--include-all-pairs",
        dest="only_merge_best",
        action="store_false",
        help="Include all AI pairs regardless of merge_best",
    )
    parser.set_defaults(only_merge_best=True)

    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)
    canonical_paths = ensure_merge_paths(runs_root, sid, create=True)
    override = Path(args.packs_dir) if args.packs_dir else None
    merge_paths = _merge_paths_with_override(canonical_paths, override)

    packs_dir = merge_paths.packs_dir
    base_dir = merge_paths.base
    index_path = merge_paths.index_file

    log.info("PACKS_DIR_USED sid=%s dir=%s", sid, packs_dir)
    log.debug("MERGE_RESULTS_DIR sid=%s dir=%s", sid, merge_paths.results_dir)

    manifest = RunManifest.for_sid(sid, allow_create=True)
    manifest.upsert_ai_packs_dir(base_dir)
    persist_manifest(manifest)
    log.info("MANIFEST_AI_PACKS_DIR_SET sid=%s dir=%s", sid, base_dir)

    packs = build_merge_ai_packs(
        sid,
        runs_root,
        only_merge_best=bool(args.only_merge_best),
        max_lines_per_side=int(args.max_lines_per_side),
    )

    existing_totals: dict[str, Any] = {}
    if index_path.exists():
        try:
            existing_index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            existing_index_payload = None
        if isinstance(existing_index_payload, Mapping):
            totals_candidate = existing_index_payload.get("totals")
            if isinstance(totals_candidate, Mapping):
                existing_totals = dict(totals_candidate)

    index_entries: list[dict[str, object]] = []
    for pack in packs:
        pair = pack.get("pair") or {}
        try:
            a_idx = int(pair.get("a"))
            b_idx = int(pair.get("b"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Pack is missing pair indices") from exc

        pack_filename = pair_pack_filename(a_idx, b_idx)
        pack_path = pair_pack_path(merge_paths, a_idx, b_idx)

        points_mode_active = bool(pack.get("points_mode"))
        score_total = coerce_score_value(
            pack.get("score_total"), points_mode=points_mode_active
        )

        # Pack payload should remain minimal; strip helper metadata before persisting.
        pack_to_write = dict(pack)
        pack_to_write.pop("points_mode", None)
        pack_to_write.pop("score_total", None)

        _write_pack(pack_path, pack_to_write)
        log.info("PACK_WRITTEN sid=%s file=%s a=%s b=%s", sid, pack_filename, a_idx, b_idx)

        index_entries.append(
            {
                "a": a_idx,
                "b": b_idx,
                "pair": [a_idx, b_idx],
                "pack_file": pack_filename,
                "lines_a": 0,
                "lines_b": 0,
                "score_total": score_total,
                "score": score_total,
            }
        )

    pairs_count = len(index_entries)
    if pairs_count > 0:
        if args.packs_dir:
            index_path = base_dir / merge_paths.index_file.name
        seen_pairs: set[tuple[int, int]] = set()
        pairs_payload: list[dict[str, object]] = []
        for entry in index_entries:
            a_idx = int(entry["a"])
            b_idx = int(entry["b"])
            score_value = entry.get("score", entry.get("score_total", 0))
            pack_file = entry.get("pack_file")
            for pair in ((a_idx, b_idx), (b_idx, a_idx)):
                if pair in seen_pairs:
                    continue
                pair_entry: dict[str, object] = {
                    "pair": [pair[0], pair[1]],
                    "score": score_value,
                }
                if pack_file:
                    pair_entry["pack_file"] = pack_file
                pairs_payload.append(pair_entry)
                seen_pairs.add(pair)

        totals_payload: dict[str, Any] = dict(existing_totals)
        totals_payload["packs_built"] = pairs_count
        totals_payload["created_packs"] = pairs_count
        if "merge_zero_packs" in totals_payload:
            totals_payload["merge_zero_packs"] = False
        if "scored_pairs" not in totals_payload:
            totals_payload["scored_pairs"] = pairs_count

        index_payload = {
            "sid": sid,
            "packs": index_entries,
            "pairs": pairs_payload,
            "pairs_count": pairs_count,
            "totals": totals_payload,
        }
        _write_index(index_path, index_payload)
        packs_in_index = len(index_payload.get("packs", []))
        log.info("INDEX_WRITTEN sid=%s index=%s pairs=%d", sid, index_path, packs_in_index)

        manifest = RunManifest.for_sid(sid).set_ai_built(base_dir, pairs_count)
        persist_manifest(manifest)
        log.info(
            "MANIFEST_AI_PACKS_UPDATED sid=%s dir=%s index=%s pairs=%d",
            sid,
            base_dir,
            index_path,
            pairs_count,
        )
    else:
        log.info("INDEX_SKIPPED_NO_PAIRS sid=%s dir=%s", sid, packs_dir)

    print(f"[BUILD] wrote {len(index_entries)} packs to {packs_dir}")

    try:
        from backend.ai.merge.sender import trigger_autosend_after_build
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "MERGE_AUTOSEND_TRIGGER_IMPORT_FAILED sid=%s", sid, exc_info=True
        )
    else:
        try:
            trigger_autosend_after_build(
                sid,
                runs_root=runs_root,
                created=pairs_count,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "MERGE_AUTOSEND_TRIGGER_FAILED sid=%s", sid, exc_info=True
            )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
