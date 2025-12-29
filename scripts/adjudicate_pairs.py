"""CLI to rebuild AI packs and adjudicate scorer-selected pairs."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from backend.core.io.tags import read_tags
from backend.core.merge.acctnum import normalize_level
from backend.core.logic.report_analysis.ai_adjudicator import (
    adjudicate_pair,
    persist_ai_decision,
)
from backend.core.logic.report_analysis.ai_pack import build_ai_pack_for_pair
from backend.pipeline.runs import RUNS_ROOT_ENV


logger = logging.getLogger(__name__)


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


def _collect_tags(accounts_root: Path) -> dict[int, list[dict[str, Any]]]:
    tags_by_account: dict[int, list[dict[str, Any]]] = {}
    for entry in sorted(accounts_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            account_idx = int(entry.name)
        except ValueError:
            continue
        tags_path = entry / "tags.json"
        tags_by_account[account_idx] = read_tags(tags_path)
    return tags_by_account


def _collect_ai_pairs(
    tags_by_account: Mapping[int, list[Mapping[str, Any]]]
) -> dict[tuple[int, int], dict[str, Any]]:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for account_idx, tags in tags_by_account.items():
        for tag in tags:
            if tag.get("kind") != "merge_pair":
                continue
            decision = str(tag.get("decision", "")).lower()
            if decision != "ai":
                continue
            partner_idx = _coerce_int(tag.get("with"))
            if partner_idx is None or partner_idx == account_idx:
                continue
            key = (min(account_idx, partner_idx), max(account_idx, partner_idx))
            if key in pairs:
                continue
            pairs[key] = {
                "a": key[0],
                "b": key[1],
                "highlights": _extract_highlights_from_tag(tag),
            }
    return pairs


def _has_merge_result(
    tags_by_account: Mapping[int, list[Mapping[str, Any]]],
    a_idx: int,
    b_idx: int,
) -> bool:
    for account_idx, partner_idx in ((a_idx, b_idx), (b_idx, a_idx)):
        tags = tags_by_account.get(account_idx) or []
        for tag in tags:
            if tag.get("kind") != "merge_result":
                continue
            partner_value = _coerce_int(tag.get("with"))
            if partner_value == partner_idx:
                return True
    return False


def adjudicate_pairs_for_sid(
    sid: str,
    *,
    runs_root: Path,
    only_missing: bool,
) -> None:
    accounts_root = runs_root / sid / "cases" / "accounts"
    if not accounts_root.exists():
        raise FileNotFoundError(f"accounts directory not found: {accounts_root}")

    tags_by_account = _collect_tags(accounts_root)
    pairs = _collect_ai_pairs(tags_by_account)

    start_log = {"sid": sid, "pairs": len(pairs)}
    logger.info("MERGE_V2_MANUAL_START %s", json.dumps(start_log, sort_keys=True))

    if not pairs:
        print(f"No scorer AI pairs found for SID {sid}.")
        return

    processed = 0
    skipped = 0
    for key in sorted(pairs.keys()):
        a_idx = pairs[key]["a"]
        b_idx = pairs[key]["b"]
        highlights = pairs[key]["highlights"]

        if only_missing and _has_merge_result(tags_by_account, a_idx, b_idx):
            skipped += 1
            skip_log = {"sid": sid, "pair": {"a": a_idx, "b": b_idx}, "reason": "merge_result_exists"}
            logger.info("MERGE_V2_MANUAL_SKIP %s", json.dumps(skip_log, sort_keys=True))
            continue

        pack = build_ai_pack_for_pair(
            sid,
            runs_root,
            a_idx,
            b_idx,
            highlights,
        )

        decision = adjudicate_pair(pack)
        persist_ai_decision(sid, runs_root, a_idx, b_idx, decision)

        processed += 1
        summary = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
            "reasons": list(decision.get("reasons", [])),
            "flags": dict(decision.get("flags", {})),
        }
        logger.info("MERGE_V2_MANUAL_DONE %s", json.dumps(summary, sort_keys=True))

        reason = decision.get("reason") or ""
        print(
            f"{sid} pair {a_idx}-{b_idx}: {decision.get('decision')} "
            f"flags={decision.get('flags', {})} reason={reason}"
        )
        reasons = decision.get("reasons") or []
        for reason in reasons:
            print(f"  - {reason}")

    finish_log = {
        "sid": sid,
        "processed": processed,
        "skipped": skipped,
        "total": len(pairs),
    }
    logger.info("MERGE_V2_MANUAL_FINISH %s", json.dumps(finish_log, sort_keys=True))
    print(f"Processed {processed} pair(s); skipped {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjudicate scorer-selected account merge pairs for a SID"
    )
    parser.add_argument("--sid", required=True, help="Case SID to adjudicate")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override runs root directory (defaults to $RUNS_ROOT or ./runs)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip pairs that already have merge_result tags",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runs_root = _resolve_runs_root(args.runs_root)
    try:
        adjudicate_pairs_for_sid(
            args.sid,
            runs_root=runs_root,
            only_missing=bool(args.only_missing),
        )
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        error_log = {"sid": args.sid, "error": exc.__class__.__name__}
        logger.error("MERGE_V2_MANUAL_ERROR %s", json.dumps(error_log, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
