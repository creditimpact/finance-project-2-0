"""Wrappers around merge scoring helpers for programmatic use."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping as TypingMapping

from backend import config as app_config
from backend.core.merge import acctnum
from scripts.score_bureau_pairs import (
    ScoreComputationResult,
    build_merge_tags,
    build_pair_rows,
    choose_best_partner_cached,
    compute_scores_for_sid,
    persist_merge_tags_to_tags,
)

_DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))

logger = logging.getLogger(__name__)

SCORER_WEIGHTS = {
    "acctnum_exact": app_config.ACCTNUM_EXACT_WEIGHT,
    "masked": app_config.ACCTNUM_MASKED_WEIGHT,
}

def _ensure_tag_levels(
    merge_tags: Mapping[int, Mapping[str, Any]] | None,
) -> Dict[int, Dict[str, Any]]:
    enriched: Dict[int, Dict[str, Any]] = {}
    if not isinstance(merge_tags, Mapping):
        return enriched

    for idx, tag in merge_tags.items():
        if not isinstance(tag, Mapping):
            continue
        tag_dict = dict(tag)
        acct_level = tag_dict.get("acctnum_level")
        if not isinstance(acct_level, str) or not acct_level:
            aux_payload = tag_dict.get("aux")
            if isinstance(aux_payload, Mapping):
                acct_level = str(aux_payload.get("acctnum_level") or "none")
            else:
                acct_level = "none"
        tag_dict["acctnum_level"] = acct_level
        try:
            key = int(idx)
        except (TypeError, ValueError):
            continue
        enriched[key] = tag_dict

    return enriched


def _load_bureau_payload(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("MERGE_V2_ACCT_LOAD_FAILED path=%s", path)
        return {}

    if not isinstance(data, Mapping):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for bureau in ("transunion", "experian", "equifax"):
        branch = data.get(bureau)
        if isinstance(branch, Mapping):
            result[bureau] = dict(branch)
        else:
            result[bureau] = {}
    return result


def _normalize_account_numbers(
    payload: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, acctnum.NormalizedAccountNumber]:
    normalized: Dict[str, acctnum.NormalizedAccountNumber] = {}
    if not isinstance(payload, Mapping):
        payload = {}
    for bureau in ("transunion", "experian", "equifax"):
        display = ""
        branch = payload.get(bureau)
        if isinstance(branch, Mapping):
            raw_display = branch.get("account_number_display")
            display = "" if raw_display is None else str(raw_display)
        normalized[bureau] = acctnum.normalize_display(display)
    return normalized


def _update_result_with_match(
    result: Dict[str, Any],
    match: acctnum.AccountNumberMatch,
) -> None:
    parts = dict(result.get("parts") or {})
    old_points = int(parts.get("account_number", 0) or 0)
    new_points = 1 if match.level == "exact_or_known_match" else 0
    parts["account_number"] = new_points
    result["parts"] = parts

    diff = new_points - old_points
    for key in ("identity_score", "identity_sum", "total"):
        try:
            current = int(result.get(key, 0) or 0)
        except (TypeError, ValueError):
            current = 0
        result[key] = current + diff

    highlights = dict(result.get("highlights") or {})
    highlights["acctnum_level"] = match.level
    result["highlights"] = highlights

    matched_fields = dict(result.get("matched_fields") or {})
    matched = match.level == "exact_or_known_match"
    matched_fields["account_number"] = matched
    result["matched_fields"] = matched_fields

    matched_pairs = dict(result.get("matched_pairs") or {})
    if match.a_bureau and match.b_bureau:
        matched_pairs["account_number"] = [match.a_bureau, match.b_bureau]
    elif "account_number" not in matched_pairs:
        matched_pairs["account_number"] = []
    result["matched_pairs"] = matched_pairs

    aux = dict(result.get("aux") or {})
    acct_aux = dict(aux.get("account_number") or {})
    acct_aux["acctnum_level"] = match.level
    acct_aux["matched"] = matched
    acct_aux["best_pair"] = [match.a_bureau, match.b_bureau] if matched else []
    acct_aux["raw_values"] = {"a": match.a.raw, "b": match.b.raw}
    acct_aux["acctnum_debug"] = {
        "a": match.a.to_debug_dict(),
        "b": match.b.to_debug_dict(),
        "visible_match": dict(match.debug),
    }
    acct_aux["acctnum_digits_len_a"] = len(match.a.digits)
    acct_aux["acctnum_digits_len_b"] = len(match.b.digits)
    aux["account_number"] = acct_aux
    result["aux"] = aux


def _apply_account_number_scoring(
    sid: str,
    indices: Iterable[int],
    scores_by_idx: TypingMapping[int, TypingMapping[int, Dict[str, Any]]],
    normalized_by_idx: Mapping[int, Dict[str, acctnum.NormalizedAccountNumber]],
) -> None:
    sorted_indices = list(sorted(indices))
    for pos, i in enumerate(sorted_indices):
        left_scores = scores_by_idx.get(i)
        if not isinstance(left_scores, Mapping):
            continue
        a_norm = normalized_by_idx.get(i, {})
        for j in sorted_indices[pos + 1 :]:
            result = left_scores.get(j)
            if not isinstance(result, dict):
                continue
            b_norm = normalized_by_idx.get(j, {})
            match = acctnum.best_account_number_match(a_norm, b_norm)
            _update_result_with_match(result, match)
            right_scores = scores_by_idx.get(j)
            if isinstance(right_scores, Mapping):
                reverse = right_scores.get(i)
                if isinstance(reverse, dict):
                    _update_result_with_match(reverse, match.swapped())
            logger.info(
                "MERGE_V2_ACCT_BEST sid=%s i=%s j=%s level=%s a_digits=%s b_digits=%s",
                sid,
                i,
                j,
                match.level,
                match.debug.get("a", {}).get("digits") if isinstance(match.debug.get("a"), Mapping) else "",
                match.debug.get("b", {}).get("digits") if isinstance(match.debug.get("b"), Mapping) else "",
            )
            logger.info(
                "MERGE_V2_ACCTNUM_MATCH sid=%s i=%s j=%s level=%s short=%s long=%s why=%s",
                sid,
                i,
                j,
                match.level,
                str(match.debug.get("short", "")),
                str(match.debug.get("long", "")),
                str(match.debug.get("why", "")),
            )


def score_bureau_pairs_cli(
    *, sid: str, write_tags: bool = False, runs_root: Path | str | None = None
) -> ScoreComputationResult:
    """Execute the score-bureau-pairs workflow similar to the CLI."""

    sid_str = str(sid)
    logger.info(
        "SCORER_WEIGHTS acctnum_exact=%s masked=%s",
        SCORER_WEIGHTS["acctnum_exact"],
        SCORER_WEIGHTS["masked"],
    )
    runs_root_path = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT

    indices, scores_by_idx = compute_scores_for_sid(sid_str, runs_root=runs_root_path)

    if indices:
        normalized_by_idx: Dict[int, Dict[str, acctnum.NormalizedAccountNumber]] = {}
        for idx in indices:
            path = (
                runs_root_path
                / sid_str
                / "cases"
                / "accounts"
                / str(idx)
                / "bureaus.json"
            )
            payload = _load_bureau_payload(path)
            normalized_by_idx[idx] = _normalize_account_numbers(payload)
        _apply_account_number_scoring(sid_str, indices, scores_by_idx, normalized_by_idx)
        best_by_idx = choose_best_partner_cached(scores_by_idx)
        rows = build_pair_rows(scores_by_idx)
        merge_tags = build_merge_tags(scores_by_idx, best_by_idx)
        if write_tags:
            merge_tags = persist_merge_tags_to_tags(
                sid_str, scores_by_idx, best_by_idx, runs_root=runs_root_path
            )
    else:
        best_by_idx = {}
        rows = []
        merge_tags = {}

    computation = ScoreComputationResult(
        sid=sid_str,
        runs_root=runs_root_path,
        indices=indices,
        scores_by_idx=scores_by_idx,
        best_by_idx=best_by_idx,
        merge_tags=merge_tags,
        rows=rows,
    )

    enriched_tags = _ensure_tag_levels(computation.merge_tags)
    if enriched_tags:
        computation = replace(computation, merge_tags=enriched_tags)
    return computation


__all__ = ["score_bureau_pairs_cli"]
