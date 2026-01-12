"""Lean problem case builder for Stage-A accounts.

This module materialises a compact set of JSON artefacts for each
problematic account detected by the analyzer.  Only the fields required by
operators are persisted which keeps case folders small and guarantees that
``triad_rows`` never land on disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from backend.pipeline.runs import RunManifest, write_breadcrumb
from backend.core.io.tags import read_tags, upsert_tag, write_tags_atomic
from backend.core.merge.acctnum import normalize_level
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.logic.report_analysis.problem_extractor import (
    build_rule_fields_from_triad,
    load_stagea_accounts_from_manifest,
)
from backend.config import STAGEA_ORIGCRED_POST_EXTRACT, HISTORY_MAIN_WIRING_ENABLED

from backend.core.logic.report_analysis.account_merge import (
    choose_best_partner,
    build_merge_best_tag,
    build_merge_pair_tag,
    gen_unordered_pairs,
    merge_v2_only_enabled,
    score_all_pairs_0_100,
)

from .keys import compute_logical_account_key

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[4]))

LEAN = os.getenv("CASES_LEAN_MODE", "1") != "0"

ALLOWED_BUREAUS_TOPLEVEL = ("transunion", "experian", "equifax")


def _candidate_manifest_paths(sid: str, root: Path | None = None) -> List[Path]:
    candidates: List[Path] = []

    env_manifest = os.getenv("REPORT_MANIFEST_PATH")
    if env_manifest:
        candidates.append(Path(env_manifest))

    runs_root_env = os.getenv("RUNS_ROOT")
    if runs_root_env:
        candidates.append(Path(runs_root_env) / sid / "manifest.json")

    if root is not None:
        candidates.append(Path(root) / "runs" / sid / "manifest.json")

    candidates.append(PROJECT_ROOT / "runs" / sid / "manifest.json")

    unique: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except RuntimeError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _load_manifest_for_sid(sid: str, root: Path | None = None) -> RunManifest | None:
    for path in _candidate_manifest_paths(sid, root=root):
        if path.exists():
            try:
                return RunManifest(path).load()
            except Exception:
                logger.debug(
                    "CASE_BUILDER manifest_load_failed sid=%s path=%s", sid, path
                )
    return None


def _write_json(path: Path, obj: Any) -> None:
    payload = obj
    if (
        path.name == "summary.json"
        and isinstance(obj, Mapping)
        and os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1"
    ):
        payload = dict(obj)
        compact_merge_sections(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_summary_json(path: Path) -> Tuple[Dict[str, Any], bool]:
    if not path.exists():
        return {}, False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False

    if isinstance(data, dict):
        return data, True
    return {}, False


def _coerce_list(value: Any) -> List[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _sanitize_bureau_fields(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {key: value for key, value in payload.items() if key != "triad_rows"}
    if payload is None:
        return {}
    return payload


def _sanitize_bureaus(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for bureau, payload in (data or {}).items():
        cleaned[bureau] = _sanitize_bureau_fields(payload)
    return cleaned


_MONTHLY_V2_REMOVED_KEYS = {
    "derived_month_num",
    "derived_year",
    "month_label_normalized",
    "year_token_raw",
}


def _clone_json_safe(obj: Any) -> Any:
    """Best-effort deep clone that preserves JSON-serializable structure."""

    try:
        return json.loads(json.dumps(obj, ensure_ascii=False))
    except Exception:
        # Fallback to shallow copy when serialization fails; caller only uses clone for filtering
        if isinstance(obj, Mapping):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def _filter_monthly_v2_for_export(bureaus_payload: Any) -> Any:
    """Return a copy of bureaus payload with monthly_v2 derived keys stripped.

    This is export-only: the input object is not mutated. Only entries under
    two_year_payment_history_monthly_tsv_v2 are touched; everything else is
    preserved verbatim.
    """

    if not isinstance(bureaus_payload, Mapping):
        return bureaus_payload

    filtered = _clone_json_safe(bureaus_payload)

    monthly_block = filtered.get("two_year_payment_history_monthly_tsv_v2")
    if not isinstance(monthly_block, Mapping):
        return filtered

    for bureau, entries in monthly_block.items():
        if not isinstance(entries, list):
            continue
        cleaned_entries = []
        for entry in entries:
            if isinstance(entry, Mapping):
                cleaned_entries.append(
                    {k: v for k, v in entry.items() if k not in _MONTHLY_V2_REMOVED_KEYS}
                )
            else:
                cleaned_entries.append(entry)
        monthly_block[bureau] = cleaned_entries

    return filtered


_ORIGCRED_DASH_TOKENS = {"--", "—", "–"}
_RAW_ORIGCRED_PATTERN = re.compile(
    r"original\s+creditor(?:\s+\d+)?(?:\s*[:：]\s*|\s+)(?P<value>.+?)\s+(?:--|—|–)\s+(?:--|—|–)\s*$",
    re.IGNORECASE,
)


def _iter_raw_line_texts(raw_lines: Iterable[Any]) -> Iterable[str]:
    for entry in raw_lines or []:
        text: Any | None = None
        if isinstance(entry, Mapping):
            text = entry.get("text")
        elif isinstance(entry, (str, bytes)):
            text = entry
        elif entry is not None:
            text = str(entry)
        if text is None:
            continue
        try:
            text_str = str(text)
        except Exception:
            continue
        if text_str.strip():
            yield text_str


def try_scan_raw_triads_for_original_creditor(raw_lines: Iterable[Any]) -> str | None:
    """Best-effort extraction of TU original creditor from raw triad rows."""

    for raw_text in _iter_raw_line_texts(raw_lines):
        normalized = " ".join(raw_text.replace("：", ":").split())
        match = _RAW_ORIGCRED_PATTERN.search(normalized)
        if not match:
            continue
        candidate = match.group("value").strip()
        if candidate and candidate not in _ORIGCRED_DASH_TOKENS:
            return candidate
    return None


def _maybe_backfill_original_creditor(
    triad_fields: Any, raw_lines: Iterable[Any]
) -> None:
    if not STAGEA_ORIGCRED_POST_EXTRACT:
        return
    if not isinstance(triad_fields, MutableMapping):
        return
    tu_fields = triad_fields.get("transunion")
    if not isinstance(tu_fields, MutableMapping):
        return

    existing = tu_fields.get("original_creditor")
    existing_str = ""
    if existing is not None:
        existing_str = str(existing).strip()

    if existing_str and existing_str not in _ORIGCRED_DASH_TOKENS:
        return

    fallback = try_scan_raw_triads_for_original_creditor(raw_lines)
    if not fallback:
        return

    tu_fields["original_creditor"] = fallback
    logger.info("TRIAD_ORIGCRED_POST_EXTRACT_FILLED value=%r", fallback)


def _normalize_str_list(values: Any) -> List[str]:
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [str(item) for item in values if item is not None]
    return []


def _build_issue_tag(summary: Mapping[str, Any]) -> Dict[str, Any] | None:
    issue_value = summary.get("primary_issue")
    if not isinstance(issue_value, str) or not issue_value.strip():
        return None

    tag: Dict[str, Any] = {
        "kind": "issue",
        "type": issue_value.strip(),
    }

    reasons = summary.get("problem_reasons")
    normalized_reasons = _normalize_str_list(reasons)
    if normalized_reasons:
        tag["details"] = {"problem_reasons": normalized_reasons}

    return tag


def _tag_sort_key(entry: Mapping[str, Any]) -> Tuple[str, int, str]:
    kind = str(entry.get("kind", ""))
    partner = entry.get("with")
    if isinstance(partner, int):
        partner_key = partner
    else:
        partner_key = -1
    tag_name = str(entry.get("tag", ""))
    return (kind, partner_key, tag_name)


def _build_bureaus_payload_from_stagea(
    acc: Mapping[str, Any] | None,
) -> OrderedDict[str, Any]:
    if not isinstance(acc, Mapping):
        acc = {}

    sanitized_bureaus = _sanitize_bureaus(acc.get("triad_fields"))

    ordered: "OrderedDict[str, Any]" = OrderedDict()
    for bureau in ALLOWED_BUREAUS_TOPLEVEL:
        value = sanitized_bureaus.get(bureau)
        if value is None:
            ordered[bureau] = {}
        else:
            ordered[bureau] = value

    seven_year = acc.get("seven_year_history") or {}

    ordered["seven_year_history"] = seven_year
    
    # Prefer TSV v2 monthly status pairs; fallback to legacy for backward compatibility
    tsv_v2_monthly = acc.get("two_year_payment_history_monthly_tsv_v2")
    if isinstance(tsv_v2_monthly, dict) and tsv_v2_monthly is not None:
        ordered["two_year_payment_history_monthly_tsv_v2"] = tsv_v2_monthly
        logger.info("CASE_BUILDER_TSV_V2_MONTHLY_INCLUDED tu=%d xp=%d eq=%d",
                   len(tsv_v2_monthly.get("transunion", [])),
                   len(tsv_v2_monthly.get("experian", [])),
                   len(tsv_v2_monthly.get("equifax", [])))
    else:
        # Fallback to legacy two_year_payment_history when monthly_v2 is absent
        two_year = acc.get("two_year_payment_history") or {}
        if two_year:
            ordered["two_year_payment_history"] = two_year
    
    ordered["order"] = list(ALLOWED_BUREAUS_TOPLEVEL)

    return ordered


def _extract_candidate_reason(cand: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    reason = cand.get("reason") if isinstance(cand, Mapping) else None

    primary_issue = None
    problem_reasons = None
    problem_tags = None

    if isinstance(reason, Mapping):
        primary_issue = reason.get("primary_issue")
        problem_reasons = reason.get("problem_reasons")
        problem_tags = reason.get("problem_tags")

    if primary_issue is None:
        primary_issue = cand.get("primary_issue")
    if problem_reasons is None:
        problem_reasons = cand.get("problem_reasons")
    if problem_tags is None:
        problem_tags = cand.get("problem_tags")

    return primary_issue, problem_reasons, problem_tags


# ---------------------------------------------------------------------------
# Legacy helpers
# ---------------------------------------------------------------------------


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
    fields = account.get("fields") or {}

    issuer = (
        fields.get("issuer")
        or fields.get("creditor")
        or fields.get("name")
        or account.get("issuer")
        or account.get("creditor")
        or account.get("name")
    )
    last4 = (
        fields.get("account_last4")
        or fields.get("last4")
        or account.get("account_last4")
        or account.get("last4")
    )
    account_type = (
        fields.get("account_type")
        or fields.get("type")
        or account.get("account_type")
        or account.get("type")
    )
    opened_date = (
        fields.get("opened_date")
        or fields.get("date_opened")
        or account.get("opened_date")
        or account.get("date_opened")
    )

    logical_key = compute_logical_account_key(issuer, last4, account_type, opened_date)
    acc_id = logical_key or f"idx-{idx:03d}"
    return re.sub(r"[^a-z0-9_-]", "_", str(acc_id).lower())


def _load_accounts(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, Mapping):
        accounts = data.get("accounts") or []
    elif isinstance(data, list):
        accounts = data
    else:
        accounts = []

    out: List[Mapping[str, Any]] = []
    for acc in accounts:
        if isinstance(acc, Mapping):
            out.append(acc)
    return out


def _build_account_lookup(
    accounts: Iterable[Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    by_key: Dict[str, Mapping[str, Any]] = {}
    for idx, acc in enumerate(accounts, start=1):
        account_index = acc.get("account_index")
        if isinstance(account_index, int):
            by_key[str(account_index)] = acc

        acc_id = _make_account_id(acc, idx)
        by_key[acc_id] = acc

    return by_key


def _resolve_inputs_from_manifest(
    sid: str, *, root: Path | None = None
) -> tuple[Path, Path, RunManifest]:
    m = _load_manifest_for_sid(sid, root=root)
    if m is None:
        raise RuntimeError(f"Run manifest not found for sid={sid}")
    try:
        accounts_path = Path(m.get("traces.accounts_table", "accounts_json")).resolve()
        general_path = Path(m.get("traces.accounts_table", "general_json")).resolve()
    except KeyError as e:
        raise RuntimeError(
            f"Run manifest missing traces.accounts_table key for sid={sid}: {e}"
        )
    return accounts_path, general_path, m


# ---------------------------------------------------------------------------
# Lean writer implementation
# ---------------------------------------------------------------------------


def _build_problem_cases_lean(
    sid: str, candidates: List[Dict[str, Any]], *, root: Path | None = None
) -> Dict[str, Any]:
    """Build lean per-account case folders under ``runs/<sid>/cases``."""

    try:
        full_accounts = load_stagea_accounts_from_manifest(sid, root=root)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Stage-A accounts for sid={sid}") from exc

    accounts_by_index: Dict[int, Dict[str, Any]] = {}
    for account in full_accounts:
        try:
            idx = int(account.get("account_index"))
        except Exception:
            continue
        accounts_by_index[idx] = account

    total = len(full_accounts)

    manifest = _load_manifest_for_sid(sid, root=root)
    if manifest is not None:
        cases_dir = manifest.ensure_run_subdir("cases_dir", "cases")
        accounts_dir = (cases_dir / "accounts").resolve()
        accounts_dir.mkdir(parents=True, exist_ok=True)
        manifest.set_base_dir("cases_accounts_dir", accounts_dir)
        write_breadcrumb(manifest.path, cases_dir / ".manifest")
        runs_root_path = cases_dir.parent.parent.resolve()
    else:
        base_root = Path(root) if root is not None else PROJECT_ROOT
        runs_root_path = (base_root / "runs").resolve()
        cases_dir = (runs_root_path / sid / "cases").resolve()
        accounts_dir = cases_dir / "accounts"
        accounts_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cases_dir / ".manifest"
        manifest_path.write_text("missing", encoding="utf-8")

    logger.info("PROBLEM_CASES start sid=%s total=%d out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}
    tag_paths: Dict[int, Path] = {}
    written_indices: List[int] = []
    merge_v2_only = merge_v2_only_enabled()

    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue

        idx_val = cand.get("account_index")
        try:
            idx = int(idx_val)
        except Exception:
            logger.warning(
                "CASE_BUILD_SKIP sid=%s reason=no_account_index cand=%s", sid, cand
            )
            continue

        account = accounts_by_index.get(idx)
        if not isinstance(account, Mapping):
            logger.warning(
                "CASE_BUILD_SKIP sid=%s idx=%s reason=no_full_account", sid, idx
            )
            continue

        account_dir = accounts_dir / str(idx)
        account_dir.mkdir(parents=True, exist_ok=True)

        pointers = {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        }

        raw_lines = list(account.get("lines") or [])
        _write_json(account_dir / pointers["raw"], raw_lines)

        _maybe_backfill_original_creditor(account.get("triad_fields"), raw_lines)

        bureaus_payload = _build_bureaus_payload_from_stagea(account)
        if HISTORY_MAIN_WIRING_ENABLED and ("two_year_payment_history_monthly" in bureaus_payload):
            heading = account.get("heading", "unknown")
            logger.info("CASE_BUILDER_HISTORY_MONTHLY_INCLUDED sid=%s account_id=%s heading=%s", sid, idx, heading)
        bureaus_payload_filtered = _filter_monthly_v2_for_export(bureaus_payload)
        _write_json(account_dir / pointers["bureaus"], bureaus_payload_filtered)

        flat_fields, _provenance = build_rule_fields_from_triad(dict(account))
        _write_json(account_dir / pointers["flat"], flat_fields)

        tags_path = account_dir / pointers["tags"]
        if not tags_path.exists():
            tags_path.write_text("[]", encoding="utf-8")
        tag_paths[idx] = tags_path

        meta = {
            "account_index": idx,
            "heading_guess": account.get("heading_guess"),
            "page_start": account.get("page_start"),
            "line_start": account.get("line_start"),
            "page_end": account.get("page_end"),
            "line_end": account.get("line_end"),
            "pointers": pointers,
        }
        account_id = cand.get("account_id") or account.get("account_id")
        if account_id is not None:
            meta["account_id"] = account_id
        _write_json(account_dir / "meta.json", meta)

        summary_path = account_dir / pointers["summary"]
        existing_summary, had_existing = _load_summary_json(summary_path)

        summary_obj: Dict[str, Any] = {
            "account_index": idx,
            "pointers": pointers,
        }
        if account_id is not None:
            summary_obj["account_id"] = account_id

        if had_existing:
            for key in ("problem_reasons", "problem_tags", "primary_issue"):
                if key in existing_summary:
                    summary_obj[key] = existing_summary[key]
            for key in ("merge_explanations", "ai_explanations", "merge_scoring"):
                if key in existing_summary:
                    summary_obj[key] = existing_summary[key]
        else:
            primary_issue, problem_reasons, problem_tags = _extract_candidate_reason(cand)
            reasons_list = _coerce_list(problem_reasons) or []
            tags_list = _coerce_list(problem_tags)

            summary_obj["problem_reasons"] = reasons_list
            if tags_list is not None:
                summary_obj["problem_tags"] = tags_list
            else:
                summary_obj["problem_tags"] = []
            if primary_issue is not None:
                summary_obj["primary_issue"] = primary_issue

        merge_tag_v2_obj: Any = (
            cand.get("merge_tag_v2") if isinstance(cand, Mapping) else None
        )
        if merge_tag_v2_obj is None and had_existing:
            merge_tag_v2_obj = existing_summary.get("merge_tag_v2")

        if not merge_v2_only:
            if isinstance(merge_tag_v2_obj, Mapping):
                try:
                    sanitized_tag_v2 = json.loads(
                        json.dumps(merge_tag_v2_obj, ensure_ascii=False)
                    )
                except TypeError:
                    sanitized_tag_v2 = dict(merge_tag_v2_obj)
                summary_obj["merge_tag_v2"] = sanitized_tag_v2
            elif merge_tag_v2_obj is not None:
                summary_obj["merge_tag_v2"] = merge_tag_v2_obj

        _write_json(summary_path, summary_obj)

        artifact_keys = {str(idx)}
        if account_id is not None:
            artifact_keys.add(str(account_id))

        issue_tag = _build_issue_tag(summary_obj)
        if issue_tag is not None:
            upsert_tag(tags_path, issue_tag, unique_keys=("kind",))

        if manifest is None:
            legacy_id = str(account_id) if account_id is not None else f"idx-{idx:03d}"
            legacy_path = accounts_dir / f"{legacy_id}.json"
            legacy_payload = dict(summary_obj)
            legacy_payload.setdefault("case_dir", str(account_dir))
            try:
                _write_json(legacy_path, legacy_payload)
            except Exception:
                logger.debug(
                    "CASE_BUILD_LEGACY_WRITE_FAILED sid=%s path=%s", sid, legacy_path
                )

        if manifest is not None:
            try:
                for key in artifact_keys:
                    manifest.set_artifact(f"cases.accounts.{key}", "dir", account_dir)
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "meta", account_dir / "meta.json"
                    )
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "raw", account_dir / pointers["raw"]
                    )
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "bureaus", account_dir / pointers["bureaus"]
                    )
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "flat", account_dir / pointers["flat"]
                    )
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "summary", summary_path
                    )
                    manifest.set_artifact(
                        f"cases.accounts.{key}", "tags", account_dir / pointers["tags"]
                    )
            except Exception:
                pass

        written_ids.append(str(idx))
        written_indices.append(idx)

    candidates_list = [c for c in candidates if isinstance(c, Mapping)]
    index_payload = {
        "sid": sid,
        "total": total,
        "problematic": len(candidates_list),
        "problematic_accounts": [c.get("account_index") for c in candidates_list],
    }
    _write_json(cases_dir / "index.json", index_payload)
    if manifest is not None:
        manifest.set_artifact("cases", "accounts_index", accounts_dir / "index.json")
        manifest.set_artifact("cases", "problematic_ids", cases_dir / "index.json")

    accounts_index = {
        "sid": sid,
        "count": len(written_ids),
        "ids": written_ids,
        "items": [],
    }
    for aid in written_ids:
        item: Dict[str, Any] = {
            "id": aid,
            "dir": str((accounts_dir / aid).resolve()),
        }
        try:
            aid_int = int(aid)
        except (TypeError, ValueError):
            aid_int = None
        if aid_int is not None:
            full_acc = accounts_by_index.get(aid_int)
            if isinstance(full_acc, Mapping):
                acc_id_val = full_acc.get("account_id")
                if acc_id_val is not None:
                    item["account_id"] = acc_id_val
        group_id = merge_groups.get(aid)
        if group_id is not None:
            item["merge_group_id"] = group_id
        accounts_index["items"].append(item)

    accounts_index_path = accounts_dir / "index.json"
    _write_json(accounts_index_path, accounts_index)
    logger.info(
        "CASES_INDEX sid=%s file=%s count=%d", sid, accounts_index_path, len(written_ids)
    )

    merge_scores: Dict[int, Dict[int, Dict[str, Any]]] = {}
    best_partners: Dict[int, Dict[str, Any]] = {}
    if written_indices:
        try:
            merge_scores = score_all_pairs_0_100(
                sid, written_indices, runs_root=runs_root_path
            )
            best_partners = choose_best_partner(merge_scores)
        except Exception:
            logger.exception("CASE_MERGE_SCORE sid=%s failed", sid)
            merge_scores = {}
            best_partners = {}

    merge_kinds = {"merge_pair", "merge_best"}
    for idx, path in tag_paths.items():
        existing_tags = read_tags(path)
        filtered_tags = [
            tag for tag in existing_tags if tag.get("kind") not in merge_kinds
        ]
        if filtered_tags != existing_tags:
            write_tags_atomic(path, filtered_tags)

    valid_decisions = {"ai", "auto"}
    for left, right in gen_unordered_pairs(written_indices):
        result = merge_scores.get(left, {}).get(right)
        if not isinstance(result, Mapping):
            continue
        pair_left = build_merge_pair_tag(right, result)
        decision = pair_left.get("decision")
        if decision not in valid_decisions:
            continue
        left_path = tag_paths.get(left)
        if left_path is not None:
            upsert_tag(left_path, pair_left, unique_keys=("kind", "with"))
        right_path = tag_paths.get(right)
        if right_path is not None:
            pair_right = build_merge_pair_tag(left, result)
            upsert_tag(right_path, pair_right, unique_keys=("kind", "with"))

    for idx in written_indices:
        best_tag = build_merge_best_tag(best_partners.get(idx, {}))
        if not best_tag:
            continue
        if best_tag.get("decision") not in valid_decisions:
            continue
        path = tag_paths.get(idx)
        if path is not None:
            upsert_tag(path, best_tag, unique_keys=("kind",))

        summary_path = accounts_dir / str(idx) / "summary.json"
        summary_data, _ = _load_summary_json(summary_path)
        if not isinstance(summary_data, dict):
            summary_data = {}

        merge_summary: dict[str, object] = {
            "best_with": best_tag.get("with"),
            "score_total": (
                float(best_tag.get("score_total") or best_tag.get("total") or 0.0)
                if best_tag.get("points_mode")
                else int(best_tag.get("score_total") or best_tag.get("total") or 0)
            ),
            "reasons": [
                str(reason)
                for reason in (best_tag.get("reasons") or [])
                if isinstance(reason, str) and reason
            ],
            "conflicts": [
                str(conflict)
                for conflict in (best_tag.get("conflicts") or [])
                if isinstance(conflict, str) and conflict
            ],
        }

        aux_payload = best_tag.get("aux")
        matched_fields: dict[str, bool] = {}
        matched_pairs: dict[str, list[str]] = {}
        acct_level_value = normalize_level(None)
        account_pair: list[str] = []
        if isinstance(aux_payload, Mapping):
            raw_matched = aux_payload.get("matched_fields")
            if isinstance(raw_matched, Mapping):
                matched_fields = {
                    str(field): bool(flag)
                    for field, flag in raw_matched.items()
                }
            acct_level_value = normalize_level(aux_payload.get("acctnum_level"))
            raw_pairs = aux_payload.get("by_field_pairs")
            if isinstance(raw_pairs, Mapping):
                for field, pair in raw_pairs.items():
                    if not isinstance(pair, (list, tuple)):
                        continue
                    values = [str(item) for item in list(pair)[:2]]
                    if len(values) != 2:
                        continue
                    field_key = str(field)
                    matched_pairs[field_key] = values
                    if field_key == "account_number":
                        account_pair = values
            digits_a = aux_payload.get("acctnum_digits_len_a")
            digits_b = aux_payload.get("acctnum_digits_len_b")
            if digits_a is not None:
                try:
                    merge_summary["acctnum_digits_len_a"] = int(digits_a)
                except (TypeError, ValueError):
                    pass
            if digits_b is not None:
                try:
                    merge_summary["acctnum_digits_len_b"] = int(digits_b)
                except (TypeError, ValueError):
                    pass
        merge_summary["acctnum_level"] = acct_level_value

        if account_pair and len(account_pair) == 2:
            matched_pairs.setdefault("account_number", account_pair)
        else:
            matched_pairs["account_number"] = []
        merge_summary["matched_pairs"] = matched_pairs
        merge_summary["matched_fields"] = matched_fields

        summary_data["merge_scoring"] = merge_summary
        _write_json(summary_path, summary_data)

    logger.info(
        "PROBLEM_CASES done sid=%s total=%d problematic=%d out=%s",
        sid,
        total,
        len(candidates_list),
        cases_dir,
    )

    return {
        "sid": sid,
        "total": total,
        "problematic": len(candidates_list),
        "out": str(cases_dir),
        "cases": {"count": len(written_ids), "dir": str(accounts_dir)},
        "merge_scoring": {"scores": merge_scores, "best": best_partners},
    }


# ---------------------------------------------------------------------------
# Legacy writer implementation
# ---------------------------------------------------------------------------


def _build_problem_cases_legacy(
    sid: str, candidates: List[Dict[str, Any]], *, root: Path | None = None
) -> Dict[str, Any]:
    """Materialise legacy problem case files for ``sid``."""

    acc_path, gen_path, manifest = _resolve_inputs_from_manifest(sid, root=root)
    logger.info(
        "ANALYZER_INPUT sid=%s accounts_json=%s general_json=%s",
        sid,
        acc_path,
        gen_path,
    )
    if not acc_path.exists():
        raise RuntimeError(
            f"accounts_from_full.json missing sid={sid} path={acc_path}"
        )

    accounts = _load_accounts(acc_path)
    total = len(accounts)
    lookup = _build_account_lookup(accounts)
    accounts_by_index: Dict[int, Mapping[str, Any]] = {}
    for acc in accounts:
        idx = acc.get("account_index")
        if isinstance(idx, int):
            accounts_by_index[idx] = acc

    general_info: Dict[str, Any] | None = None
    try:
        if gen_path and gen_path.exists():
            general_info = json.loads(gen_path.read_text(encoding="utf-8"))
    except Exception:
        general_info = None

    cases_dir = manifest.ensure_run_subdir("cases_dir", "cases")
    accounts_dir = (cases_dir / "accounts").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    manifest.set_base_dir("cases_accounts_dir", accounts_dir)
    write_breadcrumb(manifest.path, cases_dir / ".manifest")
    logger.info("CASES_OUT sid=%s accounts_dir=%s", sid, accounts_dir)

    logger.info("PROBLEM_CASES start sid=%s total=%s out=%s", sid, total, cases_dir)

    written_ids: List[str] = []
    merge_groups: Dict[str, str] = {}
    merge_v2_only = merge_v2_only_enabled()
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue

        account_id = cand.get("account_id")
        account_index = (
            cand.get("account_index") if isinstance(cand.get("account_index"), int) else None
        )

        full_acc: Mapping[str, Any] | None = None
        if account_index is not None:
            full_acc = accounts_by_index.get(account_index)
        if full_acc is None and account_id is not None:
            full_acc = lookup.get(str(account_id))
            if isinstance(full_acc, Mapping):
                idx_from_acc = full_acc.get("account_index")
                if isinstance(idx_from_acc, int):
                    account_index = idx_from_acc

        if not isinstance(account_index, int):
            logger.warning(
                "CASE_BUILD_SKIP sid=%s account_id=%s reason=no_account_index", sid, account_id
            )
            continue

        if not isinstance(full_acc, Mapping):
            full_acc = accounts_by_index.get(account_index)

        if not isinstance(full_acc, Mapping):
            logger.warning(
                "CASE_BUILD_SKIP sid=%s idx=%s reason=no_full_account", sid, account_index
            )
            continue

        account_dir = (accounts_dir / str(account_index)).resolve()
        account_dir.mkdir(parents=True, exist_ok=True)

        pointers = {
            "raw": "raw_lines.json",
            "bureaus": "bureaus.json",
            "flat": "fields_flat.json",
            "tags": "tags.json",
            "summary": "summary.json",
        }

        raw_lines = list(full_acc.get("lines") or [])
        _write_json(account_dir / pointers["raw"], raw_lines)

        bureaus_obj = _build_bureaus_payload_from_stagea(full_acc)
        bureaus_obj_filtered = _filter_monthly_v2_for_export(bureaus_obj)
        _write_json(account_dir / pointers["bureaus"], bureaus_obj_filtered)

        flat_fields, _prov = build_rule_fields_from_triad(dict(full_acc))
        _write_json(account_dir / pointers["flat"], flat_fields)

        tags_path = account_dir / pointers["tags"]
        if not tags_path.exists():
            tags_path.write_text("[]", encoding="utf-8")

        meta_obj = {
            "account_index": account_index,
            "heading_guess": full_acc.get("heading_guess"),
            "page_start": full_acc.get("page_start"),
            "line_start": full_acc.get("line_start"),
            "page_end": full_acc.get("page_end"),
            "line_end": full_acc.get("line_end"),
            "pointers": pointers,
        }
        if account_id is not None:
            meta_obj["account_id"] = account_id
        _write_json(account_dir / "meta.json", meta_obj)

        summary_path = account_dir / pointers["summary"]
        existing_summary, loaded_existing = _load_summary_json(summary_path)
        if loaded_existing:
            summary_obj = dict(existing_summary)
            if merge_v2_only:
                summary_obj.pop("merge_tag", None)
                summary_obj.pop("merge_tag_v2", None)
        else:
            summary_obj = {}

        summary_obj["sid"] = sid
        summary_obj["account_index"] = account_index
        if account_id is not None:
            summary_obj["account_id"] = account_id

        candidate_reason = cand.get("reason") if isinstance(cand.get("reason"), Mapping) else None
        candidate_primary_issue = (
            candidate_reason.get("primary_issue")
            if isinstance(candidate_reason, Mapping)
            else cand.get("primary_issue")
        )
        candidate_problem_reasons = (
            candidate_reason.get("problem_reasons")
            if isinstance(candidate_reason, Mapping)
            else cand.get("problem_reasons")
        )
        candidate_problem_tags = cand.get("problem_tags")

        coerced_tags = _coerce_list(candidate_problem_tags)
        coerced_reasons = _coerce_list(candidate_problem_reasons)

        if not loaded_existing:
            if coerced_tags is not None:
                summary_obj["problem_tags"] = coerced_tags
            else:
                summary_obj.setdefault("problem_tags", [])
            if coerced_reasons is not None:
                summary_obj["problem_reasons"] = coerced_reasons
            else:
                summary_obj.setdefault("problem_reasons", [])
            if candidate_primary_issue is not None:
                summary_obj["primary_issue"] = candidate_primary_issue
            if "confidence" in cand and cand.get("confidence") is not None:
                summary_obj["confidence"] = cand.get("confidence")
        else:
            if "problem_tags" not in summary_obj and coerced_tags is not None:
                summary_obj["problem_tags"] = coerced_tags
            if "problem_reasons" not in summary_obj and coerced_reasons is not None:
                summary_obj["problem_reasons"] = coerced_reasons
            if "primary_issue" not in summary_obj and candidate_primary_issue is not None:
                summary_obj["primary_issue"] = candidate_primary_issue

        merge_tag = cand.get("merge_tag")
        group_id: Any = None
        if isinstance(merge_tag, Mapping):
            try:
                merge_tag_obj = json.loads(
                    json.dumps(merge_tag, ensure_ascii=False)
                )
            except TypeError:
                merge_tag_obj = dict(merge_tag)
            if not merge_v2_only:
                summary_obj["merge_tag"] = merge_tag_obj
            group_id = merge_tag_obj.get("group_id")
        else:
            existing_tag: Any = None
            if not merge_v2_only:
                existing_tag = summary_obj.get("merge_tag")
            if existing_tag is None:
                legacy_tag = existing_summary.get("merge_tag")
                if isinstance(legacy_tag, Mapping):
                    existing_tag = legacy_tag
            if isinstance(existing_tag, Mapping):
                group_id = existing_tag.get("group_id")
        if isinstance(group_id, str):
            merge_groups[str(account_index)] = group_id

        merge_tag_v2 = cand.get("merge_tag_v2")
        if not merge_v2_only:
            if isinstance(merge_tag_v2, Mapping):
                try:
                    merge_tag_v2_obj = json.loads(
                        json.dumps(merge_tag_v2, ensure_ascii=False)
                    )
                except TypeError:
                    merge_tag_v2_obj = dict(merge_tag_v2)
                summary_obj["merge_tag_v2"] = merge_tag_v2_obj
            else:
                existing_v2 = summary_obj.get("merge_tag_v2")
                if existing_v2 is None:
                    legacy_v2 = existing_summary.get("merge_tag_v2")
                    if isinstance(legacy_v2, Mapping):
                        try:
                            legacy_v2_obj = json.loads(
                                json.dumps(legacy_v2, ensure_ascii=False)
                            )
                        except TypeError:
                            legacy_v2_obj = dict(legacy_v2)
                        summary_obj["merge_tag_v2"] = legacy_v2_obj
        else:
            summary_obj.pop("merge_tag_v2", None)

        summary_obj["pointers"] = pointers

        if not loaded_existing and isinstance(general_info, dict):
            extra: Dict[str, Any] = {}
            for k in (
                "client",
                "client_name",
                "report_date",
                "report_start",
                "report_end",
                "generated_at",
                "provider",
                "source",
            ):
                v = general_info.get(k)
                if v is not None:
                    extra[k] = v
            if extra:
                summary_obj["general"] = extra

        _write_json(summary_path, summary_obj)

        try:
            manifest.set_artifact(f"cases.accounts.{account_index}", "dir", account_dir)
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "meta", account_dir / "meta.json"
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "raw", account_dir / pointers["raw"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "bureaus", account_dir / pointers["bureaus"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "flat", account_dir / pointers["flat"]
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "summary", summary_path
            )
            manifest.set_artifact(
                f"cases.accounts.{account_index}", "tags", account_dir / pointers["tags"]
            )
        except Exception:
            pass

        written_ids.append(str(account_index))

    cand_list = list(candidates)
    index_data = {
        "sid": sid,
        "total": total,
        "problematic": len(cand_list),
        "problematic_accounts": [c.get("account_id") for c in cand_list],
    }
    _write_json(cases_dir / "index.json", index_data)
    manifest.set_artifact("cases", "accounts_index", accounts_dir / "index.json")
    manifest.set_artifact("cases", "problematic_ids", cases_dir / "index.json")

    acc_index = {
        "sid": sid,
        "count": len(written_ids),
        "ids": written_ids,
        "items": [],
    }
    for aid in written_ids:
        item = {
            "id": aid,
            "dir": str((accounts_dir / aid).resolve()),
        }
        try:
            aid_int = int(aid)
        except (TypeError, ValueError):
            aid_int = None
        if aid_int is not None:
            full_acc = accounts_by_index.get(aid_int)
            if isinstance(full_acc, Mapping):
                account_id_val = full_acc.get("account_id")
                if account_id_val is not None:
                    item["account_id"] = account_id_val
        group_id = merge_groups.get(aid)
        if group_id is not None:
            item["merge_group_id"] = group_id
        acc_index["items"].append(item)
    accounts_index_path = accounts_dir / "index.json"
    _write_json(accounts_index_path, acc_index)
    logger.info(
        "CASES_INDEX sid=%s file=%s count=%d", sid, accounts_index_path, len(written_ids)
    )

    logger.info(
        "PROBLEM_CASES done sid=%s total=%s problematic=%s out=%s",
        sid,
        total,
        len(cand_list),
        cases_dir,
    )

    return {
        "sid": sid,
        "total": total,
        "problematic": len(cand_list),
        "out": str(cases_dir),
        "cases": {"count": len(written_ids), "dir": str(accounts_dir)},
    }


# ---------------------------------------------------------------------------
# Public entry point with feature flag
# ---------------------------------------------------------------------------


def build_problem_cases(
    sid: str,
    candidates: List[Dict[str, Any]],
    *,
    root: Path | None = None,
) -> Dict[str, Any]:
    if candidates is None:  # pragma: no cover - defensive guard
        raise TypeError("candidates must not be None")

    cand_list = list(candidates)

    if not LEAN:
        return _build_problem_cases_legacy(sid, candidates=cand_list, root=root)
    return _build_problem_cases_lean(sid, candidates=cand_list, root=root)


__all__ = ["build_problem_cases"]
