import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Optional bootstrap to ensure repo root is on sys.path
try:  # pragma: no cover - convenience import
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:
    # Fallback: add repository root based on this file's location
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from collections import Counter
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.core.merge.acctnum import normalize_level
from backend.pipeline.runs import RunManifest


def _rebase_manifest_path(raw: str, sid: str) -> Path:
    """Map a manifest-recorded path to the local ``runs/<sid>`` layout."""

    path = Path(raw)
    if path.exists():
        return path

    normalized = str(raw).replace("\\", "/")
    if "/runs/" in normalized:
        suffix = normalized.split("/runs/", 1)[1]
        candidate = Path("runs") / suffix
        if candidate.exists():
            return candidate

    if sid and sid in normalized:
        after_sid = normalized.split(sid, 1)[1].lstrip("/\\")
        if after_sid:
            candidate = Path("runs") / sid / after_sid
            if candidate.exists():
                return candidate

    return path


def _load_stagea_accounts_with_fallback(sid: str) -> List[Dict[str, Any]]:
    """Load Stage-A accounts, rebasing manifest paths when necessary."""

    manifest = RunManifest.for_sid(sid)
    raw_path = manifest.get("traces.accounts_table", "accounts_json")
    accounts_path = _rebase_manifest_path(raw_path, sid)
    if not accounts_path.exists():
        raise FileNotFoundError(
            f"Stage-A accounts JSON not found for sid={sid!r} (expected at {accounts_path})"
        )

    with accounts_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    accounts = payload.get("accounts")
    return list(accounts) if isinstance(accounts, list) else []


def _coerce_index(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_stagea_account(accounts: Iterable[Dict[str, Any]], sid: str) -> Tuple[Dict[str, Any], int]:
    """Locate account 8 (or fallback) within Stage-A ``accounts``."""

    mask = "552433**********"
    fallback: Optional[Tuple[Dict[str, Any], int]] = None

    for acc in accounts:
        idx = _coerce_index((acc or {}).get("account_index"))
        if idx == 8:
            return acc, idx

        if fallback is None:
            lines = acc.get("lines") if isinstance(acc.get("lines"), list) else []
            for line in lines:
                text = str((line or {}).get("text") or "")
                if mask in text:
                    if idx is not None:
                        fallback = (acc, idx)
                    break

    if fallback is not None:
        return fallback

    raise AssertionError(
        f"unable to locate Stage-A account 8 for sid={sid!r}; no matching account mask found"
    )


def _read_account_index_from_dir(path: Path) -> Optional[int]:
    for name in ("meta.json", "summary.json", "account.json"):
        candidate = path / name
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = _coerce_index((payload or {}).get("account_index"))
        if idx is not None:
            return idx
    return None


def _find_cases_accounts_dir(manifest: RunManifest) -> Path:
    base_dirs = manifest.data.get("base_dirs", {}) if isinstance(manifest.data, dict) else {}
    raw_dir = base_dirs.get("cases_accounts_dir") if isinstance(base_dirs, dict) else None
    if isinstance(raw_dir, str) and raw_dir:
        candidate = _rebase_manifest_path(raw_dir, manifest.sid)
        if candidate.exists():
            return candidate

    fallback = Path("runs") / manifest.sid / "cases" / "accounts"
    return fallback


def _load_bureaus_for_account(sid: str, account_idx: int) -> Tuple[Dict[str, Any], Path]:
    manifest = RunManifest.for_sid(sid)
    cases_dir = _find_cases_accounts_dir(manifest)
    if not cases_dir.exists():
        raise FileNotFoundError(
            f"cases/accounts directory not found for sid={sid!r} (looked at {cases_dir})"
        )

    search_dirs = [p for p in sorted(cases_dir.iterdir(), key=lambda item: item.name) if p.is_dir()]
    if not search_dirs:
        raise FileNotFoundError(
            f"no account case folders found under {cases_dir} for sid={sid!r}"
        )

    selected_dir: Optional[Path] = None
    for entry in search_dirs:
        idx = _read_account_index_from_dir(entry)
        if idx == account_idx or entry.name == str(account_idx):
            selected_dir = entry
            break

    if selected_dir is None:
        # Fallback: first available directory with bureaus.json
        for entry in search_dirs:
            if (entry / "bureaus.json").exists():
                selected_dir = entry
                break

    if selected_dir is None:
        raise FileNotFoundError(
            f"unable to locate bureaus.json for account {account_idx} under {cases_dir}"
        )

    bureaus_path = selected_dir / "bureaus.json"
    if not bureaus_path.exists():
        raise FileNotFoundError(
            f"expected bureaus.json in {selected_dir}, but the file was missing"
        )

    with bureaus_path.open("r", encoding="utf-8") as fh:
        bureaus = json.load(fh)

    return bureaus, bureaus_path


def _assert_account8_values(
    sid: str, stagea_account: Dict[str, Any], bureaus: Dict[str, Any]
) -> None:
    tu_expected = {
        "account_number_display": "****",
        "high_balance": "$12,028",
        "last_verified": "11.8.2025",
        "date_of_last_activity": "30.3.2024",
        "date_reported": "11.8.2025",
        "date_opened": "20.11.2021",
        "balance_owed": "$6,217",
        "closed_date": "30.3.2024",
        "account_rating": "Derogatory",
        "account_description": "Individual",
        "dispute_status": "Account not disputed",
        "creditor_type": "Bank Credit Cards",
        "account_status": "Closed",
        "payment_status": "Collection/Chargeoff",
        "payment_amount": "$0",
        "last_payment": "18.6.2025",
        "term_length": "--",
        "past_due_amount": "$6,217",
        "account_type": "Credit Card",
        "payment_frequency": "--",
        "credit_limit": "$12,000",
    }

    triad_fields = stagea_account.get("triad_fields")
    if not isinstance(triad_fields, dict):
        raise AssertionError(f"Stage-A account for sid={sid!r} is missing triad_fields")

    stagea_tu = triad_fields.get("transunion") if isinstance(triad_fields.get("transunion"), dict) else {}
    stagea_eq = triad_fields.get("equifax") if isinstance(triad_fields.get("equifax"), dict) else {}

    bureaus_tu = bureaus.get("transunion") if isinstance(bureaus.get("transunion"), dict) else {}
    bureaus_eq = bureaus.get("equifax") if isinstance(bureaus.get("equifax"), dict) else {}

    for key, expected in tu_expected.items():
        stage_val = stagea_tu.get(key)
        if stage_val != expected:
            raise AssertionError(
                f"Stage-A transunion.{key} expected {expected!r} but found {stage_val!r}"
            )

        bureau_val = bureaus_tu.get(key)
        if bureau_val != expected:
            raise AssertionError(
                f"bureaus.json transunion.{key} expected {expected!r} but found {bureau_val!r}"
            )

    # Ensure missing values remain empty strings rather than "--".
    for bureau_key in ("transunion", "experian", "equifax"):
        stage_fields = triad_fields.get(bureau_key)
        bureau_fields = bureaus.get(bureau_key)
        if not isinstance(stage_fields, dict) or not isinstance(bureau_fields, dict):
            continue
        for key, stage_val in stage_fields.items():
            if stage_val == "":
                value = bureau_fields.get(key)
                if value != "":
                    raise AssertionError(
                        f"{bureau_key}.{key} expected empty string for missing value but found {value!r}"
                    )

    # Equifax banding sanity: ensure the closed_date value was not polluted by the next label.
    eq_closed_stage = stagea_eq.get("closed_date")
    eq_closed_bureaus = bureaus_eq.get("closed_date")
    if eq_closed_stage != eq_closed_bureaus:
        raise AssertionError(
            f"equifax.closed_date mismatch between Stage-A ({eq_closed_stage!r}) and bureaus.json ({eq_closed_bureaus!r})"
        )
    if isinstance(eq_closed_bureaus, str) and "Last Payment" in eq_closed_bureaus:
        raise AssertionError("equifax.closed_date captured tokens from the next label (contains 'Last Payment')")


def _format_reasons(reasons: Any) -> List[str]:
    formatted: List[str] = []
    if isinstance(reasons, Mapping):
        for key in sorted(reasons.keys()):
            value = reasons[key]
            if isinstance(value, bool):
                if value:
                    formatted.append(str(key))
            elif value not in (None, ""):
                formatted.append(f"{key}={value}")
        return formatted

    if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
        for entry in reasons:
            if not isinstance(entry, Mapping):
                continue
            kind = entry.get("kind") or "reason"
            extras: List[str] = []
            for key in sorted(entry.keys()):
                if key == "kind":
                    continue
                value = entry[key]
                if value in (None, ""):
                    continue
                extras.append(f"{key}={value}")
            if extras:
                formatted.append(f"{kind}({' '.join(extras)})")
            else:
                formatted.append(str(kind))
    return formatted


def _merge_reason_maps(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if key not in merged:
                merged[key] = value
            elif value:
                merged[key] = value
    return merged


def _build_merge_summary(
    candidates: List[Dict[str, Any]], *, only_ai: bool = False
) -> Dict[str, Any]:
    """Return clusters and pair table summaries from merge-tagged candidates."""

    cluster_map: Dict[str, Dict[str, Any]] = {}
    pairs: List[Dict[str, Any]] = []

    for idx, account in enumerate(candidates):
        merge_tag = account.get("merge_tag") or {}
        group_id = str(merge_tag.get("group_id") or f"g{idx + 1}")
        best_match = merge_tag.get("best_match") or {}
        parts = merge_tag.get("parts") or {}

        best_score = best_match.get("score")
        pair_decision = best_match.get("decision")
        account_decision = merge_tag.get("decision")
        aux = merge_tag.get("aux") or {}
        override_reasons = (aux or {}).get("override_reasons")
        reasons_field = merge_tag.get("reasons")
        mapping_sources = [
            source for source in (override_reasons, reasons_field) if isinstance(source, Mapping)
        ]
        combined_reasons = _merge_reason_maps(*mapping_sources)

        entry: Dict[str, Any] = {
            "idx": idx,
            "group": group_id,
            "best": best_match.get("account_index"),
            "score": best_score,
            "decision": pair_decision or account_decision,
            "acctnum_level": normalize_level(aux.get("acctnum_level")),
            "balowed_ok": bool(combined_reasons.get("balance_only_triggers_ai")),
            "reasons": [],
        }

        if isinstance(reasons_field, list):
            entry["reasons"].extend(_format_reasons(reasons_field))
        entry["reasons"].extend(_format_reasons(combined_reasons))

        if account_decision and account_decision != entry["decision"]:
            entry["account_decision"] = account_decision

        if parts:
            entry["parts"] = {key: float(value) for key, value in parts.items()}

        pairs.append(entry)

        cluster_entry = cluster_map.setdefault(
            group_id,
            {"group": group_id, "members": [], "best_matches": []},
        )
        cluster_entry["members"].append(idx)
        cluster_entry["best_matches"].append(entry)

    if only_ai:
        pairs = [entry for entry in pairs if entry.get("decision") == "ai"]

    pairs.sort(key=lambda item: item["idx"])

    clusters: List[Dict[str, Any]] = []
    for group_id in sorted(cluster_map.keys()):
        cluster_entry = cluster_map[group_id]
        cluster_entry["members"].sort()
        cluster_entry["best_matches"] = sorted(
            cluster_entry["best_matches"], key=lambda item: item["idx"]
        )
        clusters.append(cluster_entry)

    return {"clusters": clusters, "pairs": pairs}


def check_lean(sid: str) -> None:
    """Validate that case folders for the SID contain lean artifacts only."""

    base = Path("runs") / sid / "cases" / "accounts"
    if not base.exists():
        raise FileNotFoundError(f"accounts directory not found for SID {sid!r} at {base}")

    try:
        stagea_accounts = _load_stagea_accounts_with_fallback(sid)
        stagea_by_idx = {}
        for account in stagea_accounts:
            idx = account.get("account_index")
            try:
                if idx is None:
                    continue
                idx_key = int(idx)
            except (TypeError, ValueError):
                continue
            stagea_by_idx[idx_key] = account
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"[LEAN-CHECK] unable to load Stage-A accounts for sid={sid}: {exc}")
        stagea_by_idx = None

    required_files = (
        "meta.json",
        "summary.json",
        "bureaus.json",
        "fields_flat.json",
        "raw_lines.json",
        "tags.json",
    )

    lean_failures = False
    samples: List[tuple[int, Dict[str, Any]]] = []

    for account_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not account_dir.is_dir():
            continue

        for filename in required_files:
            path = account_dir / filename
            if not path.exists():
                raise AssertionError(
                    f"expected {filename} in {account_dir}, but it was missing"
                )

        summary_text = (account_dir / "summary.json").read_text(encoding="utf-8")
        if "triad_rows" in summary_text:
            raise AssertionError(
                f"triad_rows detected in summary.json for {account_dir}; expected lean output"
            )

        bureaus_path = account_dir / "bureaus.json"
        bureaus_data = json.loads(bureaus_path.read_text(encoding="utf-8"))
        missing = [
            key
            for key in ("two_year_payment_history", "seven_year_history")
            if key not in bureaus_data
        ]
        if missing:
            print(f"[LEAN-CHECK] bureaus.json missing keys: {missing}")
            lean_failures = True

        if len(samples) < 2:
            try:
                idx = int(account_dir.name)
            except ValueError:
                continue
            samples.append((idx, bureaus_data))

    if stagea_by_idx and samples:
        sample_idx, bureaus_data = samples[0]
        stagea_account = stagea_by_idx.get(sample_idx)
        if stagea_account:
            two_year_match = (
                bureaus_data.get("two_year_payment_history")
                == stagea_account.get("two_year_payment_history")
            )
            seven_year_match = (
                bureaus_data.get("seven_year_history")
                == stagea_account.get("seven_year_history")
            )
            print(
                f"[LEAN-CHECK] account {sample_idx} two_year_payment_history match: {two_year_match}"
            )
            print(
                f"[LEAN-CHECK] account {sample_idx} seven_year_history match: {seven_year_match}"
            )
            if not (two_year_match and seven_year_match):
                lean_failures = True

    if lean_failures:
        raise AssertionError("lean check failed; see [LEAN-CHECK] messages above")


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke run problem candidate detection for a run SID")
    ap.add_argument("--sid", required=True, help="Run session id (SID) to analyze")
    ap.add_argument(
        "--show-all",
        action="store_true",
        help="Show all candidates instead of first 5",
    )
    ap.add_argument(
        "--show-merge",
        action="store_true",
        help="Display merge clustering summary",
    )
    ap.add_argument(
        "--only-ai",
        action="store_true",
        help="Limit merge summary tables to pairs requiring AI review packs",
    )
    ap.add_argument(
        "--check-lean",
        action="store_true",
        help="Verify case folders contain lean artifacts with no triad_rows",
    )
    ap.add_argument(
        "--skip-stagea-asserts",
        action="store_true",
        help="Skip Stage-A account-number verification checks",
    )
    ap.add_argument("--json-out", help="Path to write full JSON payload including merge data")
    args = ap.parse_args()

    out = detect_problem_accounts(args.sid)

    # Primary issue frequency
    issue_counts = Counter([str(c.get("reason", {}).get("primary_issue")) for c in out])
    issues_line = ", ".join(f"{k}={v}" for k, v in issue_counts.items() if k and v)

    # Reason key frequency (prefix before ':')
    reason_keys = []
    for c in out:
        reasons = (c.get("reason", {}) or {}).get("problem_reasons") or []
        for r in reasons:
            s = str(r)
            key = s.split(":", 1)[0].strip() if ":" in s else s.strip()
            if key:
                reason_keys.append(key)
    reason_counts = Counter(reason_keys)
    top_reasons = ", ".join(f"{k}={v}" for k, v in reason_counts.most_common(10))

    merge_summary: Dict[str, Any] | None = None
    if args.show_merge:
        raise RuntimeError(
            "Merge preview is unavailable: the legacy merge scorer has been removed."
        )

    payload = {
        "sid": args.sid,
        "problematic": len(out),
        "sample": out if args.show_all else out[:5],
    }

    if merge_summary is not None:
        payload["merge"] = merge_summary

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if issues_line:
        print(f"issues: {issues_line}")
    if top_reasons:
        print(f"reasons: {top_reasons}")

    if merge_summary is not None:
        clusters_view: List[Dict[str, Any]] = []
        for cluster in merge_summary["clusters"]:
            clusters_view.append(
                {
                    "group": cluster["group"],
                    "members": cluster["members"],
                    "best_matches": [
                        {
                            key: entry[key]
                            for key in (
                                "idx",
                                "best",
                                "score",
                                "decision",
                                "account_decision",
                                "acctnum_level",
                                "balowed_ok",
                                "reasons",
                            )
                            if key in entry
                        }
                        for entry in cluster["best_matches"]
                    ],
                }
            )

        pairs_view = [
            {
                key: entry[key]
                for key in (
                    "idx",
                    "best",
                    "score",
                    "decision",
                    "account_decision",
                    "acctnum_level",
                    "balowed_ok",
                    "parts",
                    "reasons",
                )
                if key in entry
            }
            for entry in merge_summary["pairs"]
        ]

        print("clusters:", json.dumps(clusters_view, ensure_ascii=False, indent=2))
        print("pairs:", json.dumps(pairs_view, ensure_ascii=False, indent=2))

    json_out_path = Path(args.json_out) if args.json_out else None
    if json_out_path is not None:
        json_payload: Dict[str, Any] = {
            "sid": args.sid,
            "problematic": len(out),
            "candidates": out,
            "issue_counts": dict(issue_counts),
            "reason_counts": dict(reason_counts),
        }
        if merge_summary is not None:
            json_payload["merge"] = merge_summary

        json_out_path.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote JSON to {json_out_path}")

    if args.skip_stagea_asserts:
        print("[SMOKE] skipped Stage-A account verification (--skip-stagea-asserts)")
    else:
        try:
            stagea_accounts = _load_stagea_accounts_with_fallback(args.sid)
        except Exception as exc:
            print(f"[SMOKE] failed to load Stage-A accounts: {exc}")
            raise

        try:
            stagea_account, account_idx = _find_stagea_account(stagea_accounts, args.sid)
            bureaus, bureaus_path = _load_bureaus_for_account(args.sid, account_idx)
            _assert_account8_values(args.sid, stagea_account, bureaus)
            print(
                f"[SMOKE] PASS account {account_idx} TU/EQ smoke checks verified via {bureaus_path}"
            )
        except Exception as exc:
            print(f"[SMOKE] account 8 verification failed: {exc}")
            raise

    if args.check_lean:
        check_lean(args.sid)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
