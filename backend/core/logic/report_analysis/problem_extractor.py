from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from backend.pipeline.runs import RunManifest

from .keys import compute_logical_account_key

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[4]))


def _candidate_manifest_paths(sid: str, root: Path | None = None) -> List[Path]:
    """Yield manifest locations ordered by priority."""

    candidates: List[Path] = []

    env_manifest = os.getenv("REPORT_MANIFEST_PATH")
    if env_manifest:
        candidates.append(Path(env_manifest))

    runs_root_env = os.getenv("RUNS_ROOT")
    if runs_root_env:
        candidates.append(Path(runs_root_env) / sid / "manifest.json")

    if root is not None:
        candidates.append(Path(root) / "runs" / sid / "manifest.json")

    default_runs = PROJECT_ROOT / "runs" / sid / "manifest.json"
    candidates.append(default_runs)

    # Drop duplicates while preserving order
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


def _load_manifest_for_sid(sid: str, root: Path | None = None) -> Optional[RunManifest]:
    for path in _candidate_manifest_paths(sid, root=root):
        if path.exists():
            try:
                return RunManifest(path).load()
            except Exception:
                logger.debug(
                    "PROBLEM_EXTRACTOR manifest_load_failed sid=%s path=%s", sid, path
                )
                continue
    return None


def _load_accounts_from_path(path: Path) -> List[dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(obj, Mapping):
        accounts = obj.get("accounts")
        if isinstance(accounts, list):
            return list(accounts)
        # some fixtures may store the accounts list directly under another key
        if isinstance(obj.get("Accounts"), list):  # pragma: no cover - defensive
            return list(obj["Accounts"])
        return []
    if isinstance(obj, list):
        return list(obj)
    return []


def _fallback_stagea_paths(sid: str, root: Path | None = None) -> List[Path]:
    roots: List[Path] = []
    if root is not None:
        roots.append(Path(root))
    roots.append(PROJECT_ROOT)

    filenames = [
        "accounts_from_full.json",
        "accounts.json",
    ]

    paths: List[Path] = []
    for base in roots:
        table_dir = base / "traces" / "blocks" / sid / "accounts_table"
        for name in filenames:
            candidate = table_dir / name
            if candidate.exists():
                paths.append(candidate)
    return paths


def load_stagea_accounts_from_manifest(sid: str, *, root: Path | None = None) -> list[dict]:
    """Load Stage-A accounts using the run manifest when available.

    Falls back to legacy ``traces/blocks`` locations when the manifest or
    registered artifacts are missing.  This retains backwards compatibility for
    tests that synthesise inputs without running the full Stage-A pipeline.
    """

    manifest = _load_manifest_for_sid(sid, root=root)
    if manifest is not None:
        try:
            acc_path = Path(manifest.get("traces.accounts_table", "accounts_json"))
        except KeyError:
            acc_path = None
        else:
            if acc_path.exists():
                accounts = _load_accounts_from_path(acc_path)
                if accounts:
                    return accounts

    for fallback in _fallback_stagea_paths(sid, root=root):
        accounts = _load_accounts_from_path(fallback)
        if accounts:
            return accounts

    return []


_CURRENCY = re.compile(r"[^\d.\-]")


def _to_float(num: str | int | float | None) -> Optional[float]:
    if num is None:
        return None
    if isinstance(num, (int, float)):
        return float(num)
    s = str(num).strip()
    if not s:
        return None
    s = _CURRENCY.sub("", s)
    try:
        return float(s)
    except Exception:
        return None


def _pick_by_order(triad_order: list[str], values_by_bureau: dict) -> Optional[str]:
    for b in triad_order or []:
        v = values_by_bureau.get(b)
        if v not in (None, "", "--"):
            return v
    for b in ("transunion", "experian", "equifax"):
        v = values_by_bureau.get(b)
        if v not in (None, "", "--"):
            return v
    return None


def build_rule_fields_from_triad(account: dict) -> tuple[dict, dict]:
    triad = account.get("triad", {}) or {}
    order = triad.get("order", ["transunion", "experian", "equifax"])
    triad_fields = account.get("triad_fields", {}) or {}

    def get_k(k: str) -> tuple[Optional[str], Optional[str]]:
        vals = {
            "transunion": (triad_fields.get("transunion") or {}).get(k),
            "experian": (triad_fields.get("experian") or {}).get(k),
            "equifax": (triad_fields.get("equifax") or {}).get(k),
        }
        # preferred order first
        for b in order or []:
            v = vals.get(b)
            if v not in (None, "", "--"):
                return v, b
        # fallback canonical order
        for b in ("transunion", "experian", "equifax"):
            v = vals.get(b)
            if v not in (None, "", "--"):
                return v, b
        return None, None

    prov: dict[str, Optional[str]] = {}

    v, b = get_k("past_due_amount"); past_due_amount = _to_float(v); prov["past_due_amount"] = b
    v, b = get_k("balance_owed");    balance_owed    = _to_float(v); prov["balance_owed"] = b
    v, b = get_k("credit_limit");    credit_limit    = _to_float(v); prov["credit_limit"] = b

    v, b = get_k("payment_status"); payment_status = v; prov["payment_status"] = b
    v, b = get_k("account_status"); account_status = v; prov["account_status"] = b

    sev = account.get("seven_year_history") or {}

    def _total_lates(bureau: str) -> int:
        rec = sev.get(bureau) or {}
        return int(rec.get("late30", 0)) + int(rec.get("late60", 0)) + int(
            rec.get("late90", 0)
        )

    days_late_7y = max(
        _total_lates("transunion"),
        _total_lates("experian"),
        _total_lates("equifax"),
    )

    two_year = account.get("two_year_payment_history") or {}

    def _any_derog(tokens: list[str]) -> bool:
        return any(t for t in (tokens or []) if (t or "").upper() not in ("", "OK"))

    has_derog_2y = any(
        [
            _any_derog(two_year.get("transunion") or []),
            _any_derog(two_year.get("experian") or []),
            _any_derog(two_year.get("equifax") or []),
        ]
    )

    v, b = get_k("creditor_remarks"); creditor_remarks = v; prov["creditor_remarks"] = b
    v, b = get_k("account_type");     account_type     = v; prov["account_type"] = b

    fields = {
        "past_due_amount": past_due_amount,
        "balance_owed": balance_owed,
        "credit_limit": credit_limit,
        "payment_status": payment_status,
        "account_status": account_status,
        "days_late_7y": days_late_7y,
        "has_derog_2y": has_derog_2y,
        "account_type": account_type,
        "creditor_remarks": creditor_remarks,
    }
    return fields, prov

BAD_PAYMENT = {
    "late",
    "delinquent",
    "past due",
    "charge-off",
    "collection",
    "derog",
    "120",
    "150",
    "co",
}
BAD_ACCOUNT = {
    "collections",
    "charge-off",
    "charged off",
    "repossession",
    "foreclosure",
}


# Centralized, configurable rule thresholds
RULES_CFG: Dict[str, Dict[str, float | int]] = {
    "thresholds": {
        # Flag delinquency when past_due_amount strictly greater than this value
        "past_due_amount_gt": 0.0,
        # Flag late history when days_late_7y is greater or equal to this value
        "days_late_7y_min": 1,
    }
}


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _contains_any(haystack: str, needles: set[str]) -> str | None:
    for n in needles:
        if n and n in haystack:
            return n
    return None


def _format_amount(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def evaluate_account_problem(fields: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []

    # Numeric conditions
    try:
        pda = float(fields.get("past_due_amount") or 0)
        if pda > float(RULES_CFG["thresholds"]["past_due_amount_gt"]):
            reasons.append(
                f"past_due_amount: {_format_amount(fields.get('past_due_amount'))}"
            )
    except Exception:
        pass

    # History conditions
    try:
        days = int(fields.get("days_late_7y") or 0)
        if days >= int(RULES_CFG["thresholds"]["days_late_7y_min"]):
            reasons.append(f"late_history: days_late_7y={days}")
    except Exception:
        pass

    # Status conditions
    pay = _norm(fields.get("payment_status"))
    acc = _norm(fields.get("account_status"))

    if _contains_any(pay, BAD_PAYMENT):
        reasons.append(f"bad_payment_status: {fields.get('payment_status')}")
    if _contains_any(acc, BAD_ACCOUNT):
        reasons.append(f"bad_account_status: {fields.get('account_status')}")

    # Optional: positive balance on a closed account
    try:
        bal = float(fields.get("balance_owed") or 0)
        if bal > 0 and acc == "closed":
            reasons.append("positive_balance_on_closed")
    except Exception:
        pass

    # Primary issue selection
    primary_issue = "unknown"
    if _contains_any(pay, {"charge-off", "collection", "co"}) or _contains_any(
        acc, {"charge-off", "collections", "charged off"}
    ):
        if _contains_any(pay, {"collection"}) or _contains_any(acc, {"collections"}):
            primary_issue = "collection"
        else:
            primary_issue = "charge_off"
    elif (fields.get("past_due_amount") or 0) and float(fields.get("past_due_amount") or 0) > float(
        RULES_CFG["thresholds"]["past_due_amount_gt"]
    ):
        primary_issue = "delinquency"
    elif int(fields.get("days_late_7y") or 0) >= int(
        RULES_CFG["thresholds"]["days_late_7y_min"]
    ):
        primary_issue = "late_history"
    elif _contains_any(pay, BAD_PAYMENT) or _contains_any(acc, BAD_ACCOUNT):
        primary_issue = "status"

    decision: Dict[str, Any] = {
        "primary_issue": primary_issue,
        "issue_types": [],
        "problem_reasons": reasons,
        "decision_source": "rules",
        "confidence": 0.0,
        "tier": "none",
        "debug": {"signals": []},
    }
    return decision


def _get_field(account: Mapping[str, Any], *names: str) -> Any:
    """Return the first non-null field matching ``names`` from account or its fields."""
    fields = (
        account.get("fields") if isinstance(account.get("fields"), Mapping) else None
    )
    for name in names:
        if fields and fields.get(name) is not None:
            return fields.get(name)
        if account.get(name) is not None:
            return account.get(name)
    return None


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
    issuer = _get_field(account, "issuer", "creditor", "name")
    last4 = _get_field(account, "account_last4", "last4")
    account_type = _get_field(account, "account_type", "type")
    opened_date = _get_field(account, "opened_date", "date_opened")
    key = compute_logical_account_key(issuer, last4, account_type, opened_date)
    acc_id = key or f"idx-{idx:03d}"
    return re.sub(r"[^a-z0-9_-]", "_", acc_id.lower())


def _resolve_inputs_from_manifest(sid: str) -> tuple[Path, Path, RunManifest]:
    m = RunManifest.for_sid(sid)
    try:
        accounts_path = Path(m.get("traces.accounts_table", "accounts_json")).resolve()
        general_path = Path(m.get("traces.accounts_table", "general_json")).resolve()
    except KeyError as e:
        raise RuntimeError(f"Run manifest missing traces.accounts_table key for sid={sid}: {e}")
    return accounts_path, general_path, m


def detect_problem_accounts(sid: str, root: Path | None = None) -> List[Dict[str, Any]]:
    """Return problematic accounts for ``sid`` based on rule evaluation.

    Uses manifest-driven Stage-A accounts and adapts triad fields when
    a flat ``fields`` mapping is not present. Does not write any files.
    """
    try:
        accounts: List[Mapping[str, Any]] = list(
            load_stagea_accounts_from_manifest(sid, root=root)
        )
    except Exception:
        accounts = []
    logger.info("ANALYZER_START sid=%s accounts=%d", sid, len(accounts))

    candidates: List[Dict[str, Any]] = []
    for pos, acc in enumerate(accounts, start=1):
        if not isinstance(acc, Mapping):
            continue
        fields = acc.get("fields") if isinstance(acc.get("fields"), Mapping) else None
        prov: dict[str, Optional[str]] = {}
        if not fields:
            fields, prov = build_rule_fields_from_triad(dict(acc))
        reason = evaluate_account_problem(dict(fields))
        # Enrich debug signals with bureau provenance where applicable
        dbg_signals: List[str] = []
        try:
            pda = fields.get("past_due_amount")
            if pda is not None and float(pda) > 0:
                dbg_signals.append(
                    f"past_due_amount:{_format_amount(pda)} (bureau={prov.get('past_due_amount')})"
                )
        except Exception:
            pass
        pay = _norm(fields.get("payment_status"))
        if pay and _contains_any(pay, BAD_PAYMENT):
            dbg_signals.append(
                f"payment_status:{fields.get('payment_status')} (bureau={prov.get('payment_status')})"
            )
        acc_status = _norm(fields.get("account_status"))
        if acc_status and _contains_any(acc_status, BAD_ACCOUNT):
            dbg_signals.append(
                f"account_status:{fields.get('account_status')} (bureau={prov.get('account_status')})"
            )
        try:
            bal = fields.get("balance_owed")
            if bal is not None and float(bal) > 0 and acc_status == "closed":
                dbg_signals.append(
                    f"positive_balance_on_closed (bureau={prov.get('account_status') or prov.get('balance_owed')})"
                )
        except Exception:
            pass
        # Attach to reason.debug.signals
        try:
            reason.setdefault("debug", {})
            reason["debug"]["signals"] = list(dbg_signals)
        except Exception:
            pass
        if reason and list(reason.get("problem_reasons") or []):
            idx_raw = acc.get("account_index")
            try:
                idx_int = int(idx_raw)
            except Exception:
                idx_int = pos
            account_id = acc.get("account_id")
            if account_id is None:
                account_id = _make_account_id(acc, idx_int)
            candidates.append(
                {
                    "account_index": acc.get("account_index"),
                    "heading_guess": acc.get("heading_guess"),
                    "account_id": account_id,
                    "reason": reason,
                }
            )
            try:
                heading = acc.get("heading_guess")
                idx = acc.get("account_index")
                logger.info(
                    "ANALYZER_PROBLEM sid=%s idx=%s heading=%s reason=%s",
                    sid,
                    idx,
                    heading,
                    reason,
                )
            except Exception:
                pass
    logger.info("ANALYZER_DONE sid=%s count=%d", sid, len(candidates))
    # Aggregate summary by primary_issue for quick grepping
    try:
        counts: Dict[str, int] = {}
        for c in candidates:
            pi = None
            try:
                pi = str((c.get("reason") or {}).get("primary_issue") or "")
            except Exception:
                pi = ""
            if not pi:
                continue
            counts[pi] = counts.get(pi, 0) + 1
        logger.info("ANALYZER_SUMMARY sid=%s issues=%s", sid, counts)
    except Exception:
        pass
    return candidates
