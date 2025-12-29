from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Tuple
import time

from jsonschema import Draft7Validator, ValidationError

from backend.audit.audit import emit_event
from backend.analytics.analytics_tracker import get_counters, set_metric
from backend.telemetry.metrics import emit_counter
from backend.core.logic.utils.pii import redact_pii
from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import get_ai_client
from backend.core.logic.policy import get_precedence, precedence_version


class Rulebook(Protocol):
    """Protocol representing a rulebook with a version attribute."""

    version: str


# Regex patterns to detect admissions and their corresponding red flags and summaries
ADMISSION_PATTERNS: Tuple[Tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bmy fault\b", re.IGNORECASE),
        "admission_of_fault",
        "Creditor reports an issue; consumer requests verification.",
    ),
    (
        re.compile(r"\bi owe\b", re.IGNORECASE),
        "admission_of_debt",
        "Creditor reports a debt; consumer requests verification.",
    ),
    (
        re.compile(r"\bpagu[eé] tarde\b", re.IGNORECASE),
        "late_payment",
        "Creditor reports a late payment; consumer requests verification.",
    ),
)

_SCHEMA_PATH = Path(__file__).with_name("stage_2_5_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft7Validator(_SCHEMA)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _ai_cfg() -> Dict[str, Any]:
    return {
        "enabled": _env_bool("S2_5_ENABLE_AI_ADMISSION", False),
        "model": os.getenv("S2_5_AI_MODEL", "gpt-4o-mini"),
        "timeout_ms": _env_int("S2_5_AI_TIMEOUT_MS", 1200),
        "max_tokens": _env_int("S2_5_AI_MAX_TOKENS", 200),
        "min_chars": _env_int("S2_5_AI_MIN_CHARS", 12),
    }


AI_ALLOWED_CATEGORIES = [
    "late",
    "fault",
    "owe",
    "ownership",
    "authorization",
    "promise",
    "settlement",
    "other",
    "none",
]

_AI_CATEGORY_TO_FLAG = {
    "late": "admission_of_fault",
    "fault": "admission_of_fault",
    "owe": "admission_of_debt",
    "ownership": "admission_of_ownership",
    "authorization": "admission_of_authorization",
    "promise": "promise_to_pay",
    "settlement": "settlement_commitment",
    "other": "admission_generic",
}


def ai_admission_check(statement_masked: str, cfg: Mapping[str, Any]) -> Dict[str, Any] | None:
    start = time.perf_counter()
    emit_counter("s2_5_ai_admission_checks_total")
    try:
        ai_client = get_ai_client()
        system_prompt = (
            "You are a compliance assistant. Output JSON only. Do not fabricate facts or legal conclusions. "
            "Return exactly these keys: admission (boolean), category (string), neutral_summary (string). "
            "Categories: late, fault, owe, ownership, authorization, promise, settlement, other, none. "
            "If no admission is present, admission=false, category=\"none\", neutral_summary=\"\". "
            "Keep neutral_summary short and verification-oriented. No admissions, no promises."
        )
        user_prompt = (
            f"STATEMENT:\n{statement_masked}\n\n"
            "TASK:\n1) Does the statement include an admission of lateness, fault, owing debt, ownership,\n"
            "   authorization, promise to pay, or settlement? If yes, set admission=true and choose the best category.\n"
            "2) Provide a short neutral_summary that removes admissions and asks for verification.\n"
            "3) Output JSON only as specified.\n\n"
            "RETURN FORMAT:\n{\"admission\": <bool>, \"category\": \"<enum>\", \"neutral_summary\": \"<string>\"}"
        )
        resp = ai_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=cfg.get("model"),
            max_tokens=cfg.get("max_tokens"),
            timeout=cfg.get("timeout_ms", 0) / 1000,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        emit_counter("s2_5_ai_latency_ms", latency_ms)
        content = getattr(resp.choices[0].message, "content", "")
        data, _ = parse_json(content)
    except Exception:
        emit_counter("s2_5_ai_error_total")
        latency_ms = (time.perf_counter() - start) * 1000
        emit_counter("s2_5_ai_latency_ms", latency_ms)
        return None

    if not isinstance(data, dict):
        emit_counter("s2_5_ai_error_total")
        return None

    admission = data.get("admission")
    category = data.get("category")
    neutral_summary = data.get("neutral_summary")
    if (
        not isinstance(admission, bool)
        or not isinstance(category, str)
        or category not in AI_ALLOWED_CATEGORIES
        or not isinstance(neutral_summary, str)
    ):
        emit_counter("s2_5_ai_error_total")
        return None

    if admission:
        emit_counter("s2_5_ai_admissions_detected_total")

    return {
        "admission": admission,
        "category": category,
        "neutral_summary": neutral_summary,
    }


def _fill_defaults(data: Dict[str, Any]) -> None:
    """Populate ``data`` with default values from the schema."""

    for key, subschema in _SCHEMA.get("properties", {}).items():
        if key not in data and "default" in subschema:
            data[key] = json.loads(json.dumps(subschema["default"]))


def neutralize_admissions(statement: str, account_id: str | None = None) -> Tuple[str, list[str], bool]:
    """Return a legally safe version of ``statement``.

    Matches known admission phrases and rewrites the statement into a
    verification-focused summary. Returns the summary, any red flags detected,
    and whether a prohibited admission was present.
    """

    lowered = statement.lower()
    red_flags: list[str] = []
    summary = statement
    prohibited = False
    for pattern, flag, replacement in ADMISSION_PATTERNS:
        if pattern.search(lowered):
            red_flags.append(flag)
            summary = replacement
            prohibited = True

    if prohibited:
        emit_counter("s2_5_admissions_detected_total")
        payload = {
            "raw_statement": redact_pii(statement)[:100],
            "summary": summary,
        }
        if account_id:
            payload["account_id"] = redact_pii(str(account_id))
        emit_event("admission_neutralized", payload)

    return summary, red_flags, prohibited


def evaluate_rules(
    normalized_statement: str,
    account_facts: Dict[str, Any],
    rulebook: Rulebook,
    tri_merge: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate ``normalized_statement`` and ``account_facts`` against rules.

    ``tri_merge`` carries mismatch details that may trigger additional rules
    (e.g. presence, balance, personal info). When provided, its data is surfaced
    in the returned result and exposed to rule conditions under the
    ``tri_merge`` namespace (``tri_merge.balance``, ``tri_merge.presence``, …).
    """

    # Build accessors for rulebook data
    rules = getattr(rulebook, "rules", None)
    if rules is None and isinstance(rulebook, Mapping):
        rules = rulebook.get("rules", [])

    flags = getattr(rulebook, "flags", None)
    if flags is None and isinstance(rulebook, Mapping):
        flags = rulebook.get("flags", {})

    precedence = get_precedence(rulebook)

    exclusions = getattr(rulebook, "exclusions", None)
    if exclusions is None and isinstance(rulebook, Mapping):
        exclusions = rulebook.get("exclusions", {})

    tri_flags: Mapping[str, bool] = {
        k: True for k in (tri_merge or {}).get("mismatch_types", [])
    }

    red_flags: list[str] = []
    if "late" in normalized_statement.lower():
        red_flags.append("late_payment")

    def get_value(path: str) -> Any:
        if path == "statement" or path == "normalized_statement":
            return normalized_statement
        if path.startswith("flags."):
            target = flags
            for part in path.split(".")[1:]:
                if isinstance(target, Mapping):
                    target = target.get(part)
                else:
                    return None
            return target
        if path.startswith("tri_merge."):
            key = path.split(".", 1)[1]
            if key in tri_flags:
                return tri_flags.get(key)
            return (tri_merge or {}).get(key)
        target: Any = account_facts
        for part in path.split("."):
            if isinstance(target, Mapping):
                target = target.get(part)
            else:
                return None
        return target

    def eval_cond(cond: Mapping[str, Any]) -> bool:
        if "all" in cond:
            return all(eval_cond(c) for c in cond["all"])
        if "any" in cond:
            return any(eval_cond(c) for c in cond["any"])
        field = cond.get("field", "")
        value = get_value(field)
        if "eq" in cond:
            return value == cond["eq"]
        if "ne" in cond:
            return value != cond["ne"]
        if "lt" in cond:
            try:
                return value < cond["lt"]
            except TypeError:
                return False
        if "lte" in cond:
            try:
                return value <= cond["lte"]
            except TypeError:
                return False
        if "gt" in cond:
            try:
                return value > cond["gt"]
            except TypeError:
                return False
        if "gte" in cond:
            try:
                return value >= cond["gte"]
            except TypeError:
                return False
        return False

    triggered: Dict[str, Mapping[str, Any]] = {}
    for rule in rules or []:
        when = rule.get("when")
        if when and eval_cond(when):
            triggered[rule["id"]] = rule.get("effect", {})

    precedence_map = {rid: i for i, rid in enumerate(precedence or [])}
    sorted_hits = sorted(
        triggered.items(), key=lambda item: precedence_map.get(item[0], len(precedence_map))
    )

    final_hits: list[str] = []
    needs_evidence: list[str] = []
    suggested_dispute_frame = ""
    action_tags: list[str] = []
    suppressed: set[str] = set()
    priority_order = {"High": 3, "Medium": 2, "Low": 1}
    best_tag = ""
    best_pri = 0

    for rule_id, effect in sorted_hits:
        if rule_id in suppressed:
            emit_counter(f"rulebook.suppressed_rules.{rule_id}")
            continue
        final_hits.extend(effect.get("rule_hits", [rule_id]))
        needs_evidence.extend(effect.get("needs_evidence", []))
        if not suggested_dispute_frame and effect.get("suggested_dispute_frame"):
            suggested_dispute_frame = effect["suggested_dispute_frame"]
        tag = effect.get("action_tag")
        if tag:
            action_tags.append(tag)
            pri = priority_order.get(str(effect.get("priority", "Low")), 0)
            if pri > best_pri:
                best_pri = pri
                best_tag = tag
        for ex in effect.get("excludes", []):
            suppressed.add(ex)
        for ex in (exclusions or {}).get(rule_id, []):
            suppressed.add(ex)

    # Deduplicate while preserving order
    seen_hits: set[str] = set()
    final_hits = [x for x in final_hits if not (x in seen_hits or seen_hits.add(x))]
    seen_ev: set[str] = set()
    needs_evidence = [x for x in needs_evidence if not (x in seen_ev or seen_ev.add(x))]
    seen_tags: set[str] = set()
    action_tags = [x for x in action_tags if not (x in seen_tags or seen_tags.add(x))]

    result: Dict[str, Any] = {
        "rule_hits": final_hits,
        "needs_evidence": needs_evidence,
        "red_flags": red_flags,
        "suggested_dispute_frame": suggested_dispute_frame,
        "action_tag": best_tag or (action_tags[0] if action_tags else ""),
    }
    if tri_merge:
        result["tri_merge"] = tri_merge
    return result


def normalize_and_tag(
    account_cls: Dict[str, Any],
    account_facts: Dict[str, Any],
    rulebook: Rulebook,
    account_id: str | None = None,
) -> Dict[str, Any]:
    """Normalize user statements and tag accounts with rulebook metadata."""
    start = time.perf_counter()
    emit_counter("s2_5_accounts_total")

    raw_statement = account_cls.get("user_statement_raw") or account_facts.get(
        "user_statement_raw"
    )
    user_statement_raw = raw_statement or "No statement provided"
    legal_safe_summary, admission_flags, admission_detected = neutralize_admissions(
        user_statement_raw, account_id
    )

    cfg = _ai_cfg()
    if (
        not admission_detected
        and cfg.get("enabled")
        and raw_statement
        and len(str(raw_statement)) >= cfg.get("min_chars", 0)
    ):
        statement_masked = redact_pii(str(raw_statement))
        ai_res = ai_admission_check(statement_masked, cfg)
        emit_event(
            "admission_ai_checked",
            {
                "account_id": redact_pii(str(account_id)) if account_id else "",
                "admission": bool(ai_res and ai_res.get("admission")),
                "category": ai_res.get("category", "none") if ai_res else "none",
            },
        )
        if ai_res and ai_res.get("admission"):
            admission_detected = True
            flag = _AI_CATEGORY_TO_FLAG.get(ai_res.get("category", ""))
            if flag:
                admission_flags.append(flag)
            summary = ai_res.get("neutral_summary", "")
            if summary:
                legal_safe_summary = summary

    evaluation = evaluate_rules(
        legal_safe_summary,
        account_facts,
        rulebook,
        account_facts.get("tri_merge"),
    )
    rulebook_version = getattr(rulebook, "version", "")
    if not rulebook_version and isinstance(rulebook, Mapping):
        rulebook_version = str(rulebook.get("version", ""))

    result = evaluation.copy()
    result.update(
        {
            "legal_safe_summary": legal_safe_summary,
            "prohibited_admission_detected": admission_detected,
            "rulebook_version": rulebook_version,
            "precedence_version": precedence_version,
        }
    )
    result["red_flags"] = list(
        dict.fromkeys(result.get("red_flags", []) + admission_flags)
    )

    if result.get("action_tag"):
        emit_counter(f"rulebook.tag_selected.{result['action_tag']}")

    _fill_defaults(result)
    _VALIDATOR.validate(result)

    emit_counter("s2_5_rule_hits_total", len(result["rule_hits"]))
    emit_counter("s2_5_needs_evidence_total", len(result["needs_evidence"]))
    latency_ms = (time.perf_counter() - start) * 1000
    emit_counter("s2_5_latency_ms", latency_ms)
    counters = get_counters()
    if counters.get("s2_5_accounts_total"):
        set_metric(
            "s2_5_rule_hits_per_account",
            counters.get("s2_5_rule_hits_total", 0)
            / counters["s2_5_accounts_total"],
        )

    if account_id:
        emit_event(
            "rule_evaluated",
            {
                "account_id": redact_pii(str(account_id)),
                "rule_hits": result["rule_hits"],
                "rulebook_version": rulebook_version,
                "precedence_version": precedence_version,
            },
        )
    return result


__all__ = [
    "normalize_and_tag",
    "neutralize_admissions",
    "evaluate_rules",
    "Rulebook",
    "ValidationError",
]
