"""Deterministic polarity classifier for bureau field values."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import ast
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Literal, Sequence

import yaml


Polarity = Literal["good", "bad", "neutral", "unknown"]
Severity = Literal["low", "medium", "high"]

logger = logging.getLogger(__name__)

_ALLOWED_POLARITIES: set[str] = {"good", "bad", "neutral", "unknown"}
_ALLOWED_SEVERITIES: set[str] = {"low", "medium", "high"}

_POLARITY_CONFIG_PATH: Path = Path(__file__).with_name("polarity_config.yml")
_CONFIG_CACHE: tuple[float | None, Dict[str, Any]] | None = None


def load_polarity_config() -> Dict[str, Any]:
    """Load and cache the polarity configuration YAML file."""

    global _CONFIG_CACHE

    path = _POLARITY_CONFIG_PATH

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        logger.warning("POLARITY_CONFIG_NOT_FOUND path=%s", path)
        _CONFIG_CACHE = (None, {})
        return {}

    mtime = stat_result.st_mtime
    if _CONFIG_CACHE is not None and _CONFIG_CACHE[0] == mtime:
        return _CONFIG_CACHE[1]

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("POLARITY_CONFIG_READ_FAILED path=%s", path)
        _CONFIG_CACHE = (mtime, {})
        return {}

    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        logger.exception("POLARITY_CONFIG_PARSE_FAILED path=%s", path)
        data = {}

    if not isinstance(data, dict):
        data = {}

    _CONFIG_CACHE = (mtime, data)
    return data


def parse_money(raw: Any) -> Optional[float]:
    """Parse a currency value into a float (cents-aware)."""

    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    text = text.replace("$", "").replace(",", "")
    text = text.replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    text = text.replace("--", "")
    if not text or text in {"-", "-0", "0-"}:
        return None

    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        logger.debug("POLARITY_PARSE_MONEY_FAILED value=%r normalized=%r", raw, text)
        return None

    return float(decimal_value)


def is_blank(value: Any) -> bool:
    """Return True when ``value`` should be treated as blank."""

    if value is None:
        return True

    if isinstance(value, (int, float)):
        return False

    text = str(value).strip()
    return text == "" or text == "--"


def norm_text(value: Any) -> str:
    """Normalize ``value`` for keyword matching (casefold + collapse spaces)."""

    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    else:
        text = str(value)

    collapsed = " ".join(text.split())
    return collapsed.casefold()


def _normalize_polarity(candidate: Any) -> Polarity:
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in _ALLOWED_POLARITIES:
            return lowered  # type: ignore[return-value]
    return "unknown"


def _normalize_severity(candidate: Any) -> Severity:
    if isinstance(candidate, str):
        lowered = candidate.strip().lower()
        if lowered in _ALLOWED_SEVERITIES:
            return lowered  # type: ignore[return-value]
    return "low"


def _safe_eval_boolean(expression: str, variables: Mapping[str, Any]) -> bool:
    """Evaluate ``expression`` using a whitelist of AST nodes."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        logger.debug("POLARITY_RULE_SYNTAX_ERROR expression=%r", expression)
        return False

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError("unsupported bool op")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(_eval(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if isinstance(operator, ast.Eq) and not (left == right):
                    return False
                elif isinstance(operator, ast.NotEq) and not (left != right):
                    return False
                elif isinstance(operator, ast.Gt) and not (left > right):
                    return False
                elif isinstance(operator, ast.GtE) and not (left >= right):
                    return False
                elif isinstance(operator, ast.Lt) and not (left < right):
                    return False
                elif isinstance(operator, ast.LtE) and not (left <= right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            identifier = node.id.lower()
            if identifier == "true":
                return True
            if identifier == "false":
                return False
            if identifier in variables:
                return variables[identifier]
            raise ValueError(f"unknown identifier {node.id!r}")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):  # pragma: no cover (legacy py compat)
            return node.n
        if isinstance(node, ast.Str):  # pragma: no cover
            return node.s
        raise ValueError(f"unsupported expression: {expression!r}")

    try:
        result = _eval(parsed)
    except Exception:
        logger.debug("POLARITY_RULE_EVAL_ERROR expression=%r", expression, exc_info=True)
        return False

    return bool(result)


def _build_result(
    polarity: Polarity,
    severity: Severity,
    *,
    value_norm: Any = None,
    rule_hit: str | None = None,
    reason: str | None = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "polarity": polarity,
        "severity": severity,
        "value_norm": value_norm,
        "rule_hit": rule_hit,
        "reason": reason,
        "evidence": evidence or {},
    }
    return result


def _evaluate_money(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    parsed_value = parse_money(raw_value)
    evidence: Dict[str, Any] = {"parsed": parsed_value}

    if parsed_value is None:
        return _build_result(
            "unknown",
            "low",
            value_norm=None,
            rule_hit=None,
            reason="value missing or invalid",
            evidence=evidence,
        )

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            condition = rule.get("if")
            if not isinstance(condition, str) or not condition.strip():
                continue
            if _safe_eval_boolean(condition, {"value": parsed_value}):
                evidence["matched_rule"] = condition
                polarity = _normalize_polarity(rule.get("polarity"))
                severity = _normalize_severity(rule.get("severity"))
                rule_name = rule.get("name")
                rule_hit = str(rule_name) if isinstance(rule_name, str) else condition
                reason_value = rule.get("reason")
                if isinstance(reason_value, str) and reason_value.strip():
                    reason = reason_value.strip()
                else:
                    reason = f"condition \"{condition}\" matched"
                return _build_result(
                    polarity,
                    severity,
                    value_norm=parsed_value,
                    rule_hit=rule_hit,
                    reason=reason,
                    evidence=evidence,
                )

    return _build_result(
        "unknown",
        "low",
        value_norm=parsed_value,
        rule_hit=None,
        reason="no rule matched",
        evidence=evidence,
    )


def _evaluate_date(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    present = not is_blank(raw_value)
    evidence: Dict[str, Any] = {"parsed": present}

    rules = field_cfg.get("rules")
    if isinstance(rules, Iterable):
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            condition = rule.get("if")
            if not isinstance(condition, str) or not condition.strip():
                continue
            if _safe_eval_boolean(condition, {"is_present": present}):
                evidence["matched_rule"] = condition
                polarity = _normalize_polarity(rule.get("polarity"))
                severity = _normalize_severity(rule.get("severity"))
                rule_name = rule.get("name")
                rule_hit = str(rule_name) if isinstance(rule_name, str) else condition
                reason_value = rule.get("reason")
                if isinstance(reason_value, str) and reason_value.strip():
                    reason = reason_value.strip()
                else:
                    reason = f"condition \"{condition}\" matched"
                return _build_result(
                    polarity,
                    severity,
                    value_norm=present,
                    rule_hit=rule_hit,
                    reason=reason,
                    evidence=evidence,
                )

    polarity = "neutral" if present else "unknown"
    reason = "date present" if present else "date missing"
    return _build_result(
        polarity,
        "low",
        value_norm=present,
        rule_hit=None,
        reason=reason,
        evidence=evidence,
    )


def _match_keyword(
    normalized_value: str, raw_value: Any, keywords: Iterable[Any]
) -> Optional[Dict[str, Any]]:
    for keyword in keywords:
        if isinstance(keyword, Mapping):
            rule_id = keyword.get("id")
            if not isinstance(rule_id, str) or not rule_id.strip():
                continue

            match_on = str(keyword.get("match_on") or "normalized").strip().lower()
            if match_on == "raw":
                if raw_value is None:
                    target_value = ""
                elif isinstance(raw_value, str):
                    target_value = raw_value
                else:
                    target_value = str(raw_value)
            else:
                target_value = normalized_value

            pattern = keyword.get("pattern")
            if isinstance(pattern, str):
                try:
                    match = re.search(pattern, target_value, flags=re.IGNORECASE)
                except re.error:
                    logger.debug(
                        "POLARITY_INVALID_REGEX pattern=%r rule_id=%s", pattern, rule_id
                    )
                else:
                    if match:
                        return {
                            "rule_hit": rule_id.strip(),
                            "matched": match.group(0),
                            "pattern": pattern,
                        }

            value = keyword.get("value")
            if isinstance(value, str):
                normalized_keyword = norm_text(value)
                if normalized_keyword and normalized_keyword in normalized_value:
                    return {
                        "rule_hit": rule_id.strip(),
                        "matched": value,
                    }
            continue

        if isinstance(keyword, str):
            normalized_keyword = norm_text(keyword)
            if normalized_keyword and normalized_keyword in normalized_value:
                return {"rule_hit": keyword, "matched": keyword}
    return None


def _evaluate_text(field_cfg: Mapping[str, Any], raw_value: Any) -> Dict[str, Any]:
    normalized_value = norm_text(raw_value)
    evidence: Dict[str, Any] = {"parsed": normalized_value or None}

    weights = field_cfg.get("weights")
    weight_map = weights if isinstance(weights, Mapping) else {}

    for category in ("bad", "good", "neutral"):
        keywords = field_cfg.get(f"{category}_keywords")
        if isinstance(keywords, Iterable):
            match_info = _match_keyword(normalized_value, raw_value, keywords)
            if match_info:
                matched_keyword = str(match_info.get("matched") or "").strip()
                evidence["matched_keyword"] = matched_keyword or None
                pattern = match_info.get("pattern")
                if isinstance(pattern, str):
                    evidence["matched_pattern"] = pattern
                polarity = _normalize_polarity(category)
                severity = _normalize_severity(weight_map.get(category))
                keyword_label = matched_keyword or str(match_info.get("rule_hit"))
                reason = f"matched {category} keyword '{keyword_label}'"
                return _build_result(
                    polarity,
                    severity,
                    value_norm=normalized_value or None,
                    rule_hit=(
                        str(match_info.get("rule_hit"))
                        if match_info.get("rule_hit") is not None
                        else None
                    ),
                    reason=reason,
                    evidence=evidence,
                )

    default_polarity = _normalize_polarity(field_cfg.get("default"))
    severity = _normalize_severity(weight_map.get(default_polarity))
    reason = f"default polarity '{default_polarity}'"
    return _build_result(
        default_polarity,
        severity,
        value_norm=normalized_value or None,
        rule_hit=None,
        reason=reason,
        evidence=evidence,
    )


def classify_field_value(field: str, raw_value: Optional[str | int | float]) -> Dict[str, Any]:
    """Classify a field value using the polarity configuration."""

    config = load_polarity_config()
    fields_cfg = config.get("fields")
    if not isinstance(fields_cfg, Mapping):
        fields_cfg = {}

    field_cfg = fields_cfg.get(field)
    if not isinstance(field_cfg, Mapping):
        return _build_result(
            "unknown",
            "low",
            value_norm=None,
            rule_hit=None,
            reason="field not configured",
            evidence={"parsed": None},
        )

    field_type = str(field_cfg.get("type") or "").strip().lower()
    if field_type == "money":
        return _evaluate_money(field_cfg, raw_value)
    if field_type == "date":
        return _evaluate_date(field_cfg, raw_value)
    if field_type == "text":
        return _evaluate_text(field_cfg, raw_value)

    logger.debug("POLARITY_UNKNOWN_FIELD_TYPE field=%s type=%r", field, field_type)
    return _build_result(
        "unknown",
        "low",
        value_norm=None,
        rule_hit=None,
        reason="unsupported field type",
        evidence={"parsed": None},
    )


@dataclass(frozen=True)
class ApplyPolarityResult:
    processed_accounts: int
    updated_accounts: list[int]
    config_digest: Optional[str] = None


def _compute_config_digest() -> Optional[str]:
    try:
        raw = _POLARITY_CONFIG_PATH.read_bytes()
    except FileNotFoundError:
        logger.warning("POLARITY_CONFIG_NOT_FOUND_DIGEST path=%s", _POLARITY_CONFIG_PATH)
        return None
    except OSError:  # pragma: no cover - defensive logging
        logger.debug(
            "POLARITY_CONFIG_DIGEST_FAILED path=%s",
            _POLARITY_CONFIG_PATH,
            exc_info=True,
        )
        return None

    return hashlib.sha256(raw).hexdigest()


def _load_existing_polarity_block(account_path: Path) -> Optional[Mapping[str, Any]]:
    summary_path = account_path / "summary.json"
    try:
        raw = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        logger.debug(
            "POLARITY_EXISTING_SUMMARY_FAILED path=%s",
            summary_path,
            exc_info=True,
        )
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:  # pragma: no cover - defensive logging
        logger.debug(
            "POLARITY_EXISTING_SUMMARY_INVALID path=%s",
            summary_path,
            exc_info=True,
        )
        return None

    if not isinstance(data, Mapping):
        return None

    block = data.get("polarity_check")
    if isinstance(block, Mapping):
        return dict(block)
    return None


def apply_polarity_checks(
    accounts_dir: Path | str,
    indices: Sequence[object],
    *,
    sid: Optional[str] = None,
) -> ApplyPolarityResult:
    base_path = Path(accounts_dir)
    unique_indices: list[int] = []
    seen: set[int] = set()
    for value in indices:
        try:
            idx = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)

    unique_indices.sort()

    processed = 0
    updated: list[int] = []
    sid_value = sid or ""

    for idx in unique_indices:
        account_path = base_path / f"{idx}"
        if not account_path.exists():
            continue

        processed += 1
        before_block = _load_existing_polarity_block(account_path)

        try:
            from backend.core.logic.intra_polarity import analyze_account_polarity

            result_block = analyze_account_polarity(sid_value, account_path)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "POLARITY_APPLY_FAILED sid=%s account_dir=%s",
                sid_value,
                account_path,
            )
            continue

        if before_block != result_block:
            updated.append(idx)

    digest = _compute_config_digest()
    return ApplyPolarityResult(processed_accounts=processed, updated_accounts=updated, config_digest=digest)


__all__ = [
    "ApplyPolarityResult",
    "Polarity",
    "Severity",
    "apply_polarity_checks",
    "classify_field_value",
    "is_blank",
    "load_polarity_config",
    "norm_text",
    "parse_money",
]

