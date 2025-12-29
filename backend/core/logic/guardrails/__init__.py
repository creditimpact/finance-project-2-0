import json
import time
from typing import Any, List, Tuple

from backend.analytics.analytics_tracker import (
    log_ai_request,
    log_ai_stage,
    log_guardrail_fix,
    log_letter_without_strategy,
    log_policy_violations_prevented,
)
from backend.api.session_manager import get_session, update_session
from backend.api import config as api_config
from backend.audit.audit import emit_event
from backend.core.logic.compliance.rule_checker import RuleViolation, check_letter
from backend.core.logic.compliance.rules_loader import load_rules
from backend.core.models.letter import LetterContext
from backend.core.services.ai_client import AIClient

_INPUT_COST_PER_TOKEN = 0.01 / 1000
_OUTPUT_COST_PER_TOKEN = 0.03 / 1000


def _build_system_prompt() -> str:
    rules = load_rules()
    lines = [
        "You are a credit dispute letter generator. Follow the systemic rules provided:",
    ]
    for rule in rules:
        desc = rule.get("description")
        if desc:
            lines.append(f"- {desc}")
    lines.append("Use only neutral, factual language.")
    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()


def _record_letter(
    session_id: str,
    letter_type: str,
    text: str,
    violations: List[RuleViolation],
    iterations: int,
) -> None:
    session = get_session(session_id) or {}
    letters = session.get("letters_generated", [])
    letters.append(
        {
            "type": letter_type,
            "text": text,
            "violations": violations,
            "iterations": iterations,
        }
    )
    update_session(session_id, letters_generated=letters)


def _get_val(ctx: LetterContext | dict[str, Any], key: str) -> Any:
    if isinstance(ctx, dict):
        return ctx.get(key)
    return getattr(ctx, key, None)


def generate_letter_with_guardrails(
    user_prompt: str,
    state: str | None,
    context: LetterContext | dict[str, Any],
    session_id: str,
    letter_type: str,
    ai_client: AIClient,
) -> Tuple[str, List[RuleViolation], int]:
    """Generate a letter via LLM and ensure compliance with rule checker."""
    if (
        letter_type == "custom"
        and not (
            _get_val(context, "debt_type") and _get_val(context, "dispute_reason")
        )
    ):
        log_letter_without_strategy()
        if not api_config.ALLOW_CUSTOM_LETTERS_WITHOUT_STRATEGY:
            return "strategy_context_required", [], 0
        emit_event("strategy_applied", {"strategy_applied": False})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    iterations = 0
    text = ""
    violations: List[RuleViolation] = []
    while iterations < 2:
        iterations += 1
        start = time.perf_counter()
        response = ai_client.chat_completion(
            messages=messages,
            temperature=0.3,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)
        cost = tokens_in * _INPUT_COST_PER_TOKEN + tokens_out * _OUTPUT_COST_PER_TOKEN
        log_ai_request(tokens_in, tokens_out, cost, latency_ms)
        log_ai_stage("candidate", tokens_in + tokens_out, cost)
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()
        text, violations = check_letter(text, state, context)
        if violations:
            log_policy_violations_prevented(len(violations))
        critical = [v for v in violations if v["severity"] == "critical"]
        if not critical or iterations >= 2:
            break
        log_guardrail_fix(letter_type)
        rule_list = ", ".join(v["rule_id"] for v in critical)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": f"The draft contains violations of {rule_list}. Please fix them and return a compliant version.",
            }
        )
    _record_letter(session_id, letter_type, text, violations, iterations)
    return text, violations, iterations


def fix_draft_with_guardrails(
    draft_text: str,
    state: str | None,
    context: LetterContext | dict[str, Any],
    session_id: str,
    letter_type: str,
    ai_client: AIClient,
) -> Tuple[str, List[RuleViolation], int]:
    """Check and optionally repair an existing draft letter."""
    original_fields: dict[str, dict[str, Any]] = {}
    try:
        data = json.loads(draft_text)
        for acc in data.get("accounts", []):
            acc_id = str(acc.get("account_id", ""))
            if acc_id:
                original_fields[acc_id] = {
                    "action_tag": acc.get("action_tag"),
                    "priority": acc.get("priority"),
                    "flags": acc.get("flags"),
                }
    except Exception:
        pass
    text, violations = check_letter(draft_text, state, context)
    if violations:
        log_policy_violations_prevented(len(violations))
    if original_fields:
        idx = text.rfind("}")
        if idx != -1:
            text = text[: idx + 1]
    iterations = 1
    critical = [v for v in violations if v["severity"] == "critical"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": text},
    ]
    while critical and iterations < 2:
        log_guardrail_fix(letter_type)
        rule_list = ", ".join(v["rule_id"] for v in critical)
        messages.append(
            {
                "role": "user",
                "content": f"The draft contains violations of {rule_list}. Please fix them and return a compliant version.",
            }
        )
        start = time.perf_counter()
        response = ai_client.chat_completion(
            messages=messages,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)
        cost = tokens_in * _INPUT_COST_PER_TOKEN + tokens_out * _OUTPUT_COST_PER_TOKEN
        log_ai_request(tokens_in, tokens_out, cost, latency_ms)
        log_ai_stage("finalize", tokens_in + tokens_out, cost)
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.replace("```", "").strip()
        text, violations = check_letter(text, state, context)
        if violations:
            log_policy_violations_prevented(len(violations))
        if original_fields:
            idx = text.rfind("}")
            if idx != -1:
                text = text[: idx + 1]
        iterations += 1
        critical = [v for v in violations if v["severity"] == "critical"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": text},
        ]
    try:
        data = json.loads(text)
        for acc in data.get("accounts", []):
            acc_id = str(acc.get("account_id", ""))
            if acc_id in original_fields:
                original = original_fields[acc_id]
                for field in ("action_tag", "priority", "flags"):
                    if field in original and original[field] is not None:
                        acc[field] = original[field]
        text = json.dumps(data, indent=2)
    except Exception:
        pass
    _record_letter(session_id, letter_type, text, violations, iterations)
    return text, violations, iterations
