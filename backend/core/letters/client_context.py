from __future__ import annotations

from typing import Dict, List, Optional

import yaml

from backend.telemetry.metrics import emit_counter
from backend.assets.paths import ASSETS_ROOT
from backend.core.ai.paraphrase import paraphrase
from backend.core.logic.utils.pii import redact_pii
from backend.api.config import env_bool

_PHRASE_CACHE: Dict[str, Dict[str, str]] | None = None
_DENYLIST = ["promise to pay", "personal medical"]


def _load_phrases() -> Dict[str, Dict[str, str]]:
    global _PHRASE_CACHE
    if _PHRASE_CACHE is None:
        path = ASSETS_ROOT / "phrases" / "phrases.yaml"
        try:
            with open(path, encoding="utf-8") as f:
                _PHRASE_CACHE = yaml.safe_load(f) or {}
        except FileNotFoundError:  # pragma: no cover - missing asset
            _PHRASE_CACHE = {}
    return _PHRASE_CACHE


def choose_phrase_template(action_tag: str, policy_flags: Dict, rule_hits: List[str]) -> Dict[str, str]:
    phrases = _load_phrases()
    entry = phrases.get(action_tag, {})
    phrase = entry.get("neutral_client_sentence")
    if phrase:
        return {"key": "neutral", "template": phrase}
    return {}


def render_phrase(phrase: str, variables: Dict[str, str]) -> str:
    try:
        rendered = phrase.format(**{k: str(v) for k, v in variables.items()})
    except Exception:
        rendered = phrase
    return redact_pii(rendered)


def format_safe_client_context(
    action_tag: str,
    cleaned_client_summary: str,
    policy_findings: Dict,
    rule_hits: List[str],
) -> Optional[str]:
    if policy_findings.get("prohibited_admission_detected"):
        emit_counter(f"client_context_used.{action_tag}.neutral.policy_blocked")
        return None

    choice = choose_phrase_template(action_tag, policy_findings, rule_hits)
    phrase = choice.get("template")
    if not phrase:
        emit_counter(f"client_context_used.{action_tag}.neutral.policy_blocked")
        return None

    rendered = render_phrase(phrase, {})
    rendered = rendered.strip()
    if not rendered:
        emit_counter(f"client_context_used.{action_tag}.neutral.masked_out")
        return None
    if len(rendered) > 150:
        emit_counter(f"client_context_used.{action_tag}.neutral.too_long")
        return None
    lowered = rendered.lower()
    if any(tok in lowered for tok in _DENYLIST):
        emit_counter(f"client_context_used.{action_tag}.neutral.policy_blocked")
        return None

    if env_bool("ENABLE_CONTEXT_PARAPHRASE", False):
        paraphrased = paraphrase(rendered, banned_terms=_DENYLIST)
        if paraphrased:
            rendered = redact_pii(paraphrased)

    emit_counter(f"client_context_used.{action_tag}.neutral.ok")
    return rendered


__all__ = [
    "choose_phrase_template",
    "render_phrase",
    "format_safe_client_context",
]
