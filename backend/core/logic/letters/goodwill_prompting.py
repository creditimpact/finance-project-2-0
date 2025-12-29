"""Prompt construction and AI interaction for goodwill letters."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from backend.audit.audit import AuditLevel, AuditLogger
from backend.core.logic.utils.json_utils import parse_json
from backend.core.logic.utils.pdf_ops import gather_supporting_docs
from backend.core.services.ai_client import AIClient


def generate_goodwill_letter_draft(
    client_name: str,
    creditor: str,
    account_summaries: List[Dict[str, Any]],
    tone: str = "neutral",
    session_id: str | None = None,
    *,
    ai_client: AIClient,
    audit: AuditLogger | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Call ``ai_client`` to create a goodwill letter draft.

    Returns a tuple ``(data, doc_names)`` where ``data`` is the parsed JSON
    response and ``doc_names`` lists any supplemental document filenames
    included in the prompt.
    """

    docs_text, doc_names, _ = gather_supporting_docs(session_id or "")
    if docs_text:
        if audit and audit.level is AuditLevel.VERBOSE:
            print(
                f"[📎] Including supplemental docs for goodwill letter to {creditor}."
            )
        docs_section = (
            "\nThe following additional documents were provided by the client:\n"
            + docs_text
        )
    else:
        docs_section = ""

    prompt = f"""
Write a goodwill adjustment letter for credit reporting purposes. Write it **in the first person**, in a {tone} tone as if the client wrote it.
For each account below, craft a short story-style paragraph that blends the provided neutral_phrase with the client's structured_summary and any hardship and recovery details. Use the neutral_phrase as the legal/tone base while personalizing with the client's explanation. Do not copy the phrase or summary verbatim. Mention supporting documents by name when helpful.

Include these fields in the JSON response:
- intro_paragraph: opening lines
- hardship_paragraph: brief explanation of the hardship
- recovery_paragraph: how things improved
- accounts: list of {{name, account_number, status, paragraph}}
- closing_paragraph: polite request for goodwill adjustment

Creditor: {creditor}
Client name: {client_name}
Accounts: {json.dumps(account_summaries, indent=2)}
Supporting doc names: {', '.join(doc_names) if doc_names else 'None'}
{docs_section}

Return strictly valid JSON: all property names and strings in double quotes, no trailing commas or comments, and no text outside the JSON.
"""

    response = ai_client.chat_completion(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "goodwill_letter_prompt",
            {"creditor": creditor, "prompt": prompt, "accounts": account_summaries},
        )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    if audit and audit.level is AuditLevel.VERBOSE:
        print("\n----- GPT RAW RESPONSE -----")
        print(content)
        print("----- END RESPONSE -----\n")

    result, _ = parse_json(content)
    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "goodwill_letter_response",
            {"creditor": creditor, "response": result},
        )
    return result or {}, doc_names


__all__ = ["generate_goodwill_letter_draft"]
