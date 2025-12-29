import logging
import re
from collections import Counter
from typing import Any, Dict, List, Mapping

from backend.core.logic.utils.json_utils import parse_json
from backend.core.logic.utils.names_normalization import BUREAUS
from backend.core.services.ai_client import AIClient
from .text_provider import load_cached_text


def extract_clean_name(full_name: str) -> str:
    parts = full_name.strip().split()
    seen = set()
    unique_parts = []
    for part in parts:
        if part.lower() not in seen:
            unique_parts.append(part)
            seen.add(part.lower())
    return " ".join(unique_parts)


def normalize_name_order(name: str) -> str:
    parts = name.strip().split()
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name


def extract_bureau_info_column_refined(
    pdf_path: str,
    ai_client: AIClient,
    client_info: dict | None = None,
    use_ai: bool = False,
    *,
    session_id: str | None = None,
) -> Mapping[str, Any]:
    bureaus = BUREAUS
    data = {b: {"name": "", "dob": "", "current_address": ""} for b in bureaus}
    discrepancies: list[str] = []

    cached = load_cached_text(session_id or "") if session_id else None
    first_page = cached["pages"][0] if cached and cached.get("pages") else ""
    raw_text = first_page
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]

    def extract_name() -> str:
        for line in lines:
            if "name" in line.lower():
                return extract_clean_name(line.split(":", 1)[-1].strip())
        return ""

    def extract_dob() -> str:
        for line in lines:
            if "dob" in line.lower() or "date of birth" in line.lower():
                m = re.search(r"(19|20)\d{2}", line)
                if m:
                    return m.group(0)
        return ""

    def extract_address() -> str:
        for idx, line in enumerate(lines):
            if "address" in line.lower():
                part = line.split(":", 1)[-1].strip()
                next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
                return f"{part} {next_line}".strip()
        return ""

    name = extract_name()
    dob = extract_dob()
    address = extract_address()

    for b in bureaus:
        data[b]["name"] = name
        data[b]["dob"] = dob
        data[b]["current_address"] = address

    for field in ["name", "dob", "current_address"]:
        field_values = [data[b][field] for b in bureaus if data[b][field]]
        if len(field_values) >= 2:
            most_common = Counter(field_values).most_common(1)[0][0]
            for b in bureaus:
                if not data[b][field] or (
                    len(data[b][field].split()) < 2
                    and data[b][field].lower() not in most_common.lower()
                ):
                    data[b][field] = most_common

    # Normalize name order only once
    for b in bureaus:
        data[b]["name"] = normalize_name_order(data[b]["name"])

    for field in ["name", "dob", "current_address"]:
        values = {b: data[b][field].strip().lower() for b in bureaus if data[b][field]}
        if len(set(values.values())) > 1:
            discrepancies.append(
                f"âš ï¸ Mismatch in {field} across bureaus:\n"
                + "\n".join([f"  - {b}: {data[b][field]}" for b in bureaus])
            )

    # Compare to client-provided info
    if client_info:
        if "legal_name" in client_info:
            extracted = data["Experian"]["name"].lower().strip()
            legal = client_info["legal_name"].lower().strip()
            if set(extracted.split()) != set(legal.split()):
                discrepancies.append(
                    f"âš ï¸ Name mismatch with ID: extracted '{extracted}' vs client '{legal}'"
                )

        if "legal_address" in client_info:
            extracted = data["Experian"]["current_address"].lower().strip()
            legal = client_info["legal_address"].lower().strip()
            if extracted != legal:
                discrepancies.append(
                    f"âš ï¸ Address mismatch with ID: extracted '{extracted}' vs client '{legal}'"
                )

    if use_ai:
        try:
            print("[ðŸ¤-] Running GPT validation for personal info...")
            prompt = f"""
You are a credit repair AI assistant.
You received the first page of a SmartCredit report. Extract the following:

- Full name (most consistent one)
- Year of birth (YYYY)
- Current full address

Output JSON in this format:
{{
  "name": "...",
  "dob": "...",
  "current_address": "..."
}}

Return strictly valid JSON: use double quotes for all property names and strings, avoid trailing commas, and include no text outside the JSON.

Here is the text:
===
{raw_text}
===
"""
            response = ai_client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            ai_data, _ = parse_json(content)
            for b in bureaus:
                data[b].update(ai_data)
        except Exception as e:
            print(f"[âš ï¸] AI info extraction failed: {str(e)}")

    return {
        "data": data["Experian"],
        "discrepancies": discrepancies,
        "raw_all_bureaus": data,
    }
