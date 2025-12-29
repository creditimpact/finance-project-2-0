from __future__ import annotations

import functools
import json
import logging
import pathlib
from typing import Any


LOGGER = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE_PATH = (HERE / "seed_argument_templates.json").resolve()


@functools.lru_cache(maxsize=1)
def load_seed_templates() -> dict:
    try:
        with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
            try:
                data: Any = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse seed argument templates JSON: {exc.msg}"
                ) from exc
    except FileNotFoundError:
        LOGGER.warning(
            "Seed argument templates file missing at %s; proceeding without seed arguments",
            TEMPLATE_PATH,
        )
        return {}

    if not isinstance(data, dict):
        raise ValueError("Seed argument templates JSON must decode to an object")

    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        raise ValueError("Seed argument templates 'templates' entry must be an object")

    for field_key, field_payload in templates.items():
        if not isinstance(field_payload, dict):
            raise ValueError(
                f"Seed argument templates for field '{field_key}' must be an object"
            )
        for code_key, template_payload in field_payload.items():
            if not isinstance(template_payload, dict):
                raise ValueError(
                    "Seed argument template entry for "
                    f"'{field_key}.{code_key}' must be an object"
                )
            text_value = template_payload.get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(
                    "Seed argument template entry for "
                    f"'{field_key}.{code_key}' must include non-empty 'text'"
                )
            tone_value = template_payload.get("tone")
            if tone_value is not None and not isinstance(tone_value, str):
                raise ValueError(
                    "Seed argument template entry for "
                    f"'{field_key}.{code_key}' must have 'tone' as a string when provided"
                )

    return templates


def build_seed_argument(field: str, c_code: str) -> dict | None:
    tpl = load_seed_templates().get(field, {}).get(c_code)
    if not tpl:
        return None
    return {
        "seed": {
            "id": f"{field}__{c_code}",
            "tone": tpl.get("tone", "firm_courteous"),
            "text": tpl["text"].strip()
        }
    }
