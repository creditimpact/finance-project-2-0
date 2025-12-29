import json
import os
from typing import Any, Mapping, MutableMapping, Sequence

from backend.core.config import ENABLE_VALIDATION_REQUIREMENTS, VALIDATION_DEBUG
from backend.core.logic.validation_requirements import (
    build_validation_requirements_for_account,
)


def run(account_dir: str) -> dict:
    """
    Runs the validation-requirements builder on a single account folder.
    Writes back into summary.json (under 'validation_requirements').
    Idempotent: safe to run multiple times.
    """
    if not ENABLE_VALIDATION_REQUIREMENTS:
        return {"skipped": True, "reason": "flag_off"}

    bureaus_json = os.path.join(account_dir, "bureaus.json")
    summary_json = os.path.join(account_dir, "summary.json")
    if not (os.path.isfile(bureaus_json) and os.path.isfile(summary_json)):
        return {"skipped": True, "reason": "missing_inputs"}

    result = build_validation_requirements_for_account(account_dir) or {}

    try:
        with open(summary_json, "r+", encoding="utf-8") as f:
            summary = json.load(f)
            block = result.get("validation_requirements")
            if isinstance(block, Mapping):
                summary["validation_requirements"] = dict(block)
            else:
                summary["validation_requirements"] = {
                    "schema_version": 3,
                    "findings": [],
                }

            _aggregate_seed_arguments(summary)
            if not VALIDATION_DEBUG and "validation_debug" in summary:
                summary.pop("validation_debug", None)
            f.seek(0)
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.truncate()
    except Exception as exc:  # pragma: no cover - defensive logging path
        return {"skipped": True, "reason": f"write_error:{exc}"}

    return {
        "skipped": False,
        "findings_count": int(result.get("count") or 0),
    }


def _aggregate_seed_arguments(summary: MutableMapping[str, Any]) -> None:
    """Populate top-level seed arguments derived from validation findings."""

    block = summary.get("validation_requirements")
    if not isinstance(block, Mapping):
        return

    findings = block.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes, bytearray)):
        findings = []

    seeds: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        argument_block = finding.get("argument")
        if not isinstance(argument_block, Mapping):
            continue
        seed_entry = argument_block.get("seed")
        if not isinstance(seed_entry, Mapping):
            continue

        seed_id_raw = seed_entry.get("id")
        seed_id = str(seed_id_raw).strip() if seed_id_raw is not None else ""
        if not seed_id:
            continue

        tone_value = seed_entry.get("tone")
        tone = str(tone_value).strip() if tone_value is not None else "firm_courteous"
        if not tone:
            tone = "firm_courteous"

        text_value = seed_entry.get("text")
        text = str(text_value or "").strip()

        seeds.append({
            "id": seed_id,
            "tone": tone,
            "text": text,
        })

    seen: set[str] = set()
    seeds_dedup: list[dict[str, Any]] = []
    for seed in seeds:
        sid = seed["id"]
        if sid in seen:
            continue
        seeds_dedup.append(seed)
        seen.add(sid)

    arguments_block = summary.get("arguments")
    if not isinstance(arguments_block, MutableMapping):
        arguments_block = {}
        summary["arguments"] = arguments_block

    arguments_block["seeds"] = seeds_dedup
    arguments_block.setdefault("composites", [])
