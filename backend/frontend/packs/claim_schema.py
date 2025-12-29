from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ClaimSchemaEntry:
    key: str
    title: str
    description: str | None
    requires: tuple[str, ...]
    optional: tuple[str, ...]
    auto_attach: tuple[str, ...]
    min_uploads: int | None


@dataclass(frozen=True, slots=True)
class IssueSchemaEntry:
    issue: str
    claims: tuple[ClaimSchemaEntry, ...]


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "shared" / "claims_schema.json"


@lru_cache(maxsize=1)
def load_claims_schema() -> dict[str, Any]:
    path = _schema_path()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise ValueError("claims schema must be a mapping")
    return payload


@lru_cache(maxsize=1)
def _compiled_schema() -> tuple[list[str], dict[str, IssueSchemaEntry], dict[str, ClaimSchemaEntry]]:
    data = load_claims_schema()
    auto_attach_base = [
        entry for entry in data.get("autoAttachBase", []) if isinstance(entry, str)
    ]

    issue_map: dict[str, IssueSchemaEntry] = {}
    claim_map: dict[str, ClaimSchemaEntry] = {}

    issues = data.get("byIssue", [])
    if not isinstance(issues, list):  # pragma: no cover - defensive
        issues = []

    for issue_entry in issues:
        if not isinstance(issue_entry, dict):
            continue
        issue_key = issue_entry.get("issue")
        if not isinstance(issue_key, str) or not issue_key:
            continue

        claims_payload = issue_entry.get("claims", [])
        if not isinstance(claims_payload, list):
            continue

        compiled_claims: list[ClaimSchemaEntry] = []
        for claim_entry in claims_payload:
            if not isinstance(claim_entry, dict):
                continue
            claim_key = claim_entry.get("key")
            title = claim_entry.get("title")
            if not isinstance(claim_key, str) or not claim_key:
                continue
            if not isinstance(title, str) or not title:
                continue

            description = claim_entry.get("description")
            if not isinstance(description, str):
                description = None

            requires = tuple(
                sorted({doc for doc in claim_entry.get("requires", []) if isinstance(doc, str)})
            )
            optional = tuple(
                sorted({doc for doc in claim_entry.get("optional", []) if isinstance(doc, str)})
            )
            auto_attach = tuple(
                sorted({doc for doc in claim_entry.get("autoAttach", []) if isinstance(doc, str)})
            )
            min_uploads_value = claim_entry.get("minUploads")
            min_uploads = (
                int(min_uploads_value)
                if isinstance(min_uploads_value, int) and min_uploads_value >= 0
                else None
            )

            compiled = ClaimSchemaEntry(
                key=claim_key,
                title=title,
                description=description,
                requires=requires,
                optional=optional,
                auto_attach=auto_attach,
                min_uploads=min_uploads,
            )
            compiled_claims.append(compiled)
            if claim_key not in claim_map:
                claim_map[claim_key] = compiled

        issue_map[issue_key] = IssueSchemaEntry(
            issue=issue_key,
            claims=tuple(compiled_claims),
        )

    return auto_attach_base, issue_map, claim_map


def auto_attach_base() -> tuple[str, ...]:
    base, _, _ = _compiled_schema()
    return tuple(base)


def iter_issue_entries() -> Iterable[IssueSchemaEntry]:
    _, issue_map, _ = _compiled_schema()
    return issue_map.values()


def get_issue_entry(issue: str | None) -> IssueSchemaEntry | None:
    _, issue_map, _ = _compiled_schema()
    if not issue:
        return None
    return issue_map.get(issue)


def get_claim_entry(claim_key: str | None) -> ClaimSchemaEntry | None:
    _, _, claim_map = _compiled_schema()
    if not claim_key:
        return None
    return claim_map.get(claim_key)


def resolve_issue_claims(issue: str | None) -> tuple[str, tuple[ClaimSchemaEntry, ...]]:
    base = auto_attach_base()
    issue_entry = get_issue_entry(issue)
    if issue_entry is not None and issue_entry.claims:
        return base, issue_entry.claims

    generic_entry = get_issue_entry("generic")
    if generic_entry is not None and generic_entry.claims:
        return base, generic_entry.claims

    # Fallback to an empty tuple if schema is missing generic entry.
    return base, tuple()


def all_claim_keys() -> set[str]:
    _, _, claim_map = _compiled_schema()
    return set(claim_map.keys())


def all_doc_keys_for_claim(claim: ClaimSchemaEntry) -> set[str]:
    keys = set(claim.requires)
    keys.update(claim.optional)
    keys.update(claim.auto_attach)
    return keys


def all_doc_keys() -> set[str]:
    base, issue_map, _ = _compiled_schema()
    keys: set[str] = set(base)
    for issue_entry in issue_map.values():
        for claim in issue_entry.claims:
            keys.update(all_doc_keys_for_claim(claim))
    return keys
