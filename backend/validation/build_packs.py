"""Utilities for constructing Validation AI packs from prepared cases."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.ai.note_style_reader import get_style_metadata
from backend.core.ai.paths import (
    validation_pack_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)
from backend.validation.index_schema import (
    ValidationPackRecord,
    build_validation_index,
)


_SYSTEM_PROMPT = (
    "Project: credit-analyzer\n"
    "Module: Validation / AI Adjudication\n\n"
    "The validation system detects discrepancies between credit bureaus’ reported values for each field (Equifax, Experian, TransUnion).\n"
    "It doesn’t understand language or legal reasoning — that’s why findings are sent to the AI adjudicator.\n\n"
    "Your role is to determine whether the bureau statements conflict in a way that creates usable evidence for a consumer dispute.\n"
    "You are not deciding which bureau is correct; you are deciding whether the discrepancy is strong enough that a bureau would be obligated or reasonably expected to investigate or correct it if disputed.\n"
    "Assume the consumer asserts the version that is most favorable to them.\n"
    "The detection platform already confirmed the numerical/text mismatch — focus only on the linguistic and legal significance.\n\n"
    "Each pack represents one field (e.g., account_type, payment_status, etc.) and includes the normalized/raw values reported by each bureau.\n"
    "Evaluate these values exactly as written and produce a JSON decision that aligns with the expected schema."
)

_PROMPT_USER_TEMPLATE = (
    "Evaluate the following tri-bureau field finding and answer the core question: Is this discrepancy strong enough that a credit bureau would be obligated or reasonably expected to investigate or correct it if disputed?\n"
    "The detection system already confirmed the mismatch — concentrate on legal/linguistic significance, not detection mechanics.\n"
    "Assume the consumer claims the more favorable version is correct.\n\n"
    "Instructions:\n"
    "1) Choose one decision: strong_actionable | supportive_needs_companion | neutral_context_only | no_case.\n"
    "   Base your choice on the legal and material significance of the bureau statements, not on detection mechanics.\n"
    "2) Populate the 'checks' object:\n"
    "   - materiality: true/false\n"
    "   - supports_consumer: true/false (true when the mismatch favors the consumer’s asserted, more beneficial version)\n"
    "   - doc_requirements_met: true/false (use finding.documents; assume true when the list is non-empty)\n"
    "   - mismatch_code: copy finding.reason_code exactly (e.g., \"C4_TWO_MATCH_ONE_DIFF\")\n"
    "3) Provide a rationale (≤120 words) that explains whether the bureau narratives conflict in a legally meaningful way. Reference the mismatch code and rely only on the provided text.\n"
    "4) List citations for the exact bureau:value pairs you used (e.g., \"equifax: paid charge-off\").\n\n"
    "Constraints:\n"
    "- Use only the provided normalized/raw values; do not speculate beyond the pack.\n"
    "- Do not simply restate that a mismatch exists—focus on its legal impact.\n"
    "- Minor wording differences that keep the same meaning (e.g., \"real estate mortgage\" vs \"conventional real estate mortgage\" or \"auto loan\" vs \"vehicle loan\") usually stay neutral_context_only.\n"
    "- Clear categorical conflicts (e.g., \"secured loan\" vs \"unsecured loan\", \"charged-off\" vs \"paid as agreed\") should be treated as strong_actionable.\n"
    "- Differences like \"mortgage\" vs \"HELOC\" can be supportive_needs_companion when they suggest a different product but need corroboration.\n"
    "- Output strictly valid JSON matching the expected_output schema.\n\n"
    "Finding (verbatim JSON):\n"
    "<finding blob here>\n\n"
    "Return JSON only.\n"
    "If you cannot produce a valid object, return:\n"
    '{"decision":"no_case","rationale":"schema_mismatch","citations":["system:none"],"checks":{"materiality":false,"supports_consumer":false,"doc_requirements_met":false,"mismatch_code":"unknown"}}'
)

_EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["decision", "rationale", "citations", "checks"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "strong_actionable",
                "supportive_needs_companion",
                "neutral_context_only",
                "no_case",
            ],
        },
        "rationale": {"type": "string", "maxLength": 700},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "checks": {
            "type": "object",
            "required": [
                "materiality",
                "supports_consumer",
                "doc_requirements_met",
                "mismatch_code",
            ],
            "properties": {
                "materiality": {"type": "boolean"},
                "supports_consumer": {"type": "boolean"},
                "doc_requirements_met": {"type": "boolean"},
                "mismatch_code": {"type": "string"},
            },
        },
    },
}
_BUREAUS = ("transunion", "experian", "equifax")

_ELIGIBLE_FIELDS = frozenset(
    {
        "account_type",
        "creditor_type",
        "account_rating",
        "two_year_payment_history",
    }
)


_PACK_MAX_SIZE_ENV = "VALIDATION_PACK_MAX_SIZE_KB"


log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _bytes_to_kb(size_bytes: int) -> float:
    return size_bytes / 1024


def _pack_max_size_kb() -> float | None:
    raw = os.getenv(_PACK_MAX_SIZE_ENV)
    if raw is None:
        return None

    if isinstance(raw, str):
        raw_value = raw.strip()
    else:
        raw_value = str(raw)

    if not raw_value:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        log.warning(
            "VALIDATION_PACK_MAX_SIZE_INVALID env_value=%r", raw,
        )
        return None

    if value <= 0:
        return None

    return value


@dataclass
class _PackSizeStats:
    count: int = 0
    total_bytes: int = 0
    max_bytes: int = 0

    def observe(self, size_bytes: int) -> None:
        self.count += 1
        self.total_bytes += max(size_bytes, 0)
        if size_bytes > self.max_bytes:
            self.max_bytes = size_bytes

    def average_bytes(self) -> float:
        if not self.count:
            return 0.0
        return self.total_bytes / self.count

    def average_kb(self) -> float:
        return self.average_bytes() / 1024

    def max_kb(self) -> float:
        return self.max_bytes / 1024


@dataclass(frozen=True)
class ManifestPaths:
    """Resolved locations for validation pack inputs and outputs."""

    sid: str
    accounts_dir: Path
    packs_dir: Path
    results_dir: Path
    index_path: Path
    log_path: Path


class ValidationPackBuilder:
    """Build per-account Validation AI pack payloads."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.paths = self._resolve_manifest_paths(manifest)
        self._pack_max_size_kb = _pack_max_size_kb()
        self._pack_max_size_bytes = (
            int(self._pack_max_size_kb * 1024) if self._pack_max_size_kb is not None else None
        )
        self._runs_root = self._infer_runs_root()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> list[dict[str, Any]]:
        """Generate packs for every account referenced in the manifest."""

        records: list[ValidationPackRecord] = []
        serialized_records: list[dict[str, Any]] = []
        stats = {
            "total_accounts": 0,
            "written_accounts": 0,
            "skipped_accounts": 0,
            "errors": 0,
        }
        field_counts: Counter[str] = Counter()
        size_stats = _PackSizeStats()
        for account_id, account_dir in self._iter_accounts():
            stats["total_accounts"] += 1
            pack_path: Path | None = None
            try:
                payloads, metadata = self._build_account_pack(account_id, account_dir)
                skip_reason = (
                    metadata.get("skip_reason") if isinstance(metadata, Mapping) else None
                )
                if not payloads:
                    self._log(
                        "pack_skipped",
                        account_id=f"{account_id:03d}",
                        reason=skip_reason or "no_payloads",
                    )
                    stats["skipped_accounts"] += 1
                    continue

                serialized, size_bytes = self._serialize_payloads(payloads)
                if (
                    payloads
                    and self._pack_max_size_bytes is not None
                    and size_bytes > self._pack_max_size_bytes
                ):
                    size_kb = _bytes_to_kb(size_bytes)
                    self._log(
                        "pack_blocked_max_size",
                        account_id=f"{account_id:03d}",
                        fields=len(payloads),
                        size_bytes=size_bytes,
                        size_kb=size_kb,
                        max_size_kb=self._pack_max_size_kb,
                    )
                    log.warning(
                        "VALIDATION_PACK_SIZE_BLOCKED sid=%s account_id=%03d size_bytes=%d max_kb=%s",
                        self.paths.sid,
                        account_id,
                        size_bytes,
                        self._pack_max_size_kb,
                    )
                    stats["skipped_accounts"] += 1
                    continue

                pack_path = self._write_pack(
                    account_id,
                    serialized,
                    field_count=len(payloads),
                    size_bytes=size_bytes,
                )
                record = self._build_index_record(account_id, pack_path, payloads, metadata)
                records.append(record)
                serialized_records.append(record.to_json_payload())
                stats["written_accounts"] += 1

                for line in payloads:
                    if not isinstance(line, Mapping):
                        continue
                    field_value = line.get("field")
                    field_key = str(field_value).strip() if field_value is not None else ""
                    if field_key:
                        field_counts[field_key] += 1
                size_stats.observe(size_bytes)
            except Exception:  # pragma: no cover - defensive logging
                stats["errors"] += 1
                log.exception(
                    "VALIDATION_PACK_BUILD_FAILED sid=%s account_id=%s pack=%s account_dir=%s",
                    self.paths.sid,
                    f"{account_id:03d}" if isinstance(account_id, int) else account_id,
                    pack_path,
                    account_dir,
                )
                continue

        avg_kb = size_stats.average_kb()
        max_kb = size_stats.max_kb()
        log.info(
            "VALIDATION_PACK_BUILD_SUMMARY sid=%s total=%d written=%d skipped=%d errors=%d avg_kb=%.3f max_kb=%.3f",
            self.paths.sid,
            stats["total_accounts"],
            stats["written_accounts"],
            stats["skipped_accounts"],
            stats["errors"],
            avg_kb,
            max_kb,
        )

        self._log(
            "pack_build_summary",
            total_accounts=stats["total_accounts"],
            written_accounts=stats["written_accounts"],
            skipped_accounts=stats["skipped_accounts"],
            errors=stats["errors"],
            average_size_bytes=size_stats.average_bytes(),
            average_size_kb=avg_kb,
            max_size_bytes=size_stats.max_bytes,
            max_size_kb=max_kb,
            field_counts=dict(field_counts),
        )

        self._write_index(records)
        return serialized_records

    # ------------------------------------------------------------------
    # Account pack construction
    # ------------------------------------------------------------------
    def _iter_accounts(self) -> Iterable[tuple[int, Path]]:
        accounts_dir = self.paths.accounts_dir
        if not accounts_dir.is_dir():
            return []

        for child in sorted(accounts_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                account_id = int(child.name)
            except (TypeError, ValueError):
                continue
            yield account_id, child

    def _build_account_pack(
        self, account_id: int, account_dir: Path
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        summary = self._read_json(account_dir / "summary.json")
        if not isinstance(summary, Mapping):
            return [], {"skip_reason": "missing_summary"}

        validation_block = summary.get("validation_requirements")
        if not isinstance(validation_block, Mapping):
            return [], {"skip_reason": "missing_validation_requirements"}

        legacy_requirements = validation_block.get("requirements")
        if legacy_requirements:
            if isinstance(legacy_requirements, Sequence) and not isinstance(
                legacy_requirements, (str, bytes, bytearray)
            ):
                legacy_count = len(legacy_requirements)
            else:
                legacy_count = 1
            self._log(
                "legacy_requirements_ignored",
                account_id=f"{account_id:03d}",
                legacy_count=legacy_count,
            )

        findings = validation_block.get("findings")
        if not isinstance(findings, Sequence):
            return [], {"skip_reason": "missing_findings"}

        field_consistency = validation_block.get("field_consistency")
        if not isinstance(field_consistency, Mapping):
            field_consistency = {}

        send_to_ai_map = self._build_send_to_ai_map(findings)

        bureaus = self._read_json(account_dir / "bureaus.json")
        bureaus_map: Mapping[str, Mapping[str, Any]]
        if isinstance(bureaus, Mapping):
            normalized_bureaus: dict[str, dict[str, Any]] = {}
            for name, value in bureaus.items():
                if not isinstance(value, Mapping):
                    continue
                bureau_key = str(name).strip().lower()
                normalized_bureaus[bureau_key] = {
                    str(key): val for key, val in value.items()
                }
            bureaus_map = normalized_bureaus
        else:
            bureaus_map = {}

        style_metadata = self._style_metadata_for_account(summary)

        payloads: list[dict[str, Any]] = []
        weak_fields: list[str] = []
        included_findings: list[Mapping[str, Any]] = []
        for requirement in findings:
            if not isinstance(requirement, Mapping):
                continue

            normalized_strength = self._normalize_strength(requirement.get("strength"))

            field = requirement.get("field")
            if not field:
                continue

            canonical_field = self._canonical_field_key(field)
            if canonical_field is None:
                continue

            if canonical_field not in _ELIGIBLE_FIELDS:
                continue

            send_flag = send_to_ai_map.get(canonical_field)
            if send_flag is not True:
                continue

            field_name = str(field)
            weak_fields.append(field_name)
            included_findings.append(self._json_clone(requirement))

            line = self._build_line(
                account_id,
                requirement,
                normalized_strength,
                bureaus_map,
                field_consistency.get(str(field)),
                style_metadata,
            )
            if line is not None:
                payloads.append(line)

        if not payloads:
            reason = "no_valid_requirements" if weak_fields else "no_weak_fields"
            return payloads, {"skip_reason": reason}

        metadata = {
            "weak_fields": weak_fields,
            "field_consistency": field_consistency,
            "findings": included_findings,
            "summary": summary,
            "built_at": _utc_now(),
            "source_hash": self._build_source_hash(
                summary,
                included_findings,
                field_consistency,
                payloads,
            ),
        }

        return payloads, metadata

    @staticmethod
    def _build_send_to_ai_map(findings: Any) -> dict[str, bool]:
        if not isinstance(findings, Sequence):
            return {}

        mapping: dict[str, bool] = {}
        for entry in findings:
            if not isinstance(entry, Mapping):
                continue
            field_key = ValidationPackBuilder._canonical_field_key(entry.get("field"))
            if field_key is None or field_key not in _ELIGIBLE_FIELDS:
                continue
            send_flag = ValidationPackBuilder._coerce_bool(entry.get("send_to_ai"))
            mapping[field_key] = send_flag
        return mapping

    @staticmethod
    def _canonical_field_key(field: Any) -> str | None:
        if field is None:
            return None
        if isinstance(field, str):
            text = field.strip()
        else:
            text = str(field).strip()
        if not text:
            return None
        return text.lower()

    def _build_line(
        self,
        account_id: int,
        requirement: Mapping[str, Any],
        strength: str,
        bureaus: Mapping[str, Mapping[str, Any]],
        consistency: object,
        style_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        field = requirement.get("field")
        if not isinstance(field, str) or not field.strip():
            return None
        field_name = field.strip()
        field_key = self._field_key(field_name)
        account_key = f"{account_id:03d}"

        documents = self._normalize_string_list(requirement.get("documents"))
        category = self._coerce_optional_str(requirement.get("category"))
        min_days = self._coerce_optional_int(requirement.get("min_days"))
        min_days_business = self._coerce_optional_int(
            requirement.get("min_days_business")
        )
        duration_unit = self._coerce_optional_str(requirement.get("duration_unit"))
        min_corroboration = self._coerce_optional_int(
            requirement.get("min_corroboration")
        )
        conditional_gate = self._coerce_bool(requirement.get("conditional_gate"))

        context = self._build_context(consistency)
        extra_context = requirement.get("notes") or requirement.get("reason")
        if extra_context:
            context.setdefault("requirement_note", str(extra_context))

        bureau_values = self._build_bureau_values(field_name, bureaus, consistency)

        reason_code = self._coerce_optional_str(requirement.get("reason_code"))
        reason_label = self._coerce_optional_str(requirement.get("reason_label"))
        finding_clone = self._json_clone(requirement)
        finding_json = json.dumps(finding_clone, ensure_ascii=False, sort_keys=True)

        prompt_user = _PROMPT_USER_TEMPLATE.replace("<finding blob here>", finding_json)

        if style_metadata:
            prompt_user = self._append_style_metadata(prompt_user, style_metadata)

        payload: dict[str, Any] = {
            "sid": self.paths.sid,
            "account_id": account_id,
            "account_key": account_key,
            "id": f"acc_{account_key}__{field_key}",
            "field": field_name,
            "field_key": field_key,
            "category": category,
            "documents": documents,
            "min_days": min_days,
            "duration_unit": duration_unit,
            "strength": strength,
            "bureaus": bureau_values,
            "context": context,
            "finding": finding_clone,
            "finding_json": finding_json,
            "expected_output": _EXPECTED_OUTPUT_SCHEMA,
            "prompt": {
                "system": _SYSTEM_PROMPT,
                "user": prompt_user,
            },
            "send_to_ai": True,
        }

        if reason_code:
            payload["reason_code"] = reason_code
        if reason_label:
            payload["reason_label"] = reason_label

        if min_corroboration is not None:
            payload["min_corroboration"] = min_corroboration
        if conditional_gate:
            payload["conditional_gate"] = True

        if min_days_business is not None:
            payload["min_days_business"] = min_days_business

        return payload

    def _infer_runs_root(self) -> Path | None:
        try:
            return self.paths.packs_dir.parents[3]
        except IndexError:
            return None

    def _style_metadata_for_account(
        self, summary: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        account_label = summary.get("account_id")
        if not isinstance(account_label, str) or not account_label.strip():
            return None

        try:
            metadata = get_style_metadata(
                self.paths.sid,
                account_label.strip(),
                runs_root=self._runs_root,
            )
        except Exception:
            log.warning(
                "VALIDATION_STYLE_METADATA_FETCH_FAILED sid=%s account_id=%s",
                self.paths.sid,
                account_label,
                exc_info=True,
            )
            return None

        return self._sanitize_style_metadata(metadata)

    @staticmethod
    def _sanitize_style_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(metadata, Mapping):
            return None

        tone = str(metadata.get("tone") or "").strip() or "neutral"
        topic = str(metadata.get("topic") or "").strip() or "other"

        emphasis_raw = metadata.get("emphasis")
        emphasis: list[str] = []
        if isinstance(emphasis_raw, Sequence) and not isinstance(
            emphasis_raw, (str, bytes, bytearray)
        ):
            seen: set[str] = set()
            for entry in emphasis_raw:
                text = str(entry or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                emphasis.append(text)

        return {
            "tone": tone,
            "topic": topic,
            "emphasis": emphasis,
        }

    @staticmethod
    def _append_style_metadata(
        prompt_user: str, metadata: Mapping[str, Any]
    ) -> str:
        try:
            serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):  # defensive guard
            return prompt_user

        base = prompt_user.rstrip()
        return f"{base}\n\nSTYLE_METADATA:\n{serialized}"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _write_pack(
        self,
        account_id: int,
        serialized: str,
        *,
        field_count: int,
        size_bytes: int,
    ) -> Path:
        packs_dir = self.paths.packs_dir
        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_path = packs_dir / validation_pack_filename_for_account(account_id)

        pack_path.write_text(serialized, encoding="utf-8")

        size_kb = _bytes_to_kb(size_bytes)

        self._log(
            "pack_created",
            account_id=f"{account_id:03d}",
            pack=str(pack_path.resolve()),
            fields=field_count,
            size_bytes=size_bytes,
            size_kb=size_kb,
        )
        return pack_path

    @staticmethod
    def _serialize_payloads(
        payloads: Sequence[Mapping[str, Any]],
    ) -> tuple[str, int]:
        if not payloads:
            return "", 0

        lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payloads]
        serialized = "\n".join(lines) + "\n"
        size_bytes = len(serialized.encode("utf-8"))
        return serialized, size_bytes

    def _write_index(self, records: Sequence[ValidationPackRecord]) -> None:
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        index = build_validation_index(
            index_path=self.paths.index_path,
            sid=self.paths.sid,
            packs_dir=self.paths.packs_dir,
            results_dir=self.paths.results_dir,
            records=records,
        )
        index.write()

    def _build_index_record(
        self,
        account_id: int,
        pack_path: Path,
        payloads: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> ValidationPackRecord:
        index_dir = self.paths.index_path.parent.resolve()
        pack_rel = self._relative_to_index(pack_path, index_dir)
        jsonl_path = self.paths.results_dir / validation_result_jsonl_filename_for_account(
            account_id
        )
        summary_path = self.paths.results_dir / validation_result_summary_filename_for_account(
            account_id
        )

        result_jsonl_rel = self._relative_to_index(jsonl_path, index_dir)
        result_json_rel = self._relative_to_index(summary_path, index_dir)

        weak_fields = metadata.get("weak_fields")
        weak_fields_tuple: tuple[str, ...]
        if isinstance(weak_fields, Sequence) and not isinstance(
            weak_fields, (bytes, bytearray, str)
        ):
            weak_fields_tuple = tuple(
                str(field).strip() for field in weak_fields if str(field).strip()
            )
        else:
            weak_fields_tuple = ()

        built_at = metadata.get("built_at")
        if isinstance(built_at, str) and built_at.strip():
            built_timestamp = built_at.strip()
        else:
            built_timestamp = _utc_now()

        source_hash = metadata.get("source_hash")
        if isinstance(source_hash, str) and source_hash.strip():
            source_hash_value = source_hash.strip()
        else:
            source_hash_value = None

        extra: dict[str, Any] = {"account_key": f"{account_id:03d}"}

        return ValidationPackRecord(
            account_id=account_id,
            pack=pack_rel,
            result_jsonl=result_jsonl_rel,
            result_json=result_json_rel,
            lines=len(payloads),
            status="built",
            built_at=built_timestamp,
            weak_fields=weak_fields_tuple,
            source_hash=source_hash_value,
            extra=extra,
        )

    @staticmethod
    def _relative_to_index(path: Path, index_dir: Path) -> str:
        resolved_path = path.resolve()
        base = index_dir.resolve()
        try:
            relative = resolved_path.relative_to(base)
        except ValueError:
            try:
                relative = Path(os.path.relpath(resolved_path, base))
            except ValueError:
                return resolved_path.as_posix()
        posix_path = relative.as_posix()
        return posix_path or "."

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_manifest_paths(manifest: Mapping[str, Any]) -> ManifestPaths:
        sid = str(manifest.get("sid") or "").strip()
        if not sid:
            raise ValueError("Manifest is missing 'sid'")

        base_dirs = manifest.get("base_dirs")
        if not isinstance(base_dirs, Mapping):
            raise ValueError("Manifest is missing 'base_dirs'")

        accounts_dir_raw = base_dirs.get("cases_accounts_dir")
        if not accounts_dir_raw:
            raise ValueError("Manifest missing base_dirs.cases_accounts_dir")
        accounts_dir = Path(str(accounts_dir_raw)).resolve()

        ai_section = manifest.get("ai")
        if not isinstance(ai_section, Mapping):
            raise ValueError("Manifest missing 'ai' section")

        packs_section = ai_section.get("packs")
        if not isinstance(packs_section, Mapping):
            raise ValueError("Manifest missing ai.packs")

        validation_section = packs_section.get("validation")
        if not isinstance(validation_section, Mapping):
            raise ValueError("Manifest missing ai.packs.validation")

        packs_dir_raw = validation_section.get("packs_dir") or validation_section.get("packs")
        results_dir_raw = validation_section.get("results_dir") or validation_section.get("results")
        index_path_raw = validation_section.get("index")
        log_path_raw = (
            validation_section.get("logs")
            or validation_section.get("log")
            or validation_section.get("log_file")
        )

        if not packs_dir_raw:
            raise ValueError("Manifest missing ai.packs.validation.packs_dir")
        if not results_dir_raw:
            raise ValueError("Manifest missing ai.packs.validation.results_dir")
        if not index_path_raw:
            raise ValueError("Manifest missing ai.packs.validation.index")
        if not log_path_raw:
            raise ValueError("Manifest missing ai.packs.validation.logs")

        packs_dir = Path(str(packs_dir_raw))
        results_dir = Path(str(results_dir_raw))
        index_path = Path(str(index_path_raw))
        log_path = Path(str(log_path_raw))

        return ManifestPaths(
            sid=sid,
            accounts_dir=accounts_dir,
            packs_dir=packs_dir,
            results_dir=results_dir,
            index_path=index_path,
            log_path=log_path,
        )

    @staticmethod
    def _normalize_strength(strength: Any) -> str:
        if isinstance(strength, str):
            normalized = strength.strip().lower()
            if normalized in {"weak", "soft"}:
                return "weak"
            if normalized:
                return normalized
        return "unknown"

    @staticmethod
    def _json_clone(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _build_source_hash(
        summary: Mapping[str, Any] | None,
        findings: Sequence[Mapping[str, Any]],
        field_consistency: Mapping[str, Any],
        pack_lines: Sequence[Mapping[str, Any]],
    ) -> str:
        payload = {
            "summary": summary or {},
            "findings": list(findings),
            "field_consistency": field_consistency,
            "pack_lines": list(pack_lines),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        try:
            text = str(value)
        except Exception:
            return None
        return text.strip() or None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            return []
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @staticmethod
    def _field_key(field: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "_", field.strip().lower())
        return key.strip("_") or "field"

    @staticmethod
    def _build_context(consistency: object) -> dict[str, Any]:
        if not isinstance(consistency, Mapping):
            return {}

        context: dict[str, Any] = {}
        consensus = ValidationPackBuilder._coerce_optional_str(consistency.get("consensus"))
        if consensus:
            context["consensus"] = consensus

        disagreeing = ValidationPackBuilder._normalize_string_list(
            consistency.get("disagreeing_bureaus")
        )
        if disagreeing:
            context["disagreeing_bureaus"] = disagreeing

        missing = ValidationPackBuilder._normalize_string_list(
            consistency.get("missing_bureaus")
        )
        if missing:
            context["missing_bureaus"] = missing

        history = consistency.get("history")
        if isinstance(history, Mapping):
            context["history"] = ValidationPackBuilder._normalize_history(history)

        return context

    @staticmethod
    def _normalize_history(history: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in history.items():
            try:
                normalized[str(key)] = value
            except Exception:
                continue
        return normalized

    @staticmethod
    def _build_bureau_values(
        field: str,
        bureaus: Mapping[str, Mapping[str, Any]],
        consistency: object,
    ) -> dict[str, dict[str, Any]]:
        raw_map: Mapping[str, Any] = {}
        normalized_map: Mapping[str, Any] = {}
        if isinstance(consistency, Mapping):
            raw_values = consistency.get("raw")
            if isinstance(raw_values, Mapping):
                raw_map = raw_values
            normalized_values = consistency.get("normalized")
            if isinstance(normalized_values, Mapping):
                normalized_map = normalized_values

        values: dict[str, dict[str, Any]] = {}
        for bureau in _BUREAUS:
            bureau_data = bureaus.get(bureau, {})
            if not isinstance(bureau_data, Mapping):
                bureau_data = {}

            raw_value = ValidationPackBuilder._extract_value(raw_map.get(bureau))
            normalized_hint = None
            if raw_value is None:
                raw_value, normalized_hint = ValidationPackBuilder._extract_bureau_field_values(
                    bureau_data, field
                )

            normalized_value = ValidationPackBuilder._extract_value(normalized_map.get(bureau))
            if normalized_value is None:
                if normalized_hint is None:
                    _, normalized_hint = ValidationPackBuilder._extract_bureau_field_values(
                        bureau_data, field
                    )
                normalized_value = normalized_hint

            values[bureau] = {
                "raw": raw_value,
                "normalized": normalized_value,
            }

        return values

    @staticmethod
    def _extract_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            for candidate in ("raw", "normalized", "value", "text"):
                if candidate in value:
                    return value[candidate]
            return dict(value)
        return value

    @staticmethod
    def _extract_bureau_field_values(
        bureau_data: Mapping[str, Any], field: str
    ) -> tuple[Any, Any]:
        if not isinstance(bureau_data, Mapping):
            return None, None

        value = bureau_data.get(field)
        if isinstance(value, Mapping):
            raw_value = value.get("raw")
            normalized_value = value.get("normalized")

            if raw_value is None:
                raw_value = ValidationPackBuilder._extract_value(value)

            if normalized_value is None:
                for candidate in ("normalized_value", "value", "text"):
                    if candidate in value and value[candidate] is not None:
                        normalized_value = value[candidate]
                        break

            return raw_value, normalized_value

        if value is None:
            return None, None

        return value, None

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, event: str, **payload: Any) -> None:
        log_path = self.paths.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now(),
            "sid": self.paths.sid,
            "event": event,
        }
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_manifest_from_source(
    manifest: Mapping[str, Any] | Path | str,
) -> Mapping[str, Any]:
    """Return a manifest mapping from ``manifest`` regardless of input type."""
    if isinstance(manifest, Mapping):
        return manifest

    manifest_path = Path(manifest)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    data = json.loads(manifest_text)
    if not isinstance(data, Mapping):
        raise TypeError("Manifest root must be a mapping")
    return data


def build_validation_packs(
    manifest: Mapping[str, Any] | Path | str,
) -> list[dict[str, Any]]:
    """Build Validation AI packs for every account defined by ``manifest``."""

    manifest_data = load_manifest_from_source(manifest)
    builder = ValidationPackBuilder(manifest_data)
    return builder.build()


def resolve_manifest_paths(manifest: Mapping[str, Any]) -> ManifestPaths:
    """Return the resolved :class:`ManifestPaths` from ``manifest``."""

    return ValidationPackBuilder._resolve_manifest_paths(manifest)


__all__ = [
    "build_validation_packs",
    "load_manifest_from_source",
    "resolve_manifest_paths",
    "ValidationPackBuilder",
    "ManifestPaths",
]
