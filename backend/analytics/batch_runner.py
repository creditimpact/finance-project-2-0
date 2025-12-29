"""Batch processing utilities for generating analytics reports."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple
from uuid import uuid4

from croniter import croniter

from backend.core.letters.router import route_accounts
from backend.core.logic.utils.pii import redact_pii
from backend.audit.audit import emit_event, set_log_context

FLAGS = [
    "LETTERS_ROUTER_PHASED",
    "ENFORCE_TEMPLATE_VALIDATION",
    "SAFE_CLIENT_SENTENCE_ENABLED",
]


# ---------------------------------------------------------------------------
# Data models


@dataclass(frozen=True)
class BatchFilters:
    """Filter options for a batch analytics run."""

    action_tags: List[str]
    family_ids: Optional[List[str]] = None
    cycle_range: Optional[Tuple[int, int]] = None
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    page_size: Optional[int] = None
    page_token: Optional[str] = None


class BatchRunner:
    """Run analytics batches and persist job metadata."""

    def __init__(
        self, *, job_store: Path | None = None, output_dir: Path | None = None
    ) -> None:
        self.job_store = Path(job_store or Path("backend/analytics/batch_jobs.sqlite"))
        self.output_dir = Path(output_dir or Path("backend/analytics/batch_reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # -- public API -----------------------------------------------------
    def run(self, filters: BatchFilters, format: Literal["csv", "json"]) -> str:
        """Run a batch job and return the job id.

        The job is idempotent; subsequent invocations with the same filters and
        format will return the existing job record without reprocessing.
        """

        job_id = self._job_id(filters, format)
        with sqlite3.connect(self.job_store) as conn:
            cur = conn.execute(
                "SELECT output_path FROM batch_jobs WHERE job_id=?",
                (job_id,),
            )
            row = cur.fetchone()
            if row:
                return job_id

        audit_id = str(uuid4())
        set_log_context(audit_id=audit_id)
        emit_event(
            "batch_job_start",
            {"job_id": job_id, "filters": asdict(filters)},
        )

        try:
            samples = self._fetch_samples(filters)
            report = process_samples(samples)
            report = _redact_output(report)

            output_path = self.output_dir / f"{job_id}.{format}"
            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            else:
                _write_csv(report, output_path)

            with sqlite3.connect(self.job_store) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO batch_jobs
                    (job_id, filters, format, output_path, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        json.dumps(asdict(filters), sort_keys=True),
                        format,
                        str(output_path),
                        "completed",
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()

            emit_event(
                "batch_job_finish",
                {"job_id": job_id, "status": "completed", "filters": asdict(filters)},
            )
            return job_id
        except Exception as exc:
            emit_event(
                "batch_job_finish",
                {
                    "job_id": job_id,
                    "status": "failed",
                    "filters": asdict(filters),
                    "error": str(exc),
                },
            )
            raise

    def schedule(self, cron_expr: str) -> str:
        """Register a scheduled job and return its identifier.

        The next run time is computed using ``croniter`` and persisted with the
        job record so that external schedulers can introspect upcoming runs.
        """

        now = datetime.utcnow()
        next_run = croniter(cron_expr, now).get_next(datetime)
        job_id = sha256(cron_expr.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.job_store) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batch_jobs
                (job_id, filters, format, output_path, status, created_at, cron_expr, next_run)
                VALUES (?, ?, '', '', 'scheduled', ?, ?, ?)
                """,
                (job_id, "{}", now.isoformat(), cron_expr, next_run.isoformat()),
            )
            conn.commit()
        return job_id

    def retry(self, job_id: str) -> str:
        """Re-run a previously executed job."""

        with sqlite3.connect(self.job_store) as conn:
            cur = conn.execute(
                "SELECT filters, format FROM batch_jobs WHERE job_id=?",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"job_id {job_id} not found")
            filters = BatchFilters(**json.loads(row[0]))
            format = row[1]
            conn.execute("DELETE FROM batch_jobs WHERE job_id=?", (job_id,))
            conn.commit()

        return self.run(filters, format)  # type: ignore[arg-type]

    # -- helpers --------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.job_store) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    filters TEXT NOT NULL,
                    format TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    cron_expr TEXT,
                    next_run TEXT
                )
                """
            )
            conn.commit()

    def _job_id(self, filters: BatchFilters, format: str) -> str:
        raw = json.dumps({"filters": asdict(filters), "format": format}, sort_keys=True)
        return sha256(raw.encode("utf-8")).hexdigest()

    # The data fetcher is intentionally simple so tests can monkeypatch it.
    def _fetch_samples(
        self, filters: BatchFilters
    ) -> List[Mapping[str, object]]:  # pragma: no cover - placeholder
        return []


def _redact_output(data: Any) -> Any:
    """Recursively sanitize ``data`` by masking PII fields."""

    if isinstance(data, dict):
        redacted: Dict[str, Any] = {}
        for key, value in data.items():
            lower = key.lower()
            if lower == "full_name":
                redacted[key] = "[REDACTED]"
            elif lower == "ssn":
                redacted[key] = redact_pii(str(value))
            else:
                redacted[key] = _redact_output(value)
        return redacted
    if isinstance(data, list):
        return [_redact_output(v) for v in data]
    if isinstance(data, str):
        return redact_pii(data)
    return data


def benchmark_finalize(num: int = 1000, *, max_workers: int | None = None) -> float:
    """Return letters-per-second throughput for finalize routing."""

    os.environ.setdefault("LETTERS_ROUTER_PHASED", "1")

    base_ctx = {
        "bureau": "experian",
        "creditor_name": "Acme",
        "account_number_masked": "1234",
        "legal_safe_summary": "ok",
        "is_identity_theft": True,
        "client": {"full_name": "John Doe"},
    }
    items = [("fraud_dispute", dict(base_ctx), f"sess-{i}") for i in range(num)]

    start = time.perf_counter()
    route_accounts(items, phase="finalize", max_workers=max_workers)
    elapsed = time.perf_counter() - start
    throughput = num / elapsed if elapsed else 0.0
    print(f"throughput_lps={throughput:.2f}")
    return throughput


# ---------------------------------------------------------------------------
# Existing analytics helpers


def _percentile(values: List[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values``."""

    if not values:
        return 0.0
    vals = sorted(values)
    k = max(0, int(len(vals) * pct / 100) - 1)
    return vals[k]


def _flatten_heatmap(
    data: Mapping[str, Mapping[str, int]]
) -> Mapping[str, Mapping[str, int]]:
    """Return a nested dict with integer counts."""

    result: Dict[str, Dict[str, int]] = {}
    for tag, fields in data.items():
        bucket = result.setdefault(tag, {})
        for field, count in fields.items():
            bucket[field] = bucket.get(field, 0) + int(count)
    return result


def process_samples(samples: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Process ``samples`` and return an aggregated metrics report."""

    candidate_counts: Dict[str, int] = {}
    finalized_counts: Dict[str, int] = {}
    missing_stage1: Dict[str, Dict[str, int]] = {}
    missing_stage2: Dict[str, Dict[str, int]] = {}
    validation_breakdown: Dict[str, Dict[str, int]] = {}
    sanitizer_applied: Dict[str, int] = {}
    policy_override: Dict[str, Dict[str, int]] = {}
    total_letters: Dict[str, int] = {}
    render_times: Dict[str, List[float]] = {}
    ai_costs: Dict[str, List[float]] = {}

    for sample in samples:
        tag = str(sample.get("action_tag", ""))
        template = str(sample.get("template", ""))

        candidate_counts[tag] = candidate_counts.get(tag, 0) + 1

        for field in sample.get("candidate_missing_fields", []) or []:
            bucket = missing_stage1.setdefault(tag, {})
            bucket[field] = bucket.get(field, 0) + 1

        if sample.get("validation_failed_fields"):
            for field in sample.get("final_missing_fields", []) or []:
                bucket = missing_stage2.setdefault(tag, {})
                bucket[field] = bucket.get(field, 0) + 1
            vf = validation_breakdown.setdefault(template, {})
            for field in sample.get("validation_failed_fields", []) or []:
                vf[field] = vf.get(field, 0) + 1
        else:
            finalized_counts[tag] = finalized_counts.get(tag, 0) + 1
            for field in sample.get("final_missing_fields", []) or []:
                bucket = missing_stage2.setdefault(tag, {})
                bucket[field] = bucket.get(field, 0) + 1

        total_letters[template] = total_letters.get(template, 0) + 1

        if sample.get("sanitizer_overrides"):
            sanitizer_applied[template] = sanitizer_applied.get(template, 0) + 1
            po_bucket = policy_override.setdefault(template, {})
            for reason in sample.get("sanitizer_overrides", []) or []:
                sanitized = str(reason).replace(" ", "_")
                po_bucket[sanitized] = po_bucket.get(sanitized, 0) + 1

        render_times.setdefault(template, []).append(float(sample.get("render_ms", 0)))
        ai_costs.setdefault(template, []).append(float(sample.get("ai_cost", 0)))

    pass_rate: Dict[str, float] = {}
    for tag, count in candidate_counts.items():
        finalized = finalized_counts.get(tag, 0)
        pass_rate[tag] = finalized / count if count else 0.0

    sanitize_rate: Dict[str, float] = {}
    for template, total in total_letters.items():
        applied = sanitizer_applied.get(template, 0)
        sanitize_rate[template] = applied / total if total else 0.0

    render_avg = {tpl: sum(vals) / len(vals) for tpl, vals in render_times.items()}
    render_p95 = {tpl: _percentile(vals, 95) for tpl, vals in render_times.items()}
    ai_avg = {tpl: sum(vals) / len(vals) for tpl, vals in ai_costs.items()}
    ai_p95 = {tpl: _percentile(vals, 95) for tpl, vals in ai_costs.items()}

    report: Dict[str, object] = {
        "finalization_pass_rate": pass_rate,
        "missing_fields": {
            "after_normalization": _flatten_heatmap(missing_stage1),
            "after_strategy": _flatten_heatmap(missing_stage2),
        },
        "validation_failed": _flatten_heatmap(validation_breakdown),
        "sanitizer": {
            "applied_rate": sanitize_rate,
            "policy_override_reason": _flatten_heatmap(policy_override),
        },
        "render_latency_ms": {"avg": render_avg, "p95": render_p95},
        "ai_cost": {"avg": ai_avg, "p95": ai_p95},
    }
    return report


def _write_csv(report: Mapping[str, object], path: Path) -> None:
    rows: List[List[object]] = []
    for tag, rate in report["finalization_pass_rate"].items():
        rows.append(["finalization_pass_rate", tag, "", "", rate])
    for stage_key, stage_data in report["missing_fields"].items():
        stage_label = f"missing_fields_{stage_key}"
        for tag, fields in stage_data.items():
            for field, count in fields.items():
                rows.append([stage_label, tag, field, "", count])
    for template, fields in report["validation_failed"].items():
        for field, count in fields.items():
            rows.append(["validation_failed", template, field, "", count])
    for template, rate in report["sanitizer"]["applied_rate"].items():
        rows.append(["sanitizer_applied_rate", template, "", "", rate])
    for template, reasons in report["sanitizer"]["policy_override_reason"].items():
        for reason, count in reasons.items():
            rows.append(["policy_override_reason", template, reason, "", count])
    for template, val in report["render_latency_ms"]["avg"].items():
        rows.append(["render_latency_avg_ms", template, "", "", val])
    for template, val in report["render_latency_ms"]["p95"].items():
        rows.append(["render_latency_p95_ms", template, "", "", val])
    for template, val in report["ai_cost"]["avg"].items():
        rows.append(["ai_cost_avg", template, "", "", val])
    for template, val in report["ai_cost"]["p95"].items():
        rows.append(["ai_cost_p95", template, "", "", val])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "category", "item", "extra", "value"])
        writer.writerows(rows)


__all__ = ["BatchRunner", "BatchFilters", "process_samples", "benchmark_finalize"]
