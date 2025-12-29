from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Mapping, Any, Optional

from backend.ai.manifest import ensure_validation_section
from backend.ai.validation_builder import build_validation_packs_for_run, run_validation_send_for_sid
from backend.runflow.decider import (
    record_stage_force,
    reconcile_umbrella_barriers,
    _validation_results_progress,
    _compute_umbrella_barriers,
)
from backend.pipeline.validation_merge_helpers import apply_validation_merge_and_update_state

logger = logging.getLogger(__name__)


def _orchestrator_mode_enabled() -> bool:
    raw = os.getenv("VALIDATION_ORCHESTRATOR_MODE")
    if raw is None:
        return True
    lowered = str(raw).strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}


def _resolve_runs_root(runs_root: Optional[str | Path]) -> Path:
    if runs_root is None:
        env = os.getenv("RUNS_ROOT")
        return Path(env) if env else Path("runs")
    return Path(runs_root)


def _wait_settings() -> tuple[int, int]:
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            val = int(str(raw).strip())
        except Exception:
            return default
        return val if val >= 0 else default
    return (
        _env_int("VALIDATION_MERGE_WAIT_SECONDS", 90),
        _env_int("VALIDATION_MERGE_POLL_SECONDS", 5),
    )


class ValidationOrchestrator:
    def __init__(self, *, runs_root: Optional[str | Path] = None) -> None:
        self._runs_root = _resolve_runs_root(runs_root)

    def _marker_path(self, sid: str) -> Path:
        return self._runs_root / sid / ".locks" / "validation_orchestrator.managed"

    def _set_marker(self, sid: str, content: str) -> None:
        path = self._marker_path(sid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception:
            logger.debug("ORCHESTRATOR_MARK_SET_FAILED sid=%s path=%s", sid, path, exc_info=True)

    def _clear_marker(self, sid: str) -> None:
        path = self._marker_path(sid)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.debug("ORCHESTRATOR_MARK_CLEAR_FAILED sid=%s path=%s", sid, path, exc_info=True)

    def run_for_sid(self, sid: str) -> dict[str, Any]:
        if not _orchestrator_mode_enabled():
            logger.info("VALIDATION_ORCHESTRATOR_DISABLED sid=%s", sid)
            return {"sid": sid, "orchestrator": False}

        runs_root = self._runs_root
        run_dir = runs_root / sid
        logger.info("VALIDATION_ORCHESTRATOR_START sid=%s runs_root=%s", sid, runs_root)

        # ── MERGE_READY BARRIER CHECK ──────────────────────────────────────
        # Validation cannot start until merge is ready (AI results applied).
        # This prevents the timing bug where validation completes before merge.
        barriers = _compute_umbrella_barriers(run_dir)
        merge_ready = barriers.get("merge_ready", False)
        if not merge_ready:
            logger.info(
                "VALIDATION_ORCHESTRATOR_DEFERRED sid=%s reason=merge_not_ready barriers=%s",
                sid,
                barriers,
            )
            return {"sid": sid, "deferred": True, "reason": "merge_not_ready"}
        # ───────────────────────────────────────────────────────────────────

        self._set_marker(sid, "active")

        # Pack phase: ensure manifest section and build packs
        try:
            ensure_validation_section(sid, runs_root=runs_root)
            build_stats = build_validation_packs_for_run(sid, runs_root=runs_root)
        except Exception:
            logger.exception("VALIDATION_PACK_BUILD_FAILED sid=%s", sid)
            return {"sid": sid, "error": "pack_build_failed"}

        packs_written = sum(len(entries or []) for entries in (build_stats or {}).values()) if isinstance(build_stats, Mapping) else 0
        logger.info("VALIDATION_PACKS_BUILT sid=%s packs=%d", sid, packs_written)

        # Send phase: sequentially send and write results
        try:
            send_stats = run_validation_send_for_sid(sid, runs_root)
        except Exception:
            logger.exception("VALIDATION_SEND_FAILED sid=%s", sid)
            return {"sid": sid, "error": "send_failed", "packs": packs_written}

        logger.info("VALIDATION_SEND_DONE sid=%s stats=%s", sid, send_stats)

        # Wait phase: poll for completeness deterministically, else exit partial
        max_wait, poll_interval = _wait_settings()
        total, completed, failed, ready = _validation_results_progress(run_dir)
        if max_wait > 0 and poll_interval >= 0 and not ready and failed == 0 and total > 0:
            deadline = time.monotonic() + max_wait
            attempt = 0
            while not ready and failed == 0 and time.monotonic() < deadline:
                attempt += 1
                remaining = max(0.0, deadline - time.monotonic())
                sleep_time = min(poll_interval, remaining)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                total, completed, failed, ready = _validation_results_progress(run_dir)
                logger.info(
                    "VALIDATION_WAIT sid=%s expected=%d completed=%d missing=%d attempt=%d",
                    sid,
                    total,
                    completed,
                    max(total - completed - failed, 0),
                    attempt,
                )

        if failed > 0 or completed < total:
            missing = max(total - completed - failed, 0)
            logger.info(
                "VALIDATION_PARTIAL sid=%s expected=%d completed=%d missing=%d",
                sid,
                total,
                completed,
                missing,
            )
            return {
                "sid": sid,
                "packs": packs_written,
                "results_total": total,
                "results_completed": completed,
                "results_missing": missing,
                "partial": True,
            }

        # Finalize phase: merge once, record success, reconcile umbrella once
        try:
            merge_stats = apply_validation_merge_and_update_state(
                sid,
                runs_root=runs_root,
                source="validation_orchestrator",
            )
        except Exception:
            logger.exception("VALIDATION_FINALIZE_MERGE_FAILED sid=%s", sid)
            return {"sid": sid, "error": "finalize_merge_failed"}

        # Persist final status and latch readiness
        try:
            record_stage_force(
                sid,
                {"stages": {"validation": {
                    "status": "success",
                    "sent": True,
                    "ready_latched": True,
                    "validation_finalized": True,
                }}},
                runs_root=runs_root,
                last_writer="validation_orchestrator",
                refresh_barriers=False,
            )
        except Exception:
            logger.debug("VALIDATION_FINALIZE_STAGE_WRITE_FAILED sid=%s", sid, exc_info=True)

        # Reconcile umbrella exactly once (temporarily bypass decider gating)
        old = os.getenv("VALIDATION_ORCHESTRATOR_BYPASS")
        os.environ["VALIDATION_ORCHESTRATOR_BYPASS"] = "1"
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root)
        finally:
            if old is None:
                os.environ.pop("VALIDATION_ORCHESTRATOR_BYPASS", None)
            else:
                os.environ["VALIDATION_ORCHESTRATOR_BYPASS"] = old

        logger.info("VALIDATION_ORCHESTRATOR_DONE sid=%s", sid)
        self._clear_marker(sid)
        return {
            "sid": sid,
            "packs": packs_written,
            "results_total": total,
            "results_completed": completed,
            "merge": dict(merge_stats) if isinstance(merge_stats, Mapping) else None,
            "finalized": True,
        }
