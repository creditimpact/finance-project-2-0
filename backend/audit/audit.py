from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4
from contextvars import ContextVar

from backend.core.logic.utils.pii import mask_account_fields, redact_pii

_session_id_ctx: ContextVar[str | None] = ContextVar("session_id", default=None)
_family_id_ctx: ContextVar[str | None] = ContextVar("family_id", default=None)
_cycle_id_ctx: ContextVar[int | None] = ContextVar("cycle_id", default=None)
_audit_id_ctx: ContextVar[str | None] = ContextVar("audit_id", default=None)


def set_log_context(
    *,
    session_id: str | None = None,
    family_id: str | None = None,
    cycle_id: int | None = None,
    audit_id: str | None = None,
) -> None:
    """Bind contextual identifiers for subsequent log events."""

    if session_id is not None:
        _session_id_ctx.set(session_id)
    if family_id is not None:
        _family_id_ctx.set(family_id)
    if cycle_id is not None:
        _cycle_id_ctx.set(cycle_id)
    if audit_id is not None:
        _audit_id_ctx.set(audit_id)


class AuditLevel(Enum):
    ESSENTIAL = 1
    VERBOSE = 2


class AuditLogger:
    """Collects structured audit information for a credit repair run."""

    ESSENTIAL_STEPS = {
        "strategist_invocation",
        "strategist_raw_output",
        "strategist_failure",
        "strategy_generated",
        "strategy_merged",
        "strategy_decision",
        "strategy_rule_enforcement",
        "pre_strategy_fallback",
        "strategy_fallback",
    }

    def __init__(self, level: AuditLevel = AuditLevel.ESSENTIAL) -> None:
        self.level = level
        self.data: Dict[str, Any] = {
            "start_time": datetime.now(UTC).isoformat(),
            "steps": [],
            "accounts": {},
            "errors": [],
        }

    def log_step(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        if self.level == AuditLevel.ESSENTIAL and stage not in self.ESSENTIAL_STEPS:
            return
        audit_id = str(uuid4())
        entry = {
            "stage": stage,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": mask_account_fields(details or {}),
            "audit_id": audit_id,
        }
        self.data["steps"].append(entry)
        set_log_context(audit_id=audit_id)

    def log_account(self, account_id: Any, info: Dict[str, Any]) -> None:
        if (
            self.level == AuditLevel.ESSENTIAL
            and info.get("stage") not in self.ESSENTIAL_STEPS
        ):
            return
        acc = self.data["accounts"].setdefault(str(account_id), [])
        audit_id = str(uuid4())
        entry = {"timestamp": datetime.now(UTC).isoformat(), "audit_id": audit_id}
        entry.update(mask_account_fields(info))
        acc.append(entry)
        set_log_context(audit_id=audit_id)

    def log_error(self, message: str) -> None:
        audit_id = str(uuid4())
        self.data["errors"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "message": message,
                "audit_id": audit_id,
            }
        )
        set_log_context(audit_id=audit_id)

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "audit.json"
        with open(path, "w", encoding="utf-8") as f:
            json_data = json.dumps(self.data, indent=2)
            f.write(redact_pii(json_data))
        return path


def create_audit_logger(
    session_id: str, level: AuditLevel = AuditLevel.ESSENTIAL
) -> AuditLogger:
    audit = AuditLogger(level=level)
    audit.data["session_id"] = session_id
    set_log_context(session_id=session_id)
    return audit


_logger = logging.getLogger(__name__)


def emit_event(
    event: str, payload: Dict[str, Any], extra: Dict[str, Any] | None = None
) -> None:
    """Emit a structured audit log entry for external monitoring."""

    ctx = {
        "session_id": _session_id_ctx.get(),
        "family_id": _family_id_ctx.get(),
        "cycle_id": _cycle_id_ctx.get(),
        "audit_id": _audit_id_ctx.get(),
    }
    ctx_extra = {k: v for k, v in ctx.items() if v is not None}
    if extra:
        ctx_extra.update(extra)
    _logger.info(
        "%s %s", event, redact_pii(json.dumps(payload)), extra=ctx_extra
    )
