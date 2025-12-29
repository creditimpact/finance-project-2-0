from __future__ import annotations

from typing import Any, Mapping, Optional

import uuid

from backend.core.runflow import runflow_step


_ACTIVE_SPANS: dict[str, dict[str, Any]] = {}


def _normalise_mapping(payload: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items()}
    return None


def start_span(
    sid: str,
    stage: str,
    name: str,
    ctx: Optional[Mapping[str, Any]] = None,
    *,
    parent_span_id: Optional[str] = None,
) -> str:
    span_id = uuid.uuid4().hex
    _ACTIVE_SPANS[span_id] = {
        "sid": sid,
        "stage": stage,
        "name": name,
        "parent_span_id": parent_span_id,
    }

    runflow_step(
        sid,
        stage,
        name,
        status="start",
        metrics=_normalise_mapping(ctx),
        span_id=span_id,
        parent_span_id=parent_span_id,
    )

    return span_id


def end_span(
    span_id: str,
    *,
    status: str = "success",
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    info = _ACTIVE_SPANS.pop(span_id, None)
    if not info:
        return

    sid = info["sid"]
    stage = info["stage"]
    name = info["name"]
    parent_span_id = info.get("parent_span_id")

    runflow_step(
        sid,
        stage,
        name,
        status=status,
        metrics=_normalise_mapping(metrics),
        out=_normalise_mapping(out),
        reason=reason,
        span_id=span_id,
        parent_span_id=parent_span_id,
        error=_normalise_mapping(error),
    )


def span_step(
    sid: str,
    stage: str,
    name: str,
    *,
    parent_span_id: Optional[str] = None,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
) -> None:
    runflow_step(
        sid,
        stage,
        name,
        status=status,
        account=account,
        metrics=_normalise_mapping(metrics),
        out=_normalise_mapping(out),
        reason=reason,
        span_id=None,
        parent_span_id=parent_span_id,
        error=_normalise_mapping(error),
        substage=substage,
    )


__all__ = ["start_span", "end_span", "span_step"]
