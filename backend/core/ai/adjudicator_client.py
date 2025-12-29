from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import httpx
from jsonschema import Draft7Validator, ValidationError as SchemaValidationError
from pydantic import ValidationError

from backend.config import AI_MAX_RETRIES, AI_REQUEST_TIMEOUT_S
from backend.core.ai.models import AIAdjudicateRequest, AIAdjudicateResponse
from backend.core.case_store.telemetry import emit

AI_URL = "/internal/ai-adjudicate"

SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"
with open(SCHEMA_DIR / "ai_adjudication.json") as _f:
    _ai_response_validator = Draft7Validator(json.load(_f))


def call_adjudicator(
    session, payload: AIAdjudicateRequest
) -> Optional[AIAdjudicateResponse]:
    """Call internal AI adjudicator endpoint.

    Returns parsed ``AIAdjudicateResponse`` or ``None`` on failure. Any
    validation, timeout, or HTTP error results in ``None`` and emits telemetry
    with the error class name as status.
    """

    attempts = AI_MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        status = "ok"
        try:
            with httpx.Client(timeout=AI_REQUEST_TIMEOUT_S) as cli:
                r = cli.post(AI_URL, json=payload.model_dump(mode="json"))
            r.raise_for_status()
            data = r.json()
            resp = AIAdjudicateResponse.model_validate(data)
            _ai_response_validator.validate(resp.model_dump())
            dur = (time.perf_counter() - t0) * 1000
            emit(
                "stageA_ai_call",
                attempt=attempt,
                status=status,
                duration_ms=round(dur, 3),
                confidence=resp.confidence,
            )
            return resp
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            json.JSONDecodeError,
            ValidationError,
            SchemaValidationError,
        ) as e:
            status = e.__class__.__name__
            dur = (time.perf_counter() - t0) * 1000
            emit(
                "stageA_ai_call",
                attempt=attempt,
                status=status,
                duration_ms=round(dur, 3),
            )
            pass
    return None
