"""Internal AI adjudication endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request
from jsonschema import Draft7Validator, ValidationError as SchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from backend.config import AI_REQUEST_TIMEOUT_S
from backend.core.ai.models import AIAdjudicateRequest, AIAdjudicateResponse
from backend.core.ai.service import run_llm_prompt

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
with open(SCHEMA_DIR / "ai_adjudication.json") as _f:
    _ai_response_validator = Draft7Validator(json.load(_f))

ai_bp = Blueprint("ai_internal", __name__)


@ai_bp.post("/internal/ai-adjudicate")
def ai_adjudicate() -> Any:
    """Internal-only AI adjudication endpoint."""
    try:
        raw = request.get_json(force=True) or {}
        req = AIAdjudicateRequest.model_validate(raw)
    except PydanticValidationError:
        return jsonify({"error": "invalid_request"}), 400

    system_prompt = f"hierarchy_version={req.hierarchy_version}"
    try:
        llm_raw = run_llm_prompt(
            system_prompt,
            req.fields,
            temperature=0.0,
            timeout=AI_REQUEST_TIMEOUT_S,
        )
        data = json.loads(llm_raw)
        resp = AIAdjudicateResponse.model_validate(data)
        payload = resp.model_dump()
        _ai_response_validator.validate(payload)
        return jsonify(payload)
    except TimeoutError:
        return jsonify({"error": "timeout"}), 504
    except (
        json.JSONDecodeError,
        PydanticValidationError,
        SchemaValidationError,
    ) as exc:
        logger.debug("ai_adjudicate_schema_failure: %s", exc)
        return jsonify({"error": "SchemaValidationError"}), 502
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ai_adjudicate_failure: %s", exc)
        return jsonify({"error": "bad_gateway"}), 502
