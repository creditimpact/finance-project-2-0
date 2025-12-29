"""Endpoint for ingesting UI telemetry events."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request
from jsonschema import Draft7Validator, ValidationError

from backend.api.config import env_int, env_list
from backend.core.telemetry.ui_ingest import emit

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
with open(SCHEMA_DIR / "ui_event.json") as _f:
    _validator = Draft7Validator(json.load(_f))

UI_EVENT_MAX_BODY_BYTES = env_int("UI_EVENT_MAX_BODY_BYTES", 4096)
UI_EVENT_ACCEPTED_TYPES = env_list("UI_EVENT_ACCEPTED_TYPES") or [
    "ui_review_expand",
    "ui_review_collapse",
]

ui_event_bp = Blueprint("ui_event", __name__)


@ui_event_bp.post("/api/ui-event")
def receive_ui_event() -> Any:
    """Validate and forward UI telemetry events."""
    if request.content_length and request.content_length > UI_EVENT_MAX_BODY_BYTES:
        return "", 413
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "SchemaValidationError"}), 400
    try:
        _validator.validate(data)
    except ValidationError:
        return jsonify({"error": "SchemaValidationError"}), 400
    if data["type"] not in UI_EVENT_ACCEPTED_TYPES:
        return jsonify({"error": "SchemaValidationError"}), 400

    emit(
        data["type"],
        session_id=data["session_id"],
        account_id=data["payload"]["account_id"],
        bureau=data["payload"]["bureau"],
        decision_source=data["payload"].get("decision_source"),
        tier=data["payload"].get("tier"),
        ts=data["ts"],
    )
    logger.info(
        "ui_event_validated type=%s session=%s account=%s bureau=%s",
        data["type"],
        data["session_id"],
        data["payload"]["account_id"],
        data["payload"]["bureau"],
    )
    return "", 204
