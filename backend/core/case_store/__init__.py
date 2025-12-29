"""Case Store models and helpers."""

from .errors import CaseStoreError, IO_ERROR, NOT_FOUND, VALIDATION_FAILED
from .models import (
    AccountCase,
    AccountFields,
    Artifact,
    Bureau,
    PersonalInformation,
    ReportMeta,
    SessionCase,
    Summary,
)
from pydantic import ValidationError

__all__ = [
    "AccountCase",
    "AccountFields",
    "Artifact",
    "Bureau",
    "PersonalInformation",
    "ReportMeta",
    "SessionCase",
    "Summary",
    "CaseStoreError",
    "NOT_FOUND",
    "VALIDATION_FAILED",
    "IO_ERROR",
    "load_session_case_json",
]


def load_session_case_json(data: str) -> SessionCase:
    """Parse a SessionCase from a JSON string.

    Raises:
        CaseStoreError: if validation fails.
    """

    try:
        return SessionCase.model_validate_json(data)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise CaseStoreError(code=VALIDATION_FAILED, message=str(exc)) from exc
