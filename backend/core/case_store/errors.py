from dataclasses import dataclass


@dataclass
class CaseStoreError(Exception):
    code: str
    message: str


@dataclass
class CaseWriteConflict(CaseStoreError):
    account_id: str
    last_seen_version: int


# Known error codes
NOT_FOUND = "NOT_FOUND"
VALIDATION_FAILED = "VALIDATION_FAILED"
IO_ERROR = "IO_ERROR"
WRITE_CONFLICT = "WRITE_CONFLICT"


__all__ = [
    "CaseStoreError",
    "CaseWriteConflict",
    "NOT_FOUND",
    "VALIDATION_FAILED",
    "IO_ERROR",
    "WRITE_CONFLICT",
]
