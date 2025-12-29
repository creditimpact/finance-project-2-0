"""Deprecated shim for backward compatibility.

This module previously housed :func:`build_dispute_payload` but has been
superseded by :mod:`backend.core.logic.post_confirmation`.  Importing from this
module continues to work, but new code should use the ``post_confirmation``
module directly.
"""

from .post_confirmation import build_dispute_payload

__all__ = ["build_dispute_payload"]

