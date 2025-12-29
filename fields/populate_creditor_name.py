"""Populate creditor_name from tri-merge evidence.

Idempotently fills the ``creditor_name`` field on the provided context. Values
are pulled from a tri-merge family record which may expose the creditor name
under either ``creditor_name`` or ``name``.  The field is only set when missing
from ``ctx`` and a source value is available.
"""
from __future__ import annotations

from typing import Mapping


def populate_creditor_name(ctx: dict, tri_merge: Mapping[str, str] | None = None) -> None:
    """Populate ``creditor_name`` on ``ctx`` if missing.

    Parameters
    ----------
    ctx:
        The letter context to mutate.
    tri_merge:
        A mapping representing tri-merge evidence. Either ``creditor_name`` or
        ``name`` may be supplied.
    """

    if ctx.get("creditor_name"):
        return
    tri_merge = tri_merge or {}
    name = tri_merge.get("creditor_name") or tri_merge.get("name")
    if name:
        ctx["creditor_name"] = name
