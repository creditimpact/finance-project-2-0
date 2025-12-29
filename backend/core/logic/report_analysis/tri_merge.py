"""Utilities for merging and comparing bureau tradeline data."""

from __future__ import annotations

import collections
import hashlib
import math
import os
import re
from random import random
from typing import Dict, Iterable, List, Tuple

from backend.analytics.analytics_tracker import set_metric
from backend.telemetry.metrics import emit_counter
from backend.api.session_manager import get_session, update_session
from backend.audit.audit import emit_event
from backend.core.logic.utils.names_normalization import canonicalize_creditor
from backend.core.logic.utils.pii import mask_account

from .tri_merge_models import Mismatch, Tradeline, TradelineFamily

BUREAUS = ["Experian", "Equifax", "TransUnion"]


def _last4(account_number: str | None) -> str:
    digits = re.sub(r"\D", "", account_number or "")
    return digits[-4:]


def normalize_and_match(bureau_data: Iterable[Tradeline]) -> List[TradelineFamily]:
    """Normalize input tradelines and group likely matches.

    ``bureau_data`` is an iterable of :class:`Tradeline` items from any bureau.
    Groups are keyed by canonicalized creditor, last four digits of the account
    number, and basic date attributes. Each group becomes a
    :class:`TradelineFamily` with a stable ``family_id`` and a ``match_confidence``
    attribute indicating how reliable the grouping is.
    """

    groups: Dict[Tuple[str, str, str, str], TradelineFamily] = {}

    for tl in bureau_data:
        creditor = canonicalize_creditor(tl.creditor)
        last4 = _last4(tl.account_number)
        opened = str(tl.data.get("date_opened") or tl.data.get("open_date") or "")
        reported = str(tl.data.get("date_reported") or tl.data.get("report_date") or "")
        key = (creditor, last4, opened, reported)

        family = groups.get(key)
        if not family:
            family = TradelineFamily(
                account_number=mask_account(tl.account_number or "")
            )
            groups[key] = family
        if tl.bureau in family.tradelines:
            # Track duplicates for mismatch analysis
            family._duplicates.append(tl)
        else:
            family.tradelines[tl.bureau] = tl

    families: List[TradelineFamily] = []
    confidences: List[float] = []
    for key, family in groups.items():
        features = "|".join(key)
        family_id = hashlib.sha1(features.encode("utf-8")).hexdigest()[:10]
        feature_count = sum(1 for part in key if part)
        match_confidence = feature_count / 4.0
        family.family_id = family_id  # type: ignore[attr-defined]
        family.match_confidence = match_confidence  # type: ignore[attr-defined]
        families.append(family)
        confidences.append(match_confidence)

        bucket = int(match_confidence * 10) * 10
        emit_counter(f"tri_merge.match_confidence_hist.{bucket}")
        # Sample per-family logs to reduce volume
        if match_confidence < 0.5 and random() < 0.1:
            emit_event(
                "tri_merge.low_match_confidence",
                {
                    "family_id": family_id,
                    "creditor": key[0],
                    "confidence": match_confidence,
                },
                extra={"family_id": family_id, "cycle_id": 0},
            )

    if confidences:
        sorted_conf = sorted(confidences)
        index = int(math.ceil(0.95 * len(sorted_conf))) - 1
        p95_value = sorted_conf[index]
    else:
        p95_value = 0
    set_metric("tri_merge.match_confidence_p95", p95_value)

    for fam in families:
        for bureau in fam.tradelines.keys():
            emit_counter("tri_merge.families_total", {"bureau": bureau})
    return families


def compute_mismatches(families: Iterable[TradelineFamily]) -> List[TradelineFamily]:
    """Compare tradeline families to surface mismatches.

    Detected mismatches are appended to each family's ``mismatches`` list and a
    snapshot of the raw evidence is stored in the session under
    ``session["tri_merge"]["evidence"][family_id]``.
    """

    session_id = os.getenv("SESSION_ID", "")
    session = get_session(session_id) if session_id else None
    tri_store = (
        session.setdefault("tri_merge", {}).setdefault("evidence", {})
        if session
        else {}
    )

    def _record(fam: TradelineFamily, mismatch: Mismatch) -> None:
        fam.mismatches.append(mismatch)
        for bureau in mismatch.values:
            emit_counter("tri_merge.mismatches_total", {"bureau": bureau})
            emit_counter(f"tri_merge.mismatch.{mismatch.field}", {"bureau": bureau})

    for fam in families:
        present = set(fam.tradelines.keys())
        missing = [b for b in BUREAUS if b not in present]
        if missing:
            values = {b: (b in present) for b in BUREAUS}
            _record(fam, Mismatch(field="presence", values=values))

        def cmp(field: str, mtype: str) -> None:
            values = {b: tl.data.get(field) for b, tl in fam.tradelines.items()}
            if len(set(values.values())) > 1:
                _record(fam, Mismatch(field=mtype, values=values))

        def cmp_dates() -> None:
            opened_vals = {
                b: tl.data.get("date_opened") for b, tl in fam.tradelines.items()
            }
            reported_vals = {
                b: tl.data.get("date_reported") for b, tl in fam.tradelines.items()
            }
            if (
                len(set(opened_vals.values())) > 1
                or len(set(reported_vals.values())) > 1
            ):
                values = {
                    b: (opened_vals.get(b), reported_vals.get(b))
                    for b in fam.tradelines.keys()
                }
                _record(fam, Mismatch(field="dates", values=values))

        cmp("balance", "balance")
        cmp("status", "status")
        cmp_dates()
        cmp("remarks", "remarks")
        cmp("utilization", "utilization")
        cmp("personal_info", "personal_info")

        dups = fam._duplicates
        if dups:
            counts = collections.Counter(d.bureau for d in dups)
            _record(fam, Mismatch(field="duplicate", values=dict(counts)))

        if tri_store is not None:
            tri_store[getattr(fam, "family_id", "")] = {
                "tradelines": {b: tl.data for b, tl in fam.tradelines.items()},
                "mismatches": [m.field for m in fam.mismatches],
            }

    if session_id:
        update_session(session_id, tri_merge={"evidence": tri_store})

    return list(families)
