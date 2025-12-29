from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

EXPORT_COLUMNS = [
    "session_id",
    "family_id",
    "action_tag",
    "cycle_id",
    "outcome",
    "planner_status",
    "last_sent_at",
    "next_eligible_at",
]


@dataclass(frozen=True)
class ExportFilters:
    """Filter options for analytics exports."""

    action_tags: Sequence[str] | None = None
    family_ids: Sequence[str] | None = None
    cycle_range: tuple[int, int] | None = None
    start_ts: str | None = None
    end_ts: str | None = None


def _build_where(filters: ExportFilters) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if filters.action_tags:
        placeholders = ",".join("?" for _ in filters.action_tags)
        clauses.append(f"action_tag IN ({placeholders})")
        params.extend(filters.action_tags)
    if filters.family_ids:
        placeholders = ",".join("?" for _ in filters.family_ids)
        clauses.append(f"family_id IN ({placeholders})")
        params.extend(filters.family_ids)
    if filters.cycle_range:
        clauses.append("cycle_id BETWEEN ? AND ?")
        params.extend(filters.cycle_range)
    if filters.start_ts:
        clauses.append("last_sent_at >= ?")
        params.append(filters.start_ts)
    if filters.end_ts:
        clauses.append("last_sent_at <= ?")
        params.append(filters.end_ts)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def fetch_joined(
    conn: sqlite3.Connection,
    filters: ExportFilters,
    *,
    view_name: str = "analytics_planner_outcomes",
) -> Iterator[Mapping[str, object]]:
    """Yield joined planner/outcome rows from ``view_name`` applying ``filters``."""

    where, params = _build_where(filters)
    query = (
        "SELECT session_id, family_id, action_tag, cycle_id, outcome, "
        "planner_status, last_sent_at, next_eligible_at "
        f"FROM {view_name}{where}"
    )
    cur = conn.execute(query, params)
    cols = [c[0] for c in cur.description]
    for row in cur:
        yield dict(zip(cols, row))


def stream_csv(rows: Iterable[Mapping[str, object]]) -> Iterator[str]:
    """Yield CSV lines for ``rows`` with an ``export_version`` header."""

    yield "export_version=1\n"
    yield ",".join(EXPORT_COLUMNS) + "\n"
    for row in rows:
        yield ",".join(str(row.get(col, "")) for col in EXPORT_COLUMNS) + "\n"


def stream_json(rows: Iterable[Mapping[str, object]]) -> Iterator[str]:
    """Yield JSON chunks for ``rows`` with an ``export_version`` header."""

    yield "{\"export_version\":1,\"rows\":[\n"
    first = True
    for row in rows:
        if not first:
            yield ",\n"
        else:
            first = False
        yield json.dumps(row, separators=(",", ":"))
    yield "\n]}"
