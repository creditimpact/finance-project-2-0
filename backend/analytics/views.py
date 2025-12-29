from __future__ import annotations

import sqlite3


def create_tri_merge_view(conn: sqlite3.Connection) -> None:
    """Create ``analytics_tri_merge_view`` if it doesn't exist."""
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS analytics_tri_merge_view AS
        SELECT
            session_id,
            account_id,
            family_id,
            cycle_id,
            tri_merge_snapshot_id,
            NULL AS plan_id,
            NULL AS step_id,
            NULL AS outcome_id
        FROM tri_merge
        """
    )


def create_planner_view(conn: sqlite3.Connection) -> None:
    """Create ``analytics_planner_view`` if it doesn't exist."""
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS analytics_planner_view AS
        SELECT
            session_id,
            account_id,
            family_id,
            cycle_id,
            tri_merge_snapshot_id,
            plan_id,
            step_id,
            NULL AS outcome_id
        FROM planner
        """
    )


def create_outcome_view(conn: sqlite3.Connection) -> None:
    """Create ``analytics_outcome_view`` if it doesn't exist."""
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS analytics_outcome_view AS
        SELECT
            session_id,
            account_id,
            family_id,
            cycle_id,
            tri_merge_snapshot_id,
            plan_id,
            step_id,
            outcome_id
        FROM outcome
        """
    )


def create_views(conn: sqlite3.Connection) -> None:
    """Create all analytics views on ``conn``."""
    create_tri_merge_view(conn)
    create_planner_view(conn)
    create_outcome_view(conn)
