# Analytics Views

This document records the schemas for analytics views and their version history.

## analytics.tri_merge_view

### v1
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |

### v2
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |
| tri_merge_snapshot_id | TEXT | Snapshot provenance identifier |
| plan_id | TEXT | Planner provenance identifier |
| step_id | TEXT | Planner step provenance identifier |
| outcome_id | TEXT | Outcome provenance identifier |

## analytics.planner_view

### v1
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |
| plan_id | TEXT | Planner provenance identifier |
| step_id | TEXT | Planner step provenance identifier |

### v2
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |
| tri_merge_snapshot_id | TEXT | Snapshot provenance identifier |
| plan_id | TEXT | Planner provenance identifier |
| step_id | TEXT | Planner step provenance identifier |
| outcome_id | TEXT | Outcome provenance identifier |

## analytics.outcome_view

### v1
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |
| outcome_id | TEXT | Outcome provenance identifier |

### v2
| column | type | description |
| --- | --- | --- |
| session_id | TEXT | Session identifier |
| account_id | TEXT | Account identifier |
| family_id | TEXT | Family identifier |
| cycle_id | INTEGER | Cycle identifier |
| tri_merge_snapshot_id | TEXT | Snapshot provenance identifier |
| plan_id | TEXT | Planner provenance identifier |
| step_id | TEXT | Planner step provenance identifier |
| outcome_id | TEXT | Outcome provenance identifier |
