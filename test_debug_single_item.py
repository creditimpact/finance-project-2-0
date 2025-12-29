"""Debug script to check single-item plan behavior."""

from datetime import datetime
from zoneinfo import ZoneInfo
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding

findings = [
    Finding(
        field="account_status",
        category="status",
        min_days=10,
        duration_unit="business",
        default_decision="strong_actionable",
        bureaus=["equifax"],
        bureau_dispute_state={"equifax": "conflict"},
    ),
]

result = compute_optimal_plan(
    findings=findings,
    weekend={5, 6},
    holidays=set(),
    timezone_name="America/New_York",
    run_datetime=datetime(2025, 11, 20, 10, 0, 0, tzinfo=ZoneInfo("America/New_York")),
    last_submit_window=(0, 40),
)

master_plan = result["master"]
best_weekday = result["best_weekday"]
weekday_plan = result["weekday_plans"][best_weekday]

print("Master plan summary:")
print(master_plan.get("summary"))
print("\nWeekday plan summary:")
print(weekday_plan.get("summary"))
print("\nSequence debug:")
print(weekday_plan.get("sequence_debug"))
print("\nReason:", master_plan.get("reason"))
