import json, sys, datetime
from pathlib import Path

# Usage: python tools/simulate_overlap_optimizer_demo.py <plan_wd.json>
# Demonstrates baseline (minimal overlap=1) vs optimized (current plan) for two-item skeleton.

def main(path_str: str):
    p = Path(path_str)
    data = json.loads(p.read_text())
    seq = data.get("sequence_debug", [])
    summary = data.get("summary", {})
    if len(seq) < 2:
        print("Need at least 2 items for demo")
        return
    opener = seq[0]
    closer = seq[1]
    anchor_date = datetime.date.fromisoformat(data["anchor"]["date"])
    # Unbounded spans
    opener_submit_index = int(opener.get("calendar_day_index", 0))
    opener_sla_end_date = opener.get("sla_window", {}).get("end", {}).get("date")
    closer_submit_index_opt = int(closer.get("calendar_day_index", 0))
    closer_sla_end_date = closer.get("sla_window", {}).get("end", {}).get("date")
    opener_sla_end_index = (datetime.date.fromisoformat(opener_sla_end_date) - anchor_date).days
    closer_sla_end_index = (datetime.date.fromisoformat(closer_sla_end_date) - anchor_date).days
    opener_unbounded = int(opener.get("effective_contribution_days_unbounded", opener.get("effective_contribution_days", 0)))
    closer_unbounded = int(closer.get("effective_contribution_days_unbounded", closer.get("effective_contribution_days", 0)))
    sum_unbounded = opener_unbounded + closer_unbounded
    target_cap = int(summary.get("inbound_cap_target", 50))
    required_overlap = max(sum_unbounded - target_cap, 0)
    baseline_overlap = 1  # business invariant minimum
    baseline_submit_index_closer = opener_sla_end_index - baseline_overlap
    inbound_baseline = sum_unbounded - baseline_overlap
    optimized_overlap = int(closer.get("overlap_unbounded_days_with_prev", 0))
    inbound_optimized = sum_unbounded - optimized_overlap
    delta_submit_days = baseline_submit_index_closer - closer_submit_index_opt

    print("OVERLAP OPTIMIZER DEMO (SID {})".format(data.get("anchor", {}).get("date")))
    print()
    print("Unbounded Spans:")
    print(f"  Opener span: submit_index={opener_submit_index} end_index={opener_sla_end_index} length={opener_unbounded}")
    print(f"  Closer span (unbounded): end_index={closer_sla_end_index} length={closer_unbounded}")
    print()
    print("Derived Metrics:")
    print(f"  Sum unbounded: {sum_unbounded}")
    print(f"  Target cap: {target_cap}")
    print(f"  Required overlap: {required_overlap}")
    print()
    print("Baseline (minimum overlap=1):")
    print(f"  Closer baseline submit index: {baseline_submit_index_closer}")
    print(f"  Baseline overlap: {baseline_overlap}")
    print(f"  Baseline inbound: {inbound_baseline}")
    print()
    print("Optimized (current plan):")
    print(f"  Closer optimized submit index: {closer_submit_index_opt}")
    print(f"  Optimized overlap: {optimized_overlap}")
    print(f"  Optimized inbound: {inbound_optimized}")
    print()
    print("Adjustments:")
    print(f"  Submit index shift earlier by: {delta_submit_days} days")
    print(f"  Additional overlap gained: {optimized_overlap - baseline_overlap} days")
    print(f"  Inbound reduction: {inbound_baseline - inbound_optimized} days")
    print()
    print("Identity Checks:")
    print(f"  Baseline: {inbound_baseline} = {sum_unbounded} - {baseline_overlap}")
    print(f"  Optimized: {inbound_optimized} = {sum_unbounded} - {optimized_overlap}")
    print(f"  Summary reported inbound (unbounded): {summary.get('total_effective_days_unbounded')} matches optimized? {summary.get('total_effective_days_unbounded') == inbound_optimized}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/simulate_overlap_optimizer_demo.py <plan_wd.json>")
        sys.exit(1)
    main(sys.argv[1])
