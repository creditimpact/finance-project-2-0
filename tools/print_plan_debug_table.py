"""
Print a compact debug table for a plan JSON emphasizing unbounded overlap model.
Usage:
  python tools/print_plan_debug_table.py <plan_json_path>
"""
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/print_plan_debug_table.py <plan_json_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    plan = json.loads(path.read_text(encoding='utf-8'))

    seq = plan.get("sequence_debug", [])
    summ = plan.get("summary", {})

    print("SUMMARY:")
    print(f"  inbound_cap_sum_items_unbounded: {summ.get('inbound_cap_sum_items_unbounded')}")
    print(f"  total_effective_days_unbounded:  {summ.get('total_effective_days_unbounded')}")
    print(f"  total_overlap_unbounded_days:    {summ.get('total_overlap_unbounded_days')}")
    print(f"  _debug_sum_items_unbounded:      {summ.get('_debug_sum_items_unbounded')}")
    print(f"  _debug_calculated_inbound:       {summ.get('_debug_calculated_inbound')}")
    print(f"  _debug_identity_valid:           {summ.get('_debug_identity_valid')}")
    print()

    header = (
        "idx",
        "field",
        "submit",
        "sla_start",
        "sla_end",
        "run_unb_at_submit",
        "eff_unbounded",
        "ovlp_raw_prev",
        "ovlp_unb_prev",
        "inbound_before",
        "inbound_after",
    )
    print("\t".join(header))

    inbound_before = 0
    sum_unbounded = 0
    sum_overlaps = 0

    for i, e in enumerate(seq, start=1):
        field = e.get("field")
        submit = e.get("submit", {}).get("date")
        sla_start = e.get("sla_window", {}).get("start", {}).get("date")
        sla_end = e.get("sla_window", {}).get("end", {}).get("date")
        run_unb_at_submit = e.get("running_unbounded_at_submit")
        eff_unbounded = e.get("effective_contribution_days_unbounded")
        ovlp_raw = e.get("overlap_raw_days_with_prev") if i > 1 else "-"
        ovlp_unb = e.get("overlap_unbounded_days_with_prev") if i > 1 else "-"
        inbound_before = run_unb_at_submit
        inbound_after = e.get("running_total_days_unbounded_after")
        row = (
            str(i),
            str(field),
            str(submit),
            str(sla_start),
            str(sla_end),
            str(run_unb_at_submit),
            str(eff_unbounded),
            str(ovlp_raw),
            str(ovlp_unb),
            str(inbound_before),
            str(inbound_after),
        )
        print("\t".join(row))

if __name__ == "__main__":
    main()
