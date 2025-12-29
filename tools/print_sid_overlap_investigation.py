import json, sys, datetime
from pathlib import Path

# Usage: python tools/print_sid_overlap_investigation.py <plan_wd.json>

COLS = [
    "idx","field","calendar_day_index","timeline.from_day","timeline.to_day",
    "effective_contribution_days","effective_contribution_days_unbounded",
    "sla_start_index","sla_end_index","running_unbounded_at_submit",
    "running_total_days_unbounded_after","overlap_unbounded_days_with_prev",
    "handoff_days_before_prev_sla_end"
]

def days_index(anchor_date, target_date):
    return (target_date - anchor_date).days

def parse_date(s):
    return datetime.date.fromisoformat(s)

def main(path_str: str):
    p = Path(path_str)
    data = json.loads(p.read_text())
    anchor_date = parse_date(data["anchor"]["date"]) if "anchor" in data else None
    seq = data.get("sequence_debug", [])
    summary = data.get("summary", {})

    rows = []
    sum_unbounded = 0
    for entry in seq:
        idx = entry.get("idx")
        field = entry.get("field")
        c_idx = entry.get("calendar_day_index")
        tl = entry.get("timeline", {})
        from_day = tl.get("from_day")
        to_day = tl.get("to_day")
        eff = entry.get("effective_contribution_days")
        eff_unbounded = entry.get("effective_contribution_days_unbounded")
        sum_unbounded += int(eff_unbounded or 0)
        sla = entry.get("sla_window", {})
        start_date = parse_date(sla.get("start", {}).get("date"))
        end_date = parse_date(sla.get("end", {}).get("date"))
        sla_start_index = days_index(anchor_date, start_date)
        sla_end_index = days_index(anchor_date, end_date)
        run_unb_submit = entry.get("running_unbounded_at_submit")
        run_unb_after = entry.get("running_total_days_unbounded_after")
        ovlp_unb_prev = entry.get("overlap_unbounded_days_with_prev")
        handoff_before_prev_end = entry.get("handoff_days_before_prev_sla_end")
        rows.append([
            idx,field,c_idx,from_day,to_day,eff,eff_unbounded,
            sla_start_index,sla_end_index,run_unb_submit,run_unb_after,
            ovlp_unb_prev,handoff_before_prev_end
        ])

    total_overlap = summary.get("total_overlap_unbounded_days")
    calculated_inbound = sum_unbounded - int(total_overlap or 0)
    reported_total = summary.get("total_effective_days_unbounded")

    # Print table
    print("INVESTIGATION TABLE (CURRENT SID STATE):")
    print(" | ".join(COLS))
    for r in rows:
        print(" | ".join(str(x) if x is not None else "-" for x in r))
    print()
    print(f"sum_unbounded: {sum_unbounded}")
    print(f"total_overlap_unbounded_days: {total_overlap}")
    print(f"calculated_inbound: {calculated_inbound}")
    print(f"summary.total_effective_days_unbounded: {reported_total}")
    print(f"identity_match: {calculated_inbound == reported_total}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/print_sid_overlap_investigation.py <plan_wd.json>")
        sys.exit(1)
    main(sys.argv[1])
