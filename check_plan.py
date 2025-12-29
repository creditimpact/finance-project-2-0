import json

plan_path = r"runs\8aa0361b-a36a-40af-b907-93f87b3e9a64\cases\accounts\7\strategy\equifax\plan.json"
with open(plan_path) as f:
    p = json.load(f)

best = p["best_overall"]
seq = best.get("sequence_debug", [])
inv_header = best.get("inventory_header", {})
inv = inv_header.get("inventory_selected", [])

print(f"Sequence entries: {len(seq)}")
print(f"Inventory selected entries: {len(inv)}")
print()

if seq and inv:
    for i in range(min(len(seq), len(inv))):
        seq_entry = seq[i]
        inv_entry = inv[i]
        print(f"Entry {i+1}: {seq_entry['field']}")
        print(f"  SEQ: idx={seq_entry['calendar_day_index']}, date={seq_entry['submit']['date']}, eff={seq_entry['effective_contribution_days']}, running={seq_entry['running_total_days_after']}")
        print(f"  INV: idx={inv_entry['planned_submit_index']}, date={inv_entry['planned_submit_date']}, eff={inv_entry['effective_contribution_days']}, running={inv_entry['running_total_after']}")
        
        if (seq_entry['calendar_day_index'] != inv_entry['planned_submit_index'] or
            seq_entry['submit']['date'] != inv_entry['planned_submit_date'] or
            seq_entry['effective_contribution_days'] != inv_entry['effective_contribution_days'] or
            seq_entry['running_total_days_after'] != inv_entry['running_total_after']):
            print(f"  ❌ MISMATCH!")
        else:
            print(f"  ✅ MATCH")
        print()
