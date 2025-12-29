# Report Analysis Audit

## Problematic Accounts Source

Stage-A writes account summaries to `accounts_from_full.json` during report
analysis. After `TRACE_CLEANUP` the file remains under
`traces/blocks/<sid>/accounts_table/`.

### Flow
1. **Stage-A** – extracts accounts and saves `accounts_from_full.json`.
2. **Cleanup** – preserves the JSON alongside `_debug_full.tsv` and
   `general_info_from_full.json`.
3. **Problem Cases** – `build_problem_cases` loads
   `accounts_from_full.json` and writes case artifacts under
   `cases/<sid>/...`.

### Outputs
- `cases/<sid>/index.json` – summary of problematic accounts.
- `cases/<sid>/accounts/<account_id>.json` – individual case files.

### Optional Case Store
If a legacy Case Store session exists, the system reads problem accounts from
that path instead. The Case Store remains supported, but `accounts_from_full.json`
is the source of truth when it is absent.

## X-based Triad Parsing

Stage-A can optionally split certain rows into bureau-specific values based on
token X positions. When enabled, the parser inspects each page for the
`TransUnion / Experian / Equifax` header row. The horizontal midpoint of each
title defines four bands: the label column followed by the TransUnion, Experian,
and Equifax columns. Tokens are assigned to a band when their midpoint falls
inside the band with a tiny right-edge tolerance so that adjacent bands do not
leave gaps.

### Special Labels and Rows

* Fields normally have a trailing `:` in the label. `Account #` is treated as a
  label as well even though it ends with `#` instead of a colon.
* A row without a label is considered a continuation and is appended to the last
  opened field for whichever band produced it.
* Parsing stops when a label `Two-Year Payment History` is encountered; the raw
  lines after that label remain untouched.

### Environment Flags

```
RAW_TRIAD_FROM_X=True
RAW_JOIN_TOKENS_WITH_SPACE=True
```

`RAW_TRIAD_FROM_X` enables X-based triad parsing and adds extra debug output.
`RAW_JOIN_TOKENS_WITH_SPACE` keeps token order while inserting spaces between
neighboring tokens.

### Example Logs

```
[triad] bands: label=0-150 tu=150-300 xp=300-450 eq=450-600
[triad] field: Payment Status: Current | Current | Current
[triad] field: High Balance: 0 | 0 | 0
```

### Example JSON Output

```json
{
  "triad": {"enabled": true, "order": ["transunion", "experian", "equifax"]},
  "triad_fields": {
    "transunion": {"High Balance": "0", "Payment Status": "Current"},
    "experian": {"High Balance": "0", "Payment Status": "Current"},
    "equifax": {"High Balance": "0", "Payment Status": "Current"}
  },
  "triad_rows": [
    "High Balance: 0 0 0",
    "Payment Status: Current Current Current"
  ]
}
```

### Quick Verify

1. Run Stage-A with the environment flags above.
2. Check the Stage-A log for triad markers:

   ```bash
   grep TRIAD_HEADER_MATCH traces/blocks/<sid>/*.log
   grep TRIAD_CARRY traces/blocks/<sid>/*.log
   grep TRIAD_STOP traces/blocks/<sid>/*.log
   ```

   Hits confirm that triad parsing matched the header, carried lines, or stopped.
3. Inspect the parsed accounts output:

   ```bash
   cat traces/blocks/<sid>/accounts_table/accounts_from_full.json
   ```

   This JSON lists the extracted accounts and persists after `TRACE_CLEANUP`.
