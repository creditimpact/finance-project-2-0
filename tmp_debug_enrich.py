from backend.core.logic.report_analysis.block_exporter import enrich_block

def blk(heading, lines):
    return {"heading": heading, "lines": lines, "meta": {"block_type":"account"}, "fields": {}}

top_labels=[
    "Account #",
    "High Balance:",
    "Last Verified:",
]
lines=(
    ["SETERUS INC"]
    + top_labels
    + ["Transunion", "****2222", "$10,000", "21.10.2019"]
    + [
        "Account Status:",
        "Payment Status:",
        "Creditor Remarks:",
        "Payment Amount:",
        "Last Payment:",
        "Term Length:",
        "Past Due Amount:",
        "Account Type:",
        "Payment Frequency:",
        "Credit Limit:",
    ]
    + ["Transunion"]
    + [
        "Closed",
        "Current",
        "Transferred to another lender",
        "$0",
        "11.2.2019",
        "360 Month(s)",
        "$0",
        "Conventional real estate",
        "mortgage",
        "--",
    ]
    + ["Experian"]
    + [
        "Closed",
        "Current",
        "Transferred to another lender",
        "$0",
        "11.2.2019",
        "360 Month(s)",
        "$0",
        "Conventional real estate",
        "mortgage",
        "--",
    ]
    + ["Equifax"]
    + [
        "Closed",
        "Current",
        "Transferred to another lender",
        "$0",
        "11.2.2019",
        "360 Month(s)",
        "$0",
        "Conventional real estate",
        "mortgage",
        "--",
    ]
)
res=enrich_block(blk("SETERUS INC",lines))
import json
print(json.dumps(res["fields"], indent=2))
