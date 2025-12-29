import json, pathlib

session_id = "sess_sc_e2e"
case_path = pathlib.Path(".cases") / f"{session_id}.json"

data = json.loads(case_path.read_text())
accounts = (
    data.get("accounts")
    or data.get("accounts_map")
    or data.get("accounts_list")
    or {}
)
if isinstance(accounts, dict):
    cases_written = len(accounts)
else:
    cases_written = len(list(accounts))

issuers = set()
weak_blocks: list[int] = []

if isinstance(accounts, dict):
    iterator = accounts.values()
else:
    iterator = accounts

for acct in iterator:
    fields = acct.get("fields", acct)
    issuer = fields.get("issuer") or fields.get("creditor_type") or ""
    if issuer:
        issuers.add(issuer)
    if fields.get("_weak_fields"):
        weak_blocks.append(acct.get("block_index"))

block_count = cases_written

print(
    json.dumps(
        {
            "session_id": session_id,
            "case_file": str(case_path),
            "block_count": block_count,
            "cases_written": cases_written,
            "issuers": sorted(issuers),
            "weak_blocks": sorted([b for b in weak_blocks if b is not None]),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
