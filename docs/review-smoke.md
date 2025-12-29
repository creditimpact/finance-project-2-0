# Frontend review smoke test

Use these curl checks against a local server (default `http://localhost:5000`) with a prepared run ID, e.g. `SID-frontend-review-smoke`. All responses should return `200`.

```bash
export SID="SID-frontend-review-smoke"
export ORIGIN="http://localhost:5000"

# Verify the manifest is reachable
curl -fsS "$ORIGIN/api/runs/$SID/frontend/review/index" | jq '.packs | length'

# Inspect the pack listing (file paths must use forward slashes)
curl -fsS "$ORIGIN/api/runs/$SID/frontend/review/packs" | jq '.'

# Fetch each pack using the file path from the listing
curl -fsS "$ORIGIN/runs/$SID/frontend/review/packs/idx-001.json" | jq '.'
```

If the listing returns multiple pack entries, repeat the final curl for each `file` path returned.
