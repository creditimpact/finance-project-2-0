# Tools

Utility scripts for development and maintenance.

## process_accounts.py

Process a SmartCredit analysis JSON report into bureau-specific payloads.

```bash
python tools/process_accounts.py path/to/analyzed_report.json output_dir
```

## replay_outcomes.py

Recompute outcome events from raw bureau reports for debugging.

```bash
python tools/replay_outcomes.py report1.json [report2.json ...]
```

## migrate_frontend_packs.py

Move legacy frontend packs into the new review-stage layout.

```bash
# migrate a specific run directory
python tools/migrate_frontend_packs.py runs/S123

# migrate every run under the default runs/ root
python tools/migrate_frontend_packs.py --all

# treat positional arguments as run IDs relative to --runs-root
python tools/migrate_frontend_packs.py S555 --runs-root /data/runs
```

The script creates `frontend/review/packs/` and `frontend/review/responses/` as
needed, moves any legacy pack files into the packs directory, and rebuilds
`frontend/review/index.json` so the UI can discover the migrated packs.
