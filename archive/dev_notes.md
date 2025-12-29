## Dev Notes

- Added strict warning policy (`-W error`) to pytest and resolved all existing warnings.
- Replaced deprecated `datetime.utcnow()` calls with timezone-aware `datetime.now(UTC)`.
- Adjusted tests to avoid or capture sanitization warnings and ensured files are closed.
- Validation date normalization now reads the detected convention from `trace/date_convention.json` via the tolerance loader; the manifest is only used to locate that trace file when syncing runs between machines.
- The `last_verified` field rides the same ±5 day tolerance as other reporting dates so deterministic comparisons suppress near-matches instead of escalating C4/C5 noise.
- Keep the curated `account_rating` alias map (see `backend/core/logic/consistency.py`) focused on high-signal synonyms so the normalization remains predictable for reviewers.
