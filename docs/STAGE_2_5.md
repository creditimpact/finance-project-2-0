# Stage 2.5 Deployment Guide

Stage 2.5 introduces normalization and deterministic rule evaluation of client statements so that strategy output and audit logs capture a "legal_safe_summary" along with rule metadata.

## Inputs
- **User statement** – raw text provided by the client (`user_statement_raw`).
- **Account facts** – structured attributes used by rules (e.g., `type`, `identity_theft`, `days_since_first_contact`).
- **Rulebook** – versioned YAML file (`backend/policy/rulebook.yaml`) defining limits, precedence and exclusions.

## Outputs
Each account receives a Stage 2.5 payload matching `backend/core/logic/strategy/stage_2_5_schema.json`:
- `legal_safe_summary`
- `suggested_dispute_frame`
- `rule_hits`
- `needs_evidence`
- `red_flags`
- `prohibited_admission_detected`
- `rulebook_version`

## Rule Evaluation
1. Admissions in `user_statement_raw` are neutralized into a legally safe summary.
2. Rules from the rulebook are evaluated against the normalized statement and account facts.
3. Matches are ordered by rulebook `precedence`; earlier rules win when conflicts arise.
4. `exclusions` suppress lower‑priority rules so only compatible rule hits remain.
5. The final result deduplicates `rule_hits` and `needs_evidence` and selects the first `suggested_dispute_frame`.

### Precedence and Exclusion Rules
- **Precedence:** `E_IDENTITY`, `E_IDENTITY_NEEDS_AFFIDAVIT`, `D_VALIDATION`, `C_MOV`, `H_OBSOL`, `A_CRA_DISPUTE`, `B_DIRECT_DISPUTE`, `L_DUPLICATE_TRADELINE`, `M_UNAUTHORIZED_INQUIRY`, `J_MEDICAL`, `K_UTILIZATION_PAYDOWN`, `F_GOODWILL`, `G_PFD`, `I_CEASE`.
- **Exclusions:** Examples:
  - `E_IDENTITY` suppresses `F_GOODWILL`, `G_PFD`, `D_VALIDATION`, `A_CRA_DISPUTE`, `B_DIRECT_DISPUTE`, `H_OBSOL`.
  - `D_VALIDATION` suppresses `A_CRA_DISPUTE`, `B_DIRECT_DISPUTE`, `G_PFD`.

## Metrics
Stage 2.5 emits counters via `analytics_tracker`:
- `s2_5_accounts_total`
- `s2_5_rule_hits_total`
- `s2_5_needs_evidence_total`
- `s2_5_latency_ms`
- `s2_5_rule_hits_per_account`
- `s2_5_admissions_detected_total`
- `rulebook.tag_selected.{tag}`
- `rulebook.suppressed_rules.{rule_name}`

## Rulebook Update Workflow
1. Edit `backend/policy/rulebook.yaml` and bump its `version`.
2. Document the change in `CHANGELOG.md`.
3. Run validation tests:
   ```bash
   pytest tests/strategy/test_stage_2_5_rules.py tests/policy/test_policy_loader.py
   ```
4. Commit the updated rulebook and deploy.

## Examples
### Identity theft
Before:
```json
{ "account_id": "123", "user_statement_raw": "This account isn't mine." }
```
After:
```json
{
  "account_id": "123",
  "user_statement_raw": "This account isn't mine.",
  "stage_2_5": {
    "legal_safe_summary": "Client reports identity theft and requests verification.",
    "suggested_dispute_frame": "fraud",
    "rule_hits": ["E_IDENTITY", "E_IDENTITY_NEEDS_AFFIDAVIT"],
    "needs_evidence": ["identity_theft_affidavit"],
    "red_flags": [],
    "prohibited_admission_detected": false,
    "rulebook_version": "1.2.0"
  }
}
```

### Collection
Before:
```json
{
  "account_id": "A1",
  "type": "collection",
  "user_statement_raw": "They keep calling about a debt I don't owe."
}
```
After:
```json
{
  "account_id": "A1",
  "type": "collection",
  "user_statement_raw": "They keep calling about a debt I don't owe.",
  "stage_2_5": {
    "legal_safe_summary": "Client disputes ownership and requests validation of the debt.",
    "suggested_dispute_frame": "debt_validation",
    "rule_hits": ["D_VALIDATION"],
    "needs_evidence": ["proof_of_payment"],
    "red_flags": [],
    "prohibited_admission_detected": false,
    "rulebook_version": "1.2.0"
  }
}
```

## Pre-deploy Checks
1. **Run unit tests** to ensure the normalizer and logging behave as expected:
   ```bash
   pytest tests/strategy/test_stage_2_5_pipeline.py tests/strategy/test_rule_logging.py
   ```
2. **Execute the minimal workflow integration test** to confirm Stage 2.5 data is persisted during the full pipeline:
   ```bash
   pytest tests/test_local_workflow.py::test_skip_goodwill_when_identity_theft
   ```

## Verifying in a Sandbox
1. **Process a sample SmartCredit report** through the CLI:
   ```bash
   python main.py path/to/report.pdf user@example.com
   ```
2. After the run, inspect the generated client folder under `Clients/<YYYY-MM>/<Client>_cli/`.
   - The `strategy.json` file should contain Stage 2.5 fields for each account:
     - `legal_safe_summary`
     - `rule_hits`
     - `needs_evidence`
     - `red_flags`
   - Accounts without statements will show `"legal_safe_summary": "No statement provided"` and `"rule_hits": []`.
3. Review audit logs (or analytics counters) for `rule_evaluated` events to ensure Stage 2.5 evaluations are logged.

Once these checks pass, Stage 2.5 can be promoted to production.
