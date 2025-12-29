# Template Routing

The letter router selects an HTML template for each `action_tag` based on
available candidates and bureau evidence.

## Routing flow
1. Load candidate templates for the tag from `router/template_config.yaml`.
2. If the evidence source matches the requested credit reporting agency (CRA),
   templates whose filename includes `_bureau_<bureau>` are tried first.
3. Otherwise the router falls back to a generic template for the tag.
4. When no candidate exists, the router defaults to `default_dispute.html` and
   labels the metric tag as `default`.
5. A deterministic hash of `(action_tag, template)` is written to the audit log
   when an `AuditLogger` is supplied.

## Metrics
- `router.candidate_selected{tag}` – emitted for every selection, with `tag`
  set to the action tag or `default` on fallback.
- `router.finalized.{tag}.{template}` – emitted when a template is finalized.
- `router.missing_fields.finalize.{tag}.{field}` – required fields still missing
  after finalization.
- `router.sanitize_success.{template}` / `router.sanitize_failure.{template}` –
  outcome of HTML sanitization.

## Action tags
| action_tag | candidate templates | required fields |
|---|---|---|
| dispute | `dispute_letter_template.html` | `bureau` |
| goodwill | `goodwill_letter_template.html` | `creditor` |
| fraud_dispute | `fraud_dispute_letter_template.html` | `creditor_name`, `account_number_masked`, `bureau`, `legal_safe_summary`, `is_identity_theft` |
| personal_info_correction | `personal_info_correction_letter_template.html` | `client_name`, `client_address_lines`, `date_of_birth`, `ssn_last4`, `legal_safe_summary` |
| bureau_dispute | `bureau_dispute_letter_template.html` | `creditor_name`, `account_number_masked`, `bureau`, `legal_safe_summary` |
| mov | `mov_letter_template.html` | `creditor_name`, `account_number_masked`, `legal_safe_summary`, `cra_last_result`, `days_since_cra_result` |
| inquiry_dispute | `inquiry_dispute_letter_template.html` | `inquiry_creditor_name`, `account_number_masked`, `bureau`, `legal_safe_summary`, `inquiry_date` |
| medical_dispute | `medical_dispute_letter_template.html` | `creditor_name`, `account_number_masked`, `bureau`, `legal_safe_summary`, `amount`, `medical_status` |
| custom_letter | `general_letter_template.html` | `recipient` |
| instruction | `instruction_template.html` | `client_name`, `date`, `accounts_summary`, `per_account_actions` |
| duplicate | `duplicate_memo.html` | `memo` |

## Example
```python
from backend.core.letters.router import select_template

decision = select_template("dispute", {"bureau": "experian"}, phase="candidate")
print(decision.template_path)
```

For deterministic rule evaluation and audit log details, see
[Stage 2.5](../STAGE_2_5.md) and the
[Post-Refactor Audit](../POST_REFACTOR_AUDIT.md).
