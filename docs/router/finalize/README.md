# Finalize Routing

The finalize phase selects an HTML template after strategy results merge and validation.

## CRA-first routing
During finalization a bureau-specific template is preferred for `bureau_dispute`. When
`bureau` is provided, the router looks for a template named
`<bureau>_bureau_dispute_letter_template.html` before falling back to the generic
`bureau_dispute_letter_template.html`.

## Template selection rules
* Each `action_tag` maps to a template and set of required fields.
* On finalize, required fields and substantive markers are validated. Missing
  required fields trigger `router.finalize_errors` and replace the template with
  `default_dispute.html`.
* `duplicate` and `ignore` tags short-circuit without rendering.

### Template matrix
| action_tag | template | required fields | substance checklist |
|---|---|---|---|
| dispute | dispute_letter_template.html | bureau | fcra_611, investigation_request, account_number_masked, response_window |
| goodwill | goodwill_letter_template.html | creditor | non_promissory_tone, positive_history_reference, discretionary_request, no_admission |
| custom_letter | general_letter_template.html | recipient | — |
| instruction | instruction_template.html | client_name, date, accounts_summary, per_account_actions | — |
| fraud_dispute | fraud_dispute_letter_template.html | creditor_name, account_number_masked, bureau, legal_safe_summary, is_identity_theft | fcra_605b, ftc_report, block_or_remove_request, response_window |
| debt_validation | debt_validation_letter_template.html | collector_name, account_number_masked, bureau, legal_safe_summary, days_since_first_contact | fdcpa_1692g, validation_window_30_day |
| pay_for_delete | pay_for_delete_letter_template.html | collector_name, account_number_masked, legal_safe_summary, offer_terms | deletion_clause, payment_clause |
| mov | mov_letter_template.html | creditor_name, account_number_masked, legal_safe_summary, cra_last_result, days_since_cra_result | reinvestigation_request, method_of_verification, cra_last_result, days_since_cra_result |
| personal_info_correction | personal_info_correction_letter_template.html | client_name, client_address_lines, date_of_birth, ssn_last4, legal_safe_summary | update_request, ssn_last4, date_of_birth |
| cease_and_desist | cease_and_desist_letter_template.html | collector_name, account_number_masked, legal_safe_summary | stop_contact, collector_name |
| direct_dispute | direct_dispute_letter_template.html | creditor_name, account_number_masked, legal_safe_summary, furnisher_address | — |
| bureau_dispute | bureau_dispute_letter_template.html (CRA-specific first) | creditor_name, account_number_masked, bureau, legal_safe_summary | fcra_611, reinvestigation_request, account_number_masked |
| inquiry_dispute | inquiry_dispute_letter_template.html | inquiry_creditor_name, account_number_masked, bureau, legal_safe_summary, inquiry_date | — |
| medical_dispute | medical_dispute_letter_template.html | creditor_name, account_number_masked, bureau, legal_safe_summary, amount, medical_status | — |
| paydown_first | instruction_template.html | client_name, date, accounts_summary, per_account_actions | — |

## Fallback logic
* If a bureau-specific template is missing, the generic version is used.
* Missing required fields switch to `default_dispute.html` and emit
  `router.missing_fields.finalize.*` metrics.
* Unknown action tags raise `ValueError`.

## Sanitizer rules
`sanitize_rendered_html` cleans post-rendered HTML:
1. Collapse excess whitespace.
2. Redact PII via `redact_pii`.
3. Remove denylisted phrases per template (e.g., "promise to pay" for dispute
   and general letters; "goodwill" for collection accounts).
4. Require at least one `<p>` tag; missing structure is marked as an override.

Metrics include `sanitizer.applied.{template}`,
`policy_override_reason.{template}.{term}`, and
`router.sanitize_success.{template}`/`router.sanitize_failure.{template}`.

### Example
Input:
```html
<p>I promise to pay if you delete it.</p>
```
Output after `sanitize_rendered_html(..., 'dispute_letter_template.html', {})`:
```html
<p>I  if you delete it.</p>
```
Overrides: `["promise to pay"]`

### Example rendered letters
#### Bureau dispute
```html
<html><body>
<p>ABC Bank</p>
<p>Account ****1234 (EXPERIAN)</p>
<p>I dispute the accuracy of this account under FCRA §611 and request a reinvestigation.</p>
<p>The listed balance is incorrect.</p>
</body></html>
```

#### Goodwill
```html
<html><body>
<p>Dear XYZ Collections,</p>
<p>This is a goodwill request letter.</p>
</body></html>
```
