# Validation Field Matrix

The validation pipeline now focuses on a lean set of 21 tradeline fields so ops,
engineering, and QA have a shared contract when evaluating outcomes. This list
acts as the single source of truth for which discrepancies justify keeping or
opening a case.

## Always-investigatable fields (18)

If any bureau disagrees or a bureau is missing data for one of these fields,
open (or keep open) a case. They are split across five categories:

| Category | Field | Notes |
| --- | --- | --- |
| Open / Identification | `date_opened` | Account start date. |
|  | `closed_date` | Furnished closure date. |
|  | `account_type` | Normalized tradeline type (e.g., revolving, installment). |
|  | `creditor_type` | Furnisher classification (bank, collection, etc.). |
| Terms | `high_balance` | Highest balance ever reported. |
|  | `credit_limit` | Credit limit or original loan amount. |
|  | `term_length` | Contract length in months. |
|  | `payment_amount` | Scheduled payment obligation. |
|  | `payment_frequency` | Cadence of the scheduled payment. |
| Activity | `balance_owed` | Current balance due. |
|  | `last_payment` | Date or amount of the most recent payment (per bureau schema). |
|  | `past_due_amount` | Amount currently past due. |
|  | `date_of_last_activity` | Latest reported activity date. |
| Status / Reporting | `account_status` | High-level status (open, closed, charged-off, etc.). |
|  | `payment_status` | Delinquency bucket furnished in the Metro 2 status code. |
|  | `date_reported` | Snapshot date for the furnished trade. |
| Histories | `two_year_payment_history` | 24-month payment grid. |
|  | `seven_year_history` | Long-tail delinquency markers. |

## Conditional / soft fields (3)

These fields remain in validation but only escalate to a **strong** dispute when
the furnished content substantiates it. Otherwise, the case should stay weak or
`no_case` even if bureaus disagree.

- `creditor_remarks`
- `account_rating`
- `account_number_display`

Documented rationales should reflect why a weak or `no_case` determination was
appropriate when the evidence does not clearly support a dispute.

## Informational-only fields (out of scope)

The following fields stay in the data payload for context but do not trigger
validation actions:

- `account_description`
- `dispute_status`
- `last_verified`

Treat them as informational annotations for letter drafting or analytics only.
