# Stage 4 SmartCredit Parser Hardening

## Background
Stage 4 finalizes a fully deterministic parsing pipeline for SmartCredit PDFs. Parsing no longer calls or depends on any LLMs.

Deterministic parsing flow:

PyMuPDF text → Selective OCR (flag-gated) → Normalization → Deterministic Extractors → Case Store

Note: Stage A AI adjudication remains a separate, downstream stage and is not part of parsing.

## Problem
- Formatting variance (spacing, Unicode, OCR artifacts) can degrade deterministic extraction if not normalized.
- OCR must remain selective and controlled to avoid latency and noise.
- Observability should clearly track coverage, quality, and performance per step.

## Tasks
### Extractor Hardening
- Property-based tests for dates (`MM/DD/YYYY`, `MMM YYYY`, `YYYY`) and amounts (symbols, commas, negatives).
- Light fuzzing for tokens/sections (whitespace, line breaks, Unicode, OCR quirks).
- Targeted cases: collections, charge-off, student loans, mortgage, auto, joint, closed, dispute.

### Observability
- Dashboards: section parse times, per-field timings, coverage %, failure rate by cause.
- Alerts: SLOs for p95 parse time, failure rate, unknown-field rate.
- PII masking/anonymization in logs.

### Schema & Documentation
- Lock in SmartCredit JSON schema (see below).
- Short README: flags and flow; how to run tests and load tests.

## Acceptance Criteria
- Tests cover key format variations; load test passes within budget.
- Dashboard and alerts live with PII-safe logs.
- JSON schema finalized and documented.

## Why deterministic
- PII safety: no third-party transmission during parsing; strict data boundaries.
- Cost and latency: no per-document LLM costs; predictable p95.
- Testability: exact, reproducible outputs; easier CI coverage and diffing.

---

# SmartCredit Field Catalog
The following catalogue describes the hierarchical structure and field definitions used by the deterministic SmartCredit parser and OCR label hints.

## A) Meta & Scores (Header)
- **provider** �?" e.g. "SmartCredit / ConsumerDirect".  Labels: Provider, Source, Delivered By, SmartCredit.
- **model** �?" e.g. "VantageScore 3.0".  Labels: Model, Score Model, VantageScore.
- **report_view_timestamp** �?" timestamp printed in header/footer.  Labels: Report Viewed, Report Date/Time, Generated.
- **scores[]** (per bureau)
  - **bureau** �?" enum: TransUnion|Experian|Equifax.  Labels: TransUnion, Experian, Equifax.
  - **score** �?" integer (300�?"850 typically).  Labels: Score, Credit Score.
  - **as_of_date** �?" date the score applies to.  Labels: As of, Score Date.

*Normalization:* Dates �+' `YYYY-MM-DD` (accept `MM/DD/YYYY`, `MMM YYYY`, or just `YYYY`). Enums canonicalised.

## B) Personal Information (per bureau)
Fields repeated for each bureau (TransUnion, Experian, Equifax):
- **credit_report_date** �?" date the bureau�?Ts credit report was pulled. Labels: Credit Report, Report Date, As of.
- **name** �?" full name. Labels: Name, Consumer Name.
- **also_known_as** �?" optional AKA names. Labels: Also Known As, AKA, Other Names.
- **date_of_birth** �?" year or full date. Labels: DOB, Date of Birth, Birth Year.
- **current_address** �?" most recent address block. Labels: Current Address, Present Address.
- **previous_addresses[]** �?" list of past addresses. Labels: Previous Address(es), Prior Address(es).
- **employer** �?" current/last employer (optional). Labels: Employer, Current Employer.
- **consumer_statement** �?" consumer statement or �?oNone Reported�??. Labels: Consumer Statement, Statement, Note.

*Normalization:* Addresses kept as raw multi�?`line strings. "None Reported"/"No Data" ��' null.

## C) Summary (per bureau)
- **total_accounts** �?" int. Labels: Total Accounts, Accounts Total.
- **open_accounts** �?" int. Labels: Open Accounts, Open.
- **closed_accounts** �?" int. Labels: Closed Accounts, Closed.
- **delinquent_accounts** �?" int. Labels: Delinquent, Past Due Accounts.
- **derogatory_accounts** �?" int. Labels: Derogatory, Negative Accounts.
- **balances_total** �?" currency/float. Labels: Balances, Total Balances, Balance Total.
- **payments_total** �?" currency/float (monthly payments total). Labels: Payments, Total Payments, Monthly Payments.
- **public_records_count** �?" int. Labels: Public Records, Public Record(s).
- **inquiries_24m_count** �?" int. Labels: Inquiries (24 months), Inquiries Last 2 Years.

*Normalization:* Currency: strip $, commas �+' float. Missing/-- ��' null.

## D) Accounts
Accounts are grouped visually; keep `category_display` as seen:
- **Categories:** Revolving Accounts, Installment Accounts, Other Accounts, Collection Accounts (sometimes "Collections").

Per account (shared across bureaus):
- **creditor_name** �?" account header line. Labels: top block title, Creditor, Lender, Collections Agency.
- **category_display** �?" one of the 4 groups above.

Per account, per bureau (`per_bureau[]`):
- **bureau** �?" TransUnion|Experian|Equifax.
- **account_number_masked** �?" masked number (keep last4 if available). Labels: Account Number, Acct #, Acct No.
- **account_type** �?" high-level type (Credit Card, Auto Loan, Charge Account, Collection, Mortgage, Student Loan, Other). Labels: Type, Account Type.
- **account_type_detail** �?" optional detail. Labels: Type, Portfolio Type, Loan Type.
- **creditor_type** �?" bureau/industry classification. Labels: Creditor Type, Industry, Business Type.
- **account_status** �?" OPEN|CLOSED|PAID|DEROGATORY|COLLECTION|CHARGEOFF|SETTLED|... Labels: Account Status, Status.
- **payment_status** �?" CURRENT|LATE_30|LATE_60|LATE_90|LATE_120|COLLECTION|CHARGEOFF|... Labels: Payment Status, Pay Status.
- **account_rating** �?" high-level rating (often mirrors status). Labels: Rating, Account Rating.
- **account_description** �?" ownership/conditions (Individual|Joint|Terminated|Authorized User|...). Labels: Description, Ownership, Account Designator.
- **date_opened** �?" date. Labels: Date Opened, Opened.
- **closed_date** �?" date or null. Labels: Date Closed, Closed.
- **date_reported** �?" last reported date. Labels: Date Reported, Reported.
- **date_of_last_activity** �?" DLA if present. Labels: Date of Last Activity, DLA.
- **last_verified** �?" verification date if shown. Labels: Last Verified, Verified.
- **credit_limit** �?" for revolving; sometimes noted as "H/C".
- **high_balance** �?" highest balance / "H/C" depending on bureau.
- **balance_owed** �?" current balance. Labels: Balance, Current Balance, Amt Owed.
- **past_due_amount** �?" past due. Labels: Past Due, Amount Past Due.
- **payment_amount** �?" scheduled monthly payment. Labels: Monthly Payment, Payment.
- **last_payment** �?" date of last payment. Labels: Last Payment, Date of Last Payment.
- **term_length** �?" e.g., "54 Month(s
[... omitted 0 of 201 lines ...]

      "closed_accounts": 0,
      "delinquent_accounts": 0,
      "derogatory_accounts": 0,
      "balances_total": 0.0,
      "payments_total": 0.0,
      "public_records_count": 0,
      "inquiries_24m_count": 0
    }
  ],
  "accounts": [
    {
      "creditor_name": "string",
      "category_display": "Revolving|Installment|Other|Collection",
      "per_bureau": [
        {
          "bureau": "TransUnion",
          "account_number_masked": "string",
          "account_type": "CREDIT_CARD|AUTO_LOAN|CHARGE_ACCOUNT|MORTGAGE|STUDENT_LOAN|COLLECTION|OTHER",
          "account_type_detail": "string|null",
          "creditor_type": "string|null",
          "account_status": "OPEN|CLOSED|PAID|DEROGATORY|COLLECTION|CHARGEOFF|SETTLED|OTHER",
          "payment_status": "CURRENT|LATE_30|LATE_60|LATE_90|LATE_120|COLLECTION|CHARGEOFF|OTHER",
          "account_rating": "string|null",
          "account_description": "Individual|Joint|Authorized User|Terminated|...",
          "date_opened": "YYYY-MM-DD|null",
          "closed_date": "YYYY-MM-DD|null",
          "date_reported": "YYYY-MM-DD|null",
          "date_of_last_activity": "YYYY-MM-DD|null",
          "last_verified": "YYYY-MM-DD|null",
          "credit_limit": 0.0,
          "high_balance": 0.0,
          "balance_owed": 0.0,
          "past_due_amount": 0.0,
          "payment_amount": 0.0,
          "last_payment": "YYYY-MM-DD|null",
          "term_length": "string|null",
          "payment_frequency": "Monthly|--|null",
          "creditor_remarks": "string|null",
          "dispute_status": "string|null",
          "original_creditor": "string|null",
          "two_year_payment_history": [
            { "year": 2024, "month": "Oct", "status": "OK|30|60|90|120|CO|COLL|--" }
          ],
          "days_late_7y": { "late30": 0, "late60": 0, "late90": 0 }
        }
      ]
    }
  ],
  "public_information": {
    "items": [
      {
        "record_type": "Bankruptcy|Lien|Judgment",
        "filing_date": "YYYY-MM-DD",
        "status": "string",
        "court": "string|null",
        "amount": 0.0,
        "reference_id": "string|null"
      }
    ]
  },
  "inquiries": [
    { "creditor_name": "string", "date_of_inquiry": "YYYY-MM-DD", "bureau": "TransUnion" }
  ]
}
```

## H) OCR Label Hints
Common label variants to aid OCR:
- **Dates:** Date, Reported, As of, Opened, Closed, Verified, Last Activity, Last Payment, Filed.
- **Money:** Balance, High Balance, High Credit, Credit Limit, Past Due, Payment, Amount.
- **Statuses:** Account Status, Payment Status, Rating, Remarks, Dispute.
- **Parties:** Creditor, Original Creditor, Employer, Court.
- **Bureau markers:** TransUnion, Experian, Equifax.
