export const ALWAYS_INVESTIGATABLE_FIELDS = [
  // Open / Identification
  'date_opened',
  'closed_date',
  'account_type',
  'creditor_type',
  // Terms
  'high_balance',
  'credit_limit',
  'term_length',
  'payment_amount',
  'payment_frequency',
  // Activity
  'balance_owed',
  'last_payment',
  'past_due_amount',
  'date_of_last_activity',
  // Status / Reporting
  'account_status',
  'payment_status',
  'date_reported',
  // Histories
  'two_year_payment_history',
  'seven_year_history',
];

export const CONDITIONAL_FIELDS = [
  'account_number_display',
  'account_rating',
];

export const ALL_VALIDATION_FIELDS = [
  ...ALWAYS_INVESTIGATABLE_FIELDS,
  ...CONDITIONAL_FIELDS,
];

export const CONDITIONAL_FIELD_SET = new Set(CONDITIONAL_FIELDS);
export const ALL_VALIDATION_FIELD_SET = new Set(ALL_VALIDATION_FIELDS);

export const FIELD_LABELS = {
  date_opened: 'Date Opened',
  closed_date: 'Closed Date',
  account_type: 'Account Type',
  creditor_type: 'Creditor Type',
  high_balance: 'High Balance',
  credit_limit: 'Credit Limit',
  term_length: 'Term Length',
  payment_amount: 'Payment Amount',
  payment_frequency: 'Payment Frequency',
  balance_owed: 'Balance Owed',
  last_payment: 'Last Payment',
  past_due_amount: 'Past Due Amount',
  date_of_last_activity: 'Date of Last Activity',
  account_status: 'Account Status',
  payment_status: 'Payment Status',
  date_reported: 'Date Reported',
  two_year_payment_history: '2-Year Payment History',
  seven_year_history: '7-Year History',
  account_number_display: 'Account Number',
  account_rating: 'Account Rating',
};

export function formatValidationField(field) {
  const label = FIELD_LABELS[field] || field;
  if (CONDITIONAL_FIELD_SET.has(field)) {
    return `${label} ⚠️`;
  }
  return label;
}
