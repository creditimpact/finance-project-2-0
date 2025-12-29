export type ClaimKey =
  | 'not_mine_fraud'
  | 'paid_in_full'
  | 'settlement_done'
  | 'closed_but_reported_open'
  | 'authorized_user'
  | 'bankruptcy_discharge'
  | 'medical_ins_paid'
  | 'repo_foreclosure_cured'
  | 'judgment_satisfied'
  | 'student_loan_rehab'
  | 'wrong_dofd'
  | 'third_party_paid'
  | 'mixed_file';

export const CLAIMS: Record<ClaimKey, {
  label: string;
  requiresDocs: boolean;
  requiredDocs: string[];
  hint?: string;
}> = {
  not_mine_fraud: {
    label: 'Not mine / Identity theft',
    requiresDocs: true,
    requiredDocs: ['id_theft_report_or_police'],
    hint: 'FTC Identity Theft Report or police report',
  },
  paid_in_full: {
    label: 'Paid in full',
    requiresDocs: true,
    requiredDocs: ['pay_proof', 'payoff_letter'],
  },
  settlement_done: {
    label: 'Settled',
    requiresDocs: true,
    requiredDocs: ['settlement_letter', 'settlement_payment_proof'],
  },
  closed_but_reported_open: {
    label: 'Closed but reported open',
    requiresDocs: true,
    requiredDocs: ['closure_letter_or_official_screenshot'],
  },
  authorized_user: {
    label: 'Authorized user only',
    requiresDocs: true,
    requiredDocs: ['statement_showing_AU_or_issuer_letter'],
  },
  bankruptcy_discharge: {
    label: 'Included in bankruptcy',
    requiresDocs: true,
    requiredDocs: ['bk_discharge_order', 'bk_schedule_with_account'],
  },
  medical_ins_paid: {
    label: 'Insurance paid (medical)',
    requiresDocs: true,
    requiredDocs: ['insurance_EOB', 'payment_proof_if_any'],
  },
  repo_foreclosure_cured: {
    label: 'Repo/Foreclosure cured',
    requiresDocs: true,
    requiredDocs: ['release_or_reinstatement_letter', 'final_payment_proofs'],
  },
  judgment_satisfied: {
    label: 'Judgment satisfied/vacated',
    requiresDocs: true,
    requiredDocs: ['satisfaction_or_vacate_order'],
  },
  student_loan_rehab: {
    label: 'Student loan rehab/consol.',
    requiresDocs: true,
    requiredDocs: ['rehab_completion_or_consolidation_payoff'],
  },
  wrong_dofd: {
    label: 'Wrong DOFD / re-aging',
    requiresDocs: true,
    requiredDocs: ['original_chargeoff_letter_or_old_statements'],
  },
  third_party_paid: {
    label: 'Paid by employer/3rd party',
    requiresDocs: true,
    requiredDocs: ['third_party_payment_letter', 'payment_proof'],
  },
  mixed_file: {
    label: 'Mixed file / info mismatch',
    requiresDocs: false,
    requiredDocs: [],
    hint: 'We’ll use ID + Proof of Address already on file',
  },
};

export const BASIC_ALWAYS_INCLUDED = ['gov_id', 'proof_of_address'];
