import type claimsSchemaJson from '../../../shared/claims_schema.json';

export type PrimaryIssue =
  | 'collection'
  | 'delinquency'
  | 'chargeoff'
  | 'bankruptcy'
  | 'repossession'
  | 'foreclosure'
  | 'judgment'
  | 'student_loan'
  | 'medical'
  | 'authorized_user'
  | 'mixed_file'
  | 'wrong_dofd'
  | 'closed_but_open'
  | 'paid_in_full'
  | 'settled'
  | 'paid_by_employer'
  | 'repo_cured'
  | 'goodwill'
  | 'generic';

export type DocKey =
  | 'gov_id'
  | 'proof_of_address'
  | 'ftc_id_theft_report'
  | 'police_report'
  | 'proof_of_payment'
  | 'settlement_letter'
  | 'paid_in_full_letter'
  | 'billing_statement'
  | 'bank_statement'
  | 'employer_letter'
  | 'insurance_eob'
  | 'medical_provider_letter'
  | 'bankruptcy_docket'
  | 'bankruptcy_discharge'
  | 'servicer_transfer_letter'
  | 'auto_pay_proof'
  | 'goodwill_support'
  | 'repossession_release'
  | 'foreclosure_cure'
  | 'judgment_vacated'
  | 'student_loan_rehab';

export interface ClaimSchema {
  key: string;
  title: string;
  description?: string;
  requires: DocKey[];
  optional?: DocKey[];
  autoAttach?: DocKey[];
  minUploads?: number;
}

export interface IssueSchema {
  issue: PrimaryIssue;
  claims: ClaimSchema[];
}

export interface ReviewSchema {
  autoAttachBase: DocKey[];
  byIssue: IssueSchema[];
}

export type SharedReviewSchema = typeof claimsSchemaJson;

export type ClaimKey = ClaimSchema['key'];

export interface PackClaimsPayload {
  autoAttachBase: DocKey[];
  items: ClaimSchema[];
}

export type AttachmentsMap = Partial<Record<DocKey, string | string[]>>;
