import * as React from 'react';
import type { ClaimSchema, DocKey } from '../types/review';
import { formatDocKey, type NormalizedAttachments } from '../utils/reviewClaims';
import { cn } from '../lib/utils';

type Props = {
  claims: ClaimSchema[];
  autoAttachBase: DocKey[];
  selected: string[];
  attachments: NormalizedAttachments;
  onChange: (next: string[]) => void;
  onFilesSelected: (claim: string, docKey: DocKey, files: File[]) => void;
};

const DEFAULT_DOC_TIP = 'PDF, JPG, or PNG files work best.';

const DOC_TIPS: Record<string, string> = {
  ftc_id_theft_report: 'Upload your FTC Identity Theft Report (PDF/JPG/PNG).',
  police_report: 'Police report documenting the incident (PDF/JPG/PNG).',
  proof_of_payment: 'Receipts or statements showing payment (PDF/JPG/PNG).',
  paid_in_full_letter: 'Paid-in-full confirmation letter (PDF/JPG/PNG).',
  settlement_letter: 'Signed settlement agreement (PDF/JPG/PNG).',
  billing_statement: 'Recent billing statement (PDF/JPG/PNG).',
  bank_statement: 'Bank statement covering the disputed period (PDF/JPG/PNG).',
  employer_letter: 'Letter from employer confirming payment (PDF/JPG/PNG).',
  insurance_eob: 'Insurance Explanation of Benefits (PDF/JPG/PNG).',
  medical_provider_letter: 'Letter from the provider confirming resolution (PDF/JPG/PNG).',
  bankruptcy_docket: 'Bankruptcy docket (PDF/JPG/PNG).',
  bankruptcy_discharge: 'Bankruptcy discharge order (PDF/JPG/PNG).',
  servicer_transfer_letter: 'Servicer transfer letter (PDF/JPG/PNG).',
  auto_pay_proof: 'Screenshot or document showing auto-pay setup (PDF/JPG/PNG).',
  goodwill_support: 'Supporting note explaining the hardship (PDF/JPG/PNG).',
  repossession_release: 'Repossession release letter (PDF/JPG/PNG).',
  foreclosure_cure: 'Proof that the foreclosure was cured (PDF/JPG/PNG).',
  judgment_vacated: 'Court order showing the judgment was vacated (PDF/JPG/PNG).',
  student_loan_rehab: 'Student loan rehab or consolidation proof (PDF/JPG/PNG).',
  gov_id: 'Government-issued ID (PDF/JPG/PNG).',
  proof_of_address: 'Recent proof of address (PDF/JPG/PNG).',
};

function getDocTip(docKey: string): string {
  return DOC_TIPS[docKey] ?? DEFAULT_DOC_TIP;
}

function ClaimPicker({
  claims,
  autoAttachBase,
  selected,
  attachments,
  onChange,
  onFilesSelected,
}: Props) {
  const handleToggle = React.useCallback(
    (claimKey: string) => {
      const isSelected = selected.includes(claimKey);
      const nextClaims = isSelected
        ? selected.filter((entry) => entry !== claimKey)
        : [...selected, claimKey];
      onChange(nextClaims);
    },
    [onChange, selected]
  );

  const handleFileChange = React.useCallback(
    (claimKey: string, docKey: DocKey) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = event.target.files ? Array.from(event.target.files) : [];
      event.target.value = '';
      if (fileList.length === 0) {
        return;
      }
      onFilesSelected(claimKey, docKey, fileList);
    },
    [onFilesSelected]
  );

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {claims.map((claim) => {
          const isSelected = selected.includes(claim.key);
          return (
            <label
              key={claim.key}
              className={cn(
                'flex cursor-pointer flex-col gap-2 rounded-md border border-slate-200 bg-white p-3 transition',
                isSelected ? 'border-slate-400 ring-1 ring-slate-400' : 'hover:border-slate-300'
              )}
            >
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500"
                  checked={isSelected}
                  onChange={() => handleToggle(claim.key)}
                />
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-900">{claim.title}</p>
                  {claim.description ? (
                    <p className="text-xs text-slate-600">{claim.description}</p>
                  ) : null}
                  <p className="text-xs text-slate-500">
                    {claim.requires.length > 0
                      ? `Requires ${claim.requires.length} ${
                          claim.requires.length === 1 ? 'document' : 'documents'
                        }.`
                      : 'No extra documents required.'}
                  </p>
                </div>
              </div>
            </label>
          );
        })}
      </div>
      <p className="text-xs text-slate-500">
        {autoAttachBase.length > 0
          ? `Your basic ${autoAttachBase.map(formatDocKey).join(' & ')} will be attached automatically.`
          : 'Your basic ID & Proof of Address are automatically attached to all disputes.'}
      </p>
      <div className="space-y-4">
        {selected.map((claimKey) => {
          const claim = claims.find((entry) => entry.key === claimKey);
          if (!claim) {
            return null;
          }
          if (claim.requires.length === 0 && (!claim.optional || claim.optional.length === 0)) {
            return null;
          }
          const requiredDocs = claim.requires;
          const optionalDocs = claim.optional ?? [];
          const renderDocUpload = (docKey: DocKey, options: { required: boolean }) => {
            const uploaded = attachments[docKey] ?? [];
            const hasDocs = Array.isArray(uploaded) && uploaded.length > 0;
            return (
              <div
                key={docKey}
                className={cn(
                  'rounded-md border bg-white p-3',
                  options.required ? 'border-slate-200' : 'border-dashed border-slate-300'
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{formatDocKey(docKey)}</p>
                    <p className="text-xs text-slate-500">
                      {options.required
                        ? 'Required document'
                        : 'Optional, but helps strengthen the dispute.'}
                    </p>
                  </div>
                  {hasDocs ? (
                    <span role="img" aria-label="Document uploaded" className="text-base">
                      ✅
                    </span>
                  ) : null}
                </div>
                <input
                  type="file"
                  multiple
                  onChange={handleFileChange(claim.key, docKey)}
                  className="mt-2 block text-sm text-slate-700 file:mr-3 file:rounded-md file:border file:border-slate-300 file:bg-white file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-50"
                />
                <p className="mt-2 text-xs text-slate-500">{getDocTip(docKey)}</p>
              </div>
            );
          };
          return (
            <details key={claim.key} className="rounded-lg border border-slate-200 bg-slate-50 p-4" open>
              <summary className="cursor-pointer text-sm font-semibold text-slate-900">
                Upload documents for {claim.title}
              </summary>
              <div className="mt-3 space-y-3">
                {requiredDocs.map((docKey) => renderDocUpload(docKey, { required: true }))}
                {optionalDocs.length > 0 ? (
                  <details className="rounded-md border border-dashed border-slate-300 bg-white p-3">
                    <summary className="cursor-pointer text-sm font-semibold text-slate-900">
                      Add optional docs (recommended)
                    </summary>
                    <div className="mt-3 space-y-3">
                      {optionalDocs.map((docKey) => renderDocUpload(docKey, { required: false }))}
                    </div>
                  </details>
                ) : null}
              </div>
            </details>
          );
        })}
      </div>
    </div>
  );
}

export default ClaimPicker;
