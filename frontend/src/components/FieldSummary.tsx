import * as React from 'react';
import { cn } from '../lib/utils';
import { AgreementLevel, MISSING_VALUE } from './accountFieldTypes';
import AgreementBadge from './AgreementBadge';

export interface FieldSummaryProps {
  label: string;
  value: string | null | undefined;
  agreement: AgreementLevel;
  className?: string;
}

export function FieldSummary({ label, value, agreement, className }: FieldSummaryProps) {
  const displayValue = value && value.trim() !== '' ? value : MISSING_VALUE;
  const isMissing = displayValue === MISSING_VALUE;

  return (
    <div
      className={cn(
        'flex min-w-[160px] flex-1 flex-col gap-2 rounded-lg border border-slate-200 bg-slate-50 p-4',
        className
      )}
    >
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</span>
      <span className={cn('text-sm font-semibold', isMissing ? 'text-slate-500' : 'text-slate-900')}>
        {displayValue}
      </span>
      <AgreementBadge agreement={agreement} />
    </div>
  );
}

export default FieldSummary;
