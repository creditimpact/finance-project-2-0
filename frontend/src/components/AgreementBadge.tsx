import * as React from 'react';
import { Badge } from './ui/badge';
import { AgreementLevel } from './accountFieldTypes';
import { cn } from '../lib/utils';

const LABELS: Record<AgreementLevel, string> = {
  all: 'All agree',
  majority: '2 of 3',
  mixed: 'Mixed',
  none: '—'
};

const TONE_STYLES: Record<AgreementLevel, string> = {
  all: 'border-transparent bg-slate-200 text-slate-900',
  majority: 'border-transparent bg-sky-200 text-sky-950',
  mixed: 'border-transparent bg-amber-200 text-amber-950',
  none: 'border-transparent bg-slate-100 text-slate-700'
};

export interface AgreementBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  agreement: AgreementLevel;
}

export function AgreementBadge({ agreement, className, ...props }: AgreementBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn('px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide', TONE_STYLES[agreement], className)}
      {...props}
    >
      {LABELS[agreement]}
    </Badge>
  );
}

export default AgreementBadge;
