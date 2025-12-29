import * as React from 'react';
import { cn } from '../lib/utils';
import { AgreementLevel, BUREAUS, BUREAU_LABELS, BureauKey, MISSING_VALUE } from './accountFieldTypes';

export interface BureauGridField {
  fieldKey: string;
  label: string;
  values: Partial<Record<BureauKey, string>>;
  agreement: AgreementLevel;
}

export interface BureauGridProps {
  fields: BureauGridField[];
  className?: string;
  id?: string;
}

export function BureauGrid({ fields, className, id }: BureauGridProps) {
  const generatedId = React.useId();
  const gridId = id ?? generatedId;

  const columnHeaderIds = React.useMemo(
    () =>
      BUREAUS.reduce<Record<BureauKey, string>>((acc, bureau) => {
        acc[bureau] = `${gridId}-header-${bureau}`;
        return acc;
      }, {}),
    [gridId]
  );

  return (
    <div
      id={gridId}
      role="table"
      aria-label="Credit bureau comparison"
      className={cn('overflow-hidden rounded-lg border border-slate-200', className)}
    >
      <div
        role="rowgroup"
        className="grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))] bg-slate-50 text-sm font-medium text-slate-600"
      >
        <div role="columnheader" className="px-4 py-3 text-left">
          Field
        </div>
        {BUREAUS.map((bureau) => (
          <div
            key={bureau}
            role="columnheader"
            id={columnHeaderIds[bureau]}
            className="px-4 py-3 text-center"
          >
            {BUREAU_LABELS[bureau]}
          </div>
        ))}
      </div>
      <div role="rowgroup" className="divide-y divide-slate-200 text-sm">
        {fields.map((field) => {
          const highlight = field.agreement === 'majority' || field.agreement === 'mixed';
          const rowLabelId = `${gridId}-row-${field.fieldKey}`;

          return (
            <div
              key={field.fieldKey}
              role="row"
              className={cn(
                'grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))]',
                highlight && 'bg-amber-50'
              )}
            >
              <div
                role="rowheader"
                id={rowLabelId}
                className="px-4 py-3 font-medium text-slate-700"
              >
                {field.label}
              </div>
              {BUREAUS.map((bureau) => {
                const value = field.values[bureau] ?? MISSING_VALUE;
                const isMissing = value === MISSING_VALUE;

                return (
                  <div
                    key={bureau}
                    role="cell"
                    aria-labelledby={rowLabelId}
                    aria-describedby={columnHeaderIds[bureau]}
                    className={cn(
                      'px-4 py-3 text-center',
                      isMissing ? 'text-slate-500' : 'text-slate-800'
                    )}
                  >
                    {isMissing ? MISSING_VALUE : value}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default BureauGrid;
