import * as React from 'react';
import { cn } from '../lib/utils';

export interface AccountCardSkeletonProps {
  className?: string;
}

const SUMMARY_PLACEHOLDERS = new Array(5).fill(null);
const QUESTION_PLACEHOLDERS = new Array(4).fill(null);

export function AccountCardSkeleton({ className }: AccountCardSkeletonProps) {
  return (
    <div
      className={cn(
        'w-full rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow',
        'animate-pulse',
        className
      )}
    >
      <div className="space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-2">
            <div className="h-5 w-48 rounded bg-slate-200" />
            <div className="h-3 w-64 max-w-full rounded bg-slate-200" />
          </div>
          <div className="h-6 w-32 rounded-full bg-slate-200" />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {SUMMARY_PLACEHOLDERS.map((_, index) => (
            <div
              key={index}
              className="flex min-w-[160px] flex-1 flex-col gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4"
            >
              <div className="h-2.5 w-20 rounded bg-slate-200" />
              <div className="h-4 w-32 rounded bg-slate-200" />
              <div className="h-3 w-16 rounded bg-slate-200" />
            </div>
          ))}
        </div>

        <div className="space-y-4">
          <div className="h-4 w-48 rounded bg-slate-200" />
          <div className="grid gap-3 md:grid-cols-2">
            {QUESTION_PLACEHOLDERS.map((_, index) => (
              <div key={index} className="flex flex-col gap-3 rounded-lg border border-slate-200 p-4">
                <div className="space-y-2">
                  <div className="h-3 w-28 rounded bg-slate-200" />
                  <div className="h-2.5 w-40 rounded bg-slate-100" />
                </div>
                <div className="h-5 w-24 rounded-full bg-slate-200" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AccountCardSkeleton;
