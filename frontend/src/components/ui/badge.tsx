import * as React from 'react';
import { cn } from '../../lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'outline' | 'subtle';
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  const variants: Record<NonNullable<BadgeProps['variant']>, string> = {
    default: 'border-transparent bg-slate-900 text-white hover:bg-slate-900/90',
    outline: 'border-slate-200 text-slate-700',
    subtle: 'border-transparent bg-slate-100 text-slate-700'
  };

  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide',
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
