export type BureauKey = 'transunion' | 'experian' | 'equifax';

export const BUREAUS: BureauKey[] = ['transunion', 'experian', 'equifax'];

export const BUREAU_LABELS: Record<BureauKey, string> = {
  transunion: 'TU',
  experian: 'EX',
  equifax: 'EF'
};

export type AgreementLevel = 'all' | 'majority' | 'mixed' | 'none';

export const MISSING_VALUE = '—';
