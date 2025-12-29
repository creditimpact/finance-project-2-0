export type BureauTriple = {
  transunion?: string;
  experian?: string;
  equifax?: string;
};

export type Agreement = 'all' | 'majority' | 'mixed' | 'none';

type BureauKey = keyof BureauTriple;

const BUREAU_ORDER: BureauKey[] = ['transunion', 'experian', 'equifax'];

const MISSING_PLACEHOLDERS = new Set(['', '--']);

function normalizeValue(value?: string): string | undefined {
  if (value == null) {
    return undefined;
  }

  const trimmed = value.trim();

  if (MISSING_PLACEHOLDERS.has(trimmed)) {
    return undefined;
  }

  return trimmed;
}

function pickMajorityValue(entries: Array<[BureauKey, string]>): string | undefined {
  const counts = new Map<string, number>();

  for (const [, value] of entries) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  for (const [value, count] of counts) {
    if (count >= 2) {
      return value;
    }
  }

  return undefined;
}

function pickMixedSummary(
  entries: Array<[BureauKey, string]>,
  kind: 'account_number' | 'generic'
): string {
  if (kind === 'account_number') {
    return entries
      .slice()
      .sort((a, b) => {
        const lengthDiff = b[1].length - a[1].length;
        if (lengthDiff !== 0) {
          return lengthDiff;
        }

        return BUREAU_ORDER.indexOf(a[0]) - BUREAU_ORDER.indexOf(b[0]);
      })[0][1];
  }

  const counts = new Map<string, number>();
  let maxCount = 0;

  for (const [, value] of entries) {
    const nextCount = (counts.get(value) ?? 0) + 1;
    counts.set(value, nextCount);
    if (nextCount > maxCount) {
      maxCount = nextCount;
    }
  }

  if (maxCount > 1) {
    const [modal] = Array.from(counts.entries())
      .filter(([, count]) => count === maxCount)
      .sort(([valueA], [valueB]) => valueA.localeCompare(valueB));
    if (modal) {
      return modal[0];
    }
  }

  for (const bureau of BUREAU_ORDER) {
    const entry = entries.find(([key]) => key === bureau);
    if (entry) {
      return entry[1];
    }
  }

  return '—';
}

export function summarizeField(
  triple: BureauTriple,
  options?: { kind?: 'account_number' | 'generic' }
): { summary: string; agreement: Agreement; values: BureauTriple } {
  const kind = options?.kind ?? 'generic';
  const normalizedEntries = BUREAU_ORDER
    .map((bureau) => {
      const normalized = normalizeValue(triple[bureau]);
      return normalized ? ([bureau, normalized] as [BureauKey, string]) : undefined;
    })
    .filter((entry): entry is [BureauKey, string] => Boolean(entry));

  const normalizedValues = normalizedEntries.reduce<BureauTriple>((acc, [bureau, value]) => {
    acc[bureau] = value;
    return acc;
  }, {});

  if (normalizedEntries.length === 0) {
    return {
      summary: '—',
      agreement: 'none',
      values: normalizedValues,
    };
  }

  const uniqueValues = new Set(normalizedEntries.map(([, value]) => value));

  if (uniqueValues.size === 1) {
    const summaryValue = normalizedEntries[0][1];
    if (normalizedEntries.length === BUREAU_ORDER.length) {
      return {
        summary: summaryValue,
        agreement: 'all',
        values: normalizedValues,
      };
    }

    return {
      summary: summaryValue,
      agreement: normalizedEntries.length > 1 ? 'majority' : 'mixed',
      values: normalizedValues,
    };
  }

  const majorityValue = pickMajorityValue(normalizedEntries);
  if (majorityValue) {
    return {
      summary: majorityValue,
      agreement: 'majority',
      values: normalizedValues,
    };
  }

  return {
    summary: pickMixedSummary(normalizedEntries, kind),
    agreement: 'mixed',
    values: normalizedValues,
  };
}
