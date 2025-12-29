import * as React from 'react';
import { useParams } from 'react-router-dom';
import AccountCard, { type AccountPack } from '../components/AccountCard';
import AccountQuestions, { type AccountQuestionAnswers } from '../components/AccountQuestions';
import { Badge } from '../components/ui/badge';
import {
  BUREAUS,
  BUREAU_LABELS,
  MISSING_VALUE,
  type BureauKey,
} from '../components/accountFieldTypes';
import { cn } from '../lib/utils';
import {
  fetchFrontendReviewManifest,
  fetchFrontendReviewAccount,
  submitFrontendReviewAnswers,
} from '../api.ts';
import type { FrontendReviewManifestPack } from '../api.ts';
import { type ReviewAccountPack } from '../components/ReviewCard';
import { shouldEnableReviewClaims } from '../config/featureFlags';
import {
  hasMissingRequiredDocs,
  normalizeExistingAnswers,
  prepareAnswersPayload,
  resolveClaimsPayload,
} from '../utils/reviewClaims';
import type { ClaimSchema } from '../types/review';

const PLACEHOLDER_VALUES = new Set(['--', '—', '', 'n/a', 'N/A']);
const REVIEW_CLAIMS_ENABLED = shouldEnableReviewClaims();

function normalizeSearchTerm(term: string): string {
  return term.trim().toLowerCase();
}

function formatPrimaryIssue(issue?: string | null): string | null {
  if (!issue) {
    return null;
  }
  return issue.replace(/_/g, ' ');
}

function matchesSearch(entry: FrontendReviewManifestPack, normalizedQuery: string): boolean {
  if (!normalizedQuery) {
    return true;
  }

  const holder = (
    entry.holder_name ?? entry.display?.holder_name ?? ''
  ).toLowerCase();
  const issueRaw = entry.primary_issue ?? entry.display?.primary_issue ?? '';
  const issue = issueRaw.toLowerCase();
  const readableIssue = issueRaw.replace(/_/g, ' ').toLowerCase();

  return holder.includes(normalizedQuery) || issue.includes(normalizedQuery) || readableIssue.includes(normalizedQuery);
}

type PerBureauValues = Partial<Record<BureauKey, string | null | undefined>>;

type DisplayFieldKey =
  | 'account_number'
  | 'account_type'
  | 'status'
  | 'balance_owed'
  | 'date_opened'
  | 'closed_date';

function resolveDisplayField(
  entry: FrontendReviewManifestPack,
  key: DisplayFieldKey
): unknown {
  const displayField = entry.display?.[key as keyof NonNullable<AccountPack['display']>];
  if (displayField !== undefined) {
    return displayField;
  }
  const legacyField = (entry as Record<string, unknown>)[key];
  if (legacyField !== undefined) {
    return legacyField;
  }
  return undefined;
}

function toPerBureauValues(source: unknown): PerBureauValues {
  if (!source || typeof source !== 'object') {
    return {};
  }
  const record = source as Record<string, unknown>;
  if ('per_bureau' in record && record.per_bureau && typeof record.per_bureau === 'object') {
    return toPerBureauValues(record.per_bureau);
  }
  const result: PerBureauValues = {};
  for (const bureau of BUREAUS) {
    if (bureau in record) {
      result[bureau] = record[bureau] as string | null | undefined;
    }
  }
  return result;
}

function sanitizeValue(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim();
  if (!text || PLACEHOLDER_VALUES.has(text)) {
    return null;
  }
  return text;
}

function displayValue(value: unknown): string {
  const sanitized = sanitizeValue(value);
  return sanitized ?? MISSING_VALUE;
}

function summarizePerBureau(values: PerBureauValues): string {
  let summary: string | null = null;
  for (const bureau of BUREAUS) {
    const sanitized = sanitizeValue(values[bureau]);
    if (!sanitized) {
      continue;
    }
    if (!summary || sanitized.length > summary.length) {
      summary = sanitized;
    }
  }
  return summary ?? MISSING_VALUE;
}

function hasAnyValue(values: PerBureauValues): boolean {
  return BUREAUS.some((bureau) => sanitizeValue(values[bureau]));
}

interface SummaryFieldProps {
  label: string;
  values: PerBureauValues;
}

function SummaryField({ label, values }: SummaryFieldProps) {
  const [expanded, setExpanded] = React.useState(false);
  const summary = React.useMemo(() => summarizePerBureau(values), [values]);
  const showToggle = hasAnyValue(values);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</span>
        {showToggle ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              setExpanded((state) => !state);
            }}
            className="text-xs font-semibold text-slate-600 transition hover:text-slate-900"
          >
            {expanded ? 'Hide details' : 'View details'}
          </button>
        ) : null}
      </div>
      <div className="mt-2 text-sm font-semibold text-slate-900">{summary}</div>
      {expanded ? (
        <div className="mt-3 space-y-1 text-xs text-slate-600">
          {BUREAUS.map((bureau) => (
            <div key={bureau} className="flex items-center justify-between gap-3">
              <span className="font-medium text-slate-500">{BUREAU_LABELS[bureau]}</span>
              <span className="text-slate-800">{displayValue(values[bureau])}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

interface BureauFieldProps {
  label: string;
  values: PerBureauValues;
}

function BureauField({ label, values }: BureauFieldProps) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</span>
      <div className="mt-3 grid grid-cols-3 gap-3 text-xs sm:text-sm">
        {BUREAUS.map((bureau) => (
          <div key={bureau} className="space-y-1">
            <span className="block text-[11px] font-medium uppercase tracking-wide text-slate-500">
              {BUREAU_LABELS[bureau]}
            </span>
            <span className="block text-sm font-medium text-slate-900">{displayValue(values[bureau])}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

interface AccountListRowProps {
  entry: FrontendReviewManifestPack;
  onSelect: (accountId: string) => void;
  selected: boolean;
  answered: boolean;
}

function AccountListRow({ entry, onSelect, selected, answered }: AccountListRowProps) {
  const accountId = entry.account_id;
  const holderName = entry.holder_name ?? entry.display?.holder_name ?? 'Unknown account holder';
  const primaryIssue = formatPrimaryIssue(entry.primary_issue ?? entry.display?.primary_issue) ?? 'No primary issue';
  const accountNumberValues = toPerBureauValues(resolveDisplayField(entry, 'account_number'));
  const accountTypeValues = toPerBureauValues(resolveDisplayField(entry, 'account_type'));
  const statusValues = toPerBureauValues(resolveDisplayField(entry, 'status'));
  const balanceValues = toPerBureauValues(resolveDisplayField(entry, 'balance_owed'));
  const openedValues = toPerBureauValues(resolveDisplayField(entry, 'date_opened'));
  const closedValues = toPerBureauValues(resolveDisplayField(entry, 'closed_date'));

  const handleSelect = React.useCallback(() => {
    onSelect(accountId);
  }, [accountId, onSelect]);

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(accountId);
      }
    },
    [accountId, onSelect]
  );

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleSelect}
      onKeyDown={handleKeyDown}
      className={cn(
        'rounded-lg border bg-white p-5 shadow-sm transition hover:border-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2',
        selected ? 'border-slate-500 ring-2 ring-slate-200' : 'border-slate-200'
      )}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">{holderName}</h2>
          <p className="text-sm text-slate-600 capitalize">Primary issue: {primaryIssue}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-400">{accountId}</span>
          {answered ? <Badge className="bg-emerald-100 text-emerald-800">Answered</Badge> : null}
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <SummaryField label="Account number" values={accountNumberValues} />
        <SummaryField label="Account type" values={accountTypeValues} />
        <SummaryField label="Status" values={statusValues} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <BureauField label="Balance owed" values={balanceValues} />
        <BureauField label="Date opened" values={openedValues} />
        <BureauField label="Closed date" values={closedValues} />
      </div>
    </div>
  );
}

function LoadingRow() {
  return (
    <div className="animate-pulse rounded-lg border border-slate-200 bg-white p-5">
      <div className="h-4 w-1/3 rounded bg-slate-200" />
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, index) => (
          <div key={index} className="h-20 rounded bg-slate-100" />
        ))}
      </div>
    </div>
  );
}

export default function AccountsPage() {
  const { sid } = useParams();
  const [entries, setEntries] = React.useState<FrontendReviewManifestPack[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [selectedAccountId, setSelectedAccountId] = React.useState<string | null>(null);
  const [answeredAccounts, setAnsweredAccounts] = React.useState<Set<string>>(new Set());
  const [detailLoading, setDetailLoading] = React.useState(false);
  const [detailError, setDetailError] = React.useState<string | null>(null);
  const [selectedPack, setSelectedPack] = React.useState<ReviewAccountPack | null>(null);
  const [questionAnswers, setQuestionAnswers] = React.useState<AccountQuestionAnswers>({});
  const [submitting, setSubmitting] = React.useState(false);
  const [submitError, setSubmitError] = React.useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = React.useState(false);

  const resolvedClaims = React.useMemo(
    () => resolveClaimsPayload(selectedPack?.claims),
    [selectedPack?.claims]
  );

  const claimDefinitions = React.useMemo(() => {
    const map = new Map<string, ClaimSchema>();
    resolvedClaims.items.forEach((claim) => {
      map.set(claim.key, claim);
    });
    return map;
  }, [resolvedClaims]);

  React.useEffect(() => {
    if (!sid) {
      setError('Missing session id');
      setEntries([]);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);
    setEntries([]);

    (async () => {
      try {
        const manifest = await fetchFrontendReviewManifest(sid);
        if (!active) {
          return;
        }
        const packs = Array.isArray(manifest.packs) ? manifest.packs : [];
        setEntries(packs);
      } catch (err) {
        if (!active) {
          return;
        }
        console.error('Failed to load frontend review manifest', err);
        setError(err instanceof Error ? err.message : 'Unable to load accounts');
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, [sid]);

  React.useEffect(() => {
    if (!sid || !selectedAccountId) {
      setSelectedPack(null);
      setDetailError(null);
      setQuestionAnswers({});
      setSubmitError(null);
      setSubmitSuccess(false);
      return;
    }

    let active = true;
    setDetailLoading(true);
    setDetailError(null);
    setSelectedPack(null);
    setQuestionAnswers({});
    setSubmitError(null);
    setSubmitSuccess(false);

    (async () => {
      try {
        const manifestEntry = entries.find((entry) => entry.account_id === selectedAccountId);
        const packPath = manifestEntry?.pack_path ?? manifestEntry?.path ?? manifestEntry?.pack_path_rel;
        const pack = await fetchFrontendReviewAccount<ReviewAccountPack>(sid, selectedAccountId, {
          staticPath: typeof packPath === 'string' ? packPath : undefined,
        });
        if (!active) {
          return;
        }
        setSelectedPack(pack);
        const existingAnswers = normalizeExistingAnswers((pack as Record<string, unknown> | undefined)?.answers);
        setQuestionAnswers(existingAnswers);
      } catch (err) {
        if (!active) {
          return;
        }
        console.error('Failed to load account pack', err);
        setDetailError(err instanceof Error ? err.message : 'Unable to load account details');
      } finally {
        if (active) {
          setDetailLoading(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, [sid, selectedAccountId, entries]);

  const handleAnswerChange = React.useCallback((answers: AccountQuestionAnswers) => {
    setQuestionAnswers(answers);
    setSubmitError(null);
    setSubmitSuccess(false);
  }, []);

  const cleanedAnswers = React.useMemo(
    () => prepareAnswersPayload(questionAnswers, { includeClaims: REVIEW_CLAIMS_ENABLED }),
    [questionAnswers]
  );
  const explanationProvided = React.useMemo(() => {
    const value = questionAnswers.explanation;
    return typeof value === 'string' && value.trim() !== '';
  }, [questionAnswers.explanation]);
  const missingRequiredDocs = React.useMemo(() => {
    if (!REVIEW_CLAIMS_ENABLED) {
      return false;
    }
    return hasMissingRequiredDocs(
      questionAnswers.selectedClaims ?? [],
      questionAnswers.attachments,
      claimDefinitions
    );
  }, [REVIEW_CLAIMS_ENABLED, claimDefinitions, questionAnswers.attachments, questionAnswers.selectedClaims]);
  const canSubmit = explanationProvided && !missingRequiredDocs;

  const handleSubmitAnswers = React.useCallback(async () => {
    if (!sid || !selectedAccountId) {
      return;
    }
    if (!explanationProvided) {
      setSubmitError('Please provide an explanation before submitting.');
      return;
    }
    if (missingRequiredDocs) {
      setSubmitError('Please upload the required documents for the claims you selected.');
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      const response = await submitFrontendReviewAnswers(sid, selectedAccountId, cleanedAnswers);
      setAnsweredAccounts((prev) => {
        const next = new Set(prev);
        next.add(selectedAccountId);
        return next;
      });
      setSelectedPack((previous) => {
        if (!previous) {
          return previous;
        }
        const persistedAnswers: Record<string, unknown> = {
          answers: {
            ...(response?.answers && typeof response.answers === 'object' ? response.answers : {}),
            ...cleanedAnswers.answers,
          },
        };
        if (cleanedAnswers.claims && cleanedAnswers.claims.length > 0) {
          persistedAnswers.claims = cleanedAnswers.claims;
        }
        if (cleanedAnswers.evidence && cleanedAnswers.evidence.length > 0) {
          persistedAnswers.evidence = cleanedAnswers.evidence;
        }
        const updated: ReviewAccountPack = {
          ...previous,
          answers: persistedAnswers,
          response: response ?? null,
        };
        return updated;
      });
      setSubmitSuccess(true);
    } catch (err) {
      console.error('Failed to submit answers', err);
      setSubmitError(err instanceof Error ? err.message : 'Unable to submit answers');
    } finally {
      setSubmitting(false);
    }
  }, [
    sid,
    selectedAccountId,
    cleanedAnswers,
    explanationProvided,
    missingRequiredDocs,
  ]);

  const normalizedSearch = React.useMemo(() => normalizeSearchTerm(searchTerm), [searchTerm]);

  const filteredEntries = React.useMemo(() => {
    if (!normalizedSearch) {
      return entries;
    }
    return entries.filter((entry) => matchesSearch(entry, normalizedSearch));
  }, [entries, normalizedSearch]);

  const totalCount = entries.length;
  const visibleCount = filteredEntries.length;
  const showEmptyState = !loading && !error && totalCount === 0;
  const showFilteredEmpty = !loading && !error && totalCount > 0 && visibleCount === 0;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold text-slate-900">Accounts</h1>
          {sid ? (
            <p className="text-sm text-slate-600">Run {sid}</p>
          ) : (
            <p className="text-sm text-slate-600">No session selected</p>
          )}
          <p className="text-sm text-slate-600">
            Review account summaries, compare bureau data at a glance, and answer follow-up questions.
          </p>
        </div>
        <div className="flex w-full max-w-sm flex-col gap-2">
          <label htmlFor="account-search" className="text-sm font-medium text-slate-700">
            Search accounts
          </label>
          <input
            id="account-search"
            type="search"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search by holder or issue"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
          {totalCount > 0 && !loading ? (
            <span className="text-xs text-slate-500">
              Showing {visibleCount} of {totalCount} accounts
            </span>
          ) : null}
        </div>
      </header>

      {error ? (
        <div
          role="alert"
          className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900"
        >
          <p className="font-semibold">Unable to load accounts</p>
          <p className="mt-1 text-rose-800">{error}</p>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <section className="space-y-4">
          {loading ? (
            <>
              {Array.from({ length: 3 }).map((_, index) => (
                <LoadingRow key={index} />
              ))}
            </>
          ) : null}

          {!loading && !error
            ? filteredEntries.map((entry) => (
                <AccountListRow
                  key={entry.account_id}
                  entry={entry}
                  onSelect={setSelectedAccountId}
                  selected={entry.account_id === selectedAccountId}
                  answered={answeredAccounts.has(entry.account_id)}
                />
              ))
            : null}

          {showEmptyState ? (
            <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
              No accounts found for this run.
            </div>
          ) : null}

          {showFilteredEmpty ? (
            <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
              No accounts match your search.
            </div>
          ) : null}
        </section>

        <section className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
            {selectedAccountId === null ? (
              <p className="text-sm text-slate-600">Select an account to view details and answer questions.</p>
            ) : detailLoading ? (
              <div className="space-y-4">
                <LoadingRow />
              </div>
            ) : detailError ? (
              <div className="space-y-2 text-sm text-rose-800">
                <p className="font-semibold text-rose-900">Unable to load account</p>
                <p>{detailError}</p>
              </div>
            ) : selectedPack ? (
              <div className="space-y-6">
                <AccountCard pack={selectedPack} />
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Explain</h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Share a brief explanation to help us understand this account.
                  </p>
                </div>
                <AccountQuestions onChange={handleAnswerChange} initialAnswers={questionAnswers} />
                {submitError ? (
                  <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
                    {submitError}
                  </div>
                ) : null}
                {submitSuccess ? (
                  <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
                    Explanation saved successfully.
                  </div>
                ) : null}
                <button
                  type="button"
                  onClick={handleSubmitAnswers}
                  disabled={submitting || !canSubmit}
                  className={cn(
                    'inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-semibold shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2',
                    submitting || !canSubmit
                      ? 'cursor-not-allowed border border-slate-200 bg-slate-100 text-slate-400'
                      : 'border border-slate-900 bg-slate-900 text-white hover:bg-slate-800'
                  )}
                >
                  {submitting ? 'Submitting…' : 'Submit'}
                </button>
              </div>
            ) : (
              <p className="text-sm text-slate-600">No details available for this account.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
