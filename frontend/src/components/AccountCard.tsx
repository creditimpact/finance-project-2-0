import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import FieldSummary from './FieldSummary';
import BureauGrid from './BureauGrid';
import { AgreementLevel, BUREAUS, BureauKey, MISSING_VALUE } from './accountFieldTypes';
import { summarizeField as summarizeBureauField, type BureauTriple } from '../utils/bureauSummary';
import { QUESTION_COPY, type AccountQuestionKey } from './questionCopy';
import {
  shouldHideConsensus,
  shouldPreferLongestAccountMask,
  shouldShowBureauDetails
} from '../config/featureFlags';
import { useI18n } from '../lib/i18n';

type PerBureauBlock = {
  per_bureau?: Partial<Record<BureauKey, string | null | undefined>>;
  consensus?: string | null | undefined;
};

type DateBlock = Partial<Record<BureauKey, string | null | undefined>>;

type QuestionBlock = Partial<Record<AccountQuestionKey, string | null | undefined>>;

export type ResolvedDisplayField = {
  value?: string | null;
  source?: string | null;
  method?: string | null;
};

export type ResolvedDisplay = Partial<
  Record<
    'account_number' | 'account_type' | 'status' | 'balance_owed' | 'date_opened' | 'closed_date',
    ResolvedDisplayField
  >
>;

type AccountDisplay = {
  account_number?: PerBureauBlock;
  account_type?: PerBureauBlock;
  status?: PerBureauBlock;
  balance_owed?: PerBureauBlock;
  date_opened?: DateBlock;
  closed_date?: DateBlock;
  questions?: QuestionBlock;
  resolved?: ResolvedDisplay | null;
};

export type AccountPack = {
  holder_name?: string | null;
  primary_issue?: string | null;
  display?: AccountDisplay | null;
};

type SummaryFieldConfig = {
  key: keyof AccountDisplay;
  messageId: string;
  defaultLabel: string;
};

type FieldSummaryEntry = SummaryFieldConfig & {
  summaryValue: string;
  agreement: AgreementLevel;
  values: Partial<Record<BureauKey, string>>;
};

const SUMMARY_FIELDS: SummaryFieldConfig[] = [
  { key: 'account_type', messageId: 'accountCard.summary.accountType', defaultLabel: 'Account type' },
  { key: 'status', messageId: 'accountCard.summary.status', defaultLabel: 'Status' },
  {
    key: 'balance_owed',
    messageId: 'accountCard.summary.balanceOwed',
    defaultLabel: 'Balance owed'
  },
  { key: 'date_opened', messageId: 'accountCard.summary.dateOpened', defaultLabel: 'Date opened' },
  { key: 'closed_date', messageId: 'accountCard.summary.closedDate', defaultLabel: 'Closed date' }
];

function toBureauTriple(field: PerBureauBlock | DateBlock | undefined): BureauTriple {
  if (!field) {
    return {};
  }

  const triple: BureauTriple = {};

  if ('per_bureau' in field) {
    const perBureau = field.per_bureau ?? {};
    for (const bureau of BUREAUS) {
      const value = perBureau[bureau];
      if (value != null) {
        triple[bureau] = value;
      }
    }
    return triple;
  }

  for (const bureau of BUREAUS) {
    const value = (field as DateBlock)[bureau];
    if (value != null) {
      triple[bureau] = value;
    }
  }

  return triple;
}

function buildFieldSummary(
  field: PerBureauBlock | DateBlock | undefined,
  config: SummaryFieldConfig,
  options: { includeConsensus: boolean }
): FieldSummaryEntry {
  const summary = summarizeBureauField(toBureauTriple(field));

  let summaryValue = summary.summary;

  if (options.includeConsensus && field && 'consensus' in field) {
    const consensusValue = (field as PerBureauBlock).consensus;
    if (consensusValue && consensusValue.trim() !== '') {
      if (summaryValue === MISSING_VALUE || summaryValue.trim() === '') {
        summaryValue = consensusValue;
      }
    }
  }

  return {
    ...config,
    summaryValue,
    agreement: summary.agreement as AgreementLevel,
    values: summary.values
  };
}

function formatPrimaryIssue(issue?: string | null) {
  if (!issue) {
    return null;
  }
  return issue.replace(/_/g, ' ');
}

const ChevronDownIcon = ({ className }: { className?: string }) => (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    className={cn('h-4 w-4 text-slate-600', className)}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

export interface AccountCardProps {
  pack: AccountPack;
}

export function AccountCard({ pack }: AccountCardProps) {
  const t = useI18n();
  const display = pack.display ?? ({} as AccountDisplay);

  const includeConsensus = React.useMemo(() => !shouldHideConsensus(), []);
  const preferLongestAccountMask = React.useMemo(() => shouldPreferLongestAccountMask(), []);
  const expandDetailsByDefault = React.useMemo(() => shouldShowBureauDetails(), []);

  const fieldSummaries = React.useMemo<FieldSummaryEntry[]>(() => {
    return SUMMARY_FIELDS.map((field) =>
      buildFieldSummary(display[field.key] as PerBureauBlock | DateBlock | undefined, field, {
        includeConsensus
      })
    );
  }, [display, includeConsensus]);

  const hasDisagreement = fieldSummaries.some(
    (field) => field.agreement === 'majority' || field.agreement === 'mixed'
  );

  const [expanded, setExpanded] = React.useState(() => expandDetailsByDefault && hasDisagreement);
  const detailsId = React.useId();

  React.useEffect(() => {
    if (expandDetailsByDefault) {
      setExpanded(hasDisagreement);
    }
  }, [expandDetailsByDefault, hasDisagreement]);

  const accountNumberSummary = React.useMemo(() => {
    const triple = toBureauTriple(display.account_number);
    const result = summarizeBureauField(triple, {
      kind: 'account_number'
    });
    const values = Object.values(triple).filter(
      (value): value is string => typeof value === 'string' && value.trim() !== ''
    );

    let summaryValue = result.summary;

    if (preferLongestAccountMask && values.length > 0) {
      const longest = values.reduce((current, candidate) => {
        if (!current) {
          return candidate;
        }

        if (candidate.length > current.length) {
          return candidate;
        }

        return current;
      }, '');

      if (!summaryValue || longest.length > summaryValue.length) {
        summaryValue = longest;
      }
    }

    if (includeConsensus && display.account_number?.consensus) {
      const consensusValue = display.account_number.consensus.trim();
      if (consensusValue) {
        const shouldUseConsensus =
          summaryValue === MISSING_VALUE || summaryValue.trim() === '';
        const shouldPreferConsensus =
          preferLongestAccountMask && consensusValue.length > summaryValue.length;

        if (shouldUseConsensus || shouldPreferConsensus) {
          summaryValue = consensusValue;
        }
      }
    }

    return summaryValue;
  }, [display.account_number, includeConsensus, preferLongestAccountMask]);

  const questions: QuestionBlock = display.questions ?? {};

  return (
    <Card className="w-full">
      <CardHeader className="gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-xl font-semibold text-slate-900">
            {pack.holder_name ??
              t({ id: 'accountCard.unknownHolder', defaultMessage: 'Unknown account holder' })}
          </CardTitle>
          <p className="text-sm text-slate-600">
            {t({
              id: 'accountCard.accountNumberLabel',
              defaultMessage: 'Account number: {accountNumber}',
              values: { accountNumber: accountNumberSummary ?? MISSING_VALUE }
            })}
          </p>
        </div>
        {pack.primary_issue ? (
          <Badge variant="outline" className="whitespace-nowrap text-xs font-semibold capitalize text-slate-700">
            {formatPrimaryIssue(pack.primary_issue)}
          </Badge>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-wrap gap-4">
          {fieldSummaries.map((field) => (
            <FieldSummary
              key={field.key}
              label={t({ id: field.messageId, defaultMessage: field.defaultLabel })}
              value={field.summaryValue}
              agreement={field.agreement}
            />
          ))}
        </div>

        <div className="space-y-3">
          <button
            type="button"
            className="flex items-center gap-2 text-sm font-semibold text-slate-700 transition hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2"
            onClick={() => setExpanded((state) => !state)}
            aria-expanded={expanded}
            aria-controls={detailsId}
          >
            <span>
              {expanded
                ? t({ id: 'accountCard.details.hide', defaultMessage: 'Hide details' })
                : t({ id: 'accountCard.details.show', defaultMessage: 'See details' })}
            </span>
            <ChevronDownIcon className={cn('transition-transform', expanded ? 'rotate-180' : 'rotate-0')} />
            {hasDisagreement ? (
              <Badge className="bg-amber-100 text-amber-900">
                {t({ id: 'accountCard.details.disagreement', defaultMessage: 'Disagreement' })}
              </Badge>
            ) : null}
          </button>

          {expanded ? (
            <BureauGrid
              id={detailsId}
              fields={fieldSummaries.map((field) => ({
                fieldKey: field.key,
                label: t({ id: field.messageId, defaultMessage: field.defaultLabel }),
                values: field.values,
                agreement: field.agreement
              }))}
            />
          ) : null}
        </div>

        <div className="space-y-4">
          <h4 className="text-base font-semibold text-slate-900">
            {t({ id: 'accountCard.questions.heading', defaultMessage: 'Tell us about this account' })}
          </h4>
          <div className="grid gap-3 md:grid-cols-2">
            {Object.entries(QUESTION_COPY).map(([key, copy]) => {
              const value = questions?.[key as keyof QuestionBlock];
              return (
                <div
                  key={key}
                  className="flex flex-col gap-2 rounded-lg border border-slate-200 p-4"
                >
                  <div>
                    <p className="text-sm font-semibold text-slate-900">
                      {t({ id: `accountCard.questions.${key}.title`, defaultMessage: copy.title })}
                    </p>
                    <p className="text-xs text-slate-600">
                      {t({ id: `accountCard.questions.${key}.helper`, defaultMessage: copy.helper })}
                    </p>
                  </div>
                  <Badge variant="subtle" className="w-fit bg-slate-100 text-slate-600">
                    {value
                      ? value
                      : t({ id: 'accountCard.questions.noResponse', defaultMessage: 'No response yet' })}
                  </Badge>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default AccountCard;
