import * as React from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { type AccountQuestionAnswers } from '../components/AccountQuestions';
import ReviewCard, {
  type ReviewAccountPack,
  type ReviewCardStatus,
} from '../components/ReviewCard';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import {
  apiUrl,
  buildFrontendReviewStreamUrl,
  buildRunFrontendManifestUrl,
  fetchReviewPack,
  fetchRunFrontendManifest,
  fetchRunFrontendReviewIndex,
  fetchRunReviewPackListing,
  joinRunAsset,
  isAbsUrl,
  normalizeStaticPackPath,
  submitFrontendReviewAnswers,
  FrontendReviewAccountError,
  FetchJsonError,
} from '../api.ts';
import type {
  FrontendReviewPackListingItem,
  FrontendReviewResponse,
  RunFrontendManifestResponse,
  ReviewQuestion,
} from '../api.ts';
import { useToast } from '../components/ToastProvider';
import { REVIEW_DEBUG_ENABLED, reviewDebugLog } from '../utils/reviewDebug';
import { DEV_DIAGNOSTICS_ENABLED } from '../utils/devDiagnostics';
import { shouldEnableReviewClaims } from '../config/featureFlags';
import {
  hasMissingRequiredDocs,
  normalizeSelectedClaims,
  normalizeExistingAnswers,
  prepareAnswersPayload,
  resolveClaimsPayload,
} from '../utils/reviewClaims';

const POLL_INTERVAL_MS = 2000;
const WORKER_HINT_DELAY_MS = 30_000;
const REVIEW_CLAIMS_ENABLED = shouldEnableReviewClaims();

type CardStatus = ReviewCardStatus;

interface CardErrorDetails {
  status?: number;
  url?: string;
  responseSnippet?: string;
  message?: string;
}

interface CardState {
  status: CardStatus;
  pack: ReviewAccountPack | null;
  answers: AccountQuestionAnswers;
  error: string | null;
  success: boolean;
  response: FrontendReviewResponse | null;
  fetchStatus: string;
  errorDetails: CardErrorDetails | null;
}

type CardsState = Record<string, CardState>;

type PackListingEntry = FrontendReviewPackListingItem & {
  account_id: string;
  staticUrl?: string;
};

type Phase = 'loading' | 'waiting' | 'ready' | 'error';

type PackMap = Record<string, ReviewAccountPack>;

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return null;
}

function extractPacksCount(source: unknown): number {
  const records: Array<Record<string, unknown> | null> = [];
  const root = toRecord(source);
  const reviewRecord = getReviewSection(source);
  if (reviewRecord) {
    records.push(reviewRecord);
  }
  if (root) {
    records.push(root);
  }

  for (const record of records) {
    if (!record) {
      continue;
    }
    const direct = toNumber(record.packs_count);
    if (direct !== null) {
      return direct;
    }
    const counts = toRecord(record.counts);
    if (counts) {
      const value = toNumber(counts.packs);
      if (value !== null) {
        return value;
      }
    }
  }

  return 0;
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function getReviewSection(source: unknown): Record<string, unknown> | null {
  const root = toRecord(source);
  if (!root) {
    return null;
  }
  const frontend = toRecord(root.frontend);
  if (!frontend) {
    return null;
  }
  return toRecord(frontend.review);
}

function extractReviewQuestions(source: unknown): ReviewQuestion[] {
  const reviewRecord = getReviewSection(source);
  if (!reviewRecord) {
    return [];
  }
  const questions = reviewRecord.questions;
  if (!Array.isArray(questions)) {
    return [];
  }
  return questions.filter((entry): entry is ReviewQuestion => Boolean(entry && typeof entry === 'object'));
}

function extractReviewListingItems(source: unknown): FrontendReviewPackListingItem[] {
  const reviewRecord = getReviewSection(source);
  if (!reviewRecord) {
    return [];
  }
  const items = reviewRecord.items;
  if (!Array.isArray(items)) {
    return [];
  }

  const normalized: FrontendReviewPackListingItem[] = [];
  for (const item of items) {
    const normalizedItem = normalizePackListingItem(item);
    if (!normalizedItem) {
      continue;
    }
    normalized.push(normalizedItem);
  }

  return normalized;
}

function extractAccountId(candidate: unknown): string | undefined {
  if (typeof candidate === 'string') {
    const trimmed = candidate.trim();
    return trimmed !== '' ? trimmed : undefined;
  }
  if (!candidate || typeof candidate !== 'object') {
    return undefined;
  }
  const record = candidate as Record<string, unknown>;
  const keys = ['account_id', 'accountId', 'id'];
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim() !== '') {
      return value.trim();
    }
  }
  return undefined;
}

function normalizePackListingItem(
  item: unknown
): (FrontendReviewPackListingItem & { account_id: string }) | null {
  if (!item || typeof item !== 'object') {
    return null;
  }
  const record = item as Record<string, unknown>;

  const accountCandidates = [
    record.account_id,
    record.id,
    record.accountId,
    record.account,
    record.accountID,
  ];

  let accountId: string | undefined;
  for (const candidate of accountCandidates) {
    accountId = extractAccountId(candidate);
    if (accountId) {
      break;
    }
  }

  if (!accountId) {
    return null;
  }

  const fileKeys = ['file', 'path', 'static_path', 'staticPath', 'pack_path', 'packPath'];
  let file: string | undefined;
  for (const key of fileKeys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim() !== '') {
      file = value;
      break;
    }
  }

  if (typeof file === 'string') {
    return { account_id: accountId, file };
  }

  return { account_id: accountId };
}

function createResponseSnippet(value: string | null | undefined, maxLength = 120): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  const collapsed = value.replace(/\s+/g, ' ').trim();
  if (!collapsed) {
    return undefined;
  }
  if (collapsed.length > maxLength) {
    return `${collapsed.slice(0, maxLength)}…`;
  }
  return collapsed;
}

function extractCardErrorDetails(error: unknown): CardErrorDetails | null {
  if (error instanceof FrontendReviewAccountError) {
    const attempts = Array.isArray(error.attempts) ? error.attempts : [];
    const prioritized = attempts.filter((attempt) => {
      if (!attempt) {
        return false;
      }
      if (attempt.status != null) {
        return true;
      }
      if (typeof attempt.responseText === 'string' && attempt.responseText.trim() !== '') {
        return true;
      }
      return Boolean(attempt.error?.message);
    });
    const pick =
      prioritized.find((attempt) => attempt.kind !== 'static') ??
      prioritized[0] ??
      attempts[0];
    if (pick) {
      const snippet = createResponseSnippet(
        pick.responseText ?? pick.error?.message ?? undefined
      );
      return {
        status: pick.status,
        url: pick.url,
        responseSnippet: snippet,
        message: pick.error?.message ?? undefined,
      };
    }
    return { message: error.message };
  }

  if (error instanceof FetchJsonError) {
    const snippet = createResponseSnippet(error.responseText ?? error.message ?? undefined);
    return {
      status: error.status,
      url: error.url,
      responseSnippet: snippet,
      message: error.message,
    };
  }

  if (error instanceof Error) {
    return { message: error.message };
  }

  if (typeof error === 'string' && error.trim() !== '') {
    return { message: error };
  }

  return null;
}

function createInitialCardState(): CardState {
  return {
    status: 'idle',
    pack: null,
    answers: {},
    error: null,
    success: false,
    response: null,
    fetchStatus: 'idle',
    errorDetails: null,
  };
}

function normalizeListingFilePath(path: string | null | undefined): string | undefined {
  if (typeof path !== 'string') {
    return undefined;
  }
  return normalizeStaticPackPath(path) ?? undefined;
}

interface ReviewCardContainerProps {
  accountId: string;
  state: CardState;
  sessionId?: string;
  onChange: (answers: AccountQuestionAnswers) => void;
  onSubmit: () => void;
  onLoad: (accountId: string) => void;
  onRetry: (accountId: string) => void;
}

function ReviewCardContainer({ accountId, sessionId, state, onChange, onSubmit, onLoad, onRetry }: ReviewCardContainerProps) {
  const cardRef = React.useRef<HTMLDivElement | null>(null);
  const hasRequestedRef = React.useRef(false);

  React.useEffect(() => {
    if (state.status !== 'waiting') {
      hasRequestedRef.current = false;
      return undefined;
    }

    if (hasRequestedRef.current) {
      return undefined;
    }

    const node = cardRef.current;
    if (!node) {
      return undefined;
    }

    if (typeof window !== 'undefined' && typeof window.IntersectionObserver === 'function') {
      const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !hasRequestedRef.current) {
            hasRequestedRef.current = true;
            onLoad(accountId);
          }
        }
      }, { rootMargin: '0px 0px 200px 0px' });

      observer.observe(node);
      return () => {
        observer.disconnect();
      };
    }

    hasRequestedRef.current = true;
    onLoad(accountId);
    return undefined;
  }, [accountId, onLoad, state.status]);

  if (state.status === 'waiting') {
    return (
      <Card ref={cardRef} className="w-full">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-slate-900">Account {accountId}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="h-5 w-1/2 animate-pulse rounded bg-slate-200" />
          <div className="h-40 animate-pulse rounded bg-slate-100" />
        </CardContent>
      </Card>
    );
  }

  if (!state.pack) {
    const detail = state.errorDetails;
    const statusText = detail?.status != null ? String(detail.status) : 'Unknown';
    const urlText = detail?.url ?? 'Unknown';
    const messageSnippet = detail?.message ? createResponseSnippet(detail.message) ?? detail.message : undefined;
    const fallbackMessage = state.error ? createResponseSnippet(state.error) ?? state.error : undefined;
    const responseText =
      detail?.responseSnippet ??
      messageSnippet ??
      fallbackMessage ??
      'No response text available.';

    return (
      <Card ref={cardRef} className="w-full">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-slate-900">Account {accountId}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
            <div className="space-y-2">
              <p className="font-semibold text-rose-900">Failed to load card {accountId}</p>
              <dl className="space-y-1 text-xs text-rose-700">
                <div>
                  <dt className="font-semibold uppercase tracking-wide text-rose-800">Status</dt>
                  <dd className="text-rose-900">{statusText}</dd>
                </div>
                <div>
                  <dt className="font-semibold uppercase tracking-wide text-rose-800">URL</dt>
                  <dd className="break-all text-rose-900">{urlText}</dd>
                </div>
                <div>
                  <dt className="font-semibold uppercase tracking-wide text-rose-800">Response</dt>
                  <dd className="text-rose-900">{responseText}</dd>
                </div>
              </dl>
            </div>
          </div>
          <button
            type="button"
            onClick={() => onRetry(accountId)}
            className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:text-slate-900"
          >
            Retry
          </button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div ref={cardRef} className="w-full">
      <ReviewCard
        accountId={accountId}
        sessionId={sessionId}
        pack={state.pack}
        answers={state.answers}
        status={state.status}
        error={state.error}
        success={state.success}
        onAnswersChange={onChange}
        onSubmit={onSubmit}
      />
    </div>
  );
}

function RunReviewPageContent({ sid }: { sid: string | undefined }) {
  const { showToast } = useToast();
  const navigate = useNavigate();
  const [phase, setPhase] = React.useState<Phase>('loading');
  const [phaseError, setPhaseError] = React.useState<string | null>(null);
  const [manifest, setManifest] = React.useState<RunFrontendManifestResponse | null>(null);
  const [cards, setCards] = React.useState<CardsState>({});
  const [order, setOrder] = React.useState<string[]>([]);
  const [submittedAccounts, setSubmittedAccounts] = React.useState<Set<string>>(() => new Set());
  const [isCompleting, setIsCompleting] = React.useState(false);
  const [showWorkerHint, setShowWorkerHint] = React.useState(false);
  const [liveUpdatesUnavailable, setLiveUpdatesUnavailable] = React.useState(false);
  const [frontendMissing, setFrontendMissing] = React.useState(false);
  const [diagnosticsPacksCount, setDiagnosticsPacksCount] = React.useState<number | null>(null);
  const [packsById, setPacksById] = React.useState<PackMap>({});
  const [reviewItems, setReviewItems] = React.useState<FrontendReviewPackListingItem[]>([]);
  const [reviewReady, setReviewReady] = React.useState(false);

  const getPack = React.useCallback(
    (accountId: string) => packsById[accountId],
    [packsById]
  );

  const rememberPack = React.useCallback((accountId: string, pack: ReviewAccountPack) => {
    setPacksById((previous) => {
      if (previous[accountId] === pack) {
        return previous;
      }
      return { ...previous, [accountId]: pack };
    });
  }, []);

  const clearPacks = React.useCallback(() => {
    setPacksById({});
  }, []);

  const applyReviewIndex = React.useCallback((payload: unknown): FrontendReviewPackListingItem[] => {
    const normalizedItems = extractReviewListingItems(payload);
    setReviewItems(normalizedItems);
    setReviewReady(true);
    readyStatsRef.current = {
      total: normalizedItems.length,
      api: readyStatsRef.current.api,
      staticPacks: readyStatsRef.current.staticPacks,
    };
    return normalizedItems;
  }, []);

  const isMountedRef = React.useRef(false);
  const loadingRef = React.useRef(false);
  const loadedRef = React.useRef(false);
  const pollTimeoutRef = React.useRef<number | null>(null);
  const pollIterationRef = React.useRef(0);
  const eventSourceRef = React.useRef<EventSource | null>(null);
  const packListingRef = React.useRef<Record<string, PackListingEntry>>({});
  const globalQuestionsRef = React.useRef<ReviewQuestion[]>([]);
  const loadingAccountsRef = React.useRef<Set<string>>(new Set());
  const workerHintTimeoutRef = React.useRef<number | null>(null);
  const workerWaitingRef = React.useRef(false);
  const retryAttemptsRef = React.useRef<Record<string, number>>({});
  const retryTimeoutsRef = React.useRef<Record<string, number | undefined>>({});
  const hasShownLiveUpdateToastRef = React.useRef(false);
  const fallbackTimeoutRef = React.useRef<number | null>(null);
  const lastNetworkStatusRef = React.useRef<string | null>(null);
  const bootstrapItemsRef = React.useRef<FrontendReviewPackListingItem[]>([]);
  const readyStatsRef = React.useRef<{ total: number; api: number; staticPacks: number }>({
    total: 0,
    api: 0,
    staticPacks: 0,
  });
  const previousReadyRef = React.useRef(false);

  const stopFallbackTimeout = React.useCallback(() => {
    if (fallbackTimeoutRef.current !== null) {
      window.clearTimeout(fallbackTimeoutRef.current);
      fallbackTimeoutRef.current = null;
      reviewDebugLog('timeout:stop');
    }
  }, []);

  const startTimeout = React.useCallback(
    (delay: number, handler: () => void | Promise<void>) => {
      stopFallbackTimeout();
      reviewDebugLog('timeout:start', { delay });
      fallbackTimeoutRef.current = window.setTimeout(() => {
        fallbackTimeoutRef.current = null;
        reviewDebugLog('timeout:fire', { delay });
        void (async () => {
          try {
            await handler();
          } catch (err) {
            reviewDebugLog('timeout:error', { error: err });
            console.error('Review timeout handler failed', err);
          }
        })();
      }, delay);
    },
    [stopFallbackTimeout]
  );

  React.useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  React.useEffect(() => {
    loadedRef.current = false;
    loadingRef.current = false;
    packListingRef.current = {};
    loadingAccountsRef.current = new Set();
    bootstrapItemsRef.current = [];
    setSubmittedAccounts(new Set());
    clearPacks();
    workerWaitingRef.current = false;
    stopFallbackTimeout();
    if (workerHintTimeoutRef.current !== null) {
      window.clearTimeout(workerHintTimeoutRef.current);
      workerHintTimeoutRef.current = null;
    }
    if (isMountedRef.current) {
      setShowWorkerHint(false);
    }
    for (const key of Object.keys(retryTimeoutsRef.current)) {
      const timeoutId = retryTimeoutsRef.current[key];
      if (typeof timeoutId === 'number') {
        window.clearTimeout(timeoutId);
      }
      delete retryTimeoutsRef.current[key];
    }
    retryAttemptsRef.current = {};
    setFrontendMissing(false);
    setLiveUpdatesUnavailable(false);
    hasShownLiveUpdateToastRef.current = false;
    lastNetworkStatusRef.current = null;
    if (isMountedRef.current) {
      setPhase('loading');
      setPhaseError(null);
    }
    setDiagnosticsPacksCount(null);
    setReviewItems([]);
    setReviewReady(false);
    readyStatsRef.current = { total: 0, api: 0, staticPacks: 0 };
    previousReadyRef.current = false;
  }, [sid, clearPacks, stopFallbackTimeout]);

  const clearWorkerWait = React.useCallback(() => {
    workerWaitingRef.current = false;
    if (workerHintTimeoutRef.current !== null) {
      window.clearTimeout(workerHintTimeoutRef.current);
      workerHintTimeoutRef.current = null;
    }
    if (isMountedRef.current) {
      setShowWorkerHint(false);
    }
  }, []);

  React.useEffect(() => {
    if (!isMountedRef.current) {
      return;
    }
    const nextOrder = reviewItems.map((item) => item.account_id);
    setOrder((previous) => {
      if (previous.length === nextOrder.length && previous.every((value, index) => value === nextOrder[index])) {
        return previous;
      }
      return nextOrder;
    });
    setCards((previous) => {
      let changed = false;
      const nextState: CardsState = {};
      for (const accountId of nextOrder) {
        const existing = previous[accountId];
        if (existing) {
          nextState[accountId] = existing;
        } else {
          nextState[accountId] = { ...createInitialCardState(), status: 'waiting' };
          changed = true;
        }
      }
      if (Object.keys(previous).length !== nextOrder.length) {
        changed = true;
      }
      if (!changed) {
        const sameEntries = nextOrder.every((accountId) => previous[accountId] === nextState[accountId]);
        if (sameEntries) {
          return previous;
        }
      }
      return nextState;
    });
  }, [reviewItems]);

  const beginWorkerWait = React.useCallback(() => {
    workerWaitingRef.current = true;
    if (showWorkerHint) {
      return;
    }
    if (workerHintTimeoutRef.current !== null) {
      return;
    }
    workerHintTimeoutRef.current = window.setTimeout(() => {
      workerHintTimeoutRef.current = null;
      if (isMountedRef.current && workerWaitingRef.current) {
        setShowWorkerHint(true);
      }
    }, WORKER_HINT_DELAY_MS);
  }, [showWorkerHint]);

  const loadManifestInfo = React.useCallback(
    async (sessionId: string, options?: { isCancelled?: () => boolean }) => {
      const isCancelled = options?.isCancelled ?? (() => false);
      if (!sessionId) {
        return;
      }

      const manifestUrl = buildRunFrontendManifestUrl(sessionId);
      reviewDebugLog('manifest:fetch', { url: manifestUrl, sessionId });

      try {
        const manifestResponse = await fetchRunFrontendManifest(sessionId);
        if (isCancelled() || !isMountedRef.current) {
          return;
        }
        setManifest(manifestResponse);
        const frontendSection = manifestResponse.frontend;
        setFrontendMissing(!(frontendSection && typeof frontendSection === 'object'));
        reviewDebugLog('manifest:success', { url: manifestUrl, sessionId });
      } catch (err) {
        if (isCancelled()) {
          return;
        }
        reviewDebugLog('manifest:error', { url: manifestUrl, sessionId, error: err });
        console.warn(`[RunReviewPage] Unable to load ${manifestUrl}`, err);
      }
    },
    []
  );

  const clearRetryTimeout = React.useCallback((accountId: string) => {
    const timeoutId = retryTimeoutsRef.current[accountId];
    if (typeof timeoutId === 'number') {
      window.clearTimeout(timeoutId);
    }
    delete retryTimeoutsRef.current[accountId];
  }, []);

  const updateCard = React.useCallback((accountId: string, updater: (state: CardState) => CardState) => {
    setCards((previous) => {
      const current = previous[accountId] ?? createInitialCardState();
      const next = updater(current);
      if (next === current) {
        return previous;
      }
      return { ...previous, [accountId]: next };
    });
  }, []);

  const markSubmitted = React.useCallback((accountId: string) => {
    setSubmittedAccounts((previous) => {
      if (previous.has(accountId)) {
        return previous;
      }
      const next = new Set(previous);
      next.add(accountId);
      return next;
    });
  }, []);

  const markUnsubmitted = React.useCallback((accountId: string) => {
    setSubmittedAccounts((previous) => {
      if (!previous.has(accountId)) {
        return previous;
      }
      const next = new Set(previous);
      next.delete(accountId);
      return next;
    });
  }, []);

  const loadPackListing = React.useCallback(async (options?: { fallbackItems?: FrontendReviewPackListingItem[] }) => {
    if (!sid || loadingRef.current || loadedRef.current) {
      return;
    }
    reviewDebugLog('loadPackListing:start', { sid });
    loadingRef.current = true;
    setPhaseError(null);
    if (isMountedRef.current) {
      setPhase((state) => (state === 'ready' ? state : 'waiting'));
    }

    try {
      const packsUrl = apiUrl(`/api/runs/${encodeURIComponent(sid)}/frontend/review/packs`);
      reviewDebugLog('loadPackListing:fetch', { url: packsUrl });
      const { items } = await fetchRunReviewPackListing(sid);
      if (!isMountedRef.current) {
        return;
      }

      const fallbackItems = Array.isArray(options?.fallbackItems)
        ? options?.fallbackItems
        : bootstrapItemsRef.current;
      if (Array.isArray(options?.fallbackItems)) {
        bootstrapItemsRef.current = options.fallbackItems;
      }

      const normalizedApiItems = items
        .map((item) => normalizePackListingItem(item))
        .filter(
          (item): item is FrontendReviewPackListingItem & { account_id: string } => Boolean(item)
        );
      const normalizedFallbackItems = (fallbackItems ?? [])
        .map((item) => normalizePackListingItem(item))
        .filter(
          (item): item is FrontendReviewPackListingItem & { account_id: string } => Boolean(item)
        );

      const filteredItems =
        normalizedApiItems.length > 0 ? normalizedApiItems : normalizedFallbackItems;

      const staticCount = normalizedApiItems.length > 0 ? 0 : normalizedFallbackItems.length;

      reviewDebugLog('loadPackListing:received', {
        url: packsUrl,
        count: filteredItems.length,
        apiCount: normalizedApiItems.length,
        fallbackCount: normalizedFallbackItems.length,
        source: normalizedApiItems.length > 0 ? 'api' : 'fallback',
      });

      const basePath = `/runs/${encodeURIComponent(sid)}`;
      const baseUrl = apiUrl(basePath);
      const listingMap: Record<string, PackListingEntry> = {};
      const sanitizedBootstrapItems: FrontendReviewPackListingItem[] = [];
      for (const item of filteredItems) {
        const normalizedFile = normalizeListingFilePath(item.file);
        let staticUrl: string | undefined;
        if (normalizedFile && normalizedFile.trim() !== '') {
          if (isAbsUrl(normalizedFile)) {
            staticUrl = normalizedFile;
          } else if (
            normalizedFile.startsWith('/runs/') ||
            normalizedFile.startsWith('runs/')
          ) {
            staticUrl = normalizedFile.startsWith('/') ? normalizedFile : `/${normalizedFile}`;
          } else {
            staticUrl = joinRunAsset(baseUrl, normalizedFile);
          }
        }
        listingMap[item.account_id] = normalizedFile
          ? { ...item, file: normalizedFile, staticUrl }
          : { ...item, staticUrl };
        sanitizedBootstrapItems.push(
          normalizedFile ? { account_id: item.account_id, file: normalizedFile } : { account_id: item.account_id }
        );
      }
      packListingRef.current = listingMap;
      if (sanitizedBootstrapItems.length > 0) {
        bootstrapItemsRef.current = sanitizedBootstrapItems;
      }

      setReviewItems(filteredItems);
      setReviewReady(true);
      setOrder(filteredItems.map((item) => item.account_id));
      const initialSubmitted = new Set<string>();
      setCards(() => {
        const initial: CardsState = {};
        for (const item of filteredItems) {
          const cached = getPack(item.account_id);
          if (cached) {
            const normalizedAnswers = normalizeExistingAnswers((cached as Record<string, unknown> | undefined)?.answers);
            const hasResponse = Boolean(cached.response);
            if (hasResponse) {
              initialSubmitted.add(item.account_id);
            }
            initial[item.account_id] = {
              ...createInitialCardState(),
              status: 'ready',
              fetchStatus: 'cached',
              pack: cached,
              answers: normalizedAnswers,
              error: null,
              errorDetails: null,
              success: hasResponse,
              response: cached.response ?? null,
            };
          } else {
            initial[item.account_id] = {
              ...createInitialCardState(),
              status: 'waiting',
            };
          }
        }
        return initial;
      });
      setSubmittedAccounts(initialSubmitted);
      clearWorkerWait();
      setDiagnosticsPacksCount((previous) => (previous == null ? filteredItems.length : previous));
      readyStatsRef.current = {
        total: filteredItems.length,
        api: normalizedApiItems.length,
        staticPacks: staticCount,
      };

      if (isMountedRef.current) {
        stopFallbackTimeout();
        setPhase('ready');
        loadedRef.current = true;
      }
      reviewDebugLog('loadPackListing:ready', {
        count: filteredItems.length,
        apiCount: normalizedApiItems.length,
        fallbackCount: normalizedFallbackItems.length,
        source: normalizedApiItems.length > 0 ? 'api' : 'fallback',
      });
    } catch (err) {
      if (!isMountedRef.current) {
        return;
      }
      const message = err instanceof Error ? err.message : 'Unable to load review packs';
      clearWorkerWait();
      setPhase('error');
      setPhaseError(message);
    } finally {
      loadingRef.current = false;
    }
  }, [clearWorkerWait, getPack, sid, stopFallbackTimeout]);

  const loadAccountPack = React.useCallback(
    async (accountId: string, options?: { force?: boolean; resetAttempts?: boolean }) => {
      if (!sid) {
        return;
      }

      const cachedPack = getPack(accountId);
      if (cachedPack && !options?.force) {
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          pack: cachedPack,
          answers: state.answers && Object.keys(state.answers).length > 0
            ? state.answers
            : normalizeExistingAnswers((cachedPack as Record<string, unknown> | undefined)?.answers),
          error: null,
          errorDetails: null,
          success: Boolean(cachedPack.response),
          response: cachedPack.response ?? null,
          fetchStatus: 'cached',
        }));
        if (cachedPack.response) {
          markSubmitted(accountId);
        } else {
          markUnsubmitted(accountId);
        }
        return;
      }

      if (loadingAccountsRef.current.has(accountId)) {
        return;
      }

      loadingAccountsRef.current.add(accountId);
      if (options?.resetAttempts) {
        delete retryAttemptsRef.current[accountId];
      }
      clearRetryTimeout(accountId);
      updateCard(accountId, (state) => ({
        ...state,
        status: state.pack && !options?.force ? state.status : 'waiting',
        error: null,
        errorDetails: null,
        fetchStatus: 'fetching',
      }));

      try {
        const listing = packListingRef.current[accountId];
        const fallbackPath =
          typeof listing?.file === 'string' && listing.file.trim() !== ''
            ? listing.file
            : typeof listing?.staticUrl === 'string' && listing.staticUrl.trim() !== ''
            ? listing.staticUrl
            : undefined;
        const pack = await fetchReviewPack(sid, accountId, fallbackPath, globalQuestionsRef.current);
        if (!isMountedRef.current) {
          return;
        }
        rememberPack(accountId, pack);
        delete retryAttemptsRef.current[accountId];
        clearRetryTimeout(accountId);
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          pack,
          answers: normalizeExistingAnswers((pack as Record<string, unknown> | undefined)?.answers),
          error: null,
          errorDetails: null,
          success: Boolean(pack.response),
          response: pack.response ?? null,
          fetchStatus: 'success',
        }));
        if (pack.response) {
          markSubmitted(accountId);
        } else {
          markUnsubmitted(accountId);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to load account details';
        if (isMountedRef.current) {
          updateCard(accountId, (state) => ({
            ...state,
            status: 'ready',
            error: message,
            errorDetails: extractCardErrorDetails(err),
            pack: state.pack,
            fetchStatus: `error: ${message}`,
          }));
        }
        const nextAttempt = (retryAttemptsRef.current[accountId] ?? 0) + 1;
        retryAttemptsRef.current[accountId] = nextAttempt;
        const delay = Math.min(30_000, 1_000 * 2 ** (nextAttempt - 1));
        clearRetryTimeout(accountId);
        retryTimeoutsRef.current[accountId] = window.setTimeout(() => {
          delete retryTimeoutsRef.current[accountId];
          if (!isMountedRef.current) {
            return;
          }
          void loadAccountPack(accountId);
        }, delay);
      } finally {
        loadingAccountsRef.current.delete(accountId);
      }
    },
    [clearRetryTimeout, getPack, rememberPack, sid, updateCard, markSubmitted, markUnsubmitted]
  );

  const stopPolling = React.useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
      reviewDebugLog('poll:stop');
    }
  }, []);

  const stopStream = React.useCallback(() => {
    if (eventSourceRef.current) {
      reviewDebugLog('sse:close-request', {
        readyState: eventSourceRef.current.readyState,
      });
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      reviewDebugLog('sse:closed');
    }
  }, []);

  React.useEffect(() => {
    reviewDebugLog('RunReviewPage session', { sid, debug: REVIEW_DEBUG_ENABLED });
  }, [sid]);

  React.useEffect(() => {
    if (!reviewReady) {
      return;
    }
    for (const item of reviewItems) {
      const accountId = item?.account_id;
      if (!accountId) {
        continue;
      }
      void loadAccountPack(accountId);
    }
  }, [reviewReady, reviewItems, loadAccountPack]);

  const schedulePoll = React.useCallback(
    (sessionId: string, options?: { immediate?: boolean; reason?: string }) => {
      stopPolling();
      const immediate = Boolean(options?.immediate);
      reviewDebugLog('poll:schedule', { sessionId, immediate, reason: options?.reason ?? null });
      pollIterationRef.current = 0;
      if (isMountedRef.current) {
        setPhase((state) => {
          if (state === 'ready' || state === 'error') {
            return state;
          }
          return 'waiting';
        });
      }
      beginWorkerWait();

      const poll = async () => {
        if (!isMountedRef.current) {
          return;
        }
        const iteration = pollIterationRef.current + 1;
        pollIterationRef.current = iteration;
        reviewDebugLog('poll:tick', { sessionId, iteration });
        try {
          const payload = await fetchRunFrontendReviewIndex(sessionId);
          if (!isMountedRef.current) {
            return;
          }
          globalQuestionsRef.current = extractReviewQuestions(payload);
          const reviewSection = getReviewSection(payload);
          const reviewItems = Array.isArray(reviewSection?.items) ? reviewSection.items : [];
          const normalizedReviewItems = applyReviewIndex(payload);
          const packsCount = extractPacksCount(payload);
          reviewDebugLog('poll:packs-count', { sessionId, iteration, packsCount });
          if (packsCount > 0 || reviewItems.length > 0 || normalizedReviewItems.length > 0) {
            reviewDebugLog('poll:packs-ready', { sessionId, iteration, packsCount });
            stopPolling();
            clearWorkerWait();
            if (normalizedReviewItems.length > 0) {
              bootstrapItemsRef.current = normalizedReviewItems;
            }
            await loadPackListing({ fallbackItems: normalizedReviewItems });
            return;
          }
          lastNetworkStatusRef.current = `Poll #${iteration} returned no review packs (packs_count=${packsCount}).`;
          beginWorkerWait();
        } catch (err) {
          reviewDebugLog('poll:error', { sessionId, error: err });
          console.warn('Review poll failed', err);
          const message = err instanceof Error ? err.message : 'Poll failed';
          lastNetworkStatusRef.current = `Poll error: ${message}`;
        }

        pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
        reviewDebugLog('poll:scheduled-next', { sessionId, iteration, delay: POLL_INTERVAL_MS });
      };

      if (immediate) {
        reviewDebugLog('poll:initial-immediate', { sessionId });
        void poll();
        return;
      }

      pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
      reviewDebugLog('poll:initial-timeout', { sessionId, delay: POLL_INTERVAL_MS });
    },
    [beginWorkerWait, clearWorkerWait, loadPackListing, applyReviewIndex, stopPolling]
  );

  const startStream = React.useCallback(
    (sessionId: string) => {
      stopStream();
      try {
        const url = buildFrontendReviewStreamUrl(sessionId);
        reviewDebugLog('sse:connect', { url, sessionId });
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;
        if (isMountedRef.current) {
          setPhase((state) => {
            if (state === 'ready' || state === 'loading' || state === 'error') {
              return state;
            }
            return 'waiting';
          });
        }
        beginWorkerWait();

        eventSource.onopen = () => {
          reviewDebugLog('sse:open', { url, sessionId });
        };

        eventSource.addEventListener('packs_ready', async (event) => {
          reviewDebugLog('sse:event', { url, sessionId, type: 'packs_ready', data: event?.data });
          try {
            if (!isMountedRef.current) {
              return;
            }
            stopPolling();
            clearWorkerWait();
            lastNetworkStatusRef.current = 'Received packs_ready event.';
            await loadPackListing({ fallbackItems: bootstrapItemsRef.current });
          } catch (err) {
            reviewDebugLog('sse:event-error', { url, sessionId, error: err });
            console.error('Failed to load packs after packs_ready', err);
          }
        });

        eventSource.onerror = () => {
          reviewDebugLog('sse:error', { url, sessionId });
          eventSource.close();
          eventSourceRef.current = null;
          reviewDebugLog('sse:error-closed', { url, sessionId });
          if (!isMountedRef.current) {
            return;
          }
          setLiveUpdatesUnavailable(true);
          lastNetworkStatusRef.current = 'Stream disconnected. Retrying via poll immediately.';
          if (!hasShownLiveUpdateToastRef.current) {
            hasShownLiveUpdateToastRef.current = true;
            showToast({
              variant: 'warning',
              title: 'Stream disconnected',
              description: 'The live updates stream disconnected. Retrying via poll…',
            });
          }
          reviewDebugLog('sse:fallback-poll', { sessionId, immediate: true });
          schedulePoll(sessionId, { immediate: true, reason: 'sse-error' });
        };
      } catch (err) {
        reviewDebugLog('sse:connect-error', { sessionId, error: err });
        console.warn('Unable to open review stream', err);
        setLiveUpdatesUnavailable(true);
        const message = err instanceof Error ? err.message : 'Unable to open review stream';
        lastNetworkStatusRef.current = `Stream connection failed: ${message}`;
        if (!hasShownLiveUpdateToastRef.current) {
          hasShownLiveUpdateToastRef.current = true;
          showToast({
            variant: 'warning',
            title: 'Live updates unavailable',
            description: 'Live updates unavailable, falling back to polling…',
          });
        }
        schedulePoll(sessionId);
      }
    },
    [beginWorkerWait, clearWorkerWait, loadPackListing, schedulePoll, showToast, stopStream]
  );

  React.useEffect(() => {
    if (!sid) {
      setPhase('error');
      setPhaseError('Missing run id.');
      return () => {
        stopPolling();
        stopStream();
        clearWorkerWait();
        stopFallbackTimeout();
      };
    }

    setPhase('loading');
    setPhaseError(null);
    setManifest(null);
    setCards({});
    setOrder([]);
    globalQuestionsRef.current = [];

    let cancelled = false;
    const isCancelled = () => cancelled;

    void loadManifestInfo(sid, { isCancelled });

    const initialize = async () => {
      const indexUrl = apiUrl(`/api/runs/${encodeURIComponent(sid)}/frontend/index`);
      if (isMountedRef.current) {
        setPhase((state) => (state === 'ready' ? state : 'waiting'));
      }
      reviewDebugLog('bootstrap:fetch', { url: indexUrl, sessionId: sid });
      try {
        const payload = await fetchRunFrontendReviewIndex(sid);
        if (isCancelled() || !isMountedRef.current) {
          return;
        }

        globalQuestionsRef.current = extractReviewQuestions(payload);
        const packsCount = extractPacksCount(payload);
        setDiagnosticsPacksCount(packsCount);
        const payloadRecord =
          payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : null;
        const packsField = payloadRecord?.packs;
        const reviewSection = getReviewSection(payload);
        const reviewItems = Array.isArray(reviewSection?.items) ? reviewSection.items : [];
        const normalizedReviewItems = applyReviewIndex(payload);
        const reviewPacksField = reviewSection?.packs;
        const hasPacks =
          packsCount > 0 ||
          reviewItems.length > 0 ||
          normalizedReviewItems.length > 0 ||
          (Array.isArray(reviewPacksField) && reviewPacksField.length > 0) ||
          (Array.isArray(packsField) && packsField.length > 0);

        reviewDebugLog('bootstrap:packs-count', {
          url: indexUrl,
          sessionId: sid,
          packsCount,
          hasPacks,
        });

        if (normalizedReviewItems.length > 0) {
          bootstrapItemsRef.current = normalizedReviewItems;
        }

        if (hasPacks) {
          reviewDebugLog('bootstrap:packs-ready', { url: indexUrl, sessionId: sid, packsCount });
          await loadPackListing({ fallbackItems: normalizedReviewItems });
          return;
        }

        lastNetworkStatusRef.current = `Initial index fetch returned no review packs (packs_count=${packsCount}).`;
        beginWorkerWait();
      } catch (err) {
        if (isCancelled() || !isMountedRef.current) {
          return;
        }
        reviewDebugLog('bootstrap:error', { url: indexUrl, sessionId: sid, error: err });
        console.warn('index fetch failed, falling back to stream/poll', err);
        const message = err instanceof Error ? err.message : 'Unable to load review index';
        lastNetworkStatusRef.current = `Initial index fetch failed: ${message}`;
      }

      if (isCancelled() || !isMountedRef.current) {
        return;
      }

      startTimeout(15_000, async () => {
        if (!isMountedRef.current || loadedRef.current) {
          return;
        }
        reviewDebugLog('timeout:refetch', { url: indexUrl, sessionId: sid });
        try {
          const retryPayload = await fetchRunFrontendReviewIndex(sid);
          if (!isMountedRef.current || loadedRef.current) {
            return;
          }
          globalQuestionsRef.current = extractReviewQuestions(retryPayload);
          const retryPacksCount = extractPacksCount(retryPayload);
          const retryRecord =
            retryPayload && typeof retryPayload === 'object' ? (retryPayload as Record<string, unknown>) : null;
          const retryPacksField = retryRecord?.packs;
          const retryReviewSection = getReviewSection(retryPayload);
          const retryReviewItems = Array.isArray(retryReviewSection?.items) ? retryReviewSection.items : [];
          const normalizedRetryItems = applyReviewIndex(retryPayload);
          const retryReviewPacks = retryReviewSection?.packs;
          const hasPacksRetry =
            retryPacksCount > 0 ||
            retryReviewItems.length > 0 ||
            normalizedRetryItems.length > 0 ||
            (Array.isArray(retryReviewPacks) && retryReviewPacks.length > 0) ||
            (Array.isArray(retryPacksField) && retryPacksField.length > 0);
          lastNetworkStatusRef.current = hasPacksRetry
            ? `Timeout refetch found review packs (packs_count=${retryPacksCount}).`
            : `Timeout refetch returned no review packs (packs_count=${retryPacksCount}).`;
          if (hasPacksRetry) {
            reviewDebugLog('timeout:packs-ready', { url: indexUrl, sessionId: sid, retryPacksCount });
            if (normalizedRetryItems.length > 0) {
              bootstrapItemsRef.current = normalizedRetryItems;
            }
            await loadPackListing({ fallbackItems: normalizedRetryItems });
            if (isMountedRef.current) {
              setPhase('ready');
            }
            return;
          }
          stopPolling();
          stopStream();
          clearWorkerWait();
          if (isMountedRef.current) {
            setPhase('error');
            setPhaseError(
              lastNetworkStatusRef.current ?? 'No review packs received after waiting for updates.'
            );
          }
        } catch (err) {
          reviewDebugLog('timeout:refetch-error', { url: indexUrl, sessionId: sid, error: err });
          const message = err instanceof Error ? err.message : 'Unable to load review packs';
          lastNetworkStatusRef.current = `Timeout refetch failed: ${message}`;
          stopPolling();
          stopStream();
          clearWorkerWait();
          if (isMountedRef.current) {
            setPhase('error');
            setPhaseError(message);
          }
        }
      });

      reviewDebugLog('bootstrap:fallback', { sessionId: sid });
      startStream(sid);
      schedulePoll(sid);
    };

    void initialize();

    return () => {
      cancelled = true;
      stopPolling();
      stopStream();
      clearWorkerWait();
      stopFallbackTimeout();
    };
  }, [
    sid,
    beginWorkerWait,
    clearWorkerWait,
    loadManifestInfo,
    loadPackListing,
    applyReviewIndex,
    schedulePoll,
    startStream,
    stopPolling,
    stopStream,
    stopFallbackTimeout,
    startTimeout,
  ]);

  const handleAnswerChange = React.useCallback(
    (accountId: string, answers: AccountQuestionAnswers) => {
      updateCard(accountId, (state) => ({
        ...state,
        answers,
        status: state.status === 'done' ? 'ready' : state.status,
        error: null,
        errorDetails: null,
        success: false,
      }));
      markUnsubmitted(accountId);
    },
    [markUnsubmitted, updateCard]
  );

  const handleSubmit = React.useCallback(
    async (accountId: string) => {
      const card = cards[accountId];
      if (!sid || !card || card.status === 'saving') {
        return;
      }
      const explanation = card.answers.explanation;
      const hasExplanation =
        typeof explanation === 'string' && explanation.trim() !== '';
      const resolvedClaimsPayload = resolveClaimsPayload(card.pack?.claims);
      const claimDefinitions = new Map(
        resolvedClaimsPayload.items.map((claim) => [claim.key, claim] as const)
      );
      const normalizedClaims = REVIEW_CLAIMS_ENABLED
        ? normalizeSelectedClaims(card.answers.selectedClaims ?? [])
        : [];
      if (!hasExplanation && normalizedClaims.length === 0) {
        updateCard(accountId, (state) => ({
          ...state,
          error: 'Please add an explanation or select at least one claim before submitting.',
          errorDetails: null,
        }));
        return;
      }

      if (
        REVIEW_CLAIMS_ENABLED &&
        hasMissingRequiredDocs(normalizedClaims, card.answers.attachments, claimDefinitions)
      ) {
        updateCard(accountId, (state) => ({
          ...state,
          error: 'Please upload the required documents for the claims you selected.',
          errorDetails: null,
        }));
        return;
      }

      const cleaned = prepareAnswersPayload(card.answers, {
        includeClaims: REVIEW_CLAIMS_ENABLED,
      });

      updateCard(accountId, (state) => ({
        ...state,
        status: 'saving',
        error: null,
        errorDetails: null,
        success: true,
      }));

      try {
        const response = await submitFrontendReviewAnswers(sid, accountId, cleaned);
        let updatedPack: ReviewAccountPack | null = null;
        updateCard(accountId, (state) => {
          const persistedAnswers: Record<string, unknown> = {
            answers: {
              ...(response?.answers && typeof response.answers === 'object' ? response.answers : {}),
              ...cleaned.answers,
            },
          };
          if (cleaned.claims && cleaned.claims.length > 0) {
            persistedAnswers.claims = cleaned.claims;
          }
          if (cleaned.evidence && cleaned.evidence.length > 0) {
            persistedAnswers.evidence = cleaned.evidence;
          }
          const nextPack = state.pack
            ? {
                ...state.pack,
                answers: persistedAnswers,
                response,
              }
            : state.pack;
          updatedPack = nextPack ?? null;
          return {
            ...state,
            status: 'done',
            success: true,
            error: null,
            errorDetails: null,
            response: response ?? null,
            pack: nextPack,
          };
        });
        if (updatedPack) {
          rememberPack(accountId, updatedPack);
        }
        markSubmitted(accountId);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to submit answers';
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          error: message,
          errorDetails: null,
          success: false,
        }));
        showToast({
          variant: 'error',
          title: 'Save failed',
          description: message,
        });
      }
    },
    [cards, sid, updateCard, rememberPack, showToast, markSubmitted]
  );

  React.useEffect(() => {
    reviewDebugLog('cards:rendered', { count: order.length, accounts: order });
  }, [order]);

  const orderedCards = React.useMemo(
    () =>
      order.map((accountId) => ({
        accountId,
        state: cards[accountId] ?? { ...createInitialCardState(), status: 'waiting' },
      })),
    [order, cards]
  );
  const diagnosticsCards = React.useMemo(
    () =>
      orderedCards.map(({ accountId, state }) => ({
        accountId,
        status: state?.fetchStatus ?? state?.status ?? 'idle',
      })),
    [orderedCards]
  );

  const readyCount = React.useMemo(
    () => orderedCards.filter(({ state }) => state?.status === 'ready' || state?.status === 'done').length,
    [orderedCards]
  );
  const submittedCount = submittedAccounts.size;
  const totalCards = orderedCards.length;
  const ready = reviewReady;

  const handleCardLoad = React.useCallback(
    (accountId: string) => {
      void loadAccountPack(accountId);
    },
    [loadAccountPack]
  );

  const handleRetryLoad = React.useCallback(
    (accountId: string) => {
      void loadAccountPack(accountId, { force: true, resetAttempts: true });
    },
    [loadAccountPack]
  );

  const allDone = totalCards > 0 && submittedCount === totalCards;
  const isLoadingPhase = !ready && (phase === 'loading' || phase === 'waiting');
  const loaderMessage = phase === 'waiting'
    ? showWorkerHint
      ? 'Waiting for worker…'
      : 'Waiting for review packs…'
    : 'Loading review packs…';

  const handleFinishReview = React.useCallback(() => {
    if (!sid) {
      return;
    }
    setIsCompleting(true);
    navigate(`/runs/${encodeURIComponent(sid)}/review/complete`);
  }, [sid, navigate]);

  React.useEffect(() => {
    const wasReady = previousReadyRef.current;
    if (reviewReady && !wasReady) {
      const stats = readyStatsRef.current;
      console.log(
        `[frontend-review] ready=true items=${reviewItems.length}, apiPacks=${stats.api}, staticPacks=${stats.staticPacks}`
      );
    }
    previousReadyRef.current = reviewReady;
  }, [reviewReady, reviewItems.length]);

  return (
    <div className={`mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8 ${allDone ? 'pb-32' : ''}`}>
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900">Run review</h1>
        {sid ? <p className="text-sm text-slate-600">Run {sid}</p> : null}
        {manifest?.frontend?.review && readyCount > 0 ? (
          <p className="text-sm text-slate-600">
            {readyCount} {readyCount === 1 ? 'card' : 'cards'} ready for review.
          </p>
        ) : null}
        {liveUpdatesUnavailable ? (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            Live updates unavailable, falling back to polling…
          </div>
        ) : null}
        {isLoadingPhase ? (
          <div className="flex items-center gap-2 text-sm text-slate-600" role="status" aria-live="polite">
            <span
              aria-hidden="true"
              className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-transparent"
            />
            <span>{loaderMessage}</span>
          </div>
        ) : null}
        {phaseError && phase === 'error' ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">{phaseError}</div>
        ) : null}
      </header>

      {frontendMissing && sid ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          Frontend manifest block missing for run <span className="font-mono">{sid}</span>. Waiting for worker to publish review
          metadata.
        </div>
      ) : null}

      {ready ? (
        orderedCards.length > 0 ? (
          <div className="review-grid space-y-6">
            {orderedCards.map(({ accountId, state }) => (
              <ReviewCardContainer
                key={accountId}
                accountId={accountId}
                sessionId={sid}
                state={state ?? createInitialCardState()}
                onChange={(answers) => handleAnswerChange(accountId, answers)}
                onSubmit={() => handleSubmit(accountId)}
                onLoad={handleCardLoad}
                onRetry={handleRetryLoad}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">No review cards yet</div>
        )
      ) : null}

      {ready && allDone ? (
        <div className="flex flex-col items-start gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-semibold">All set!</p>
            <p className="text-emerald-800">You answered every review card.</p>
          </div>
          {sid ? (
            <Link
              to={`/runs/${encodeURIComponent(sid)}/accounts`}
              className="inline-flex items-center justify-center rounded-md border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-500"
            >
              View accounts
            </Link>
          ) : null}
        </div>
      ) : null}

      {ready && orderedCards.length > 0 ? (
        <p className="text-xs text-slate-500">
          {submittedCount} of {orderedCards.length} cards submitted.
        </p>
      ) : null}

      {ready && allDone ? (
        <div className="fixed bottom-0 left-0 right-0 border-t border-slate-200 bg-white shadow-lg">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
            <div>
              <p className="text-sm font-semibold text-slate-900">All cards submitted</p>
              <p className="text-sm text-slate-600">You can finish the review now.</p>
            </div>
            <button
              type="button"
              onClick={handleFinishReview}
              className="inline-flex items-center justify-center rounded-md border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:border-emerald-300 disabled:bg-emerald-300"
              disabled={isCompleting}
            >
              {isCompleting ? 'Finishing…' : 'Finish review'}
            </button>
          </div>
        </div>
      ) : null}

      {DEV_DIAGNOSTICS_ENABLED ? (
        <footer className="mt-8 border-t border-dashed border-slate-300 pt-4 text-xs text-slate-500">
          <p className="font-semibold text-slate-600">Diagnostics</p>
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            <div>
              <span className="font-medium">SID:</span>{' '}
              <span className="font-mono text-slate-700">{sid ?? 'n/a'}</span>
            </div>
            <div>
              <span className="font-medium">packs_count:</span>{' '}
              <span className="font-mono text-slate-700">
                {diagnosticsPacksCount != null ? diagnosticsPacksCount : 'unknown'}
              </span>
            </div>
          </div>
          <div className="mt-3 space-y-1">
            <p className="font-medium text-slate-600">Card fetch status</p>
            {diagnosticsCards.length > 0 ? (
              <ul className="space-y-1">
                {diagnosticsCards.map(({ accountId, status }) => (
                  <li key={accountId} className="font-mono text-slate-700">
                    {accountId}: {status}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="font-mono text-slate-400">No cards loaded</p>
            )}
          </div>
        </footer>
      ) : null}
    </div>
  );
}

export default function RunReviewPage() {
  const { sid } = useParams();

  return <RunReviewPageContent sid={sid} />;
}
