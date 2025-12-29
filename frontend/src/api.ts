function getImportMetaEnv(): Record<string, string | boolean | undefined> {
  try {
    if (typeof import.meta !== 'undefined' && (import.meta as any)?.env) {
      return (import.meta as any).env;
    }
  } catch {}
  return {};
}

const metaEnv = getImportMetaEnv();

import type { AccountPack } from './components/AccountCard';
import type { AttachmentsMap } from './types/review';
import { REVIEW_DEBUG_ENABLED, reviewDebugLog } from './utils/reviewDebug';

const metaEnvConfiguredApiBaseRaw =
  (metaEnv.VITE_API_BASE_URL ?? metaEnv.VITE_API_URL) ??
  (typeof process !== 'undefined'
    ? process.env?.VITE_API_BASE_URL ?? process.env?.VITE_API_URL
    : undefined);

const trimmedMetaEnvConfiguredApiBase =
  typeof metaEnvConfiguredApiBaseRaw === 'string'
    ? metaEnvConfiguredApiBaseRaw.trim()
    : typeof metaEnvConfiguredApiBaseRaw === 'number' ||
        typeof metaEnvConfiguredApiBaseRaw === 'boolean'
      ? String(metaEnvConfiguredApiBaseRaw).trim()
      : '';

const rawConfiguredApiBase = trimmedMetaEnvConfiguredApiBase;

const trimmedConfiguredApiBase =
  typeof rawConfiguredApiBase === 'string' ? rawConfiguredApiBase.trim() : '';

const metaEnvDev = (metaEnv as Record<string, unknown>).DEV;
const isMetaEnvDev =
  typeof metaEnvDev === 'boolean'
    ? metaEnvDev
    : typeof metaEnvDev === 'string'
      ? metaEnvDev.toLowerCase() === 'true'
      : false;

const processEnv = typeof process !== 'undefined' ? process.env : undefined;
const nodeEnv = processEnv?.NODE_ENV;
const isProcessDev = typeof nodeEnv === 'string' ? nodeEnv !== 'production' : false;

const fallbackApiBase =
  !trimmedConfiguredApiBase && (isMetaEnvDev || isProcessDev) ? 'http://127.0.0.1:5000' : '';

const effectiveApiBaseInput = trimmedConfiguredApiBase || fallbackApiBase;

export const API_BASE_URL = effectiveApiBaseInput
  ? effectiveApiBaseInput.replace(/\/+$/, '')
  : '';

export const API_BASE_CONFIGURED = trimmedMetaEnvConfiguredApiBase.length > 0;
export const API_BASE_INFERRED = !API_BASE_CONFIGURED && API_BASE_URL.length > 0;

if (API_BASE_INFERRED && typeof console !== 'undefined') {
  console.warn(
    '[api] Falling back to default API base URL. Configure VITE_API_BASE_URL to point to your backend.'
  );
}

const API_BASE = API_BASE_URL;

export const apiUrl = (path: string) =>
  `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;

export interface SubmitReviewAnswers {
  explanation?: string;
  selected_claims?: string[];
  attachments?: AttachmentsMap;
  [key: string]: unknown;
}

export interface SubmitReviewPayload {
  answers: SubmitReviewAnswers;
}

function encodePathSegments(path: string): string {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

export function joinRunAsset(base: string, rel: string): string {
  const b = base.replace(/\/+$/, '');
  const r = rel.replace(/\\/g, '/').replace(/^\/+/, '');
  return `${b}/${r}`;
}

function buildRunAssetUrl(sessionId: string, relativePath: string): string {
  const basePath = `/runs/${encodeURIComponent(sessionId)}`;
  const baseUrl = apiUrl(basePath);
  if (!relativePath) {
    return baseUrl;
  }
  const normalizedPath = relativePath.replace(/\\/g, '/');
  const encodedPath = encodePathSegments(normalizedPath);
  return joinRunAsset(baseUrl, encodedPath);
}

function trimSlashes(input?: string | null): string {
  if (typeof input !== 'string') {
    return '';
  }
  let result = input.trim();
  while (result.startsWith('/')) {
    result = result.slice(1);
  }
  while (result.endsWith('/')) {
    result = result.slice(0, -1);
  }
  return result;
}
function ensureFrontendPath(candidate: string | null | undefined, fallback: string): string {
  const trimmed = trimSlashes(candidate);
  const base = trimmed || trimSlashes(fallback);
  if (!base) {
    return 'frontend';
  }
  if (base.startsWith('frontend/')) {
    return base;
  }
  return `frontend/${base}`;
}

function stripFrontendPrefix(path: string | null | undefined): string {
  const trimmed = trimSlashes(path);
  if (trimmed.startsWith('frontend/')) {
    return trimmed.slice('frontend/'.length);
  }
  return trimmed;
}

function joinFrontendPath(base: string, child: string): string {
  return [trimSlashes(base), trimSlashes(child)].filter(Boolean).join('/');
}

function stripStagePrefix(path: string | null | undefined, stageName: string): string {
  const trimmed = trimSlashes(path);
  const normalizedStage = trimSlashes(stageName) || 'review';
  const stagePrefix = `frontend/${normalizedStage}/`;
  if (trimmed.startsWith(stagePrefix)) {
    return trimmed.slice(stagePrefix.length);
  }
  if (trimmed.startsWith('frontend/')) {
    return trimmed.slice('frontend/'.length);
  }
  return trimmed;
}

export function isAbsUrl(s: string): boolean {
  return /^https?:\/\//i.test(s);
}

export function normalizeStaticPackPath(p?: string | null): string | null {
  const s = (p || '').replace(/\\/g, '/').trim();
  if (!s) return null;

  if (isAbsUrl(s)) return s;

  if (s.startsWith('/runs/') || s.startsWith('runs/')) {
    return s.startsWith('/') ? s : `/${s}`;
  }

  if (s.startsWith('/frontend/') || s.startsWith('frontend/')) {
    return s.startsWith('/') ? s : `/${s}`;
  }

  return `/frontend/${s.replace(/^\/+/, '')}`;
}

function buildRunApiUrl(sessionId: string, path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return apiUrl(`/api/runs/${encodeURIComponent(sessionId)}${normalized}`);
}

export function buildRunFrontendManifestUrl(sessionId: string): string {
  const baseUrl = buildRunApiUrl(sessionId, '/frontend/manifest');
  return baseUrl.includes('?') ? `${baseUrl}&section=frontend` : `${baseUrl}?section=frontend`;
}

function buildFrontendReviewAccountUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/accounts')}/${encodeURIComponent(accountId)}`;
}

function buildFrontendReviewResponseUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/response')}/${encodeURIComponent(accountId)}`;
}

function buildFrontendReviewUploadUrl(sessionId: string): string {
  return buildRunApiUrl(sessionId, '/frontend/review/uploads');
}

export function buildFrontendReviewStreamUrl(sessionId: string): string {
  return buildRunApiUrl(sessionId, '/frontend/review/stream');
}

export interface FetchJsonErrorInfo {
  status?: number;
  statusText?: string;
  url?: string;
  responseText?: string;
}

export class FetchJsonError extends Error {
  status?: number;
  statusText?: string;
  url?: string;
  responseText?: string;

  constructor(message: string, info: FetchJsonErrorInfo = {}) {
    super(message);
    this.name = 'FetchJsonError';
    this.status = info.status;
    this.statusText = info.statusText;
    this.url = info.url;
    this.responseText = info.responseText;
  }
}

interface FetchJsonResult<T> {
  data: T;
  response: Response;
  rawBody: string | null;
}

async function fetchJsonWithStatus<T>(url: string, init?: RequestInit): Promise<FetchJsonResult<T>> {
  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:start', { url, init });
  }
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new FetchJsonError(`Request failed: ${detail}`, { url });
  }
  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:response', {
      url,
      status: response.status,
      ok: response.ok,
    });
  }

  const contentType = response.headers.get('Content-Type') ?? undefined;
  let rawBody: string | null = null;
  let data: unknown = null;
  let parseError: unknown = null;

  try {
    rawBody = await response.text();
  } catch (err) {
    parseError = err;
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:parse-error', { url, error: err, contentType });
    }
  }

  if (rawBody != null && parseError == null) {
    try {
      data = JSON.parse(rawBody);
    } catch (err) {
      parseError = err;
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetch:parse-error', {
          url,
          error: err,
          contentType,
          snippet: rawBody.slice(0, 200),
        });
      }
    }
  }

  if (!response.ok) {
    const bodyRecord = (data ?? null) && typeof data === 'object' && !Array.isArray(data)
      ? (data as Record<string, unknown>)
      : null;
    const detail =
      bodyRecord?.message ??
      bodyRecord?.error ??
      response.statusText ??
      (typeof rawBody === 'string' && rawBody ? rawBody.slice(0, 200) : undefined);
    const suffix = detail ? `: ${detail}` : '';
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:error', { url, status: response.status, detail });
    }
    throw new FetchJsonError(`Request failed (${response.status})${suffix}`, {
      status: response.status,
      statusText: response.statusText ?? undefined,
      url,
      responseText: typeof rawBody === 'string' ? rawBody : undefined,
    });
  }

  if (parseError) {
    const snippet = typeof rawBody === 'string' ? rawBody.slice(0, 200) : '';
    console.error(
      `[frontend-review] JSON parse failed (${url}) - content-type: ${contentType ?? 'unknown'}; snippet(first 200 bytes): ${snippet}`
    );
    const detail = parseError instanceof Error ? parseError.message : String(parseError);
    const suffix = detail ? `: ${detail}` : '';
    throw new FetchJsonError(`Failed to parse JSON${suffix}`, {
      url,
      responseText: typeof rawBody === 'string' ? rawBody : undefined,
    });
  }

  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:success', { url, body: data });
  }

  return { data: data as T, response, rawBody };
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const { data } = await fetchJsonWithStatus<T>(url, init);
  return data;
}

export interface FrontendReviewManifestPack {
  account_id: string;
  holder_name?: string | null;
  primary_issue?: string | null;
  display?: AccountPack['display'];
  account_number?: unknown;
  account_type?: unknown;
  status?: unknown;
  balance_owed?: unknown;
  date_opened?: unknown;
  closed_date?: unknown;
  path?: string;
  pack_path?: string;
  has_questions?: boolean;
  pack_path_rel?: string;
}

export interface FrontendReviewPackIndexEntry {
  account: string;
  file: string;
}

export interface FrontendReviewManifest {
  sid?: string;
  stage?: string;
  schema_version?: string | number;
  counts?: { packs?: number; responses?: number };
  packs?: FrontendReviewManifestPack[];
  generated_at?: string;
  built_at?: string;
  index_rel?: string;
  index_path?: string;
  packs_dir_rel?: string;
  packs_dir?: string;
  packs_dir_path?: string;
  responses_dir_rel?: string;
  responses_dir?: string;
  responses_dir_path?: string;
  packs_index?: FrontendReviewPackIndexEntry[];
  packs_count?: number;
  status?: string;
  queued?: boolean;
}

interface FrontendStageDescriptor {
  stage?: string;
  schema_version?: string | number;
  generated_at?: string;
  index?: string;
  index_rel?: string;
  packs_dir?: string;
  packs_dir_rel?: string;
  responses_dir?: string;
  responses_dir_rel?: string;
  counts?: { packs?: number; responses?: number };
  packs?: FrontendReviewManifestPack[];
}

interface FrontendReviewIndexApiResponse {
  status?: string;
  queued?: boolean;
  frontend?: {
    review?: FrontendStageDescriptor | FrontendReviewManifest | null;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface RunFrontendManifestResponse {
  sid?: string;
  frontend?: {
    review?: FrontendStageDescriptor | null;
    [key: string]: unknown;
  } | null;
}

export async function fetchRunFrontendManifest(
  sessionId: string,
  init?: RequestInit
): Promise<RunFrontendManifestResponse> {
  const url = buildRunFrontendManifestUrl(sessionId);
  const res = await fetchJson<RunFrontendManifestResponse>(url, init);

  const frontendSection = res?.frontend;
  const reviewCandidate =
    (frontendSection as { review?: FrontendStageDescriptor | null } | null | undefined)?.review ??
    (frontendSection as FrontendStageDescriptor | null | undefined) ??
    (res as { review?: FrontendStageDescriptor | null } | null | undefined)?.review ??
    null;

  const normalizedReview =
    reviewCandidate && typeof reviewCandidate === 'object' && !Array.isArray(reviewCandidate)
      ? (reviewCandidate as FrontendStageDescriptor)
      : null;

  let normalizedFrontend: RunFrontendManifestResponse['frontend'];

  if (frontendSection && typeof frontendSection === 'object' && !Array.isArray(frontendSection)) {
    normalizedFrontend = {
      ...(frontendSection as Record<string, unknown>),
      review: normalizedReview,
    } as RunFrontendManifestResponse['frontend'];
  } else if (frontendSection === null) {
    normalizedFrontend = null;
  } else if (normalizedReview) {
    normalizedFrontend = { review: normalizedReview };
  } else {
    normalizedFrontend = frontendSection as RunFrontendManifestResponse['frontend'];
  }

  return {
    ...res,
    frontend: normalizedFrontend,
  };
}

function normalizeFrontendReviewManifestPayload(
  payload: FrontendReviewManifest | FrontendStageDescriptor | null | undefined
): FrontendReviewManifest {
  const base: FrontendReviewManifest =
    payload && typeof payload === 'object' && !Array.isArray(payload)
      ? ({ ...payload } as FrontendReviewManifest)
      : ({} as FrontendReviewManifest);

  const stageName =
    typeof base.stage === 'string' && base.stage.trim() ? base.stage.trim() : 'review';

  const indexPath = ensureFrontendPath(
    (base as { index_path?: string }).index_path ??
      (base as { index?: string }).index ??
      `${stageName}/index.json`,
    `${stageName}/index.json`
  );

  const packsDirPath = ensureFrontendPath(
    (base as { packs_dir?: string }).packs_dir ??
      (base as { packs_dir_path?: string }).packs_dir_path ??
      (base as { packs_dir_rel?: string }).packs_dir_rel ??
      `${stageName}/packs`,
    `${stageName}/packs`
  );

  const responsesDirPath = ensureFrontendPath(
    (base as { responses_dir?: string }).responses_dir ??
      (base as { responses_dir_path?: string }).responses_dir_path ??
      (base as { responses_dir_rel?: string }).responses_dir_rel ??
      `${stageName}/responses`,
    `${stageName}/responses`
  );

  const packs = Array.isArray(base.packs)
    ? base.packs.map((entry) => {
        const pack: FrontendReviewManifestPack = { ...entry };
        const rawPath =
          typeof entry.pack_path === 'string'
            ? entry.pack_path
            : typeof entry.path === 'string'
            ? entry.path
            : typeof (entry as { file?: string }).file === 'string'
            ? (entry as { file?: string }).file
            : undefined;
        const defaultPath = joinFrontendPath(packsDirPath, `${entry.account_id}.json`);
        const normalizedPath =
          normalizeStaticPackPath(rawPath) ??
          normalizeStaticPackPath(defaultPath) ??
          `/frontend/${stripFrontendPrefix(defaultPath)}`;

        const finalPath = normalizedPath || `/frontend/${stripFrontendPrefix(defaultPath)}`;
        pack.pack_path = finalPath;
        pack.pack_path_rel = stripFrontendPrefix(finalPath);
        pack.path = finalPath;
        return pack;
      })
    : [];

  const counts =
    base.counts && typeof base.counts === 'object' && !Array.isArray(base.counts)
      ? ({ ...base.counts } as { packs?: number; responses?: number })
      : {};

  const packCount =
    typeof base.packs_count === 'number'
      ? base.packs_count
      : typeof counts.packs === 'number'
      ? counts.packs
      : packs.length;

  const responsesCount =
    typeof counts.responses === 'number' ? counts.responses : 0;

  counts.packs = packCount;
  counts.responses = responsesCount;

  const packsIndexSource = Array.isArray((base as { packs_index?: unknown }).packs_index)
    ? (((base as { packs_index?: FrontendReviewPackIndexEntry[] }).packs_index ?? []) as FrontendReviewPackIndexEntry[])
    : null;

  const fallbackIndex = packs.map((pack) => ({
    account: pack.account_id,
    file: stripStagePrefix(joinFrontendPath(packsDirPath, `${pack.account_id}.json`), stageName),
  }));

  const packsIndex = (packsIndexSource ?? fallbackIndex)
    .map((entry) => {
      if (!entry) {
        return null;
      }
      const accountCandidate =
        typeof (entry as { account?: unknown }).account === 'string'
          ? (entry as { account?: string }).account
          : typeof (entry as { account_id?: unknown }).account_id === 'string'
          ? ((entry as unknown as { account_id?: string }).account_id as string)
          : '';
      const account = (accountCandidate || '').trim();
      if (!account) {
        return null;
      }

      const fileCandidate =
        typeof entry.file === 'string' && entry.file
          ? entry.file
          : typeof (entry as { path?: string }).path === 'string'
          ? ((entry as unknown as { path?: string }).path as string)
          : joinFrontendPath(packsDirPath, `${account}.json`);

      const normalizedStatic =
        normalizeStaticPackPath(fileCandidate) ??
        normalizeStaticPackPath(joinFrontendPath(packsDirPath, `${account}.json`)) ??
        `/frontend/${stripFrontendPrefix(joinFrontendPath(packsDirPath, `${account}.json`))}`;

      const file =
        stripStagePrefix(normalizedStatic, stageName) ||
        stripStagePrefix(joinFrontendPath(packsDirPath, `${account}.json`), stageName);

      return { account, file };
    })
    .filter((entry): entry is FrontendReviewPackIndexEntry => Boolean(entry));

  const builtAtCandidate =
    typeof base.built_at === 'string'
      ? base.built_at
      : typeof base.generated_at === 'string'
      ? base.generated_at
      : undefined;

  const manifest: FrontendReviewManifest = {
    ...base,
    stage: stageName,
    index_path: indexPath,
    index_rel: stripFrontendPrefix(indexPath),
    packs_dir: packsDirPath,
    packs_dir_rel: stripFrontendPrefix(packsDirPath),
    responses_dir: responsesDirPath,
    responses_dir_rel: stripFrontendPrefix(responsesDirPath),
    packs,
    counts,
    packs_count: packCount,
    packs_index: packsIndex,
  };

  if (builtAtCandidate && !manifest.built_at) {
    manifest.built_at = builtAtCandidate;
  }
  if (manifest.built_at && !manifest.generated_at) {
    manifest.generated_at = manifest.built_at;
  }

  return manifest;
}

export async function fetchFrontendReviewManifest(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendReviewManifest> {
  const indexUrl = buildRunApiUrl(sessionId, '/frontend/review/index');
  const { data, response } = await fetchJsonWithStatus<FrontendReviewIndexApiResponse>(
    indexUrl,
    init
  );

  const frontendSection = data?.frontend;
  const reviewCandidate =
    (frontendSection && typeof frontendSection === 'object' && !Array.isArray(frontendSection)
      ? (frontendSection as { review?: FrontendStageDescriptor | FrontendReviewManifest | null })
          .review ?? null
      : null) ?? null;

  const manifestPayload =
    (reviewCandidate && typeof reviewCandidate === 'object'
      ? (reviewCandidate as FrontendReviewManifest | FrontendStageDescriptor)
      : null) ?? (data as unknown as FrontendReviewManifest | FrontendStageDescriptor | null);

  const manifest = normalizeFrontendReviewManifestPayload(manifestPayload);

  if (response.status === 202) {
    const statusValue =
      typeof data?.status === 'string' && data.status.trim() ? data.status : 'building';
    const queuedValue =
      typeof data?.queued === 'boolean' ? (data.queued as boolean) : undefined;

    return {
      ...manifest,
      status: statusValue,
      queued: queuedValue,
      counts: {
        packs: manifest.packs_count ?? manifest.counts?.packs ?? manifest.packs?.length ?? 0,
        responses: manifest.counts?.responses ?? 0,
      },
    };
  }

  return manifest;
}

export interface FrontendReviewPackListingItem {
  account_id: string;
  file?: string;
}

export interface FrontendReviewPackListingResponse {
  items: FrontendReviewPackListingItem[];
}

export async function fetchRunFrontendReviewIndex(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendStageDescriptor | Record<string, unknown>> {
  return fetchJson<FrontendStageDescriptor | Record<string, unknown>>(
    buildRunApiUrl(sessionId, '/frontend/index'),
    init
  );
}

export async function fetchRunReviewPackListing(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendReviewPackListingResponse> {
  const response = await fetchJson<{ items?: FrontendReviewPackListingItem[] | null }>(
    buildRunApiUrl(sessionId, '/frontend/review/packs'),
    init
  );
  const items = Array.isArray(response.items) ? response.items : [];
  const normalizedItems = items.map((item) => {
    if (!item || typeof item !== 'object') {
      return item as FrontendReviewPackListingItem;
    }
    const file = typeof item.file === 'string' ? item.file.replace(/\\/g, '/') : item.file;
    if (file === item.file) {
      return item as FrontendReviewPackListingItem;
    }
    return { ...item, file } as FrontendReviewPackListingItem;
  });
  return { items: normalizedItems };
}

export type ReviewQuestion = {
  id?: string;
  prompt?: string;
  required?: boolean | string | null;
  [key: string]: unknown;
};

type FetchFrontendReviewAccountOptions = RequestInit & {
  staticPath?: string | null | undefined;
  globalQuestions?: ReviewQuestion[] | null;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

type ExtractedAccountPack = AccountPack & {
  account_id?: string | null;
  answers?: Record<string, string> | null;
  response?: FrontendReviewResponse | null;
};

function unwrapPackCandidate(source: unknown): Record<string, unknown> | null {
  if (!isRecord(source)) {
    return null;
  }

  const visited = new Set<Record<string, unknown>>();
  let current: Record<string, unknown> | null = source;

  while (current && !visited.has(current)) {
    visited.add(current);
    const nested = (current as { pack?: unknown }).pack;
    if (isRecord(nested)) {
      current = nested;
      continue;
    }
    break;
  }

  return current;
}

function extractPackPayload(
  candidate: unknown,
  fallbackAccountId?: string
): ExtractedAccountPack | null {
  const packCandidate = unwrapPackCandidate(candidate);
  if (!packCandidate) {
    return null;
  }

  const result = { ...packCandidate } as ExtractedAccountPack;
  const rootRecord = isRecord(candidate) ? (candidate as Record<string, unknown>) : null;

  const accountId = result.account_id;
  if (typeof accountId !== 'string' || accountId.trim() === '') {
    const normalizedFallback =
      typeof fallbackAccountId === 'string' ? fallbackAccountId.trim() : '';
    if (!normalizedFallback) {
      return null;
    }
    result.account_id = normalizedFallback;
  }

  if (result.answers == null && rootRecord) {
    const rootAnswers = rootRecord.answers;
    if (isRecord(rootAnswers)) {
      result.answers = rootAnswers as Record<string, string>;
    }
  }

  if (result.response == null && rootRecord) {
    const rootResponse = rootRecord.response;
    if (isRecord(rootResponse)) {
      result.response = rootResponse as FrontendReviewResponse;
    }
  }

  return result;
}

function normalizeAccountPackPayload(
  candidate: unknown,
  fallbackAccountId: string
): ExtractedAccountPack | null {
  const pack = extractPackPayload(candidate, fallbackAccountId);
  if (!pack) {
    return null;
  }

  const normalizedAccountId =
    typeof pack.account_id === 'string' ? pack.account_id.trim() : '';
  if (!normalizedAccountId) {
    return null;
  }

  pack.account_id = normalizedAccountId;
  return pack;
}

type FrontendReviewAccountAttemptKind = 'account' | 'pack' | 'static';

interface FrontendReviewAccountAttempt {
  kind: FrontendReviewAccountAttemptKind;
  url: string;
  label: string;
  error?: Error;
  status?: number;
  responseText?: string;
}

export interface FrontendReviewAccountAttemptResult {
  kind: FrontendReviewAccountAttemptKind;
  url: string;
  label: string;
  error?: Error;
  status?: number;
  responseText?: string;
}

export class FrontendReviewAccountError extends Error {
  readonly attempts: ReadonlyArray<FrontendReviewAccountAttemptResult>;

  constructor(message: string, attempts: FrontendReviewAccountAttempt[]) {
    super(message);
    this.name = 'FrontendReviewAccountError';
    this.attempts = attempts.map((attempt) => ({
      kind: attempt.kind,
      url: attempt.url,
      label: attempt.label,
      error: attempt.error,
      status: attempt.status,
      responseText: attempt.responseText,
    }));
  }
}

function ensurePackQuestions(
  pack: ExtractedAccountPack,
  globalQuestions: ReviewQuestion[]
): ExtractedAccountPack {
  if (!Array.isArray(pack.questions) || pack.questions.length === 0) {
    if (globalQuestions.length > 0) {
      pack.questions = globalQuestions;
    }
  }
  return pack;
}

function computeStaticUrl(
  sessionId: string,
  staticPath: string | null
): { url: string; label: string } | null {
  if (!staticPath) {
    return null;
  }

  const normalized = normalizeStaticPackPath(staticPath);
  if (!normalized) {
    return null;
  }

  const label = stripFrontendPrefix(normalized) || normalized;
  const url = isAbsUrl(normalized)
    ? normalized
    : normalized.startsWith('/runs/') || normalized.startsWith('runs/')
      ? normalized.startsWith('/')
        ? normalized
        : `/${normalized}`
      : buildRunAssetUrl(sessionId, normalized);

  return { url, label };
}

export async function fetchFrontendReviewAccount<T = AccountPack>(
  sessionId: string,
  accountId: string,
  initOrOptions?: RequestInit | FetchFrontendReviewAccountOptions
): Promise<T> {
  if (!accountId) {
    throw new Error('Missing account id');
  }

  let init: RequestInit | undefined;
  let staticPath: string | null = null;
  let globalQuestions: ReviewQuestion[] = [];

  if (initOrOptions && typeof initOrOptions === 'object' && 'staticPath' in initOrOptions) {
    const {
      staticPath: providedStaticPath,
      globalQuestions: providedGlobalQuestions,
      ...rest
    } = initOrOptions as FetchFrontendReviewAccountOptions;
    if (typeof providedStaticPath === 'string' && providedStaticPath.trim() !== '') {
      staticPath = normalizeStaticPackPath(providedStaticPath);
    }
    if (Array.isArray(providedGlobalQuestions)) {
      globalQuestions = providedGlobalQuestions;
    }
    init = rest as RequestInit;
  } else {
    init = initOrOptions as RequestInit | undefined;
  }

  if (!staticPath) {
    staticPath = normalizeStaticPackPath(joinFrontendPath('frontend/review/packs', `${accountId}.json`));
  }

  const attemptRecords: FrontendReviewAccountAttempt[] = [];
  const accountUrl = buildFrontendReviewAccountUrl(sessionId, accountId);

  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetchFrontendReviewAccount:attempt', {
      accountId,
      label: 'accounts/:id',
      url: accountUrl,
    });
  }

  try {
    const payload = await fetchJson<unknown>(accountUrl, init);
    const pack = normalizeAccountPackPayload(payload, accountId);
    if (!pack) {
      throw new Error('No pack payload in response');
    }
    ensurePackQuestions(pack, globalQuestions);
    console.info(`[frontend-review] Pack ${accountId}: loaded via API`);
    return pack as T;
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    const attempt: FrontendReviewAccountAttempt = {
      kind: 'account',
      url: accountUrl,
      label: 'accounts/:id',
      error,
    };
    if (err instanceof FetchJsonError) {
      attempt.status = err.status;
      attempt.responseText = err.responseText ?? undefined;
    }
    attemptRecords.push(attempt);
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetchFrontendReviewAccount:attempt-error', {
        accountId,
        label: attempt.label,
        url: attempt.url,
        error,
      });
    }
  }

  const staticTarget = computeStaticUrl(sessionId, staticPath);
  if (staticTarget) {
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetchFrontendReviewAccount:attempt', {
        accountId,
        label: `static(${staticTarget.label})`,
        url: staticTarget.url,
      });
    }

    try {
      const staticInit: RequestInit = { ...(init ?? {}), cache: 'no-store' };
      const payload = await fetchJson<unknown>(staticTarget.url, staticInit);
      const pack = normalizeAccountPackPayload(payload, accountId);
      if (!pack) {
        throw new Error('No pack payload in response');
      }
      ensurePackQuestions(pack, globalQuestions);
      console.info(
        `[frontend-review] Pack ${accountId}: loaded from static fallback (${staticTarget.url})`
      );
      return pack as T;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      const attempt: FrontendReviewAccountAttempt = {
        kind: 'static',
        url: staticTarget.url,
        label: `static(${staticTarget.label})`,
        error,
      };
      if (err instanceof FetchJsonError) {
        attempt.status = err.status;
        attempt.responseText = err.responseText ?? undefined;
      }
      attemptRecords.push(attempt);
      console.error(
        `[frontend-review] Pack ${accountId}: static fallback failed (${staticTarget.url}) - ${error.message}`
      );
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetchFrontendReviewAccount:attempt-error', {
          accountId,
          label: attempt.label,
          url: attempt.url,
          error,
        });
      }
    }
  }

  const detail = attemptRecords.length
    ? attemptRecords
        .map((attempt) => `${attempt.label}: ${attempt.error?.message ?? 'failed'}`)
        .join('; ')
    : 'No pack payload found.';

  throw new FrontendReviewAccountError(`Pack ${accountId}: ${detail}`, attemptRecords);
}

export async function fetchReviewPack(
  sessionId: string,
  accountId: string,
  relPath: string | null | undefined,
  globalQuestions: ReviewQuestion[] = []
): Promise<AccountPack> {
  return fetchFrontendReviewAccount<AccountPack>(sessionId, accountId, {
    staticPath: relPath ?? undefined,
    globalQuestions,
  });
}

export interface FrontendReviewResponseClientMeta {
  user_agent?: string | null;
  tz?: string | null;
  ts?: string | null;
  [key: string]: unknown;
}

export interface FrontendReviewResponse {
  account_id?: string | null;
  answers?: Record<string, unknown> | null;
  client_meta?: FrontendReviewResponseClientMeta | null;
  saved_at?: string | null;
  [key: string]: unknown;
}

export interface FrontendReviewUploadDocInfo {
  id: string;
  claim: string;
  doc_key: string;
  account_id?: string;
  filename?: string;
  stored_filename?: string;
  uploaded_at?: string;
  size?: number;
  [key: string]: unknown;
}

export interface FrontendReviewUploadResponse {
  ok: boolean;
  doc?: FrontendReviewUploadDocInfo;
  error?: string;
  message?: string;
}

function resolveUserAgent(): string | undefined {
  if (typeof navigator !== 'undefined' && typeof navigator.userAgent === 'string') {
    return navigator.userAgent;
  }
  return undefined;
}

function resolveTimeZone(): string | undefined {
  if (typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function') {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      if (typeof tz === 'string' && tz.trim() !== '') {
        return tz;
      }
    } catch (err) {
      console.warn('Unable to resolve timezone', err);
    }
  }
  return undefined;
}

export async function submitFrontendReviewAnswers(
  sessionId: string,
  accountId: string,
  payload: SubmitReviewPayload,
  init?: RequestInit
): Promise<FrontendReviewResponse> {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  const requestBody = {
    ...payload,
    client_meta: {
      user_agent: resolveUserAgent() ?? 'unknown',
      tz: resolveTimeZone() ?? 'UTC',
      ts: new Date().toISOString(),
    },
  };
  return fetchJson<FrontendReviewResponse>(
    buildFrontendReviewResponseUrl(sessionId, accountId),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      body: JSON.stringify(requestBody),
      ...init,
    }
  );
}

export async function uploadReviewDoc(
  sessionId: string,
  accountId: string,
  claim: string,
  docKey: string,
  files: File[]
): Promise<{ doc_ids: string[] }> {
  const form = new FormData();
  form.append('sid', sessionId);
  form.append('account_id', accountId);
  form.append('claim', claim);
  form.append('doc_key', docKey);
  files.forEach((file) => form.append('files', file));

  const response = await fetch(`${API_BASE}/api/runs/${sessionId}/frontend/review/uploads`, {
    method: 'POST',
    body: form,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }

  return response.json();
}

export async function uploadFrontendReviewEvidence(
  sessionId: string,
  accountId: string,
  claim: ClaimKey,
  docKey: string,
  file: File,
  init?: RequestInit
): Promise<FrontendReviewUploadResponse> {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  if (!claim) {
    throw new Error('Missing claim key');
  }
  if (!docKey) {
    throw new Error('Missing document key');
  }

  const form = new FormData();
  form.append('account_id', accountId);
  form.append('claim', claim);
  form.append('doc_key', docKey);
  form.append('file', file);

  const response = await fetch(buildFrontendReviewUploadUrl(sessionId), {
    method: 'POST',
    body: form,
    ...init,
  });

  const data = (await response.json().catch(() => ({}))) as FrontendReviewUploadResponse | undefined;

  if (!response.ok) {
    const errorMessage = data?.error ?? data?.message ?? `Upload failed (${response.status})`;
    throw new Error(errorMessage);
  }

  return data ?? { ok: true };
}

export async function completeFrontendReview(sessionId: string, init?: RequestInit): Promise<void> {
  const response = await fetch(buildRunApiUrl(sessionId, '/frontend/review/complete'), {
    method: 'POST',
    ...(init ?? {}),
  });

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const data = await response.json();
      detail = data?.message ?? data?.error ?? undefined;
    } catch (err) {
      // Ignore parse errors for non-JSON responses
    }
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }
}

export async function uploadReport(email: string, file: File) {
  const fd = new FormData();
  fd.append('email', email);
  fd.append('file', file);

  const res = await fetch(apiUrl('/api/upload'), { method: 'POST', body: fd });
  if (res.status === 404) {
    throw new Error(
      'Could not reach backend (404 from dev server). Did you configure VITE_API_BASE_URL or Vite proxy?'
    );
  }
  let data: any = {};
  try {
    data = await res.json();
  } catch (_) {}
  if (!res.ok || !data?.ok || !data?.session_id) {
    const msg = data?.message || `Upload failed (status ${res.status})`;
    throw new Error(msg);
  }
  return data as { ok: true; status: string; session_id: string; task_id?: string };
}

export interface PollResultResponse {
  ok?: boolean;
  status?: string;
  result?: unknown;
  message?: string;
}

export async function pollResult(
  sessionId: string,
  abortSignal?: AbortSignal
): Promise<PollResultResponse> {
  const url = apiUrl(`/api/result?session_id=${encodeURIComponent(sessionId)}`);
  try {
    const res = await fetch(url, { signal: abortSignal });
    let data: PollResultResponse | null = null;
    try {
      data = (await res.json()) as PollResultResponse;
    } catch (err) {
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetch:parse-error', {
          url,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (res.status === 404) {
      return { ok: true, status: 'processing' };
    }

    if (!res.ok) {
      throw new Error(data?.message || `Result request failed (${res.status})`);
    }

    return data ?? { ok: true, status: 'processing' };
  } catch (err) {
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:error', {
        url,
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return { ok: true, status: 'processing' };
  }
}

export async function getAccount(sessionId: string, accountId: string): Promise<any> {
  const res = await fetch(
    apiUrl(`/api/accounts/${encodeURIComponent(sessionId)}/${encodeURIComponent(accountId)}`)
  );
  const data: any = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `Get account failed (${res.status})`);
  }
  return data.account;
}

export interface SubmitExplanationsPayload {
  [key: string]: unknown;
}

export async function submitExplanations(payload: SubmitExplanationsPayload): Promise<any> {
  const response = await fetch(apiUrl('/api/submit-explanations'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Failed to submit explanations');
  }

  return response.json();
}

export async function getSummaries(sessionId: string): Promise<any> {
  const response = await fetch(apiUrl(`/api/summaries/${encodeURIComponent(sessionId)}`));

  if (!response.ok) {
    throw new Error('Failed to fetch summaries');
  }

  return response.json();
}
