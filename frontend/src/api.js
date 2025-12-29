function getImportMetaEnv() {
  try {
    if (typeof import.meta !== 'undefined' && import.meta?.env) {
      return import.meta.env;
    }
  } catch {}
  return {};
}

const metaEnv = getImportMetaEnv();

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

const metaEnvDev = metaEnv.DEV;
const isMetaEnvDev =
  typeof metaEnvDev === 'boolean'
    ? metaEnvDev
    : typeof metaEnvDev === 'string'
      ? metaEnvDev.toLowerCase() === 'true'
      : false;

const nodeEnv = typeof process !== 'undefined' ? process.env?.NODE_ENV : undefined;
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

export const apiUrl = (path) =>
  `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;

function encodePathSegments(path = '') {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

export function joinRunAsset(base, rel) {
  const b = base.replace(/\/+$/, '');
  const r = rel.replace(/\\/g, '/').replace(/^\/+/, '');
  return `${b}/${r}`;
}

function buildRunAssetUrl(sessionId, relativePath) {
  const basePath = `/runs/${encodeURIComponent(sessionId)}`;
  const baseUrl = apiUrl(basePath);
  if (!relativePath) {
    return baseUrl;
  }
  const normalizedPath = (typeof relativePath === 'string' ? relativePath : '').replace(/\\/g, '/');
  const encodedPath = encodePathSegments(normalizedPath);
  return joinRunAsset(baseUrl, encodedPath);
}

function trimSlashes(input = '') {
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

function ensureFrontendPath(candidate, fallback) {
  const trimmed = trimSlashes(typeof candidate === 'string' ? candidate : '');
  const base = trimmed || trimSlashes(typeof fallback === 'string' ? fallback : '');
  if (!base) {
    return 'frontend';
  }
  if (base.startsWith('frontend/')) {
    return base;
  }
  return `frontend/${base}`;
}

function stripFrontendPrefix(path) {
  const trimmed = trimSlashes(typeof path === 'string' ? path : '');
  if (trimmed.startsWith('frontend/')) {
    return trimmed.slice('frontend/'.length);
  }
  return trimmed;
}

function joinFrontendPath(base, child) {
  return [trimSlashes(base), trimSlashes(child)].filter(Boolean).join('/');
}

export function isAbsUrl(s) {
  return /^https?:\/\//i.test(typeof s === 'string' ? s : '');
}

export function normalizeStaticPackPath(p) {
  const s = (typeof p === 'string' ? p : '').replace(/\\/g, '/').trim();
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

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function unwrapPackCandidate(source) {
  if (!isRecord(source)) {
    return null;
  }

  const visited = new Set();
  let current = source;

  while (current && !visited.has(current)) {
    visited.add(current);
    const nested = current.pack;
    if (isRecord(nested)) {
      current = nested;
      continue;
    }
    break;
  }

  return current;
}

function extractPackPayload(candidate, fallbackAccountId) {
  const packCandidate = unwrapPackCandidate(candidate);
  if (!packCandidate) {
    return null;
  }

  const result = { ...packCandidate };
  const rootRecord = isRecord(candidate) ? candidate : null;

  let accountId = typeof result.account_id === 'string' ? result.account_id.trim() : '';
  if (!accountId) {
    const normalizedFallback =
      typeof fallbackAccountId === 'string' ? fallbackAccountId.trim() : '';
    if (!normalizedFallback) {
      return null;
    }
    accountId = normalizedFallback;
  }
  result.account_id = accountId;

  if (result.answers == null && rootRecord && isRecord(rootRecord.answers)) {
    result.answers = rootRecord.answers;
  }

  if (result.response == null && rootRecord && isRecord(rootRecord.response)) {
    result.response = rootRecord.response;
  }

  return result;
}

function normalizeAccountPackPayload(candidate, fallbackAccountId) {
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

function buildFrontendReviewAccountUrl(sessionId, accountId) {
  return apiUrl(`/api/runs/${encodeURIComponent(sessionId)}/frontend/review/accounts/${encodeURIComponent(accountId)}`);
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  let data = null;
  let parseError = null;
  try {
    data = await response.json();
  } catch (error) {
    parseError = error;
  }

  if (!response.ok) {
    const detail = (data && (data.message || data.error)) || response.statusText;
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }

  if (data === null && parseError) {
    const detail = parseError instanceof Error ? parseError.message : String(parseError);
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Failed to parse JSON${suffix}`);
  }

  return data;
}

export async function fetchFrontendReviewManifest(sessionId, init) {
  const rootIndex = await fetchJson(buildRunAssetUrl(sessionId, 'frontend/index.json'), init);

  const stage = (rootIndex && rootIndex.review) || {};
  const indexPath = ensureFrontendPath(
    stage.index_rel || stage.index || 'review/index.json',
    'review/index.json'
  );
  const packsDirPath = ensureFrontendPath(
    stage.packs_dir_rel || stage.packs_dir || 'review/packs',
    'review/packs'
  );
  const responsesDirPath = ensureFrontendPath(
    stage.responses_dir_rel || stage.responses_dir || 'review/responses',
    'review/responses'
  );

  let manifestPayload = stage;
  if (!manifestPayload || !Array.isArray(manifestPayload.packs)) {
    manifestPayload = await fetchJson(buildRunAssetUrl(sessionId, indexPath));
  }

  const packs = Array.isArray(manifestPayload?.packs)
    ? manifestPayload.packs.map((entry) => {
        const pack = { ...entry };
        const rawPath =
          typeof entry.pack_path === 'string'
            ? entry.pack_path
            : typeof entry.path === 'string'
            ? entry.path
            : undefined;

        const defaultPath = joinFrontendPath(packsDirPath, `${entry.account_id}.json`);
        const normalizedPath =
          normalizeStaticPackPath(rawPath) ?? normalizeStaticPackPath(defaultPath) ?? defaultPath;

        pack.pack_path = normalizedPath;
        pack.pack_path_rel = stripFrontendPrefix(normalizedPath);
        pack.path = normalizedPath;
        return pack;
      })
    : [];

  return {
    sid: manifestPayload?.sid || rootIndex?.sid,
    stage: manifestPayload?.stage || stage.stage || 'review',
    schema_version: manifestPayload?.schema_version || stage.schema_version,
    counts: manifestPayload?.counts || stage.counts,
    generated_at: manifestPayload?.generated_at || stage.generated_at,
    packs,
    index_rel: stripFrontendPrefix(indexPath),
    index_path: indexPath,
    packs_dir_rel: stripFrontendPrefix(packsDirPath),
    packs_dir_path: packsDirPath,
    responses_dir_rel: stripFrontendPrefix(responsesDirPath),
    responses_dir_path: responsesDirPath,
  };
}

export async function fetchFrontendReviewAccount(sessionId, accountId, init) {
  if (!accountId) {
    throw new Error('Missing account id');
  }

  let requestInit = init;
  let staticPath = null;

  if (init && typeof init === 'object' && Object.prototype.hasOwnProperty.call(init, 'packPath')) {
    const { packPath: candidate, ...rest } = init;
    requestInit = rest;
    if (typeof candidate === 'string' && candidate.trim()) {
      staticPath = normalizeStaticPackPath(candidate);
    }
  } else if (
    init &&
    typeof init === 'object' &&
    Object.prototype.hasOwnProperty.call(init, 'staticPath')
  ) {
    const { staticPath: candidate, ...rest } = init;
    requestInit = rest;
    if (typeof candidate === 'string' && candidate.trim()) {
      staticPath = normalizeStaticPackPath(candidate);
    }
  }

  if (!staticPath) {
    const manifest = await fetchFrontendReviewManifest(sessionId);
    const match = manifest.packs?.find((entry) => entry.account_id === accountId);
    const candidate =
      (typeof match?.pack_path === 'string' && match.pack_path) ||
      (typeof match?.path === 'string' && match.path) ||
      joinFrontendPath(
        manifest.packs_dir_path || ensureFrontendPath('review/packs', 'review/packs'),
        `${accountId}.json`
      );
    staticPath = normalizeStaticPackPath(candidate);
  }

  if (!staticPath) {
    throw new Error(`Unable to resolve pack path for account ${accountId}`);
  }

  let url;
  if (isAbsUrl(staticPath)) {
    url = staticPath;
  } else if (staticPath.startsWith('/runs/') || staticPath.startsWith('runs/')) {
    url = staticPath.startsWith('/') ? staticPath : `/${staticPath}`;
  } else {
    url = buildRunAssetUrl(sessionId, staticPath);
  }

  const payload = await fetchJson(url, requestInit);
  const pack = normalizeAccountPackPayload(payload, accountId);
  if (!pack) {
    throw new Error(`Pack ${accountId}: No pack payload found`);
  }
  return pack;
}

export async function submitFrontendReviewAnswers(sessionId, accountId, answers, init) {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  const payload = {
    answers,
    client_ts: new Date().toISOString(),
  };
  return fetchJson(`${buildFrontendReviewAccountUrl(sessionId, accountId)}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    body: JSON.stringify(payload),
    ...init,
  });
}

export async function completeFrontendReview(sessionId, init) {
  const response = await fetch(
    apiUrl(`/api/runs/${encodeURIComponent(sessionId)}/frontend/review/complete`),
    {
      method: 'POST',
      ...(init || {}),
    }
  );

  if (!response.ok) {
    let detail;
    try {
      const data = await response.json();
      detail = data?.message || data?.error;
    } catch {
      // Ignore parse errors for non-JSON responses
    }
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }
}

export async function startProcess(email, file) {
  const formData = new FormData();
  formData.append('email', email);
  formData.append('file', file);

  const response = await fetch(apiUrl('/api/start-process'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to start process');
  }

  return response.json();
}

// Async upload API (preferred)
export async function uploadReport(email, file) {
  const fd = new FormData();
  fd.append('email', email);
  fd.append('file', file);

  const res = await fetch(apiUrl('/api/upload'), { method: 'POST', body: fd });
  if (res.status === 404) {
    throw new Error(
      'Could not reach backend (404 from dev server). Did you configure VITE_API_BASE_URL or Vite proxy?'
    );
  }
  let data = {};
  try {
    data = await res.json();
  } catch {
    // swallow JSON parse errors to craft a useful message below
  }
  if (!res.ok || !data?.ok || !data?.session_id) {
    const msg = data?.message || `Upload failed (status ${res.status})`;
    throw new Error(msg);
  }
  return data; // { ok:true, status:"queued", session_id, task_id }
}

export async function pollResult(sessionId, abortSignal) {
  // One attempt of polling. Treat 404 as in-progress (session not yet materialized)
  try {
    const res = await fetch(
      apiUrl(`/api/result?session_id=${encodeURIComponent(sessionId)}`),
      { signal: abortSignal }
    );
    let data = null;
    try {
      data = await res.json();
    } catch {
      // ignore JSON parse errors; handle via status code below
    }
    if (res.status === 404) {
      return { ok: true, status: 'processing' };
    }
    if (!res.ok) {
      throw new Error(data?.message || `Result request failed (${res.status})`);
    }
    return data;
  } catch {
    // Network/reset: surface as in-progress to keep UI tolerant
    return { ok: true, status: 'processing' };
  }
}

export async function listAccounts(sessionId) {
  const res = await fetch(apiUrl(`/api/accounts/${encodeURIComponent(sessionId)}`));
  const data = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `List accounts failed (${res.status})`);
  }
  return data.accounts || [];
}

export async function getAccount(sessionId, accountId) {
  const res = await fetch(
    apiUrl(`/api/accounts/${encodeURIComponent(sessionId)}/${encodeURIComponent(accountId)}`)
  );
  const data = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `Get account failed (${res.status})`);
  }
  return data.account;
}

export async function submitExplanations(payload) {
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

export async function getSummaries(sessionId) {
  const response = await fetch(apiUrl(`/api/summaries/${sessionId}`));
  if (!response.ok) {
    throw new Error('Failed to fetch summaries');
  }
  return response.json();
}
