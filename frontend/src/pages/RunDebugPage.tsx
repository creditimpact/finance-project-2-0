import * as React from 'react';
import { useParams } from 'react-router-dom';
import { apiUrl, joinRunAsset } from '../api.ts';
import { reviewDebugLog } from '../utils/reviewDebug';

interface FetchState {
  url: string;
  loading: boolean;
  status?: number;
  ok?: boolean;
  error?: string | null;
  body?: unknown;
  timestamp?: number;
}

function createInitialState(url: string): FetchState {
  return { url, loading: true };
}

function formatBody(body: unknown): string {
  if (body === null || body === undefined) {
    return 'null';
  }
  if (typeof body === 'string') {
    return body;
  }
  try {
    return JSON.stringify(body, null, 2);
  } catch (err) {
    return String(body);
  }
}

async function fetchEndpoint(url: string): Promise<FetchState> {
  reviewDebugLog('debug-page:fetch', { url });
  try {
    const response = await fetch(url);
    const text = await response.text();
    let body: unknown = text;
    try {
      body = JSON.parse(text);
    } catch (err) {
      // leave body as text
    }
    return {
      url,
      loading: false,
      status: response.status,
      ok: response.ok,
      body,
      error: response.ok ? null : response.statusText || 'Request failed',
      timestamp: Date.now(),
    };
  } catch (err) {
    return {
      url,
      loading: false,
      error: err instanceof Error ? err.message : 'Request failed',
      timestamp: Date.now(),
    };
  }
}

function resolvePackUrl(sid: string, packsBody: unknown): string | null {
  if (!packsBody || typeof packsBody !== 'object') {
    return null;
  }
  const items = (packsBody as { items?: unknown }).items;
  if (!Array.isArray(items)) {
    return null;
  }
  for (const entry of items) {
    if (entry && typeof entry === 'object' && 'file' in entry) {
      const file = (entry as { file?: unknown }).file;
      if (typeof file === 'string' && file.trim() !== '') {
        const basePath = `/runs/${encodeURIComponent(sid)}`;
        const base = apiUrl(basePath);
        const url = joinRunAsset(base, file);
        if (url.includes('idx-')) {
          return url;
        }
      }
    }
  }
  return null;
}

function useRunDebugData(sid: string | undefined) {
  const [indexState, setIndexState] = React.useState<FetchState | null>(null);
  const [packsState, setPacksState] = React.useState<FetchState | null>(null);
  const [packState, setPackState] = React.useState<FetchState | null>(null);

  const load = React.useCallback(async () => {
    if (!sid) {
      return;
    }
    const encodedSid = encodeURIComponent(sid);
    const indexUrl = apiUrl(`/api/runs/${encodedSid}/frontend/index`);
    const packsUrl = apiUrl(`/api/runs/${encodedSid}/frontend/review/packs`);
    setIndexState(createInitialState(indexUrl));
    setPacksState(createInitialState(packsUrl));
    setPackState(null);

    const nextIndex = await fetchEndpoint(indexUrl);
    setIndexState(nextIndex);

    const nextPacks = await fetchEndpoint(packsUrl);
    setPacksState(nextPacks);

    const packUrl = resolvePackUrl(sid, nextPacks.body);
    if (packUrl) {
      setPackState(createInitialState(packUrl));
      const nextPack = await fetchEndpoint(packUrl);
      setPackState(nextPack);
    } else {
      const fallbackUrl = apiUrl(`/runs/${encodedSid}/frontend/review/packs/idx-000.json`);
      setPackState({
        url: fallbackUrl,
        loading: false,
        error: 'No pack listing available yet.',
        timestamp: Date.now(),
      });
    }
  }, [sid]);

  return {
    indexState,
    packsState,
    packState,
    reload: load,
  };
}

function DebugResponseCard({ title, state }: { title: string; state: FetchState | null }) {
  if (!state) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        <p className="text-sm text-slate-600">Waiting for request…</p>
      </section>
    );
  }

  const formattedBody = formatBody(state.body);
  const timeLabel = state.timestamp ? new Date(state.timestamp).toLocaleTimeString() : null;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
      <dl className="mt-2 space-y-1 text-sm text-slate-700">
        <div>
          <dt className="font-medium text-slate-600">URL</dt>
          <dd className="font-mono text-xs text-slate-800 break-all">{state.url}</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-600">Status</dt>
          <dd>
            {state.loading
              ? 'Loading…'
              : state.status !== undefined
                ? `${state.status}${state.ok ? ' (ok)' : ' (error)'}`
                : 'Not requested'}
          </dd>
        </div>
        {timeLabel ? (
          <div>
            <dt className="font-medium text-slate-600">Updated</dt>
            <dd>{timeLabel}</dd>
          </div>
        ) : null}
        {state.error ? (
          <div>
            <dt className="font-medium text-slate-600">Error</dt>
            <dd className="text-rose-700">{state.error}</dd>
          </div>
        ) : null}
      </dl>
      <pre className="mt-3 max-h-80 overflow-auto rounded bg-slate-950/90 p-3 text-xs text-slate-100">
        {formattedBody}
      </pre>
    </section>
  );
}

export default function RunDebugPage() {
  const { sid } = useParams<{ sid: string }>();
  const { indexState, packsState, packState, reload } = useRunDebugData(sid);

  React.useEffect(() => {
    void reload();
  }, [reload]);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900">Run debug</h1>
        {sid ? <p className="text-sm text-slate-600">Run {sid}</p> : <p className="text-sm text-rose-600">Missing sid</p>}
        <button
          type="button"
          onClick={() => reload()}
          className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:text-slate-900"
        >
          Refresh
        </button>
      </header>

      <div className="space-y-4">
        <DebugResponseCard title="/api/runs/:sid/frontend/index" state={indexState} />
        <DebugResponseCard title="/api/runs/:sid/frontend/review/packs" state={packsState} />
        <DebugResponseCard title="/runs/:sid/frontend/review/packs/idx-*.json" state={packState} />
      </div>
    </div>
  );
}
