import sampleAccountPack from '../__fixtures__/sampleAccountPack.json';
import type { AccountPack } from '../components/AccountCard';
import type { FrontendReviewManifest, FrontendReviewManifestPack } from '../api.ts';

interface FrontendReviewMockOptions {
  sessionId?: string;
  accountId?: string;
}

let teardownMock: (() => void) | null = null;

function createResponse(payload: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
}

function normalizeUrl(input: RequestInfo | URL): string {
  if (typeof input === 'string') {
    return input;
  }
  if (input instanceof URL) {
    return input.toString();
  }
  if (input instanceof Request) {
    return input.url;
  }
  return String(input);
}

export function enableFrontendReviewMock(options: FrontendReviewMockOptions = {}) {
  if (typeof window === 'undefined' || typeof window.fetch !== 'function' || typeof Response === 'undefined') {
    return () => {};
  }
  if (teardownMock) {
    return teardownMock;
  }

  const sessionId = options.sessionId ?? 'LOCAL-000';
  const accountId = options.accountId ?? 'idx-001';
  const encodedSid = encodeURIComponent(sessionId);
  const encodedAccountId = encodeURIComponent(accountId);

  const answers: Record<string, string> = {};

  const manifestPack: FrontendReviewManifestPack = {
    account_id: accountId,
    holder_name: sampleAccountPack.holder_name ?? sampleAccountPack.display?.holder_name ?? 'Sample Holder',
    primary_issue: sampleAccountPack.primary_issue ?? sampleAccountPack.display?.primary_issue ?? 'unknown_issue',
    display: sampleAccountPack.display as AccountPack['display'],
    has_questions: true,
  };

  const manifest: FrontendReviewManifest = {
    sid: sessionId,
    stage: 'review',
    schema_version: 'mock-1.0',
    counts: { packs: 1, responses: 0 },
    packs: [
      {
        ...manifestPack,
        pack_path: `frontend/review/packs/${accountId}.json`,
        pack_path_rel: `packs/${accountId}.json`,
        path: `frontend/review/packs/${accountId}.json`,
      },
    ],
    index_path: 'frontend/review/index.json',
    index_rel: 'review/index.json',
    packs_dir: 'frontend/review/packs',
    packs_dir_path: 'frontend/review/packs',
    packs_dir_rel: 'review/packs',
    responses_dir: 'frontend/review/responses',
    responses_dir_path: 'frontend/review/responses',
    responses_dir_rel: 'review/responses',
    packs_index: [{ account: accountId, file: `packs/${accountId}.json` }],
  };
  manifest.generated_at = manifest.generated_at ?? new Date().toISOString();
  manifest.built_at = manifest.generated_at;

  const rootIndex = {
    review: {
      stage: 'review',
      schema_version: manifest.schema_version,
      index_rel: 'review/index.json',
      packs_dir_rel: 'review/packs',
      responses_dir_rel: 'review/responses',
      counts: manifest.counts,
    },
  };

  const basePack: AccountPack & { account_id: string } = {
    account_id: accountId,
    holder_name: sampleAccountPack.holder_name ?? 'Sample Holder',
    primary_issue: sampleAccountPack.primary_issue ?? 'unknown_issue',
    display: sampleAccountPack.display as AccountPack['display'],
  };

  const originalFetch = window.fetch.bind(window);

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const method = (init?.method ?? (input instanceof Request ? input.method : 'GET') ?? 'GET').toUpperCase();
    const urlString = normalizeUrl(input);

    let url: URL | null = null;
    try {
      url = new URL(urlString, window.location.origin);
    } catch (err) {
      // Ignore malformed URLs and fall back to the original fetch.
    }

    const pathname = url?.pathname ?? '';

    if (method === 'GET' && pathname === `/runs/${encodedSid}/frontend/index.json`) {
      return createResponse(rootIndex);
    }

    if (method === 'GET' && pathname === `/api/runs/${encodedSid}/frontend/review/index`) {
      return createResponse({ frontend: { review: manifest } });
    }

    if (method === 'GET' && pathname === `/runs/${encodedSid}/frontend/review/index.json`) {
      return createResponse(manifest);
    }

    if (
      method === 'GET' &&
      pathname === `/runs/${encodedSid}/frontend/review/packs/${accountId}.json`
    ) {
      const pack = {
        ...basePack,
        answers: Object.keys(answers).length > 0 ? { ...answers } : undefined,
      };
      return createResponse(pack);
    }

    if (method === 'POST' && pathname === `/api/runs/${encodedSid}/frontend/review/accounts/${encodedAccountId}/answer`) {
      let payload: any = null;
      if (typeof init?.body === 'string') {
        try {
          payload = JSON.parse(init.body);
        } catch (err) {
          payload = null;
        }
      } else if (init?.body && typeof init.body === 'object') {
        payload = init.body;
      } else if (input instanceof Request) {
        try {
          payload = await input.clone().json();
        } catch (err) {
          payload = null;
        }
      }

      if (payload && payload.answers && typeof payload.answers === 'object') {
        for (const [key, value] of Object.entries(payload.answers)) {
          if (typeof value === 'string' && value.trim()) {
            answers[key] = value;
          } else {
            delete answers[key as string];
          }
        }
      }

      return createResponse({ ok: true });
    }

    if (method === 'POST' && pathname === `/api/runs/${encodedSid}/frontend/review/complete`) {
      return createResponse({ ok: true });
    }

    return originalFetch(input as RequestInfo, init);
  };

  teardownMock = () => {
    window.fetch = originalFetch;
  };

  return teardownMock;
}

export function disableFrontendReviewMock() {
  if (teardownMock) {
    teardownMock();
    teardownMock = null;
  }
}
