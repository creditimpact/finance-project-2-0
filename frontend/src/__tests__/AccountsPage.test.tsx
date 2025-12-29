import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import AccountsPage from '../pages/Accounts';
import samplePack from '../__fixtures__/sampleAccountPack.json';

const originalFetch = global.fetch;

type FetchResponseInit = {
  ok?: boolean;
  status?: number;
  statusText?: string;
};

function createFetchResponse(data: unknown, init: FetchResponseInit = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: jest.fn().mockResolvedValue(data),
  } as unknown as Response;
}

function buildFrontendIndex(manifest: { counts?: { packs?: number; responses?: number } }) {
  return {
    review: {
      stage: 'review',
      index_rel: 'review/index.json',
      packs_dir_rel: 'review/packs',
      responses_dir_rel: 'review/responses',
      counts: manifest.counts ?? { packs: 0, responses: 0 },
    },
  };
}

function renderWithRouter() {
  return render(
    <MemoryRouter initialEntries={['/runs/S123/accounts']}>
      <Routes>
        <Route path="/runs/:sid/accounts" element={<AccountsPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('AccountsPage', () => {
  beforeEach(() => {
    (global as typeof globalThis & { fetch: jest.Mock }).fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  afterAll(() => {
    if (originalFetch) {
      global.fetch = originalFetch;
    } else {
      delete (global as typeof globalThis).fetch;
    }
  });

  it('renders manifest entries from the review stage', async () => {
    const manifest = {
      counts: { packs: 1, responses: 0 },
      packs: [
        {
          account_id: 'acct-1',
          holder_name: 'John Doe',
          primary_issue: 'wrong_account',
          display: samplePack.display,
          path: 'frontend/review/packs/acct-1.json',
        },
      ],
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce(createFetchResponse(buildFrontendIndex(manifest)))
      .mockResolvedValueOnce(createFetchResponse(manifest));

    renderWithRouter();

    await waitFor(() => expect(screen.getByText('John Doe')).toBeInTheDocument());
    expect(screen.getByText(/Primary issue:/i)).toHaveTextContent('wrong account');
    expect(screen.getByText('Balance owed')).toBeInTheDocument();
  });

  it('filters accounts using the search box', async () => {
    const manifest = {
      counts: { packs: 2, responses: 0 },
      packs: [
        {
          account_id: 'acct-1',
          holder_name: 'First Bank',
          primary_issue: 'wrong_account',
          display: samplePack.display,
          path: 'frontend/review/packs/acct-1.json',
        },
        {
          account_id: 'acct-2',
          holder_name: 'Second Bank',
          primary_issue: 'identity_theft',
          display: samplePack.display,
          path: 'frontend/review/packs/acct-2.json',
        },
      ],
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce(createFetchResponse(buildFrontendIndex(manifest)))
      .mockResolvedValueOnce(createFetchResponse(manifest));

    renderWithRouter();

    await waitFor(() => expect(screen.getByText('First Bank')).toBeInTheDocument());
    expect(screen.getByText('Second Bank')).toBeInTheDocument();

    const input = screen.getByLabelText('Search accounts');

    fireEvent.change(input, { target: { value: 'identity' } });

    await waitFor(() => expect(screen.queryByText('First Bank')).not.toBeInTheDocument());
    expect(screen.getByText('Second Bank')).toBeInTheDocument();

    fireEvent.change(input, { target: { value: 'zzz' } });

    await waitFor(() =>
      expect(screen.getByText('No accounts match your search.')).toBeInTheDocument()
    );
  });

  it('loads account details and submits answers', async () => {
    const manifest = {
      counts: { packs: 1, responses: 0 },
      packs: [
        {
          account_id: 'acct-1',
          holder_name: 'John Doe',
          primary_issue: 'wrong_account',
          display: samplePack.display,
          path: 'frontend/review/packs/acct-1.json',
        },
      ],
    };

    const detailPack = {
      ...samplePack,
      account_id: 'acct-1',
      questions: samplePack.display?.questions,
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce(createFetchResponse(buildFrontendIndex(manifest)))
      .mockResolvedValueOnce(createFetchResponse(manifest))
      .mockResolvedValueOnce(createFetchResponse(detailPack))
      .mockResolvedValueOnce(createFetchResponse({ ok: true }));

    renderWithRouter();

    await waitFor(() => expect(screen.getByText('John Doe')).toBeInTheDocument());

    fireEvent.click(screen.getByText('John Doe'));

    await waitFor(() => expect(screen.getByLabelText('Explain')).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText('Explain'), {
      target: { value: 'Some explanation' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() =>
      expect((global.fetch as jest.Mock).mock.calls[3][0]).toContain(
        '/api/runs/S123/frontend/review/response/acct-1'
      )
    );

    expect(await screen.findByText('Explanation saved successfully.')).toBeInTheDocument();
    expect(screen.getByText('Answered')).toBeInTheDocument();
  });

  it('shows an error message when fetching the manifest fails', async () => {
    const manifest = {
      counts: { packs: 0, responses: 0 },
    };

    (global.fetch as jest.Mock)
      .mockResolvedValueOnce(createFetchResponse(buildFrontendIndex(manifest)))
      .mockResolvedValueOnce(
        createFetchResponse({ message: 'boom' }, { ok: false, status: 500, statusText: 'Server error' })
      );

    renderWithRouter();

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByText(/Unable to load accounts/i)).toBeInTheDocument();
    expect(screen.getByText(/boom/i)).toBeInTheDocument();
  });
});
