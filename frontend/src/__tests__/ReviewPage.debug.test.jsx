import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ReviewPage from '../pages/ReviewPage';
import { emitUiEvent } from '../telemetry/uiTelemetry';

jest.mock('../telemetry/uiTelemetry', () => ({
  emitUiEvent: jest.fn(),
}));
jest.mock('../api.ts', () => ({
  API_BASE_CONFIGURED: true,
  API_BASE_URL: 'http://127.0.0.1:5000',
  submitExplanations: jest.fn(),
  getSummaries: jest.fn().mockResolvedValue({ summaries: {} }),
}));

const baseUploadData = {
  session_id: 'sess1',
  filename: 'file.pdf',
  email: 'test@example.com',
};

const account = {
  account_id: 'acc1',
  bureau: 'Equifax',
  name: 'Account 1',
  normalized_name: 'account 1',
  primary_issue: 'late_payment',
  issue_types: ['late_payment'],
  decision_meta: {
    decision_source: 'ai',
    tier: 'Tier1',
    confidence: 0.91,
    fields_used: ['f1', 'f2', 'f3'],
  },
  stage_a_decision: { result: 'ok' },
  problem_reasons: ['late_payment'],
};

beforeEach(() => {
  process.env.REACT_APP_UI_DEV_DEBUG_PANEL = 'true';
  process.env.REACT_APP_UI_SHOW_FIELDS_USED = 'true';
  process.env.REACT_APP_UI_MAX_FIELDS_USED = '2';
  jest.clearAllMocks();
});

test('debug panel shows meta and capped fields', async () => {
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [account] } };
  const { container } = render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const toggle = await screen.findByText('Debug');
  fireEvent.click(toggle);
  expect(screen.getByText(/source:/)).toHaveTextContent('ai');
  expect(screen.getByText(/tier:/)).toHaveTextContent('Tier1');
  expect(screen.getByText(/confidence:/)).toHaveTextContent('0.91');
  expect(screen.getByText('f1')).toBeInTheDocument();
  expect(screen.getByText('f2')).toBeInTheDocument();
  expect(screen.queryByText('f3')).not.toBeInTheDocument();
  const panel = container.querySelector('.debug-panel');
  expect(panel).toMatchSnapshot();
  const text = panel.textContent;
  expect(text).not.toMatch(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/);
  expect(text).not.toMatch(/\b\d{3}-\d{2}-\d{4}\b/);
  expect(text).not.toMatch(/\b\d{3}-\d{3}-\d{4}\b/);
});

test('debug panel hidden when flag off', async () => {
  process.env.REACT_APP_UI_DEV_DEBUG_PANEL = 'false';
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [account] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  expect(screen.queryByText('Debug')).not.toBeInTheDocument();
});

test('telemetry fires on expand and collapse', async () => {
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [account] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const toggle = await screen.findByLabelText(/Show how the system understood/i);
  fireEvent.click(toggle);
  expect(emitUiEvent).toHaveBeenCalledWith('ui_review_expand', {
    session_id: 'sess1',
    account_id: 'acc1',
    bureau: 'Equifax',
    decision_source: 'ai',
    tier: 'Tier1',
  });
  fireEvent.click(toggle);
  expect(emitUiEvent).toHaveBeenCalledWith('ui_review_collapse', {
    session_id: 'sess1',
    account_id: 'acc1',
    bureau: 'Equifax',
  });
});

