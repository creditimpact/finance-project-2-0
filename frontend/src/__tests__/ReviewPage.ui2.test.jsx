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

beforeEach(() => {
  process.env.REACT_APP_UI_DECISION_BADGES = 'true';
  process.env.REACT_APP_UI_MAX_REASON_CHIPS = '4';
  process.env.REACT_APP_UI_CONFIDENCE_DECIMALS = '2';
  process.env.REACT_APP_UI_SHOW_FIELDS_USED = 'false';
  jest.clearAllMocks();
});

test('AI account shows badge, tier colors, confidence tooltip and chips', async () => {
  const acc = {
    account_id: 'a1',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
    problem_reasons: ['late_payment'],
    primary_issue: 'collection',
    issue_types: ['collection'],
    decision_meta: {
      decision_source: 'ai',
      tier: 'Tier1',
      confidence: 0.82,
      fields_used: ['payment_status'],
    },
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const badge = await screen.findByText('AI decision');
  expect(badge).toHaveClass('bg-red-100');
  expect(screen.getByTitle('AI confidence: 0.82')).toBeInTheDocument();
  expect(screen.getByText('late_payment')).toHaveClass('chip');
});

test('rules account shows rule-based badge and tooltip', async () => {
  const acc = {
    account_id: 'a2',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
    problem_reasons: ['over_limit'],
    primary_issue: 'collection',
    issue_types: ['collection'],
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  expect(await screen.findByText('Rule-based')).toBeInTheDocument();
  expect(screen.getByTitle('No AI used (rules-only)')).toBeInTheDocument();
});

test('chips truncation with +N more', async () => {
  const acc = {
    account_id: 'a3',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
    problem_reasons: ['r1','r2','r3','r4','r5','r6','r7'],
    primary_issue: 'collection',
    issue_types: ['collection'],
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  const { container } = render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const chips = container.querySelectorAll('.chip');
  expect(chips.length).toBe(5); // 4 + "more"
  expect(screen.getByText('+3')).toBeInTheDocument();
});

test('flag off hides badges and tooltip', async () => {
  process.env.REACT_APP_UI_DECISION_BADGES = 'false';
  const acc = {
    account_id: 'a4',
    bureau: 'Equifax',
    decision_source: 'ai',
    tier: 'Tier2',
    confidence: 0.5,
    problem_reasons: ['reason'],
    primary_issue: 'collection',
    issue_types: ['collection'],
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  expect(screen.queryByText('AI decision')).not.toBeInTheDocument();
  expect(screen.queryByTitle(/AI confidence/i)).not.toBeInTheDocument();
  expect(screen.getByText('reason')).toBeInTheDocument();
});

test('PII strings rendered plainly', async () => {
  const acc = {
    account_id: 'a5',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
    problem_reasons: ['user@example.com','555-123-4567','123-45-6789'],
    primary_issue: 'collection',
    issue_types: ['collection'],
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  expect(await screen.findByText('user@example.com')).toBeInTheDocument();
  expect(screen.getByText('555-123-4567')).toBeInTheDocument();
  expect(screen.getByText('123-45-6789')).toBeInTheDocument();
});

test('telemetry emitted on expand', async () => {
  const acc = {
    account_id: 'a6',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
    problem_reasons: [],
    primary_issue: 'collection',
    issue_types: ['collection'],
  };
  const uploadData = { ...baseUploadData, accounts: { problem_accounts: [acc] } };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const toggle = await screen.findByLabelText(/Show how the system understood/i);
  fireEvent.click(toggle);
  expect(emitUiEvent).toHaveBeenCalledWith('ui_review_expand', {
    session_id: 'sess1',
    account_id: 'a6',
    bureau: 'Equifax',
    decision_source: 'rules',
    tier: 'none',
  });
});
