import { render, screen, fireEvent } from '@testing-library/react';
import AccountCard, { type AccountPack } from '../components/AccountCard';
import samplePack from '../__fixtures__/sampleAccountPack.json';

describe('AccountCard', () => {
  const ORIGINAL_FLAGS = {
    showDetails: process.env.VITE_SHOW_BUREAU_DETAILS,
    hideConsensus: process.env.VITE_HIDE_CONSENSUS,
    preferLongestMask: process.env.VITE_ACCOUNT_NUMBER_PREFER_LONGEST_MASK
  };

  afterEach(() => {
    process.env.VITE_SHOW_BUREAU_DETAILS = ORIGINAL_FLAGS.showDetails;
    process.env.VITE_HIDE_CONSENSUS = ORIGINAL_FLAGS.hideConsensus;
    process.env.VITE_ACCOUNT_NUMBER_PREFER_LONGEST_MASK = ORIGINAL_FLAGS.preferLongestMask;
  });

  it('renders the summary view and bureau grid', () => {
    render(<AccountCard pack={samplePack} />);

    expect(screen.getByRole('heading', { name: 'John Doe' })).toBeInTheDocument();

    const [accountTypeLabel] = screen.getAllByText('Account type');
    const accountTypeCell = accountTypeLabel.closest('div');
    expect(accountTypeCell).toHaveTextContent('Credit Card');
    expect(accountTypeCell).toHaveTextContent('2 of 3');

    const [statusLabel] = screen.getAllByText('Status');
    const statusCell = statusLabel.closest('div');
    expect(statusCell).toHaveTextContent('Closed');
    expect(statusCell).toHaveTextContent('2 of 3');

    expect(screen.getByText('2023-01-02')).toBeInTheDocument();

    const toggle = screen.getByRole('button', { name: /details/i });
    fireEvent.click(toggle);

    expect(screen.queryByText('2023-01-02')).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.getByText('2023-01-02')).toBeInTheDocument();
  });

  it('shows placeholders for unanswered questions', () => {
    render(<AccountCard pack={samplePack} />);

    expect(screen.getAllByText('No response yet')).toHaveLength(4);
  });

  it('ignores consensus fields when computing summaries', () => {
    process.env.VITE_HIDE_CONSENSUS = '1';

    const pack: AccountPack = {
      holder_name: 'Jane Doe',
      display: {
        account_type: {
          per_bureau: {
            transunion: 'Auto',
            experian: 'Auto',
            equifax: 'Boat Loan'
          },
          consensus: 'Boat Loan'
        }
      }
    };

    render(<AccountCard pack={pack} />);

    const [accountTypeLabel] = screen.getAllByText('Account type');
    const accountTypeCell = accountTypeLabel.closest('div');

    expect(accountTypeCell).toHaveTextContent('Auto');
    expect(accountTypeCell).not.toHaveTextContent('Boat Loan');
  });

  it('defaults the bureau details to expanded when the feature flag is enabled', () => {
    process.env.VITE_SHOW_BUREAU_DETAILS = '1';

    render(<AccountCard pack={samplePack} />);

    const toggle = screen.getByRole('button', { name: /details/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
  });

  it('defaults the bureau details to collapsed when the feature flag is disabled', () => {
    process.env.VITE_SHOW_BUREAU_DETAILS = '0';

    render(<AccountCard pack={samplePack} />);

    const toggle = screen.getByRole('button', { name: /details/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  it('falls back to consensus values when allowed', () => {
    process.env.VITE_HIDE_CONSENSUS = '0';

    const pack: AccountPack = {
      holder_name: 'Jane Doe',
      display: {
        account_type: {
          consensus: 'Personal Loan'
        }
      }
    };

    render(<AccountCard pack={pack} />);

    const [accountTypeLabel] = screen.getAllByText('Account type');
    const accountTypeCell = accountTypeLabel.closest('div');

    expect(accountTypeCell).toHaveTextContent('Personal Loan');
  });

  it('prefers the longest masked account number when the flag is enabled', () => {
    process.env.VITE_ACCOUNT_NUMBER_PREFER_LONGEST_MASK = '1';

    const pack: AccountPack = {
      holder_name: 'Masked',
      display: {
        account_number: {
          per_bureau: {
            transunion: '***12',
            experian: '********0012'
          }
        }
      }
    };

    render(<AccountCard pack={pack} />);

    expect(screen.getByText(/Account number:/i)).toHaveTextContent('********0012');
  });
});
