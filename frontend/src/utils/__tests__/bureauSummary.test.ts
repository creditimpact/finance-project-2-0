import { summarizeField } from '../bureauSummary';

describe('summarizeField', () => {
  it('returns none when all bureau values are missing', () => {
    const result = summarizeField({
      transunion: '--',
      experian: '',
      equifax: undefined,
    });

    expect(result).toEqual({
      summary: '—',
      agreement: 'none',
      values: {},
    });
  });

  it('returns all agreement when every bureau matches', () => {
    const result = summarizeField({
      transunion: 'Closed',
      experian: 'Closed',
      equifax: 'Closed',
    });

    expect(result).toEqual({
      summary: 'Closed',
      agreement: 'all',
      values: {
        transunion: 'Closed',
        experian: 'Closed',
        equifax: 'Closed',
      },
    });
  });

  it('returns majority when two bureaus agree', () => {
    const result = summarizeField({
      transunion: 'Open',
      experian: 'Open',
      equifax: 'Closed',
    });

    expect(result).toEqual({
      summary: 'Open',
      agreement: 'majority',
      values: {
        transunion: 'Open',
        experian: 'Open',
        equifax: 'Closed',
      },
    });
  });

  it('picks the longest mask for account numbers when values differ', () => {
    const result = summarizeField(
      {
        transunion: '******7890',
        experian: '***7890',
        equifax: '*7890',
      },
      { kind: 'account_number' }
    );

    expect(result.summary).toBe('******7890');
    expect(result.agreement).toBe('mixed');
    expect(result.values).toEqual({
      transunion: '******7890',
      experian: '***7890',
      equifax: '*7890',
    });
  });
});
