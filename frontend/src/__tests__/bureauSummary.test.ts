import { summarizeField, type Agreement, type BureauTriple } from '../utils/bureauSummary';

describe('summarizeField', () => {
  const summarize = (values: BureauTriple, options?: Parameters<typeof summarizeField>[1]) =>
    summarizeField(values, options);

  it('returns all agreement when all bureaus match', () => {
    const triple: BureauTriple = {
      transunion: 'Open',
      experian: 'Open',
      equifax: 'Open',
    };

    const result = summarize(triple);

    expect(result).toEqual({
      summary: 'Open',
      agreement: 'all' satisfies Agreement,
      values: {
        transunion: 'Open',
        experian: 'Open',
        equifax: 'Open',
      },
    });
  });

  it('prefers the majority value when two bureaus agree', () => {
    const triple: BureauTriple = {
      transunion: 'Closed',
      experian: 'Closed',
      equifax: 'Open',
    };

    const result = summarize(triple);

    expect(result).toEqual({
      summary: 'Closed',
      agreement: 'majority' satisfies Agreement,
      values: {
        transunion: 'Closed',
        experian: 'Closed',
        equifax: 'Open',
      },
    });
  });

  it('returns mixed when all bureaus disagree and selects first in bureau order for generic fields', () => {
    const triple: BureauTriple = {
      transunion: 'Installment',
      experian: 'Mortgage',
      equifax: 'Auto',
    };

    const result = summarize(triple);

    expect(result).toEqual({
      summary: 'Installment',
      agreement: 'mixed' satisfies Agreement,
      values: {
        transunion: 'Installment',
        experian: 'Mortgage',
        equifax: 'Auto',
      },
    });
  });

  it('returns none when every value is missing', () => {
    const triple: BureauTriple = {
      transunion: '--',
      experian: '   ',
      equifax: '',
    };

    const result = summarize(triple);

    expect(result).toEqual({
      summary: '—',
      agreement: 'none' satisfies Agreement,
      values: {},
    });
  });

  it('treats shared values with a missing bureau as a majority agreement', () => {
    const triple: BureauTriple = {
      transunion: 'Closed',
      experian: 'Closed',
      equifax: '--',
    };

    const result = summarize(triple);

    expect(result).toEqual({
      summary: 'Closed',
      agreement: 'majority' satisfies Agreement,
      values: {
        transunion: 'Closed',
        experian: 'Closed',
      },
    });
  });

  it('selects the most informative masked value for account numbers when mixed', () => {
    const triple: BureauTriple = {
      transunion: '****',
      experian: '440066**********',
      equifax: '44******',
    };

    const result = summarize(triple, { kind: 'account_number' });

    expect(result).toEqual({
      summary: '440066**********',
      agreement: 'mixed' satisfies Agreement,
      values: {
        transunion: '****',
        experian: '440066**********',
        equifax: '44******',
      },
    });
  });

  it('breaks ties for equally informative account numbers using bureau order', () => {
    const triple: BureauTriple = {
      transunion: '***1111',
      experian: '***2222',
      equifax: '***3333',
    };

    const result = summarize(triple, { kind: 'account_number' });

    expect(result).toEqual({
      summary: '***1111',
      agreement: 'mixed' satisfies Agreement,
      values: {
        transunion: '***1111',
        experian: '***2222',
        equifax: '***3333',
      },
    });
  });
});
