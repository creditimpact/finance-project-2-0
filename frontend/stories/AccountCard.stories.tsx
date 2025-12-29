import type { Meta, StoryObj } from '@storybook/react';
import AccountCard, { type AccountPack } from '../src/components/AccountCard';

type Story = StoryObj<typeof AccountCard>;

const meta: Meta<typeof AccountCard> = {
  title: 'Accounts/AccountCard',
  component: AccountCard,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

const baseQuestions: NonNullable<AccountPack['display']>['questions'] = {
  ownership: 'Yes, solely mine',
  recognize: 'Absolutely',
  explanation: 'Current with lender',
  identity_theft: 'No concerns',
};

const allAgreePack: AccountPack = {
  holder_name: 'Alex Johnson',
  primary_issue: 'wrong_account',
  display: {
    account_number: {
      per_bureau: {
        transunion: '****7890',
        experian: '****7890',
        equifax: '****7890',
      },
    },
    account_type: {
      per_bureau: {
        transunion: 'Auto Loan',
        experian: 'Auto Loan',
        equifax: 'Auto Loan',
      },
    },
    status: {
      per_bureau: {
        transunion: 'Open',
        experian: 'Open',
        equifax: 'Open',
      },
    },
    balance_owed: {
      per_bureau: {
        transunion: '$12,450',
        experian: '$12,450',
        equifax: '$12,450',
      },
    },
    date_opened: {
      transunion: '2020-01-15',
      experian: '2020-01-15',
      equifax: '2020-01-15',
    },
    closed_date: {
      transunion: '2023-05-01',
      experian: '2023-05-01',
      equifax: '2023-05-01',
    },
    questions: baseQuestions,
  },
};

export const AllAgreeOnEverything: Story = {
  args: {
    pack: allAgreePack,
  },
};

const mixedAccountNumberPack: AccountPack = {
  holder_name: 'Jordan Smith',
  primary_issue: 'balance_wrong',
  display: {
    account_number: {
      per_bureau: {
        transunion: '******2222',
        experian: '***2222',
        equifax: '**22',
      },
    },
    account_type: {
      per_bureau: {
        transunion: 'Credit Card',
        experian: 'Credit Card',
        equifax: 'Credit Card',
      },
    },
    status: {
      per_bureau: {
        transunion: 'Open',
        experian: 'Open',
        equifax: 'Open',
      },
    },
    balance_owed: {
      per_bureau: {
        transunion: '$3,200',
        experian: '$3,200',
        equifax: '$3,200',
      },
    },
    date_opened: {
      transunion: '2021-05-10',
      experian: '2021-05-10',
      equifax: '2021-05-10',
    },
    closed_date: {
      transunion: null,
      experian: null,
      equifax: null,
    },
    questions: {
      ...baseQuestions,
      explanation: 'Different masks reported',
    },
  },
};

export const MixedAccountNumberOnly: Story = {
  args: {
    pack: mixedAccountNumberPack,
  },
};

const mixedDatesPack: AccountPack = {
  holder_name: 'Morgan Lee',
  primary_issue: 'late_payment',
  display: {
    account_number: {
      per_bureau: {
        transunion: '****1111',
        experian: '****1111',
        equifax: '****1111',
      },
    },
    account_type: {
      per_bureau: {
        transunion: 'Mortgage',
        experian: 'Mortgage',
        equifax: 'Mortgage',
      },
    },
    status: {
      per_bureau: {
        transunion: 'Closed',
        experian: 'Closed',
        equifax: 'Closed',
      },
    },
    balance_owed: {
      per_bureau: {
        transunion: '$0',
        experian: '$0',
        equifax: '$0',
      },
    },
    date_opened: {
      transunion: '2018-02-01',
      experian: '2018-03-15',
      equifax: '2018-02-28',
    },
    closed_date: {
      transunion: '2023-06-30',
      experian: '2023-07-02',
      equifax: '2023-06-25',
    },
    questions: {
      ...baseQuestions,
      explanation: 'Dates vary by bureau',
    },
  },
};

export const MixedDatesOnly: Story = {
  args: {
    pack: mixedDatesPack,
  },
};

const manyMissingValuesPack: AccountPack = {
  holder_name: 'Taylor Green',
  primary_issue: 'not_mine',
  display: {
    account_number: {
      per_bureau: {
        transunion: '--',
        experian: '',
        equifax: null,
      },
    },
    account_type: {
      per_bureau: {
        transunion: null,
        experian: undefined,
        equifax: '',
      },
    },
    status: {
      per_bureau: {
        transunion: '--',
        experian: null,
        equifax: undefined,
      },
    },
    balance_owed: {
      per_bureau: {
        transunion: '$1,050',
        experian: null,
        equifax: undefined,
      },
    },
    date_opened: {
      transunion: undefined,
      experian: null,
      equifax: '',
    },
    closed_date: {
      transunion: null,
      experian: undefined,
      equifax: '',
    },
    questions: {
      ownership: null,
      recognize: null,
      explanation: '',
      identity_theft: null,
    },
  },
};

export const ManyMissingValues: Story = {
  args: {
    pack: manyMissingValuesPack,
  },
};

const identityTheftCheckedPack: AccountPack = {
  holder_name: 'Riley Chen',
  primary_issue: 'identity_theft',
  display: {
    account_number: {
      per_bureau: {
        transunion: '***9087',
        experian: '***9087',
        equifax: '***9087',
      },
    },
    account_type: {
      per_bureau: {
        transunion: 'Personal Loan',
        experian: 'Personal Loan',
        equifax: 'Personal Loan',
      },
    },
    status: {
      per_bureau: {
        transunion: 'In collections',
        experian: 'In collections',
        equifax: 'In collections',
      },
    },
    balance_owed: {
      per_bureau: {
        transunion: '$6,775',
        experian: '$6,775',
        equifax: '$6,775',
      },
    },
    date_opened: {
      transunion: '2022-11-04',
      experian: '2022-11-04',
      equifax: '2022-11-04',
    },
    closed_date: {
      transunion: null,
      experian: null,
      equifax: null,
    },
    questions: {
      ownership: 'No, not mine',
      recognize: 'Never seen this account',
      explanation: 'Disputed after fraud alert',
      identity_theft: 'Yes, identity theft',
    },
  },
};

export const IdentityTheftPrechecked: Story = {
  args: {
    pack: identityTheftCheckedPack,
  },
};
