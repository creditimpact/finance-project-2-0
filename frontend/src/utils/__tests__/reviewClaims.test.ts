import {
  prepareAnswersPayload,
  normalizeExistingAnswers,
  getMissingRequiredDocs,
  resolveClaimsPayload,
  sanitizeClaimSchema,
} from '../reviewClaims';
import type { AccountQuestionAnswers } from '../../components/AccountQuestions';
import type { ClaimSchema } from '../../types/review';

function buildClaimDefinition(overrides: Partial<ClaimSchema>): ClaimSchema {
  return sanitizeClaimSchema({
    key: 'test-claim',
    title: 'Test Claim',
    requires: [],
    ...overrides,
  });
}

describe('prepareAnswersPayload', () => {
  it('trims explanations and emits selected claims + attachments when requested', () => {
    const answers: AccountQuestionAnswers = {
      explanation: '  This needs attention.  ',
      selectedClaims: ['paid_in_full', 'wrong_dofd', 'paid_in_full' as unknown as string],
      attachments: {
        proof_of_payment: [' runs/documents/pay.pdf ', '', 'runs/documents/pay.pdf'],
        paid_in_full_letter: 'runs/documents/letter.pdf',
        billing_statement: ['runs/docs/billing1.pdf', 'runs/docs/billing2.pdf'],
      },
    };

    expect(prepareAnswersPayload(answers, { includeClaims: true })).toEqual({
      answers: {
        explanation: 'This needs attention.',
        selected_claims: ['paid_in_full', 'wrong_dofd'],
        attachments: {
          proof_of_payment: 'runs/documents/pay.pdf',
          paid_in_full_letter: 'runs/documents/letter.pdf',
          billing_statement: ['runs/docs/billing1.pdf', 'runs/docs/billing2.pdf'],
        },
      },
    });
  });

  it('omits claim metadata when includeClaims is disabled', () => {
    const answers: AccountQuestionAnswers = {
      explanation: '  explain  ',
      selectedClaims: ['paid_in_full'],
      attachments: {
        proof_of_payment: ['runs/documents/pay.pdf'],
      },
    };

    expect(prepareAnswersPayload(answers)).toEqual({
      answers: { explanation: 'explain' },
    });
  });
});

describe('normalizeExistingAnswers', () => {
  it('merges legacy claim metadata into the new selectedClaims + attachments shape', () => {
    const normalized = normalizeExistingAnswers({
      explanation: 'root wins',
      answers: {
        explanation: 'ignored',
        selected_claims: ['paid_in_full', ' paid_in_full ', 'not_mine'],
        attachments: {
          proof_of_payment: ' runs/documents/root-pay.pdf ',
        },
        claimDocuments: {
          paid_in_full: {
            paid_in_full_letter: ['runs/documents/letter.pdf'],
          },
        },
        evidence: [
          {
            docs: [
              { doc_key: 'proof_of_payment', doc_ids: ['runs/documents/evidence-pay.pdf'] },
              { doc_key: 'billing_statement', doc_ids: ['runs/documents/evidence-bill.pdf'] },
            ],
          },
        ],
      },
      claimDocuments: {
        wrong_dofd: {
          billing_statement: [' runs/documents/claimdoc-bill.pdf '],
        },
      },
    });

    expect(normalized).toEqual({
      explanation: 'root wins',
      selectedClaims: ['paid_in_full', 'not_mine'],
      attachments: {
        proof_of_payment: [
          'runs/documents/evidence-pay.pdf',
          'runs/documents/root-pay.pdf',
        ],
        paid_in_full_letter: ['runs/documents/letter.pdf'],
        billing_statement: [
          'runs/documents/evidence-bill.pdf',
          'runs/documents/claimdoc-bill.pdf',
        ],
      },
    });
  });
});

describe('getMissingRequiredDocs', () => {
  it('returns missing doc keys grouped by claim', () => {
    const claimDefinitions = new Map<string, ClaimSchema>([
      ['paid_in_full', buildClaimDefinition({ key: 'paid_in_full', requires: ['proof_of_payment'] })],
      [
        'wrong_dofd',
        buildClaimDefinition({ key: 'wrong_dofd', requires: ['billing_statement', 'bank_statement'] }),
      ],
    ]);

    const missing = getMissingRequiredDocs(
      ['paid_in_full', 'wrong_dofd'],
      {
        proof_of_payment: ['runs/documents/pay.pdf'],
        billing_statement: [],
      },
      claimDefinitions
    );

    expect(missing).toEqual({
      wrong_dofd: ['billing_statement', 'bank_statement'],
    });
  });
});

describe('resolveClaimsPayload', () => {
  it('falls back to the generic schema when items are missing', () => {
    const payload = resolveClaimsPayload({
      autoAttachBase: ['gov_id'],
      items: [],
    });

    expect(payload.autoAttachBase).toEqual(['gov_id']);
    expect(payload.items.length).toBeGreaterThan(0);
    expect(payload.items.map((entry) => entry.key)).toContain('explanation_only');
  });

  it('sanitizes malformed claim definitions', () => {
    const payload = resolveClaimsPayload({
      autoAttachBase: ['gov_id', 'proof_of_address'],
      items: [
        {
          key: 'custom_claim',
          title: 'Custom Claim',
          requires: ['proof_of_payment', ''],
          optional: ['billing_statement', 42 as unknown as never],
          autoAttach: ['gov_id', null as unknown as never],
          minUploads: -1,
          description: 15 as unknown as string,
        } as ClaimSchema,
      ],
    });

    expect(payload).toEqual({
      autoAttachBase: ['gov_id', 'proof_of_address'],
      items: [
        {
          key: 'custom_claim',
          title: 'Custom Claim',
          requires: ['proof_of_payment'],
          optional: ['billing_statement'],
          autoAttach: ['gov_id'],
        },
      ],
    });
  });
});
