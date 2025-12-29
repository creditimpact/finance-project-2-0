import * as React from 'react';
import { QUESTION_COPY, type AccountQuestionKey } from './questionCopy';

export type AccountAttachments = Partial<Record<string, string[]>>;

export interface AccountQuestionAnswers extends Partial<Record<AccountQuestionKey, string>> {
  selectedClaims?: string[];
  attachments?: AccountAttachments;
}

export interface AccountQuestionsProps {
  onChange?: (answers: AccountQuestionAnswers) => void;
  initialAnswers?: AccountQuestionAnswers;
}

const MAX_EXPLANATION_LENGTH = 1500;

function normalizeAnswer(value: string): string | undefined {
  return value.trim() === '' ? undefined : value;
}

export function AccountQuestions({ onChange, initialAnswers }: AccountQuestionsProps) {
  const [explanation, setExplanation] = React.useState(() => initialAnswers?.explanation ?? '');

  React.useEffect(() => {
    setExplanation(initialAnswers?.explanation ?? '');
  }, [initialAnswers?.explanation]);

  const handleExplanationChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextValue = event.target.value.slice(0, MAX_EXPLANATION_LENGTH);
      setExplanation(nextValue);
      onChange?.({ explanation: normalizeAnswer(nextValue) });
    },
    [onChange]
  );

  const helperId = 'account-question-explanation-helper';
  const countId = 'account-question-explanation-count';
  const describedBy = `${helperId} ${countId}`;

  return (
    <div className="space-y-2">
      <label htmlFor="account-question-explanation" className="text-sm font-semibold text-slate-900">
        {QUESTION_COPY.explanation.title}
      </label>
      <p id={helperId} className="text-xs text-slate-600">
        {QUESTION_COPY.explanation.helper}
      </p>
      <textarea
        id="account-question-explanation"
        name="account-question-explanation"
        value={explanation}
        onChange={handleExplanationChange}
        maxLength={MAX_EXPLANATION_LENGTH}
        aria-describedby={describedBy}
        rows={4}
        className="block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500"
      />
      <p id={countId} className="text-xs text-slate-500 text-right">
        {explanation.length} / {MAX_EXPLANATION_LENGTH} characters
      </p>
    </div>
  );
}

export default AccountQuestions;
