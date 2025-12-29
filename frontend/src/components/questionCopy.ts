export type AccountQuestionKey = 'explanation';

export type QuestionCopy = {
  title: string;
  helper: string;
};

export const QUESTION_COPY: Record<AccountQuestionKey, QuestionCopy> = {
  explanation: {
    title: 'Explain',
    helper: 'Share a quick note that helps us understand this account.'
  }
};
