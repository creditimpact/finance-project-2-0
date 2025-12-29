import { fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import AccountQuestions, { type AccountQuestionAnswers } from '../components/AccountQuestions';

describe('AccountQuestions', () => {
  it('renders initial explanation when provided', () => {
    const initial: AccountQuestionAnswers = {
      explanation: 'Initial note'
    };

    render(<AccountQuestions initialAnswers={initial} />);

    expect(screen.getByLabelText(/Explain/i)).toHaveValue('Initial note');
  });

  it('bubbles explanation changes to parent', () => {
    const handleChange = jest.fn();
    render(<AccountQuestions onChange={handleChange} />);

    fireEvent.change(screen.getByLabelText(/Explain/i), {
      target: { value: 'Some explanation' }
    });

    expect(handleChange).toHaveBeenLastCalledWith({ explanation: 'Some explanation' });
  });

  it('limits the explanation to 1500 characters', () => {
    const handleChange = jest.fn();
    render(<AccountQuestions onChange={handleChange} />);

    const longText = 'a'.repeat(1600);
    const textarea = screen.getByLabelText(/Explain/i) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: longText } });

    expect(textarea.value).toHaveLength(1500);
    expect(handleChange).toHaveBeenLastCalledWith({ explanation: 'a'.repeat(1500) });
  });
});
