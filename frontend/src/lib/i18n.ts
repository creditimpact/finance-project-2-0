import * as React from 'react';

export interface TranslateDescriptor {
  id: string;
  defaultMessage: string;
  values?: Record<string, string | number | undefined>;
}

export type TranslateFn = (descriptor: TranslateDescriptor) => string;

function formatMessage(defaultMessage: string, values?: TranslateDescriptor['values']): string {
  if (!values) {
    return defaultMessage;
  }

  return defaultMessage.replace(/\{(\w+)\}/g, (match, key) => {
    const value = values[key];
    if (value == null) {
      return match;
    }

    return String(value);
  });
}

export function useI18n(): TranslateFn {
  return React.useCallback((descriptor: TranslateDescriptor) => {
    return formatMessage(descriptor.defaultMessage, descriptor.values);
  }, []);
}
