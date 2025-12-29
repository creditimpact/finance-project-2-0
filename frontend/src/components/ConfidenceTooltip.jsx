import React from 'react';
import {
  ALL_VALIDATION_FIELD_SET,
  CONDITIONAL_FIELD_SET,
  formatValidationField,
} from '../utils/validationFields';

const CONDITIONAL_ICON = '⚠️';

export default function ConfidenceTooltip({ decisionSource, confidence, fieldsUsed, showFieldsUsed = false, decimals = 2 }) {
  let lines = [];
  if (decisionSource === 'ai') {
    const value = Number(confidence);
    if (!Number.isNaN(value)) {
      lines.push(`AI confidence: ${value.toFixed(decimals)}`);
    } else {
      lines.push('AI confidence: N/A');
    }
  } else {
    lines.push('No AI used (rules-only)');
  }
  if (showFieldsUsed && Array.isArray(fieldsUsed) && fieldsUsed.length > 0) {
    const filtered = fieldsUsed.filter((field) => ALL_VALIDATION_FIELD_SET.has(field));
    if (filtered.length > 0) {
      const formattedFields = filtered.map((field) => formatValidationField(field));
      lines.push(`Fields used: ${formattedFields.join(', ')}`);
      if (filtered.some((field) => CONDITIONAL_FIELD_SET.has(field))) {
        lines.push(`${CONDITIONAL_ICON} Investigates only with corroboration`);
      }
    }
  }
  return (
    <span className="info-icon" title={lines.join('\n')}>
      ℹ️
    </span>
  );
}
