import React from 'react';
import { tierColors } from '../utils/tierColors';

export default function DecisionBadge({ decisionSource, tier }) {
  const colors = tierColors[tier] || tierColors.none;
  const label = decisionSource === 'ai'
    ? 'AI decision'
    : decisionSource === 'rules'
    ? 'Rule-based'
    : 'Unknown';
  const classes = `inline-flex items-center text-xs font-medium rounded ${colors.bg} ${colors.text} border-l-4 ${colors.border} pl-2 pr-2 py-1`;
  return <span className={classes}>{label}</span>;
}
