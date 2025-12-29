import React from 'react';

function truncate(str) {
  return str.length > 48 ? str.slice(0, 45) + '...' : str;
}

export default function ReasonChips({ reasons = [], max = 4 }) {
  if (!reasons || reasons.length === 0) return null;
  const chips = reasons.slice(0, max).map((r, i) => (
    <span key={i} className="chip">
      {truncate(r)}
    </span>
  ));
  const extra = reasons.length - max;
  if (extra > 0) {
    chips.push(
      <span key="more" className="chip" title={reasons.slice(max).join(', ')}>
        +{extra}
      </span>
    );
  }
  return <div className="reason-chips">{chips}</div>;
}
