import React from 'react';

export default function DecisionDebugPanel({ meta, decision }) {
  const showPanel =
    (process.env.NODE_ENV !== 'production'
      ? process.env.REACT_APP_UI_DEV_DEBUG_PANEL !== 'false'
      : process.env.REACT_APP_UI_DEV_DEBUG_PANEL === 'true');

  if (!showPanel) {
    return null;
  }

  const showFields = process.env.REACT_APP_UI_SHOW_FIELDS_USED === 'true';
  const maxFields = Number(process.env.REACT_APP_UI_MAX_FIELDS_USED || 8);
  const fields = meta?.fields_used ? meta.fields_used.slice(0, maxFields) : [];
  const source = meta?.decision_source ?? meta?.source;
  const tier = meta?.tier;
  const confidence = meta?.confidence;

  // determine which decision artifact to show
  const stageADecision =
    decision?.stageA_decision ||
    decision?.stage_a_decision ||
    decision?.decision_json ||
    decision?.decision ||
    decision;

  return (
    <details className="debug-panel">
      <summary>Debug</summary>
      {meta && (
        <div className="decision-meta">
          <div>source: {source}</div>
          <div>tier: {tier}</div>
          <div>confidence: {confidence}</div>
        </div>
      )}
      {showFields && fields.length > 0 && (
        <div className="fields-used">
          <div>fields_used:</div>
          <ul>
            {fields.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      )}
      {stageADecision && (
        <pre className="debug-json">
          {JSON.stringify(stageADecision, null, 2)}
        </pre>
      )}
    </details>
  );
}

