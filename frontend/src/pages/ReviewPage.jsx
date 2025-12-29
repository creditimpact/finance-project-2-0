import { useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { submitExplanations, getSummaries } from '../api.ts';
import DecisionBadge from '../components/DecisionBadge';
import ReasonChips from '../components/ReasonChips';
import ConfidenceTooltip from '../components/ConfidenceTooltip';
import DecisionDebugPanel from '../components/DecisionDebugPanel';
import { emitUiEvent } from '../telemetry/uiTelemetry';

export default function ReviewPage() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const uploadData = state?.uploadData;
  const [explanations, setExplanations] = useState({});
  const [summaries, setSummaries] = useState({});
  const [showSummary, setShowSummary] = useState({});

  const UI_DECISION_BADGES = process.env.REACT_APP_UI_DECISION_BADGES !== 'false';
  const MAX_REASON_CHIPS = Number(process.env.REACT_APP_UI_MAX_REASON_CHIPS || 4);
  const CONF_DECIMALS = Number(process.env.REACT_APP_UI_CONFIDENCE_DECIMALS || 2);
  const SHOW_FIELDS_USED = process.env.REACT_APP_UI_SHOW_FIELDS_USED === 'true';
  const debugEvidence = (() => {
    try {
      return (
        process.env.VITE_DEBUG_EVIDENCE === '1' ||
        new Function('return import.meta.env?.VITE_DEBUG_EVIDENCE')() === '1'
      );
    } catch {
      return false;
    }
  })();

  useEffect(() => {
    if (uploadData?.session_id) {
      (async () => {
        try {
          const res = await getSummaries(uploadData.session_id);
          setSummaries(res?.summaries ?? {});
        } catch (err) {
          console.warn('failed to fetch summaries', err);
          setSummaries({});
        }
      })();
    }
  }, [uploadData?.session_id]);

  if (!uploadData) {
    return <p>No upload data available.</p>;
  }

  const accounts = uploadData.accounts?.problem_accounts ?? [];

  // Debug: log first card's props
  if (accounts[0]) {
    console.debug('review-card-props', {
      primary_issue: accounts[0].primary_issue,
      issue_types: accounts[0].issue_types,
      last4: accounts[0].account_number_last4,
      original_creditor: accounts[0].original_creditor,
    });
  }

  const dedupedAccounts = Array.from(
    accounts
      .reduce((map, acc) => {
        const identifier = acc.account_number_last4 ?? acc.account_fingerprint ?? '';
        const key = `${
          acc.normalized_name ?? acc.name?.toLowerCase() ?? ''
        }|${identifier}`;
        const existing = map.get(key);
        if (existing) {
          existing.late_payments = {
            ...(existing.late_payments ?? {}),
            ...(acc.late_payments ?? {}),
          };
          return map;
        }
        map.set(key, acc);
        return map;
      }, new Map())
      .values(),
  );

  const formatIssueType = (type) => {
    switch (type) {
      case 'late_payment':
        return 'Late Payment';
      case 'collection':
        return 'Collection';
      case 'charge_off':
        return 'Charge-Off';
      default:
        return type ? type.charAt(0).toUpperCase() + type.slice(1) : type;
    }
  };

  const handleChange = (key, value) => {
    setExplanations((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    try {
      await submitExplanations({
        session_id: uploadData.session_id,
        filename: uploadData.filename,
        email: uploadData.email,
        explanations,
      });
      navigate('/status');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="container">
      <h2>Explain Your Situation</h2>
      {dedupedAccounts.map((acc, idx) => {
        const issues = acc.issue_types ?? [];
        const primary = acc.primary_issue;
        const idLast4 = acc.account_number_last4 ?? null;
        const fingerprint = acc.account_fingerprint ?? null;
        const displayId = idLast4 ? `••••${idLast4}` : fingerprint ?? '';
        const secondaryIssues = issues.filter((t) => t !== primary);
        const source = acc.decision_meta?.decision_source ?? acc.decision_source;
        const tier = acc.decision_meta?.tier ?? acc.tier ?? 'none';
        const confidence = acc.decision_meta?.confidence ?? acc.confidence;
        const fields = acc.decision_meta?.fields_used;
        const handleExpandChange = (e) => {
          const checked = e.target.checked;
          setShowSummary((prev) => ({ ...prev, [acc.account_id]: checked }));
          if (checked) {
            emitUiEvent('ui_review_expand', {
              session_id: uploadData.session_id,
              account_id: acc.account_id,
              bureau: acc.bureau,
              decision_source: source,
              tier,
            });
          } else {
            emitUiEvent('ui_review_collapse', {
              session_id: uploadData.session_id,
              account_id: acc.account_id,
              bureau: acc.bureau,
            });
          }
        };
        return (
          <div key={idx} className="account-block">
            <p>
              <strong>{acc.name}</strong>
              {displayId && ` ${displayId}`}
              {acc.original_creditor && ` - ${acc.original_creditor}`}
            </p>
            {UI_DECISION_BADGES && (
              <div className="decision-row">
                <DecisionBadge decisionSource={source} tier={tier} />
                <ConfidenceTooltip
                  decisionSource={source}
                  confidence={confidence}
                  fieldsUsed={fields}
                  showFieldsUsed={SHOW_FIELDS_USED}
                  decimals={CONF_DECIMALS}
                />
              </div>
            )}
            <div className="issue-badges">
              <span className="badge">{primary ? formatIssueType(primary) : 'Unknown'}</span>
              {secondaryIssues.map((type, i) => (
                <span key={i} className="chip">
                  {formatIssueType(type)}
                </span>
              ))}
            </div>
            {acc.problem_reasons && acc.problem_reasons.length > 0 && (
              <ReasonChips reasons={acc.problem_reasons} max={MAX_REASON_CHIPS} />
            )}
            <textarea
              value={explanations[acc.name] || ''}
              onChange={(e) => handleChange(acc.name, e.target.value)}
              placeholder="Your explanation"
            />
            <small className="helper-text">
              We’ll use your explanation as context to better understand your case. It will
              not be copied word-for-word into your dispute letter.
            </small>
            <div className="summary-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={!!showSummary[acc.account_id]}
                  onChange={handleExpandChange}
                />
                Show how the system understood your explanation
              </label>
            </div>
            {showSummary[acc.account_id] && summaries[acc.account_id] && (
              <pre className="summary-box">
                {JSON.stringify(summaries[acc.account_id], null, 2)}
              </pre>
            )}
            {debugEvidence && (
              <details className="evidence-toggle">
                <summary>View evidence</summary>
                <pre className="summary-box">
                  {JSON.stringify(
                    {
                      account_trace: acc.account_trace ?? {},
                      bureau_details: acc.bureau_details ?? {},
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            )}
            <DecisionDebugPanel meta={acc.decision_meta} decision={acc} />
          </div>
        );
      })}
      <button onClick={handleSubmit}>Generate Letters</button>
    </div>
  );
}
