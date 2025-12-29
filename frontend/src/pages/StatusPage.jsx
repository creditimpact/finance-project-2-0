import React, { useEffect, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { fetchRunFrontendReviewIndex } from '../api.ts';

export default function StatusPage() {
  const [phase, setPhase] = useState('waiting');
  const navigate = useNavigate();
  const { sid: sidFromParams } = useParams();
  const [sp] = useSearchParams();
  const sid = sidFromParams || sp.get('sid');

  useEffect(() => {
    if (!sid) return undefined;

    let isMounted = true;

    const checkReviewPacks = async () => {
      try {
        const idx = await fetchRunFrontendReviewIndex(sid);
        const count = idx?.packs_count ?? idx?.items?.length ?? 0;
        if (isMounted && count > 0) {
          setPhase('redirecting');
          navigate(`/runs/${encodeURIComponent(sid)}/review`, { replace: true });
        }
      } catch {
        // Ignore polling errors and retry on the next interval.
      }
    };

    checkReviewPacks();
    const t = setInterval(checkReviewPacks, 2000);
    return () => {
      isMounted = false;
      clearInterval(t);
    };
  }, [sid, navigate]);

  return (
    <div className="container">
      <h2>Processing</h2>
      <p>
        {phase === 'redirecting'
          ? 'Review packs are ready. Redirecting you to the review experience...'
          : 'Your letters are being generated and will be emailed to you shortly.'}
      </p>
    </div>
  );
}
