import * as React from 'react';
import { Link, useParams } from 'react-router-dom';

export default function RunReviewCompletePage() {
  const { sid } = useParams();

  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0 });
    }
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-16 text-center sm:px-6 lg:px-8">
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
        <span className="text-2xl font-semibold">✓</span>
      </div>
      <div className="space-y-3">
        <h1 className="text-3xl font-semibold text-slate-900">Review complete</h1>
        <p className="text-base text-slate-600">
          Thank you for reviewing your accounts. We&apos;ll start processing your answers right away.
        </p>
      </div>
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
        {sid ? (
          <Link
            to={`/runs/${encodeURIComponent(sid)}/accounts`}
            className="inline-flex items-center justify-center rounded-md border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-500"
          >
            View accounts
          </Link>
        ) : null}
        <Link
          to="/"
          className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          Return home
        </Link>
      </div>
    </div>
  );
}
