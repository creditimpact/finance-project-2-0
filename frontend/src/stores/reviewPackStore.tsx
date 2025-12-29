import * as React from 'react';
import type { ReviewAccountPack } from '../components/ReviewCard';

interface ReviewPackStoreValue {
  packs: Record<string, ReviewAccountPack>;
  getPack: (accountId: string) => ReviewAccountPack | undefined;
  setPack: (accountId: string, pack: ReviewAccountPack) => void;
  clear: () => void;
}

const ReviewPackStoreContext = React.createContext<ReviewPackStoreValue | null>(null);

export function ReviewPackStoreProvider({ children }: { children: React.ReactNode }) {
  const [packs, setPacks] = React.useState<Record<string, ReviewAccountPack>>({});

  const setPack = React.useCallback((accountId: string, pack: ReviewAccountPack) => {
    setPacks((previous) => {
      const existing = previous[accountId];
      if (existing === pack) {
        return previous;
      }
      return {
        ...previous,
        [accountId]: pack,
      };
    });
  }, []);

  const clear = React.useCallback(() => {
    setPacks({});
  }, []);

  const getPack = React.useCallback(
    (accountId: string) => {
      return packs[accountId];
    },
    [packs]
  );

  const value = React.useMemo<ReviewPackStoreValue>(
    () => ({ packs, getPack, setPack, clear }),
    [packs, getPack, setPack, clear]
  );

  return <ReviewPackStoreContext.Provider value={value}>{children}</ReviewPackStoreContext.Provider>;
}

export function useReviewPackStore(): ReviewPackStoreValue {
  const context = React.useContext(ReviewPackStoreContext);
  if (!context) {
    throw new Error('useReviewPackStore must be used within a ReviewPackStoreProvider');
  }
  return context;
}
