import * as React from 'react';
import { cn } from '../lib/utils';

type ToastVariant = 'info' | 'success' | 'error';

type ToastOptions = {
  title?: string;
  description?: string;
  variant?: ToastVariant;
  duration?: number;
};

type ToastMessage = ToastOptions & { id: number };

type ToastContextValue = {
  showToast: (options: ToastOptions) => void;
};

const ToastContext = React.createContext<ToastContextValue | null>(null);

const DEFAULT_DURATION_MS = 6000;

function useStableTimeoutRemover() {
  const timersRef = React.useRef<Map<number, number>>(new Map());

  const clearTimer = React.useCallback((id: number) => {
    const handle = timersRef.current.get(id);
    if (handle != null) {
      window.clearTimeout(handle);
      timersRef.current.delete(id);
    }
  }, []);

  const registerTimer = React.useCallback((id: number, duration: number, onExpire: () => void) => {
    clearTimer(id);
    const handle = window.setTimeout(() => {
      timersRef.current.delete(id);
      onExpire();
    }, duration);
    timersRef.current.set(id, handle);
  }, [clearTimer]);

  React.useEffect(() => {
    return () => {
      timersRef.current.forEach((handle) => {
        window.clearTimeout(handle);
      });
      timersRef.current.clear();
    };
  }, []);

  return { registerTimer, clearTimer };
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastMessage[]>([]);
  const { registerTimer, clearTimer } = useStableTimeoutRemover();

  const removeToast = React.useCallback((id: number) => {
    setToasts((previous) => previous.filter((toast) => toast.id !== id));
    clearTimer(id);
  }, [clearTimer]);

  const showToast = React.useCallback(
    (options: ToastOptions) => {
      const id = window.crypto?.getRandomValues?.(new Uint32Array(1))?.[0] ?? Date.now();
      const toast: ToastMessage = { id, ...options };
      setToasts((previous) => [...previous, toast]);
      registerTimer(id, options.duration ?? DEFAULT_DURATION_MS, () => removeToast(id));
    },
    [registerTimer, removeToast]
  );

  const contextValue = React.useMemo<ToastContextValue>(() => ({ showToast }), [showToast]);

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex justify-center px-4 sm:px-6">
        <div className="flex w-full max-w-sm flex-col gap-3">
          {toasts.map((toast) => {
            const variant = toast.variant ?? 'info';
            return (
              <div
                key={toast.id}
                className={cn(
                  'pointer-events-auto flex flex-col gap-1 rounded-lg border p-4 shadow-lg',
                  variant === 'error'
                    ? 'border-rose-200 bg-white text-rose-900'
                    : variant === 'success'
                      ? 'border-emerald-200 bg-white text-emerald-900'
                      : 'border-slate-200 bg-white text-slate-900'
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    {toast.title ? <p className="text-sm font-semibold">{toast.title}</p> : null}
                    {toast.description ? (
                      <p className="text-sm leading-relaxed text-slate-600">{toast.description}</p>
                    ) : null}
                  </div>
                  <button
                    type="button"
                    onClick={() => removeToast(toast.id)}
                    className="mt-0.5 rounded-md p-1 text-xs text-slate-400 transition hover:text-slate-600"
                  >
                    <span className="sr-only">Dismiss</span>
                    ×
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = React.useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}
