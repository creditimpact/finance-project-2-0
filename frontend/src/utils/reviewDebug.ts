function resolveImportMetaEnv(): Record<string, string | undefined> {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch (err) {
    return {};
  }
}

const metaEnv = resolveImportMetaEnv();

function resolveEnvVar(name: string): string | undefined {
  if (name in metaEnv && typeof metaEnv[name] === 'string') {
    return metaEnv[name];
  }
  if (typeof process !== 'undefined' && process.env && typeof process.env[name] === 'string') {
    return process.env[name];
  }
  return undefined;
}

export const REVIEW_DEBUG_ENABLED = resolveEnvVar('VITE_REVIEW_DEBUG') === '1';

export function reviewDebugLog(...args: unknown[]): void {
  if (!REVIEW_DEBUG_ENABLED || typeof console === 'undefined') {
    return;
  }
  const prefix = '[review-debug]';
  if (typeof console.debug === 'function') {
    console.debug(prefix, ...args);
    return;
  }
  if (typeof console.log === 'function') {
    console.log(prefix, ...args);
  }
}
