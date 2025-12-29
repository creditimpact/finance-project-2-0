const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on']);
const FALSE_VALUES = new Set(['0', 'false', 'no', 'off']);

function readImportMetaEnv(key: string): string | undefined {
  try {
    return new Function(
      'k',
      'return typeof import !== "undefined" && import.meta && import.meta.env ? import.meta.env[k] : undefined;'
    )(key);
  } catch (_error) {
    return undefined;
  }
}

function readEnvValue(key: string): string | undefined {
  const metaValue = readImportMetaEnv(key);
  if (metaValue !== undefined) {
    return metaValue;
  }

  if (typeof process !== 'undefined' && process.env) {
    return process.env[key];
  }

  return undefined;
}

function parseBooleanFlag(value: string | undefined, fallback: boolean): boolean {
  if (value == null) {
    return fallback;
  }

  const normalized = value.toString().trim().toLowerCase();
  if (TRUE_VALUES.has(normalized)) {
    return true;
  }
  if (FALSE_VALUES.has(normalized)) {
    return false;
  }

  return fallback;
}

export function getBooleanFlag(key: string, defaultValue: boolean): boolean {
  return parseBooleanFlag(readEnvValue(key), defaultValue);
}

export function shouldShowBureauDetails(): boolean {
  return getBooleanFlag('VITE_SHOW_BUREAU_DETAILS', true);
}

export function shouldHideConsensus(): boolean {
  return getBooleanFlag('VITE_HIDE_CONSENSUS', true);
}

export function shouldPreferLongestAccountMask(): boolean {
  return getBooleanFlag('VITE_ACCOUNT_NUMBER_PREFER_LONGEST_MASK', false);
}

export function shouldEnableReviewClaims(): boolean {
  return getBooleanFlag('VITE_REVIEW_CLAIMS', false);
}
