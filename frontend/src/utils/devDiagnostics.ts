const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on']);
const DEV_VALUES = new Set(['1', 'true', 'yes', 'on', 'dev', 'development']);

function resolveImportMetaEnv(): Record<string, unknown> {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch (err) {
    return {};
  }
}

const metaEnv = resolveImportMetaEnv();

function resolveEnvVar(name: string): unknown {
  if (name in metaEnv) {
    return metaEnv[name];
  }
  if (typeof process !== 'undefined' && process.env && name in process.env) {
    return process.env[name];
  }
  return undefined;
}

function toNormalizedString(value: unknown): string | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toString();
  }
  if (typeof value === 'string') {
    return value;
  }
  return undefined;
}

function isTrueValue(value: unknown, allowedValues: Set<string>): boolean {
  const text = toNormalizedString(value);
  if (!text) {
    return false;
  }
  const normalized = text.trim().toLowerCase();
  return allowedValues.has(normalized);
}

function resolveIsDev(): boolean {
  if (isTrueValue(metaEnv.DEV, DEV_VALUES)) {
    return true;
  }
  if (isTrueValue(metaEnv.MODE, DEV_VALUES)) {
    return true;
  }
  if (isTrueValue(resolveEnvVar('DEV'), DEV_VALUES)) {
    return true;
  }
  if (isTrueValue(resolveEnvVar('MODE'), DEV_VALUES)) {
    return true;
  }
  if (isTrueValue(resolveEnvVar('NODE_ENV'), DEV_VALUES)) {
    return true;
  }
  return false;
}

function resolveDiagnosticsFlag(): boolean {
  const value = resolveEnvVar('VITE_DEV_DIAGNOSTICS');
  return isTrueValue(value, TRUE_VALUES);
}

export const DEV_DIAGNOSTICS_ENABLED = resolveIsDev() && resolveDiagnosticsFlag();

