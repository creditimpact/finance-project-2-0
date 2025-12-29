import type { AccountAttachments, AccountQuestionAnswers } from '../components/AccountQuestions';
import type { SubmitReviewPayload } from '../api';
import type { ClaimSchema, AttachmentsMap, PackClaimsPayload, DocKey } from '../types/review';
import claimsSchema from '../../../shared/claims_schema.json';

export type NormalizedAttachments = AccountAttachments;

export function normalizeSelectedClaims(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const entry of value) {
    if (typeof entry !== 'string') {
      continue;
    }
    const trimmed = entry.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    normalized.push(trimmed);
  }
  return normalized;
}

export function normalizeAttachments(value: unknown): NormalizedAttachments {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }
  const record = value as Record<string, unknown>;
  const normalized: NormalizedAttachments = {};
  for (const [rawKey, rawValue] of Object.entries(record)) {
    if (typeof rawKey !== 'string') {
      continue;
    }
    const docKey = rawKey.trim();
    if (!docKey) {
      continue;
    }
    const values = Array.isArray(rawValue) ? rawValue : [rawValue];
    const collected: string[] = [];
    for (const entry of values) {
      if (typeof entry !== 'string') {
        continue;
      }
      const trimmed = entry.trim();
      if (!trimmed || collected.includes(trimmed)) {
        continue;
      }
      collected.push(trimmed);
    }
    if (collected.length > 0) {
      normalized[docKey] = collected;
    }
  }
  return normalized;
}

function mergeAttachments(
  primary?: NormalizedAttachments,
  secondary?: NormalizedAttachments
): NormalizedAttachments {
  if (!primary && !secondary) {
    return {};
  }
  const merged: NormalizedAttachments = {};
  const apply = (source?: NormalizedAttachments) => {
    if (!source) {
      return;
    }
    for (const [docKey, docIds] of Object.entries(source)) {
      if (!Array.isArray(docIds) || docIds.length === 0) {
        continue;
      }
      const existing = merged[docKey] ? new Set(merged[docKey]) : new Set<string>();
      for (const docId of docIds) {
        if (typeof docId !== 'string') {
          continue;
        }
        const trimmed = docId.trim();
        if (!trimmed) {
          continue;
        }
        existing.add(trimmed);
      }
      if (existing.size > 0) {
        merged[docKey] = Array.from(existing);
      }
    }
  };

  apply(secondary);
  apply(primary);

  return merged;
}

function convertLegacyClaimDocuments(value: unknown): NormalizedAttachments {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }
  const record = value as Record<string, unknown>;
  const attachments: NormalizedAttachments = {};
  for (const claimValue of Object.values(record)) {
    if (!claimValue || typeof claimValue !== 'object') {
      continue;
    }
    const docMap = claimValue as Record<string, unknown>;
    for (const [docKey, docValue] of Object.entries(docMap)) {
      const normalized = normalizeAttachments({ [docKey]: docValue });
      if (normalized[docKey]) {
        attachments[docKey] = normalized[docKey];
      }
    }
  }
  return attachments;
}

function convertLegacyEvidence(value: unknown): NormalizedAttachments {
  if (!Array.isArray(value)) {
    return {};
  }
  const attachments: NormalizedAttachments = {};
  for (const entry of value) {
    if (!entry || typeof entry !== 'object') {
      continue;
    }
    const record = entry as { docs?: unknown };
    const docs = Array.isArray(record.docs) ? record.docs : [];
    for (const docEntry of docs) {
      if (!docEntry || typeof docEntry !== 'object') {
        continue;
      }
      const docRecord = docEntry as { doc_key?: unknown; doc_ids?: unknown };
      const docKey = typeof docRecord.doc_key === 'string' ? docRecord.doc_key : undefined;
      if (!docKey) {
        continue;
      }
      const normalized = normalizeAttachments({ [docKey]: docRecord.doc_ids });
      if (normalized[docKey]) {
        attachments[docKey] = normalized[docKey];
      }
    }
  }
  return attachments;
}

export function formatDocKey(docKey: string): string {
  if (!docKey) {
    return 'Document';
  }
  return docKey
    .split('_')
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ');
}

export function getMissingRequiredDocs(
  selectedClaims: string[] | undefined,
  attachments: NormalizedAttachments | undefined,
  claimDefinitions: Map<string, ClaimSchema>
): Record<string, string[]> {
  const missing: Record<string, string[]> = {};
  if (!selectedClaims || selectedClaims.length === 0) {
    return missing;
  }
  for (const claimKey of selectedClaims) {
    const definition = claimDefinitions.get(claimKey);
    if (!definition) {
      continue;
    }
    const requiredDocs = definition.requires ?? [];
    if (!requiredDocs || requiredDocs.length === 0) {
      continue;
    }
    const attachmentsForClaim = attachments ?? {};
    const missingDocs = requiredDocs.filter((docKey) => {
      const values = attachmentsForClaim[docKey];
      return !Array.isArray(values) || values.length === 0;
    });
    if (missingDocs.length > 0) {
      missing[claimKey] = missingDocs;
    }
  }
  return missing;
}

export function hasMissingRequiredDocs(
  selectedClaims: string[] | undefined,
  attachments: NormalizedAttachments | undefined,
  claimDefinitions: Map<string, ClaimSchema>
): boolean {
  return Object.keys(getMissingRequiredDocs(selectedClaims, attachments, claimDefinitions)).length > 0;
}

export function normalizeExistingAnswers(source: unknown): AccountQuestionAnswers {
  if (!source || typeof source !== 'object') {
    return {};
  }

  const record = source as Record<string, unknown>;
  const answersSection =
    record.answers && typeof record.answers === 'object' && !Array.isArray(record.answers)
      ? (record.answers as Record<string, unknown>)
      : undefined;

  const explanationFromRoot = typeof record.explanation === 'string' ? record.explanation : undefined;
  const explanationFromSection =
    answersSection && typeof answersSection.explanation === 'string'
      ? (answersSection.explanation as string)
      : undefined;

  const selectedClaims = normalizeSelectedClaims(
    answersSection?.selected_claims ?? answersSection?.claims ?? record.claims
  );

  const attachmentsFromAnswers = normalizeAttachments(answersSection?.attachments);
  const legacyClaimDocs = mergeAttachments(
    convertLegacyClaimDocuments(answersSection?.claimDocuments ?? answersSection?.claim_documents),
    convertLegacyClaimDocuments(record.claimDocuments ?? record.claim_documents)
  );
  const legacyEvidence = convertLegacyEvidence(answersSection?.evidence ?? record.evidence);
  const attachments = mergeAttachments(
    mergeAttachments(attachmentsFromAnswers, legacyClaimDocs),
    mergeAttachments(legacyEvidence, normalizeAttachments(record.attachments))
  );

  const normalized: AccountQuestionAnswers = {};

  const explanation = explanationFromRoot ?? explanationFromSection;
  if (typeof explanation === 'string' && explanation.trim() !== '') {
    normalized.explanation = explanation;
  }

  if (selectedClaims.length > 0) {
    normalized.selectedClaims = selectedClaims;
  }

  if (Object.keys(attachments).length > 0) {
    normalized.attachments = attachments;
  }

  return normalized;
}

export function prepareAnswersPayload(
  answers: AccountQuestionAnswers,
  options?: { includeClaims?: boolean }
): SubmitReviewPayload {
  const payloadAnswers: Record<string, unknown> = {};

  if (typeof answers.explanation === 'string') {
    const trimmed = answers.explanation.trim();
    if (trimmed) {
      payloadAnswers.explanation = trimmed;
    }
  }

  for (const [key, value] of Object.entries(answers)) {
    if (key === 'selectedClaims' || key === 'attachments' || key === 'explanation') {
      continue;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed) {
        payloadAnswers[key] = trimmed;
      }
    } else if (value != null && typeof value !== 'object') {
      payloadAnswers[key] = value;
    }
  }

  if (options?.includeClaims) {
    const selectedClaims = normalizeSelectedClaims(
      answers.selectedClaims ?? (answers as Record<string, unknown>).selected_claims ?? []
    );
    if (selectedClaims.length > 0) {
      payloadAnswers.selected_claims = selectedClaims;
    }

    const normalizedAttachments = normalizeAttachments(answers.attachments);
    if (Object.keys(normalizedAttachments).length > 0) {
      const attachmentsPayload: AttachmentsMap = {};
      for (const [docKey, docIds] of Object.entries(normalizedAttachments)) {
        if (!docIds || docIds.length === 0) {
          continue;
        }
        attachmentsPayload[docKey] = docIds.length === 1 ? docIds[0] : docIds;
      }
      if (Object.keys(attachmentsPayload).length > 0) {
        payloadAnswers.attachments = attachmentsPayload;
      }
    }
  }

  return { answers: payloadAnswers };
}

function sanitizeDocKeys(raw: unknown): DocKey[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const result: DocKey[] = [];
  const seen = new Set<string>();
  for (const value of raw) {
    if (typeof value !== 'string') {
      continue;
    }
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    result.push(trimmed as DocKey);
  }
  return result;
}

function sanitizeOptionalDocKeys(raw: unknown): DocKey[] | undefined {
  const docKeys = sanitizeDocKeys(raw);
  return docKeys.length > 0 ? docKeys : undefined;
}

export function sanitizeClaimSchema(entry: ClaimSchema): ClaimSchema {
  const requires = sanitizeDocKeys(entry.requires);
  const optional = sanitizeOptionalDocKeys(entry.optional);
  const autoAttach = sanitizeOptionalDocKeys(entry.autoAttach);
  const minUploads =
    typeof entry.minUploads === 'number' && entry.minUploads >= 0 ? entry.minUploads : undefined;
  const description =
    typeof entry.description === 'string' && entry.description.trim() !== ''
      ? entry.description
      : undefined;

  const sanitized: ClaimSchema = {
    key: entry.key,
    title: entry.title,
    requires,
  };

  if (optional) {
    sanitized.optional = optional;
  }

  if (autoAttach) {
    sanitized.autoAttach = autoAttach;
  }

  if (typeof minUploads === 'number') {
    sanitized.minUploads = minUploads;
  }

  if (description) {
    sanitized.description = description;
  }

  return sanitized;
}

export function resolveClaimsPayload(raw: unknown): PackClaimsPayload {
  const fallbackGeneric = claimsSchema.byIssue.find((issue) => issue.issue === 'generic');
  const fallback: PackClaimsPayload = {
    autoAttachBase: [...claimsSchema.autoAttachBase],
    items: fallbackGeneric ? fallbackGeneric.claims.map(sanitizeClaimSchema) : [],
  };

  if (!raw || typeof raw !== 'object') {
    return fallback;
  }

  const record = raw as PackClaimsPayload;
  const autoAttachBase = Array.isArray(record.autoAttachBase)
    ? record.autoAttachBase.filter((value): value is DocKey => typeof value === 'string')
    : fallback.autoAttachBase;

  const items = Array.isArray(record.items)
    ? record.items
        .filter((entry): entry is ClaimSchema => entry != null && typeof entry === 'object')
        .map(sanitizeClaimSchema)
    : [];

  if (items.length === 0) {
    return { autoAttachBase, items: fallback.items };
  }

  return { autoAttachBase, items };
}
