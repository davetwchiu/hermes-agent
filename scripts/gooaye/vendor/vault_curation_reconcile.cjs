'use strict';

const { isDeepStrictEqual } = require('node:util');

const BRIDGE_OWNER = 'davetwchiu';
const BRIDGE_REPO = 'hermes-bridge';
const RAW_INBOX_PREFIXES = ['00 Inbox/', 'Inbox/', 'inbox/'];
const SCHEDULED_RUN_PREFIX = '00 Inbox/Scheduled Runs/';
const SOURCE_RESOLUTION_MARKER = '<!-- hermes-source-resolution:v1';
const SUCCESSFUL_RECEIPT_RESULTS = new Set([
  'created', 'updated', 'noop', 'no-op', 'processed',
  'ledger_only', 'working_context_updated',
  'NO_ALERT', 'CHANGE', 'IMPORTANT', 'MAJOR', 'ERROR',
]);
const TERMINAL_OR_QUEUE_LABELS = new Set([
  'queued',
  'running',
  'completed',
  'failed',
  'needs-approval',
]);

function isRawInboxPath(path) {
  return RAW_INBOX_PREFIXES.some((prefix) => path.startsWith(prefix));
}

function parseBridgeTaskId(prBody) {
  const body = prBody || '';
  const match = body.match(
    /(?:https:\/\/github\.com\/davetwchiu\/hermes-bridge\/issues\/|davetwchiu\/hermes-bridge#)(\d+)/i,
  );
  if (!match) throw new Error('Eligible curation PR must reference one hermes-bridge task.');
  return Number(match[1]);
}

function parseExactMaterials(issueBody) {
  const body = issueBody || '';
  const match = body.match(/###\s+Exact changed material\s*```json\s*([\s\S]*?)```/i);
  if (!match) throw new Error('Bridge task is missing the Exact changed material JSON block.');

  let materials;
  try {
    materials = JSON.parse(match[1]);
  } catch (error) {
    throw new Error(`Bridge task Exact changed material JSON is invalid: ${error.message}`);
  }

  if (!Array.isArray(materials) || materials.length === 0) {
    throw new Error('Bridge task Exact changed material must be a non-empty array.');
  }

  return materials.map((item, index) => {
    if (!item || typeof item !== 'object') {
      throw new Error(`Material ${index + 1} must be an object.`);
    }
    if (typeof item.path !== 'string' || !isRawInboxPath(item.path)) {
      throw new Error(`Material ${index + 1} has an invalid Inbox path.`);
    }
    if (!/^[0-9a-f]{40}$/i.test(item.blob_sha || '')) {
      throw new Error(`Material ${index + 1} has an invalid blob_sha.`);
    }
    if (item.path.startsWith(SCHEDULED_RUN_PREFIX) && item.dedupe_key !== undefined &&
        (typeof item.dedupe_key !== 'string' || !item.dedupe_key)) {
      throw new Error(`Scheduled material ${index + 1} is missing dedupe_key.`);
    }
    return item;
  });
}

function parseSourceResolution(prBody, materials) {
  const body = prBody || '';
  const markerCount = body.split(SOURCE_RESOLUTION_MARKER).length - 1;
  if (markerCount === 0) return null;
  const matches = [...body.matchAll(
    /<!-- hermes-source-resolution:v1 source_id=(source-[0-9a-f]{64}) path=([^\s]+) blob=([0-9a-f]{40}) prior_pr=(\d+) approved_by=davetwchiu disposition=processed-with-correction -->/g,
  )];
  if (markerCount !== 1 || matches.length !== 1) {
    throw new Error('Source resolution marker is malformed or duplicated.');
  }
  const [, sourceId, path, blobSha, priorPrText] = matches[0];
  const priorPr = Number(priorPrText);
  const matchesMaterial = materials.filter((material) => (
    material.source_id === sourceId && material.path === path && material.blob_sha === blobSha
  ));
  if (!Number.isSafeInteger(priorPr) || priorPr <= 0 || matchesMaterial.length !== 1) {
    throw new Error('Source resolution must identify one exact Bridge material and prior PR.');
  }
  return { source_id: sourceId, path, blob_sha: blobSha, prior_pr: priorPr };
}

function rawReceipt(ledger, material, includeHistoryBatches = false) {
  const items = ledger?.items || {};
  const compositeKey = `${material.path}#${material.blob_sha}`;
  const stores = [items];
  if (includeHistoryBatches) {
    for (const [name, batch] of Object.entries(ledger || {})) {
      if (/^chatgpt_history_batch\d+$/.test(name) && batch && typeof batch === 'object') {
        stores.push(batch);
        if (batch.items && typeof batch.items === 'object') stores.push(batch.items);
      }
    }
  }
  for (const store of stores) {
    for (const key of [compositeKey, material.path]) {
      const receipt = store[key];
      if (receipt?.blob_sha === material.blob_sha &&
          (!receipt.path || receipt.path === material.path)) return { key, receipt };
    }
  }
  const key = Object.hasOwn(items, compositeKey) ? compositeKey : material.path;
  return { key, receipt: items[key] };
}

function verifySourceResolutionBase({ ledger, materials, taskId, sourceResolution }) {
  const problems = [];
  for (const material of materials) {
    const { receipt } = rawReceipt(ledger, material);
    if (!receipt) {
      problems.push(`missing prior receipt for ${material.path}`);
      continue;
    }
    if (receipt.blob_sha !== material.blob_sha) problems.push(`prior blob mismatch for ${material.path}`);
    if (material.source_commit && receipt.source_commit !== material.source_commit) {
      problems.push(`prior source commit mismatch for ${material.path}`);
    }
    if (receipt.status !== 'processed') problems.push(`prior status is not processed for ${material.path}`);
    if (Number(receipt.task_id) !== Number(taskId)) problems.push(`prior task id mismatch for ${material.path}`);
    if (Number(receipt.pr) !== sourceResolution.prior_pr) problems.push(`prior PR mismatch for ${material.path}`);
    if (material.path === sourceResolution.path && material.blob_sha === sourceResolution.blob_sha &&
        receipt.result !== 'review_required') {
      problems.push(`source resolution target was not review_required for ${material.path}`);
    }
  }
  if (!materials.some((material) => (
    material.source_id === sourceResolution.source_id && material.path === sourceResolution.path &&
    material.blob_sha === sourceResolution.blob_sha
  ))) problems.push('source resolution target is not an exact Bridge material');
  return { ok: problems.length === 0, problems };
}

function verifySourceResolutionDiff({ baseLedger, ledger, sourceResolution }) {
  const problems = [];
  const material = {
    path: sourceResolution.path,
    blob_sha: sourceResolution.blob_sha,
  };
  const before = rawReceipt(baseLedger, material);
  const after = rawReceipt(ledger, material);
  if (!before.receipt || !after.receipt || before.key !== after.key) {
    problems.push('source resolution must update its existing receipt key');
    return { ok: false, problems };
  }
  const beforeItems = { ...(baseLedger.items || {}) };
  const afterItems = { ...(ledger.items || {}) };
  delete beforeItems[before.key];
  delete afterItems[after.key];
  const restBefore = { ...baseLedger, items: beforeItems };
  const restAfter = { ...ledger, items: afterItems };
  if (!isDeepStrictEqual(restBefore, restAfter)) {
    problems.push('source resolution PR changed ledger entries outside the exact target receipt');
  }
  if (after.receipt.result !== 'processed' || Number(after.receipt.pr) <= 0 ||
      !String(after.receipt.resolution_note || '').includes('processed-with-correction') ||
      !String(after.receipt.resolution_note || '').includes('source_conflict')) {
    problems.push('updated target receipt lacks the processed-with-correction source_conflict note');
  }
  return { ok: problems.length === 0, problems };
}

function verifyLedgerReceipt({ ledger, materials, taskId, prNumber, sourceResolution = null }) {
  const problems = [];
  const items = ledger?.items || {};
  const scheduledRuns = ledger?.scheduled_runs || {};

  for (const material of materials) {
    if (material.path.startsWith(SCHEDULED_RUN_PREFIX) && material.dedupe_key) {
      const receipt = scheduledRuns[material.dedupe_key];
      if (!receipt) {
        problems.push(`missing scheduled_runs receipt for ${material.dedupe_key}`);
        continue;
      }
      if (receipt.path !== material.path) problems.push(`scheduled path mismatch for ${material.dedupe_key}`);
      if (receipt.blob_sha !== material.blob_sha) problems.push(`scheduled blob mismatch for ${material.dedupe_key}`);
      if (receipt.status !== 'processed') problems.push(`scheduled status is not processed for ${material.dedupe_key}`);
      if (Number(receipt.hermes_task_id) !== Number(taskId)) problems.push(`scheduled task id mismatch for ${material.dedupe_key}`);
      if (Number(receipt.pr) !== Number(prNumber)) problems.push(`scheduled PR mismatch for ${material.dedupe_key}`);
      if (!SUCCESSFUL_RECEIPT_RESULTS.has(receipt.result)) {
        problems.push(`scheduled result is not a terminal success for ${material.dedupe_key}`);
      }
      continue;
    }

    const { receipt } = rawReceipt(ledger, material, true);
    if (!receipt) {
      problems.push(`missing items receipt for ${material.path}`);
      continue;
    }
    const isResolutionTarget = sourceResolution && material.path === sourceResolution.path &&
      material.blob_sha === sourceResolution.blob_sha;
    const expectedPr = isResolutionTarget ? prNumber : sourceResolution?.prior_pr || prNumber;
    if (receipt.blob_sha !== material.blob_sha) problems.push(`blob mismatch for ${material.path}`);
    if (receipt.status !== 'processed') problems.push(`status is not processed for ${material.path}`);
    if (Number(receipt.task_id) !== Number(taskId)) problems.push(`task id mismatch for ${material.path}`);
    if (Number(receipt.pr) !== Number(expectedPr)) problems.push(`PR mismatch for ${material.path}`);
    if (!SUCCESSFUL_RECEIPT_RESULTS.has(receipt.result)) {
      problems.push(`result is not a terminal success for ${material.path}`);
    }
    if (isResolutionTarget && (receipt.result !== 'processed' ||
        !String(receipt.resolution_note || '').includes('processed-with-correction') ||
        !String(receipt.resolution_note || '').includes('source_conflict'))) {
      problems.push(`source resolution is incomplete for ${material.path}`);
    }
  }

  return { ok: problems.length === 0, problems };
}

function classifyReconciliation({
  state = 'open',
  merged = false,
  mergeable = null,
  mergeableState = 'unknown',
  receiptVerified = null,
}) {
  if (merged) {
    return receiptVerified
      ? { action: 'complete', reason: 'merged_with_receipt' }
      : { action: 'requeue', reason: 'merged_without_receipt' };
  }

  if (state === 'closed') return { action: 'requeue', reason: 'closed_unmerged' };
  if (mergeable === false || mergeableState === 'dirty') {
    return { action: 'requeue', reason: 'merge_blocked' };
  }
  if (mergeableState === 'unknown' || mergeable === null) {
    return { action: 'wait', reason: 'mergeability_pending' };
  }
  return { action: 'wait', reason: 'merge_pending' };
}

function exactWorkflowCheckState({ run, checks = [], exactHead }) {
  if (!run || run.head_sha !== exactHead || run.event !== 'pull_request') {
    return { status: 'pending', reason: 'run_not_found_on_exact_head' };
  }
  if (run.status !== 'completed') return { status: 'pending', reason: 'run_incomplete' };
  if (run.conclusion !== 'success') return { status: 'failed', reason: run.conclusion || 'no_conclusion' };

  const check = checks.find((candidate) => (
    candidate.name === 'test' &&
    candidate.status === 'completed' &&
    candidate.conclusion === 'success' &&
    candidate.app?.slug === 'github-actions' &&
    candidate.head_sha === exactHead &&
    candidate.check_suite?.id === run.check_suite_id
  ));
  return check
    ? { status: 'success', reason: 'exact_head_check_verified', check }
    : { status: 'pending', reason: 'matching_check_not_completed' };
}

function desiredBridgeLabels(currentLabels, action) {
  const labels = new Set((currentLabels || []).filter((label) => !TERMINAL_OR_QUEUE_LABELS.has(label)));
  labels.add('hermes-task');
  if (action === 'complete') labels.add('completed');
  else if (action === 'requeue') labels.add('queued');
  else labels.add('running');
  return [...labels].sort();
}

function bridgeIssueStatePatch(currentLabels, action, currentState) {
  const labels = currentLabels || [];
  if (labels.includes('needs-approval')) {
    const keep = new Set(labels.filter((label) => !['queued', 'running', 'completed', 'failed'].includes(label)));
    keep.add('hermes-task');
    keep.add('needs-approval');
    return {
      state: 'open',
      ...(currentState === 'closed' ? { state_reason: 'reopened' } : {}),
      labels: [...keep].sort(),
    };
  }
  if (action === 'complete') {
    return {
      state: 'closed',
      state_reason: 'completed',
      labels: desiredBridgeLabels(labels, action),
    };
  }
  if (!['wait', 'requeue'].includes(action)) return null;
  return {
    state: 'open',
    ...(currentState === 'closed' ? { state_reason: 'reopened' } : {}),
    labels: desiredBridgeLabels(labels, action),
  };
}

function reconciliationMarker(prNumber, reason) {
  return `<!-- vault-e2e:pr=${Number(prNumber)} reason=${reason} -->`;
}

function hasReconciliationMarker(comments, marker) {
  return (comments || []).some((comment) => String(comment?.body || '').includes(marker));
}

function latestTrustedCompletionResult(comments, { taskId, trustedAuthor, vaultOwner, vaultRepo }) {
  const candidates = (comments || [])
    .filter((comment) => comment?.user?.login === trustedAuthor &&
      String(comment.body || '').includes('reason=merged_with_receipt'))
    .sort((a, b) => {
      const time = (comment) => {
        const parsed = Date.parse(comment?.created_at || '');
        return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
      };
      return time(a) - time(b);
    });
  const latest = candidates.at(-1);
  if (!latest) return null;

  const body = String(latest.body || '');
  const match = body.match(/<!-- vault-e2e:pr=(\d+) reason=merged_with_receipt -->/);
  const prNumber = Number(match?.[1]);
  const normalizedTaskId = Number(taskId);
  const required = [
    '<!-- hermes-result:v1 -->',
    'Status: completed',
    `Task ID: ${normalizedTaskId}`,
    `- Vault PR: https://github.com/${vaultOwner}/${vaultRepo}/pull/${prNumber}`,
    '- The curation PR is merged into `main`.',
    '- Every exact immutable source path/blob from the bridge task has a matching processed receipt on `main`.',
    `- The receipt records Hermes task #${normalizedTaskId} and Vault PR #${prNumber}.`,
    'Limitations:\n- None.',
  ];
  const createdAt = Date.parse(latest.created_at || '');
  if (!Number.isSafeInteger(prNumber) || prNumber <= 0 || !Number.isSafeInteger(normalizedTaskId) ||
      normalizedTaskId <= 0 || !Number.isFinite(createdAt) || required.some((line) => !body.includes(line))) {
    return null;
  }
  return { prNumber, taskId: normalizedTaskId, createdAt: latest.created_at };
}

function hasNewerVerifiedTerminalCompletion({
  bridgeIssue,
  currentPrCreatedAt,
  completion,
  completionPr,
  completionTaskId,
  taskId,
  receiptVerified,
}) {
  const normalizedTaskId = Number(taskId);
  if (bridgeIssue?.state !== 'closed' || bridgeIssue.state_reason !== 'completed' ||
      !completion || completion.taskId !== normalizedTaskId || receiptVerified !== true ||
      !completionPr || completionPr.number !== completion.prNumber || completionTaskId !== normalizedTaskId ||
      completionPr.state !== 'closed' || completionPr.merged !== true ||
      !/^[0-9a-f]{40}$/i.test(completionPr.merge_commit_sha || '')) return false;

  const eventCreatedAt = Date.parse(currentPrCreatedAt || '');
  const mergedAt = Date.parse(completionPr.merged_at || '');
  const resultCreatedAt = Date.parse(completion.createdAt || '');
  return Number.isFinite(eventCreatedAt) && Number.isFinite(mergedAt) && Number.isFinite(resultCreatedAt) &&
    mergedAt > eventCreatedAt && resultCreatedAt >= mergedAt;
}

const RESOLVER_HANDOFF_MARKER = '<!-- hermes-provenance-resolver:v1 -->';

function parseResolverHandoff(comments, { prNumber, headSha, trustedAuthor }) {
  const matches = [];
  for (const comment of comments || []) {
    if (comment?.user?.login !== trustedAuthor) continue;
    const body = String(comment?.body || '');
    if (!body.includes(RESOLVER_HANDOFF_MARKER)) continue;
    const jsonMatch = body.match(/```json\s*([\s\S]*?)```/i);
    if (!jsonMatch) continue;
    try {
      const evidence = JSON.parse(jsonMatch[1]);
      if (Number(evidence.pr_number) !== Number(prNumber)) continue;
      if (String(evidence.head_sha).toLowerCase() !== String(headSha).toLowerCase()) continue;
      if (evidence.status !== 'verified_mapping') continue;
      if (!Number.isInteger(Number(evidence.bridge_task_number)) || Number(evidence.bridge_task_number) <= 0) continue;
      if (typeof evidence.source_path !== 'string' || typeof evidence.source_blob !== 'string') continue;
      if (!/^[0-9a-f]{40}$/i.test(evidence.source_blob)) continue;
      matches.push({ comment, evidence });
    } catch {
      // Ignore malformed or stale comments and fail closed.
    }
  }
  matches.sort((a, b) => new Date(a.comment.created_at || 0) - new Date(b.comment.created_at || 0));
  return matches.length > 0 ? matches[matches.length - 1] : null;
}

module.exports = {
  BRIDGE_OWNER,
  BRIDGE_REPO,
  parseBridgeTaskId,
  parseExactMaterials,
  parseSourceResolution,
  verifyLedgerReceipt,
  verifySourceResolutionBase,
  verifySourceResolutionDiff,
  classifyReconciliation,
  exactWorkflowCheckState,
  desiredBridgeLabels,
  bridgeIssueStatePatch,
  reconciliationMarker,
  hasReconciliationMarker,
  latestTrustedCompletionResult,
  hasNewerVerifiedTerminalCompletion,
  RESOLVER_HANDOFF_MARKER,
  parseResolverHandoff,
};
