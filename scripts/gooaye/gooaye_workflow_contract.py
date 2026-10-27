#!/usr/bin/env python3
"""Deterministic, storage-neutral contract for the shared Gooaye EPnnn runner.

The local runner owns the ledger and external calls. This module owns the
identity/checkpoint rules so a restart cannot decide to repeat a completed
side-effect from prompt text. Callers must atomically persist the returned
state while holding the existing episode lock.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 2
DEFAULT_MAX_ATTEMPTS = 3
STAGES = (
    "source_identity",
    "audio_chunks",
    "transcript_brief",
    "transcript_notice",
    "publication",
    "dispatch",
    "trusted_receipt",
    "terminal_delivery",
)
LEGACY_STAGES = tuple(stage for stage in STAGES if stage != "transcript_notice")
EPISODE_RE = re.compile(r"^EP[0-9]{3,}$")
URL_RE = re.compile(r"^https?://[^\s]+$")
STATUSES = {"pending", "running", "outcome_unknown", "completed", "failed"}


class WorkflowContractError(ValueError):
    """Base class for fail-closed workflow state errors."""


class IdentityConflict(WorkflowContractError):
    pass


class ReceiptConflict(WorkflowContractError):
    pass


class LockConflict(WorkflowContractError):
    pass


class OutcomeUnknown(WorkflowContractError):
    pass


def _text(value: Any, field: str) -> str:
    if value is None:
        raise IdentityConflict(f"missing identity field: {field}")
    result = str(value).strip()
    if not result:
        raise IdentityConflict(f"empty identity field: {field}")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def derive_identity(metadata: dict[str, Any]) -> dict[str, str]:
    """Derive one logical episode identity from actual feed/Telegram metadata."""
    episode_number = _text(metadata.get("episode_number"), "episode_number").upper()
    if not EPISODE_RE.fullmatch(episode_number):
        raise IdentityConflict("episode_number must match EPnnn")
    rss_guid = _text(metadata.get("rss_guid"), "rss_guid")
    enclosure_url = _text(metadata.get("enclosure_url"), "enclosure_url")
    if not URL_RE.fullmatch(enclosure_url):
        raise IdentityConflict("enclosure_url must be an http(s) URL")
    telegram_post_id = _text(metadata.get("telegram_post_id"), "telegram_post_id")
    # Keep the source identity exact. Episode number is included as a checked
    # display identity so a changed label cannot silently reuse another record.
    identity = {
        "episode_number": episode_number,
        "rss_guid": rss_guid,
        "enclosure_url": enclosure_url,
        "telegram_post_id": telegram_post_id,
    }
    identity["logical_id"] = "gooaye:" + hashlib.sha256(_canonical(identity).encode()).hexdigest()
    return identity


def new_state(metadata: dict[str, Any]) -> dict[str, Any]:
    identity = derive_identity(metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "stages": {
            stage: {"status": "pending", "attempts": 0}
            for stage in STAGES
        },
        "lock": None,
    }


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(state)
    stages = result.get("stages")
    if (result.get("schema_version") == 1 and isinstance(stages, dict)
            and set(stages) == set(LEGACY_STAGES)):
        # An old in-flight final send may already contain the brief.
        terminal_status = stages["terminal_delivery"].get("status")
        already_notified = terminal_status in {"running", "outcome_unknown", "completed"}
        result["schema_version"] = SCHEMA_VERSION
        result["stages"] = {
            stage: ({"status": "completed", "attempts": 0,
                     "result": {"delivery_status": "legacy_terminal_may_include_brief"}}
                    if stage == "transcript_notice" and already_notified else
                    {"status": "pending", "attempts": 0}
                    if stage == "transcript_notice" else stages[stage])
            for stage in STAGES
        }
    if result.get("schema_version") != SCHEMA_VERSION:
        raise WorkflowContractError("unsupported workflow state schema")
    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise IdentityConflict("workflow state has no durable episode identity")
    # Recompute the logical identity from the durable source fields on every
    # transition. A corrupt/tampered ledger must fail closed before any side
    # effect can be resumed.
    identity_metadata = {
        field: identity.get(field)
        for field in ("episode_number", "rss_guid", "enclosure_url", "telegram_post_id")
    }
    expected_identity = derive_identity(identity_metadata)
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise IdentityConflict(f"workflow identity mismatch: {field}")
    if set(result.get("stages", {})) != set(STAGES):
        raise WorkflowContractError("workflow state has incomplete stage set")
    for stage in STAGES:
        status = result["stages"][stage].get("status")
        if status not in STATUSES:
            raise WorkflowContractError(f"invalid status for {stage}")
    return result


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Validate and migrate a durable workflow state without mutating its caller."""
    return _copy_state(state)


def assert_identity(state: dict[str, Any], metadata: dict[str, Any]) -> None:
    expected = derive_identity(metadata)
    actual = state.get("identity") or {}
    if actual != expected:
        raise IdentityConflict("episode identity does not match the durable ledger")


def acquire_lock(state: dict[str, Any], owner: str, token: str) -> dict[str, Any]:
    owner = _text(owner, "lock owner")
    token = _text(token, "lock token")
    result = _copy_state(state)
    current = result.get("lock")
    if current is not None and current != {"owner": owner, "token": token}:
        raise LockConflict("episode is owned by another active runner")
    result["lock"] = {"owner": owner, "token": token}
    return result


def release_lock(state: dict[str, Any], owner: str, token: str) -> dict[str, Any]:
    result = _copy_state(state)
    if result.get("lock") != {"owner": owner, "token": token}:
        raise LockConflict("lock release does not match the active owner")
    result["lock"] = None
    return result


def resume_stage(state: dict[str, Any], *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> str:
    """Return the only safe next action; unknown side effects require reconcile."""
    if max_attempts < 1:
        raise WorkflowContractError("max_attempts must be positive")
    result = _copy_state(state)
    for stage in STAGES:
        record = result["stages"][stage]
        status = record["status"]
        if status == "completed":
            continue
        if status in {"running", "outcome_unknown"}:
            return f"reconcile:{stage}"
        if status == "failed" and (
            record.get("retryable", True) is False
            or int(record.get("attempts", 0)) >= max_attempts
        ):
            return f"failed:{stage}"
        return stage
    return "done"


def _require_predecessors(result: dict[str, Any], stage: str) -> None:
    index = STAGES.index(stage)
    incomplete = [name for name in STAGES[:index] if result["stages"][name]["status"] != "completed"]
    if incomplete:
        raise WorkflowContractError(f"cannot advance {stage}; incomplete predecessor: {incomplete[0]}")


def start_stage(
    state: dict[str, Any],
    stage: str,
    attempt_id: str,
    *,
    expected_hash: str | None = None,
    operation_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise WorkflowContractError(f"unknown stage: {stage}")
    attempt_id = _text(attempt_id, "attempt_id")
    if max_attempts < 1:
        raise WorkflowContractError("max_attempts must be positive")
    result = _copy_state(state)
    record = result["stages"][stage]
    _require_predecessors(result, stage)
    if record["status"] == "completed":
        return result
    if record["status"] == "outcome_unknown":
        raise OutcomeUnknown(f"reconcile {stage} before retry")
    if record["status"] == "failed" and (
        record.get("retryable", True) is False
        or int(record.get("attempts", 0)) >= max_attempts
    ):
        raise WorkflowContractError(f"retry budget exhausted for {stage}")
    if record["status"] == "running":
        if record.get("attempt_id") == attempt_id:
            return result
        raise LockConflict(f"stage {stage} already has an active attempt")
    record.update({"status": "running", "attempt_id": attempt_id})
    record["attempts"] = int(record.get("attempts", 0)) + 1
    if expected_hash is not None:
        record["expected_hash"] = _text(expected_hash, "expected_hash")
    if operation_id is not None:
        record["operation_id"] = _text(operation_id, "operation_id")
    return result


def record_blocker(
    state: dict[str, Any], stage: str, reason: str, *, failure_type: str = "unknown",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a task-specific blocker fingerprint beside its exact precondition evidence."""
    result = _copy_state(state)
    record = result["stages"].get(stage)
    if record is None or record["status"] not in {"running", "outcome_unknown", "failed"}:
        raise WorkflowContractError(f"stage {stage} has no active or blocked operation")
    reason = _text(reason, "failure reason")
    evidence = copy.deepcopy(evidence or {})
    fingerprint = {
        "logical_id": result["identity"]["logical_id"],
        "stage": stage,
        "operation_id": record.get("operation_id"),
        "failure_type": _text(failure_type, "failure_type"),
        "reason": reason,
        "evidence": evidence,
    }
    record.update(
        failure_reason=reason,
        failure_type=fingerprint["failure_type"],
        blocker_evidence=evidence,
        blocker_fingerprint=hashlib.sha256(_canonical(fingerprint).encode()).hexdigest(),
    )
    return result


def fail_stage(
    state: dict[str, Any], stage: str, reason: str, *, retryable: bool = True,
    failure_type: str = "unknown", evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _copy_state(state)
    record = result["stages"].get(stage)
    if record is None or record["status"] != "running":
        raise WorkflowContractError(f"stage {stage} must be running before failure")
    record["status"] = "failed"
    record["retryable"] = bool(retryable)
    return record_blocker(result, stage, reason, failure_type=failure_type, evidence=evidence)


def complete_stage(
    state: dict[str, Any],
    stage: str,
    *,
    artifact_hash: str | None = None,
    external_id: str | None = None,
) -> dict[str, Any]:
    result = _copy_state(state)
    record = result["stages"].get(stage)
    if record is None:
        raise WorkflowContractError(f"unknown stage: {stage}")
    if record["status"] == "completed":
        if artifact_hash is not None and record.get("artifact_hash") != artifact_hash:
            raise IdentityConflict(f"completed {stage} hash differs")
        if external_id is not None and record.get("external_id") not in {None, external_id}:
            raise IdentityConflict(f"completed {stage} external_id differs")
        return result
    if record["status"] != "running":
        raise WorkflowContractError(f"stage {stage} is not running")
    if record.get("expected_hash") and artifact_hash != record["expected_hash"]:
        raise IdentityConflict(f"{stage} artifact hash differs from the recorded expectation")
    record["status"] = "completed"
    for key in ("failure_reason", "failure_type", "blocker_evidence", "blocker_fingerprint", "retryable"):
        record.pop(key, None)
    if artifact_hash is not None:
        record["artifact_hash"] = _text(artifact_hash, "artifact_hash")
    if external_id is not None:
        record["external_id"] = _text(external_id, "external_id")
    return result


def mark_outcome_unknown(state: dict[str, Any], stage: str) -> dict[str, Any]:
    result = _copy_state(state)
    record = result["stages"].get(stage)
    if record is None or record["status"] != "running":
        raise WorkflowContractError(f"stage {stage} must be running before outcome-unknown")
    if not record.get("operation_id"):
        raise OutcomeUnknown(f"stage {stage} has no durable operation_id to reconcile")
    record["status"] = "outcome_unknown"
    return result


def reconcile_receipt(
    state: dict[str, Any],
    stage: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Converge an uncertain or blocked operation only on its exact trusted receipt."""
    result = _copy_state(state)
    record = result["stages"].get(stage)
    blocked_recheck = (record is not None and record["status"] == "failed"
                       and record.get("retryable") is False and record.get("blocker_fingerprint"))
    if record is None or (record["status"] not in {"running", "outcome_unknown"} and not blocked_recheck):
        raise WorkflowContractError(f"stage {stage} has no uncertain or fingerprinted blocked operation")
    if receipt.get("logical_id") != result["identity"].get("logical_id"):
        raise ReceiptConflict("receipt logical_id does not match episode identity")
    if not record.get("operation_id"):
        raise ReceiptConflict("uncertain operation has no recorded operation_id")
    if receipt.get("operation_id") != record["operation_id"]:
        raise ReceiptConflict("receipt operation_id does not match")
    if record.get("expected_hash") and receipt.get("content_hash") != record["expected_hash"]:
        raise ReceiptConflict("receipt content_hash does not match")
    if not receipt.get("operation_id"):
        raise ReceiptConflict("trusted receipt is missing operation_id")
    record["status"] = "completed"
    record["external_id"] = _text(receipt.get("external_id"), "receipt external_id")
    record["receipt_id"] = _text(receipt.get("receipt_id"), "receipt_id")
    record["content_hash"] = _text(receipt.get("content_hash"), "receipt content_hash")
    for key in ("failure_reason", "failure_type", "blocker_evidence", "blocker_fingerprint",
                "retryable", "absence_proof"):
        record.pop(key, None)
    return result


def requeue_after_confirmed_absence(
    state: dict[str, Any], stage: str, proof: str
) -> dict[str, Any]:
    """Allow a retry only after the existing publisher proves no write exists."""
    result = _copy_state(state)
    record = result["stages"].get(stage)
    blocked_recheck = (record is not None and record["status"] == "failed"
                       and record.get("retryable") is False and record.get("blocker_fingerprint"))
    if record is None or (record["status"] != "outcome_unknown" and not blocked_recheck):
        raise WorkflowContractError(f"stage {stage} has no uncertain or fingerprinted blocked operation")
    proof = _text(proof, "absence proof")
    record["status"] = "pending"
    record["absence_proof"] = proof
    for key in ("attempt_id", "operation_id", "failure_reason", "failure_type",
                "blocker_evidence", "blocker_fingerprint", "retryable"):
        record.pop(key, None)
    return result


def checkout_action(
    dirty_paths: list[str],
    relevant_paths: list[str],
    active_owner: str | None = None,
) -> str:
    """Choose a safe existing execution route without touching dirty work."""
    # An active owner only blocks the in-place route when the required
    # operation actually overlaps the dirty checkout. Unrelated edits must
    # never turn into the old blanket "ask the original owner" stop: use the
    # existing clean publisher/native API path instead.
    dirty = set(dirty_paths)
    relevant = set(relevant_paths)
    if dirty & relevant:
        return "blocked_active_owner" if active_owner else "blocked_relevant_conflict"
    if dirty:
        return "use_existing_clean_publisher"
    if active_owner:
        return "blocked_active_owner"
    return "proceed_in_place"
