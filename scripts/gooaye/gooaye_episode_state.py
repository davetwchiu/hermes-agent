#!/usr/bin/env python3
"""Durable, one-shot state for Gooaye EP transcript follow-up."""
from __future__ import annotations

import argparse
import copy
import hashlib
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

STATE_PATH = get_hermes_home() / "state" / "gooaye-podcast-episodes.json"
CURATION_ENABLED = True
CURATION_BLOCKER = "Curation gate is closed; do not bypass the trusted Bridge and receipt checks."
EPISODE_RE = re.compile(r"(?<![A-Za-z0-9])EP\s*0*(\d{3,})\b", re.I)
SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.I)
CONTENT_SHA_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
AUDIO_PHASES = {"discovered", "transcribing", "retryable", "review_required", "transcript_ready", "curation_pending", "processed"}


def _now(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _iso(value: datetime) -> str:
    return _now(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _now(parsed)


def _episode(value: str | int) -> str:
    match = re.fullmatch(r"(?:EP\s*)?0*(\d{3,})", str(value).strip(), re.I)
    if not match:
        raise ValueError(f"invalid episode number: {value}")
    return str(int(match.group(1)))


def ingestion_identity(head_sha: str) -> str:
    if not SHA_RE.fullmatch(head_sha or ""):
        raise ValueError("head_sha must be a full 40-character commit SHA")
    return f"vault-ingest:{head_sha.lower()}"


def _empty() -> dict:
    return {"schema_version": 1, "updated_at": None, "last_seen_post_id": None, "episodes": {}}


def _load(path: Path) -> dict:
    if not path.exists():
        return _empty()
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or not isinstance(state.get("episodes"), dict):
        raise ValueError(f"unsupported or invalid episode ledger: {path}")
    return state


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield _load(path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _refs(entry: dict, ref: dict) -> None:
    refs = entry.setdefault("source_refs", [])
    if not any(x.get("url") == ref.get("url") for x in refs):
        refs.append(ref)


def observe_messages(messages: list[dict], *, now: datetime | None = None, path: Path = STATE_PATH) -> dict:
    current = _now(now)
    cutoff = current - timedelta(hours=72)
    with _locked(path) as state:
        episodes = state["episodes"]
        valid = []
        for message in messages:
            post_id = str(message.get("id", ""))
            if post_id.isdigit():
                valid.append((int(post_id), message))
        valid.sort(key=lambda item: item[0])
        watermark = state.get("last_seen_post_id")
        first_observation = watermark is None
        new_posts = []
        for numeric_id, message in valid:
            if first_observation:
                published = _time(message.get("datetime"))
                if published and cutoff <= published <= current + timedelta(minutes=5):
                    new_posts.append((numeric_id, message))
            elif numeric_id > int(watermark):
                new_posts.append((numeric_id, message))

        added = []
        for numeric_id, message in new_posts:
            post_id = str(numeric_id)
            url = f"https://t.me/Gooaye/{post_id}"
            published_at = message.get("datetime")
            for match in EPISODE_RE.finditer(str(message.get("text", ""))):
                number = str(int(match.group(1)))
                key = f"EP{number}"
                entry = episodes.get(number)
                if entry is None:
                    entry = {
                        "episode": key,
                        "status": "detected",
                        "first_seen": _iso(current),
                        "published_at": published_at,
                        "next_check": _iso(current),
                        "expires_at": None,
                        "attempts": 0,
                        "source_refs": [],
                        "candidate_source": None,
                        "handoff": None,
                        "blocked_reason": None,
                    }
                    episodes[number] = entry
                    added.append(key)
                _refs(entry, {"type": "telegram_post", "post_id": post_id, "url": url, "published_at": published_at})

        if valid:
            state["last_seen_post_id"] = str(max(item[0] for item in valid))
        state["updated_at"] = _iso(current)
        _save(path, state)
        return {"new_episodes": added, "last_seen_post_id": state.get("last_seen_post_id")}


def work(*, now: datetime | None = None, path: Path = STATE_PATH) -> dict:
    current = _now(now)
    due, handoffs, expired, attention = [], [], [], []
    changed = False
    with _locked(path) as state:
        for number, entry in sorted(state["episodes"].items(), key=lambda item: int(item[0])):
            if entry["status"] in {"detected", "pending"}:
                audio = entry.get("audio_pipeline") or {}
                phase = audio.get("phase")
                retry_at = _time(audio.get("retry_after"))
                if phase == "review_required":
                    attention.append(copy.deepcopy(entry))
                elif phase == "retryable" and retry_at and current < retry_at:
                    continue
                else:
                    if entry["status"] == "detected":
                        entry["status"] = "pending"
                        changed = True
                    due.append(copy.deepcopy(entry))
            elif entry["status"] == "handoff_pending":
                handoffs.append(copy.deepcopy(entry))
        if changed:
            state["updated_at"] = _iso(current)
            _save(path, state)
    return {"due": due, "audio_due": due, "handoff_pending": handoffs, "expired": expired,
            "audio_attention": attention, "curation_enabled": CURATION_ENABLED,
            "curation_blocker": None if CURATION_ENABLED else CURATION_BLOCKER}


def get_audio_pipeline(episode: str, *, path: Path = STATE_PATH) -> dict | None:
    number = _episode(episode)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        return copy.deepcopy(entry.get("audio_pipeline")) if entry else None


def save_gooaye_workflow(episode: str, workflow: dict, *, now: datetime | None = None,
                         path: Path = STATE_PATH) -> dict:
    """Persist shared workflow checkpoints under the existing episode-ledger lock."""
    from gooaye_workflow_contract import _copy_state

    number, current = _episode(episode), _now(now)
    checked = _copy_state(workflow)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] in {"blocked", "expired"}:
            raise ValueError("workflow requires an existing unfinished episode")
        prior = entry.get("gooaye_workflow")
        if prior:
            prior = _copy_state(prior)
        if prior and prior.get("identity") != checked.get("identity"):
            raise ValueError("workflow identity conflicts with the existing episode checkpoint")
        if entry["status"] == "processed":
            # Only finalize the existing delivery checkpoint; all processed provenance stays frozen.
            terminal = checked["stages"]["terminal_delivery"]
            previous = prior["stages"]["terminal_delivery"] if prior else {}
            if (not prior or checked["identity"] != prior["identity"]
                    or any(checked["stages"][stage] != prior["stages"][stage]
                           for stage in checked["stages"] if stage != "terminal_delivery")
                    or previous.get("status") not in {"running", "outcome_unknown", "completed"}
                    or terminal.get("operation_id") != previous.get("operation_id")
                    or terminal.get("status") != "completed"
                    or terminal.get("result", {}).get("delivery_status") != "delivered"
                    or terminal.get("result", {}).get("receipt_id") != previous.get("operation_id")):
                raise ValueError("processed workflow permits only matching delivery checkpoint finalization")
        entry["gooaye_workflow"] = checked
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(checked)


def get_episode_record(episode: str, *, path: Path = STATE_PATH) -> dict | None:
    number = _episode(episode)
    with _locked(path) as state:
        return copy.deepcopy(state["episodes"].get(number))


def ledger_snapshot(*, path: Path = STATE_PATH) -> dict:
    with _locked(path) as state:
        return copy.deepcopy(state)


def save_audio_pipeline(episode: str, pipeline: dict, *, now: datetime | None = None,
                        path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    if pipeline.get("phase") not in AUDIO_PHASES:
        raise ValueError("invalid audio pipeline phase")
    source = pipeline.get("source")
    if source is not None:
        if (source.get("episode") != f"EP{number}" or not source.get("guid")
                or "Gooaye" not in str(source.get("feed_title", ""))
                or urlparse(str(source.get("feed_url", ""))).scheme != "https"
                or not urlparse(str(source.get("feed_url", ""))).netloc
                or urlparse(str(source.get("audio_url", ""))).scheme != "https"
                or not urlparse(str(source.get("audio_url", ""))).netloc
                or type(source.get("duration_seconds")) not in {int, float}
                or source["duration_seconds"] <= 0):
            raise ValueError("audio source requires matching episode, Gooaye RSS identity, HTTPS URLs, and duration")
    chunks = pipeline.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("audio chunks must be a list")
    indexes = [chunk.get("index") for chunk in chunks]
    if len(indexes) != len(set(indexes)) or any(type(index) is not int or index < 0 for index in indexes):
        raise ValueError("audio chunks require unique nonnegative indexes")
    if pipeline.get("phase") in {"transcript_ready", "curation_pending"}:
        if (not source or not chunks or any(chunk.get("state") != "completed" for chunk in chunks)
                or not CONTENT_SHA_RE.fullmatch(str(pipeline.get("source_audio_sha256", "")))
                or not CONTENT_SHA_RE.fullmatch(str(pipeline.get("transcript_sha256", "")))
                or not pipeline.get("transcript_path")):
            raise ValueError("a ready transcript requires complete chunks and source/transcript artifacts")
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] in {"blocked", "expired", "processed"}:
            raise ValueError("audio work requires an existing nonterminal episode")
        prior = entry.get("audio_pipeline") or {}
        prior_source = prior.get("source") or {}
        if source and prior_source.get("guid") and prior_source["guid"] != source["guid"]:
            raise ValueError("RSS GUID changed for an existing episode identity")
        if prior.get("phase") in {"transcript_ready", "curation_pending", "processed"} and pipeline.get("phase") not in {"transcript_ready", "curation_pending", "processed"}:
            raise ValueError("completed transcript state cannot move backward")
        new_chunks = {chunk["index"]: chunk for chunk in chunks}
        for old in prior.get("chunks", []):
            if old.get("state") == "completed":
                new = new_chunks.get(old["index"])
                keys = ("audio_sha256", "transcript_sha256", "cost_usd", "generation_id", "offset_seconds", "duration_seconds", "overlap_seconds")
                if not new or new.get("state") != "completed" or any(new.get(key) != old.get(key) for key in keys):
                    raise ValueError("completed audio chunk is immutable; refusing duplicate transcription")
        entry["audio_pipeline"] = copy.deepcopy(pipeline)
        entry["audio_pipeline"]["updated_at"] = _iso(current)
        if source:
            _refs(entry, {"type": "podcast_audio", "url": source["audio_url"], "guid": source["guid"]})
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def accept_audio_source(episode: str, source: dict, *, now: datetime | None = None,
                        path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    digest = str(source.get("transcript_sha256", "")).lower()
    if (source.get("kind") != "asr_generated_transcript" or source.get("transcription_state") != "completed"
            or source.get("episode_match") is not True or not CONTENT_SHA_RE.fullmatch(digest)
            or source.get("model") != "openai/whisper-large-v3"
            or source.get("provider") != "OpenRouter"):
        raise ValueError("audio source requires completed ASR provenance for the approved model/provider")
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        audio = entry.get("audio_pipeline") if entry else None
        if not entry or entry["status"] not in {"detected", "pending", "handoff_pending"} or not audio:
            raise ValueError("audio source can be accepted only for an eligible episode with a ledgered transcript")
        audio_source = audio.get("source") or {}
        if (audio.get("phase") not in {"transcript_ready", "curation_pending"}
                or source.get("rss_guid") != audio_source.get("guid")
                or source.get("audio_url") != audio_source.get("audio_url")
                or source.get("source_audio_sha256") != audio.get("source_audio_sha256")
                or source.get("transcript_path") != audio.get("transcript_path")
                or digest != audio.get("transcript_sha256")):
            raise ValueError("audio source provenance does not match the completed episode transcript")
        expected_dir = (path.parent / "gooaye-audio" / f"EP{number}").resolve()
        transcript_path = Path(source["transcript_path"]).resolve()
        if not transcript_path.is_relative_to(expected_dir) or not transcript_path.is_file():
            raise ValueError("transcript path is outside the episode's private audio artifact directory")
        actual_digest = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
        if actual_digest != digest:
            raise ValueError("transcript artifact hash does not match the ledger")
        candidate = {**source, "verified_at": _iso(current)}
        prior = entry.get("candidate_source")
        if prior:
            if prior.get("kind") == candidate["kind"] and prior.get("transcript_sha256") == digest:
                return copy.deepcopy(entry)
            raise ValueError("another source is already accepted; preserve it and do not replace it")
        entry.update(status="handoff_pending", candidate_source=candidate, next_check=None, blocked_reason=None)
        audio["phase"] = "curation_pending"
        audio["updated_at"] = _iso(current)
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def mark_audio_review_required(episode: str, reason: str, *, now: datetime | None = None,
                              path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] not in {"detected", "pending"}:
            raise ValueError("audio review state requires an unfinished episode")
        audio = entry.setdefault("audio_pipeline", {"source": None, "chunks": []})
        if audio.get("phase") in {"transcript_ready", "curation_pending", "processed"}:
            raise ValueError("completed transcript cannot be moved to review-required state")
        audio.update(phase="review_required", blocked_reason=reason, updated_at=_iso(current))
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def accept_source(episode: str, source: dict, *, now: datetime | None = None, path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    kind = source.get("kind")
    parsed = urlparse(str(source.get("url", "")))
    common = (
        kind in {"human_written_transcript", "reliable_summary"}
        and parsed.scheme == "https" and bool(parsed.netloc)
        and bool(str(source.get("title", "")).strip())
        and bool(str(source.get("author_or_publisher", "")).strip())
        and source.get("episode_match") is True
        and source.get("full_text_verified") is True
        and source.get("substantive") is True
        and source.get("read_route") == "jina.read_url"
        and 20 <= len(str(source.get("evidence_excerpt", "")).strip()) <= 500
    )
    specific = (
        source.get("human_authorship_verified") is True and bool(str(source.get("authorship_basis", "")).strip())
        if kind == "human_written_transcript"
        else source.get("attribution_verified") is True and bool(str(source.get("reliability_basis", "")).strip())
    )
    if not common or not specific:
        raise ValueError("source rejected: require a verified human transcript or attributed substantive summary read via Jina")
    candidate = {**source, "verified_at": _iso(current)}
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] != "pending":
            raise ValueError("source can be accepted only for an eligible pending episode")
        if entry.get("candidate_source"):
            if entry["candidate_source"].get("url") == candidate["url"]:
                return copy.deepcopy(entry)
            raise ValueError("a different accepted source is already recorded")
        entry.update(status="handoff_pending", candidate_source=candidate, next_check=None, blocked_reason=None)
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def seed_legacy(episode: str, status: str, reference: dict, *, path: Path = STATE_PATH) -> dict:
    number = _episode(episode)
    if status not in {"processed", "blocked"}:
        raise ValueError("legacy seed status must be processed or blocked")
    with _locked(path) as state:
        entry = state["episodes"].get(number, {
            "episode": f"EP{number}", "first_seen": None, "published_at": None, "next_check": None,
            "expires_at": None, "attempts": 0, "source_refs": [], "candidate_source": None, "handoff": None,
        })
        entry.update(status=status, next_check=None, candidate_source=None, handoff=None, legacy=True)
        if status == "blocked":
            entry["blocked_reason"] = reference.get("reason", "existing manual-review item")
        else:
            entry["result"] = "legacy_already_processed"
        _refs(entry, {"type": "legacy_vault_pr", "url": reference["url"]})
        state["episodes"][number] = entry
        state["updated_at"] = _iso(datetime.now(timezone.utc))
        _save(path, state)
        return copy.deepcopy(entry)


def record_blocker(episode: str, reason: str, *, now: datetime | None = None, path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] != "handoff_pending":
            raise ValueError("blocker applies only to an accepted source awaiting handoff")
        changed = entry.get("blocked_reason") != reason
        entry["blocked_reason"] = reason
        entry["last_gate_check"] = _iso(current)
        state["updated_at"] = _iso(current)
        _save(path, state)
        return {"changed": changed, "episode": entry["episode"], "blocked_reason": reason}


def stage_handoff(episode: str, data: dict, *, now: datetime | None = None, path: Path = STATE_PATH, enforce_gate: bool = True) -> dict:
    if enforce_gate and not CURATION_ENABLED:
        raise ValueError(CURATION_BLOCKER)
    number, current = _episode(episode), _now(now)
    head = str(data.get("head_sha", "")).lower()
    base = str(data.get("base_sha", "")).lower()
    blob = str(data.get("blob_sha", "")).lower()
    source_path = str(data.get("source_path", ""))
    branch = str(data.get("branch", ""))
    if not all(SHA_RE.fullmatch(x) for x in (head, base, blob)):
        raise ValueError("handoff requires full base, head, and blob SHAs")
    if not source_path.startswith("00 Inbox/Gooaye Podcast/") or not source_path.endswith(".md"):
        raise ValueError("source_path must be a Markdown item under 00 Inbox/Gooaye Podcast/")
    if not branch.startswith(f"hermes/vault-ingest-gooaye-EP{number}"):
        raise ValueError("branch must be a fresh Gooaye ingestion branch for this episode")
    handoff = {"base_sha": base, "head_sha": head, "blob_sha": blob, "source_path": source_path,
               "branch": branch, "identity": ingestion_identity(head), "staged_at": _iso(current)}
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] != "handoff_pending" or not entry.get("candidate_source"):
            raise ValueError("handoff requires an accepted source")
        prior = entry.get("handoff")
        if prior:
            compare = {k: prior.get(k) for k in handoff if k != "staged_at"}
            if compare != {k: v for k, v in handoff.items() if k != "staged_at"}:
                raise ValueError("conflicting handoff already staged; reuse its exact commit and identity")
            return copy.deepcopy(entry)
        entry.update(handoff=handoff, blocked_reason=None)
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def complete_handoff(episode: str, data: dict, *, now: datetime | None = None, path: Path = STATE_PATH, enforce_gate: bool = True) -> dict:
    if enforce_gate and not CURATION_ENABLED:
        raise ValueError(CURATION_BLOCKER)
    number, current = _episode(episode), _now(now)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        handoff = entry.get("handoff") if entry else None
        if not handoff:
            raise ValueError("verified Bridge task requires a staged handoff")
        if entry["status"] not in {"handoff_pending", "processed"}:
            raise ValueError("only an accepted, pending handoff can be completed")
        expected = {"identity": handoff["identity"], "source_path": handoff["source_path"],
                    "source_blob": handoff["blob_sha"], "source_commit": handoff["head_sha"]}
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"Bridge task {key} does not match staged source")
        labels = set(data.get("labels", []))
        if (data.get("terminal_status") != "completed"
                or not {"hermes-task", "completed"}.issubset(labels) or "failed" in labels):
            raise ValueError("Bridge task must have a completed terminal result")
        if data.get("receipt_verified") is not True or data.get("receipt_result") != "processed":
            raise ValueError("exact processed source receipt must be read back from Vault main")
        receipt_fields = {"receipt_source_path": "source_path", "receipt_source_blob": "source_blob",
                          "receipt_source_commit": "source_commit"}
        for field, expected_field in receipt_fields.items():
            if data.get(field) != expected[expected_field]:
                raise ValueError(f"Vault receipt {field} does not match staged source")
        bridge_issue, vault_pr = data.get("bridge_issue"), data.get("vault_pr")
        if (type(bridge_issue) is not int or bridge_issue < 1 or type(vault_pr) is not int or vault_pr < 1
                or data.get("receipt_bridge_issue") != bridge_issue or data.get("receipt_vault_pr") != vault_pr):
            raise ValueError("verified receipt must identify the same Bridge issue and merged Vault PR")
        if (data.get("workflow_name") != "Auto-merge and reconcile Hermes curation"
                or data.get("workflow_conclusion") != "success"):
            raise ValueError("trusted curation and reconciliation workflow must succeed")
        workflow_url = str(data.get("workflow_url", ""))
        pull_request_url = str(data.get("pull_request_url", ""))
        if not re.fullmatch(r"https://github\.com/davetwchiu/david-knowledge-vault/actions/runs/\d+(?:/job/\d+)?", workflow_url):
            raise ValueError("trusted workflow URL must identify a Vault Actions run")
        pr_match = re.fullmatch(r"https://github\.com/davetwchiu/david-knowledge-vault/pull/(\d+)", pull_request_url)
        if not pr_match or int(pr_match.group(1)) != vault_pr or data.get("pull_request_state") != "MERGED":
            raise ValueError("Vault PR URL must identify the merged curation PR")
        done = {**expected, "bridge_issue": bridge_issue, "vault_pr": vault_pr,
                "workflow_url": workflow_url, "pull_request_url": pull_request_url,
                "receipt_result": data["receipt_result"], "receipt_verified": True,
                "completed_at": _iso(current)}
        if entry["status"] == "processed":
            prior = entry.get("handoff_result", {})
            stable = ("identity", "source_path", "source_blob", "source_commit", "bridge_issue", "vault_pr",
                      "workflow_url", "pull_request_url", "receipt_result", "receipt_verified")
            if any(prior.get(key) != done.get(key) for key in stable):
                raise ValueError("episode already processed with a different Bridge result")
            return copy.deepcopy(entry)
        entry.update(status="processed", result="bridge_handoff_verified", next_check=None,
                     handoff_result=done, blocked_reason=None)
        if entry.get("audio_pipeline"):
            entry["audio_pipeline"].update(phase="processed", processed_at=_iso(current))
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def mark_noop(episode: str, reason: str, *, now: datetime | None = None, path: Path = STATE_PATH) -> dict:
    number, current = _episode(episode), _now(now)
    with _locked(path) as state:
        entry = state["episodes"].get(number)
        if not entry or entry["status"] != "handoff_pending" or not entry.get("candidate_source"):
            raise ValueError("no-op completion requires an accepted source")
        entry.update(status="processed", result="no_op", result_reason=reason, processed_at=_iso(current), next_check=None)
        state["updated_at"] = _iso(current)
        _save(path, state)
        return copy.deepcopy(entry)


def status(*, path: Path = STATE_PATH) -> dict:
    with _locked(path) as state:
        counts = {}
        for entry in state["episodes"].values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        audio = {entry["episode"]: {"phase": entry.get("audio_pipeline", {}).get("phase"),
                                     "rss_guid": (entry.get("audio_pipeline", {}).get("source") or {}).get("guid"),
                                     "transcript_sha256": entry.get("audio_pipeline", {}).get("transcript_sha256"),
                                     "cost_usd": entry.get("audio_pipeline", {}).get("cost_usd")}
                 for entry in state["episodes"].values() if entry.get("audio_pipeline")}
        return {"state_path": str(path), "last_seen_post_id": state.get("last_seen_post_id"),
                "episodes": {entry["episode"]: entry["status"] for entry in state["episodes"].values()},
                "counts": counts, "audio": audio, "curation_enabled": CURATION_ENABLED,
                "curation_blocker": None if CURATION_ENABLED else CURATION_BLOCKER}


def self_test() -> None:
    import tempfile

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory(prefix="gooaye-state-canary-") as directory:
        path = Path(directory) / "episodes.json"
        old = (t0 - timedelta(days=4)).isoformat().replace("+00:00", "Z")
        recent = (t0 - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        observe_messages([
            {"id": "100", "datetime": old, "text": "EP998"},
            {"id": "101", "datetime": recent, "text": "Gooaye EP999"},
            {"id": "102", "datetime": recent, "text": "EP999 transcript update"},
        ], now=t0, path=path)
        s = status(path=path)
        assert s["episodes"] == {"EP999": "detected"}
        entry = _load(path)["episodes"]["999"]
        assert entry["first_seen"] == _iso(t0) and entry["next_check"] == _iso(t0) and entry["expires_at"] is None
        assert entry["attempts"] == 0 and len(entry["source_refs"]) == 2
        observe_messages([{"id": "102", "datetime": recent, "text": "EP999"}], now=t0 + timedelta(minutes=1), path=path)
        assert len(_load(path)["episodes"]) == 1
        due = work(now=t0 + timedelta(minutes=1), path=path)["audio_due"]
        assert len(due) == 1 and due[0]["attempts"] == 0
        assert work(now=t0 + timedelta(minutes=1), path=path)["due"] == due
        bad = {"kind": "automatic_transcript", "url": "https://example.org/ep999", "title": "EP999",
               "author_or_publisher": "Example", "episode_match": True, "full_text_verified": True,
               "substantive": True, "read_route": "jina.read_url", "evidence_excerpt": "Auto captions text."}
        try:
            accept_source("EP999", bad, now=t0 + timedelta(hours=12), path=path)
            raise AssertionError("automatic transcript was accepted")
        except ValueError as exc:
            assert "source rejected" in str(exc)
        good = {"kind": "human_written_transcript", "url": "https://example.org/ep999", "title": "EP999 transcript",
                "author_or_publisher": "Example editor", "episode_match": True, "full_text_verified": True,
                "human_authorship_verified": True, "authorship_basis": "Byline on the synthetic page.",
                "substantive": True, "read_route": "jina.read_url",
                "evidence_excerpt": "A verified human-written transcript excerpt for synthetic EP999."}
        accepted = accept_source("EP999", good, now=t0 + timedelta(hours=12), path=path)
        assert accepted["status"] == "handoff_pending"
        assert work(now=t0 + timedelta(hours=18), path=path)["due"] == []
        handoff = {"base_sha": "a" * 40, "head_sha": "b" * 40, "blob_sha": "c" * 40,
                   "source_path": "00 Inbox/Gooaye Podcast/EP999.md", "branch": "hermes/vault-ingest-gooaye-EP999"}
        first = stage_handoff("EP999", handoff, now=t0 + timedelta(hours=18), path=path)
        second = stage_handoff("EP999", handoff, now=t0 + timedelta(hours=18), path=path)
        assert first["handoff"]["identity"] == ingestion_identity("b" * 40)
        assert second["handoff"]["identity"] == first["handoff"]["identity"]
        complete = {"identity": ingestion_identity("b" * 40), "source_path": handoff["source_path"],
                    "source_blob": "c" * 40, "source_commit": "b" * 40, "bridge_issue": 999999,
                    "labels": ["hermes-task", "completed"], "terminal_status": "completed",
                    "workflow_name": "Auto-merge and reconcile Hermes curation",
                    "workflow_conclusion": "success", "workflow_url": "https://github.com/davetwchiu/david-knowledge-vault/actions/runs/999999",
                    "pull_request_url": "https://github.com/davetwchiu/david-knowledge-vault/pull/999999", "vault_pr": 999999,
                    "pull_request_state": "MERGED",
                    "receipt_verified": True, "receipt_result": "processed", "receipt_source_path": handoff["source_path"],
                    "receipt_source_blob": "c" * 40, "receipt_source_commit": "b" * 40,
                    "receipt_bridge_issue": 999999, "receipt_vault_pr": 999999}
        queued = {**complete, "labels": ["hermes-task", "queued"]}
        try:
            complete_handoff("EP999", queued, path=path)
            raise AssertionError("queued ingestion was treated as completed curation")
        except ValueError as exc:
            assert "terminal result" in str(exc)
        assert complete_handoff("EP999", complete, path=path)["status"] == "processed"
        assert complete_handoff("EP999", complete, path=path)["handoff_result"]["bridge_issue"] == 999999
        observe_messages([{"id": "103", "datetime": recent, "text": "EP1000"}], now=t0 + timedelta(minutes=1), path=path)
        work(now=t0 + timedelta(minutes=1), path=path)
        overdue = work(now=t0 + timedelta(hours=73), path=path)
        assert [x["episode"] for x in overdue["audio_due"]] == ["EP1000"] and overdue["expired"] == []
        assert status(path=path)["episodes"]["EP1000"] == "pending"
        observe_messages([{"id": "104", "datetime": recent, "text": "EP1001"}], now=t0 + timedelta(minutes=2), path=path)
        summary_due = work(now=t0 + timedelta(minutes=2), path=path)["due"]
        assert {x["episode"] for x in summary_due} == {"EP1000", "EP1001"}
        summary = {"kind": "reliable_summary", "url": "https://example.org/ep1001", "title": "EP1001 notes",
                   "author_or_publisher": "Example official show notes", "episode_match": True,
                   "full_text_verified": True, "attribution_verified": True, "substantive": True,
                   "reliability_basis": "Official show notes with named publisher.",
                   "read_route": "jina.read_url", "evidence_excerpt": "A substantive, attributed synthetic summary of EP1001."}
        assert accept_source("EP1001", summary, now=t0 + timedelta(hours=12, minutes=2), path=path)["status"] == "handoff_pending"

        observe_messages([{"id": "105", "datetime": recent, "text": "EP1002"}], now=t0 + timedelta(minutes=3), path=path)
        audio_dir = path.parent / "gooaye-audio" / "EP1002"
        audio_dir.mkdir(parents=True)
        transcript_path = audio_dir / "transcript.md"
        transcript_path.write_text("# Synthetic private ASR transcript\n\n測試文字。\n", encoding="utf-8")
        transcript_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
        audio_source = {"episode": "EP1002", "guid": "guid-1002", "feed_title": "Gooaye 股癌",
                        "feed_url": "https://feeds.soundon.fm/test.xml", "audio_url": "https://rss.soundon.fm/test.mp3",
                        "title": "EP1002 | synthetic", "duration_seconds": 240}
        completed_chunk = {"index": 0, "state": "completed", "audio_sha256": "a" * 64,
                           "transcript_sha256": "b" * 64, "cost_usd": 0.001,
                           "generation_id": "generation-1", "offset_seconds": 0, "duration_seconds": 240}
        pipeline = {"phase": "transcript_ready", "source": audio_source,
                    "source_audio_sha256": "c" * 64, "transcript_path": str(transcript_path),
                    "transcript_sha256": transcript_hash, "cost_usd": 0.001, "chunks": [completed_chunk]}
        save_audio_pipeline("EP1002", pipeline, now=t0 + timedelta(minutes=3), path=path)
        asr_source = {"kind": "asr_generated_transcript", "transcription_state": "completed",
                      "episode_match": True, "model": "openai/whisper-large-v3", "provider": "OpenRouter",
                      "rss_guid": "guid-1002", "audio_url": audio_source["audio_url"],
                      "source_audio_sha256": "c" * 64, "transcript_path": str(transcript_path),
                      "transcript_sha256": transcript_hash}
        accepted_audio = accept_audio_source("EP1002", asr_source, now=t0 + timedelta(minutes=3), path=path)
        assert accepted_audio["status"] == "handoff_pending"
        assert accept_audio_source("EP1002", asr_source, now=t0 + timedelta(minutes=4), path=path)["candidate_source"]["transcript_sha256"] == transcript_hash
        observe_messages([{"id": "106", "datetime": recent, "text": "EP1003"}], now=t0 + timedelta(minutes=5), path=path)
        mark_audio_review_required("EP1003", "synthetic uncertain request", now=t0 + timedelta(minutes=5), path=path)
        review = work(now=t0 + timedelta(minutes=5), path=path)
        assert [x["episode"] for x in review["audio_attention"]] == ["EP1003"]
        assert not any(x["episode"] == "EP1003" for x in review["audio_due"])
        assert CURATION_ENABLED
    print("Gooaye episode-state canaries passed; no Vault or GitHub writes made.")


def _json_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("work")
    sub.add_parser("self-test")
    for command in ("accept-source", "accept-audio", "stage-handoff", "complete-handoff", "noop", "blocker"):
        item = sub.add_parser(command)
        item.add_argument("episode")
        item.add_argument("input_json")
    legacy = sub.add_parser("seed-legacy")
    legacy.add_argument("episode")
    legacy.add_argument("status", choices=("processed", "blocked"))
    legacy.add_argument("input_json")
    args = parser.parse_args()
    if args.command == "status":
        result = status()
    elif args.command == "work":
        result = work()
    elif args.command == "self-test":
        self_test()
        return
    elif args.command == "accept-source":
        result = accept_source(args.episode, _json_file(args.input_json))
    elif args.command == "accept-audio":
        result = accept_audio_source(args.episode, _json_file(args.input_json))
    elif args.command == "stage-handoff":
        result = stage_handoff(args.episode, _json_file(args.input_json))
    elif args.command == "complete-handoff":
        result = complete_handoff(args.episode, _json_file(args.input_json))
    elif args.command == "noop":
        result = mark_noop(args.episode, _json_file(args.input_json).get("reason", "no durable material"))
    elif args.command == "blocker":
        result = record_blocker(args.episode, _json_file(args.input_json).get("reason", "pipeline unavailable"))
    else:
        result = seed_legacy(args.episode, args.status, _json_file(args.input_json))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
