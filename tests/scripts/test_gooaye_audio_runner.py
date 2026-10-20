import copy
import hashlib
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gooaye"))

import gooaye_audio_first as runner  # noqa: E402
import gooaye_episode_state as ledger  # noqa: E402
import gooaye_workflow_contract as contract  # noqa: E402


STAGES = contract.STAGES[2:]


def test_runtime_pins_match_the_release_lock_and_canonical_vault_helper():
    repo_root = Path(__file__).resolve().parents[2]
    lock = (repo_root / "uv.lock").read_bytes()
    assert hashlib.sha256(lock).hexdigest() == runner.RUNTIME_PINS["python"]["uv_lock_sha256"]

    helper = runner._reconcile_helper_path().read_bytes()
    assert hashlib.sha256(helper).hexdigest() == runner.RUNTIME_PINS["vault_helper"]["sha256"]
    git_blob = hashlib.sha1(b"blob " + str(len(helper)).encode() + b"\0" + helper).hexdigest()
    assert git_blob == runner.RUNTIME_PINS["vault_helper"]["git_blob_sha"]
    assert runner.RUNTIME_PINS["vault_helper"]["source_commit"] == "32b5b5e81f9ea2018123ceafc11fb6712d067e43"


def test_pinned_runtime_executable_checks_version_and_binary(monkeypatch, tmp_path):
    executable = tmp_path / "node"
    binary = b"pinned fake executable"
    executable.write_bytes(binary)
    monkeypatch.setitem(runner.RUNTIME_PINS["executables"], "node", {
        "version": "v1.2.3", "sha256": hashlib.sha256(binary).hexdigest(),
    })
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout="v1.2.3\n", stderr=""))

    assert runner._pinned_executable("node") == str(executable)
    runner.RUNTIME_PINS["executables"]["node"]["sha256"] = "0" * 64
    with pytest.raises(runner.AudioIssue, match="release lock"):
        runner._pinned_executable("node")


def test_ledger_state_path_follows_each_profile_home(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_dir = repo_root / "scripts" / "gooaye"
    homes = [tmp_path / "profiles" / name for name in ("a", "b", "a")]
    for home in homes:
        env = {
            "HOME": str(tmp_path),
            "HERMES_HOME": str(home),
            "PYTHONPATH": os.pathsep.join((str(repo_root), str(script_dir))),
        }
        result = subprocess.run(
            [sys.executable, "-c", "import gooaye_episode_state as s; print(s.STATE_PATH)"],
            cwd=repo_root, env=env, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == str(home / "state" / "gooaye-podcast-episodes.json")


class UnknownOutcome(RuntimeError):
    pass


class Adapter:
    def __init__(self):
        self.calls = []
        self.receipts = {}
        self.outcomes = {}

    def execute(self, *, stage, episode, operation_id, state, context):
        self.calls.append((stage, episode, operation_id))
        outcome = self.outcomes.get(stage)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome or {
            "artifact_hash": context.get("expected_hash") or f"hash-{stage}-{episode}",
            "external_id": f"external-{stage}-{episode}",
        }

    def reconcile(self, *, stage, episode, operation_id, state, context):
        return self.receipts.get(stage, {"status": "unknown"})


def setup_episode(path, episode):
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    ledger.observe_messages([{
        "id": str(6400 + int(episode)), "datetime": now.isoformat(), "text": f"Gooaye EP{episode}"
    }], now=now, path=path)
    metadata = {
        "episode_number": f"EP{episode}", "rss_guid": f"guid-{episode}",
        "enclosure_url": f"https://rss.soundon.fm/{episode}.mp3",
        "telegram_post_id": str(6400 + int(episode)),
    }
    state = contract.new_state(metadata)
    for stage in contract.STAGES[:2]:
        state = contract.start_stage(state, stage, f"test-{stage}-{episode}")
        state = contract.complete_stage(state, stage, artifact_hash=f"hash-{stage}-{episode}")
    ledger.save_gooaye_workflow(episode, state, path=path)
    return metadata


def test_runner_advances_all_shared_stages_per_episode_and_is_idempotent(tmp_path):
    state_path = tmp_path / "episodes.json"
    adapters = {stage: Adapter() for stage in STAGES}
    for episode in ("701", "702", "703"):
        metadata = setup_episode(state_path, episode)
        result = runner.run_shared_stages(episode, metadata, adapters, state_path=state_path)
        assert result["status"] == "done"
        repeated = runner.run_shared_stages(episode, metadata, adapters, state_path=state_path)
        assert repeated["status"] == "done"
    for stage, adapter in adapters.items():
        assert [call[1] for call in adapter.calls] == ["701", "702", "703"]
        assert len({call[2] for call in adapter.calls}) == 3


def test_production_brief_adapter_receives_source_metadata_from_shared_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_original_job", lambda: {"provider": "openai-codex", "model": "gpt-6-luna"})
    monkeypatch.setattr(runner, "_generate_edited_brief", lambda text: "faithful edited brief")
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "710")
    transcript = tmp_path / "transcript.md"
    transcript.write_text("# EP710\n\n完整的測試逐字稿。\n", encoding="utf-8")
    raw = transcript.read_bytes()
    metadata.update(transcript_path=str(transcript),
                    transcript_sha256=runner.hashlib.sha256(raw).hexdigest())
    brief_adapter = runner.production_stage_adapters()["transcript_brief"]

    result = runner.run_shared_stages(
        "710", metadata, {"transcript_brief": brief_adapter}, state_path=state_path)

    assert result["status"] == "blocked"
    assert result["stage"] == "publication"
    saved = ledger.get_episode_record("710", path=state_path)["gooaye_workflow"]
    brief = saved["stages"]["transcript_brief"]
    assert brief["status"] == "completed"
    assert brief["result"]["brief_provenance"]["source_transcript_sha256"] == metadata["transcript_sha256"]


def test_unknown_publication_is_reconciled_after_restart_without_new_head(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "704")
    adapters = {stage: Adapter() for stage in STAGES}
    adapters["publication"].outcomes["publication"] = UnknownOutcome("connection dropped after publish")

    first = runner.run_shared_stages("704", metadata, adapters, state_path=state_path)
    assert first["status"] == "outcome_unknown"
    publication_calls = len(adapters["publication"].calls)

    # A new runner invocation must reconcile the durable operation before doing work.
    restarted = runner.run_shared_stages("704", metadata, adapters, state_path=state_path)
    assert restarted["status"] == "outcome_unknown"
    assert len(adapters["publication"].calls) == publication_calls

    operation_id = adapters["publication"].calls[0][2]
    adapters["publication"].receipts["publication"] = {
        "status": "completed", "receipt": {
            "logical_id": contract.derive_identity(metadata)["logical_id"],
            "operation_id": operation_id, "external_id": "source-head-704",
            "receipt_id": "source-commit-704", "content_hash": f"hash-publication-704",
        }
    }
    recovered = runner.run_shared_stages("704", metadata, adapters, state_path=state_path)
    assert recovered["status"] == "done"
    assert len(adapters["publication"].calls) == publication_calls


def test_production_publication_reconciliation_preserves_exact_handoff_fields(tmp_path, monkeypatch):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "711")
    workflow = ledger.get_episode_record("711", path=state_path)["gooaye_workflow"]
    workflow = contract.start_stage(workflow, "transcript_brief", "brief-op-711")
    workflow = contract.complete_stage(workflow, "transcript_brief", artifact_hash="brief-hash-711",
                                      external_id="transcript-711")
    workflow["stages"]["transcript_brief"]["result"] = {
        "brief_provenance": {"kind": "edited_brief", "text": "brief 711",
                              "source_transcript_sha256": "a" * 64},
    }
    workflow = contract.start_stage(
        workflow, "publication", "publication-op-711", operation_id="publication-op-711")
    workflow = contract.mark_outcome_unknown(workflow, "publication")
    ledger.save_gooaye_workflow("711", workflow, path=state_path)

    raw = b"exact source bytes 711"
    base_sha, head_sha, blob_sha = "b" * 40, "a" * 40, "c" * 40
    ref = {"object": {"sha": head_sha}}
    source = {"sha": blob_sha, "encoding": "base64",
              "content": runner.base64.b64encode(raw).decode("ascii")}
    monkeypatch.setattr(runner, "_find_source_branch", lambda episode: ref)
    monkeypatch.setattr(runner, "_read_source_at_ref", lambda episode, branch: source)
    monkeypatch.setattr(runner, "_source_markdown", lambda *args: raw)
    monkeypatch.setattr(runner, "source_metadata", lambda episode, **kwargs: metadata)
    monkeypatch.setattr(runner, "_github_api", lambda *args, **kwargs: {"parents": [{"sha": base_sha}]})

    with ledger._locked(state_path) as snapshot:
        snapshot["episodes"]["711"].update(status="handoff_pending", candidate_source={"kind": "asr_generated_transcript"})
        ledger._save(state_path, snapshot)
    monkeypatch.setattr(runner, "_existing_publication", lambda episode: None)
    result = runner.run_shared_stages(
        "711", metadata, {"publication": runner.production_stage_adapters()["publication"]},
        state_path=state_path)

    assert result["status"] == "blocked"
    saved = ledger.get_episode_record("711", path=state_path)["gooaye_workflow"]
    publication = saved["stages"]["publication"]["result"]
    assert {key: publication.get(key) for key in (
        "base_sha", "head_sha", "blob_sha", "source_path", "branch", "ingestion_identity"
    )} == {
        "base_sha": base_sha, "head_sha": head_sha, "blob_sha": blob_sha,
        "source_path": "00 Inbox/Gooaye Podcast/EP711.md",
        "branch": "hermes/vault-ingest-gooaye-EP711",
        "ingestion_identity": f"vault-ingest:{head_sha}",
    }


def test_unknown_dispatch_never_creates_a_second_source_head(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "705")
    adapters = {stage: Adapter() for stage in STAGES}
    adapters["dispatch"].outcomes["dispatch"] = UnknownOutcome("workflow API timed out")

    result = runner.run_shared_stages("705", metadata, adapters, state_path=state_path)
    assert result["status"] == "outcome_unknown"
    publication_calls = len(adapters["publication"].calls)
    dispatched_calls = len(adapters["dispatch"].calls)
    again = runner.run_shared_stages("705", metadata, adapters, state_path=state_path)
    assert again["status"] == "outcome_unknown"
    assert len(adapters["publication"].calls) == publication_calls == 1
    assert len(adapters["dispatch"].calls) == dispatched_calls == 1


def test_mismatched_receipt_fails_closed_without_repeating_external_call(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "706")
    adapters = {stage: Adapter() for stage in STAGES}
    adapters["publication"].outcomes["publication"] = UnknownOutcome("uncertain")
    runner.run_shared_stages("706", metadata, adapters, state_path=state_path)
    op = adapters["publication"].calls[0][2]
    adapters["publication"].receipts["publication"] = {
        "status": "completed", "receipt": {
            "logical_id": "gooaye:wrong", "operation_id": op, "external_id": "wrong-head",
            "receipt_id": "wrong-receipt", "content_hash": "wrong-hash",
        }
    }
    with pytest.raises(contract.ReceiptConflict):
        runner.run_shared_stages("706", metadata, adapters, state_path=state_path)
    assert len(adapters["publication"].calls) == 1


def test_confirmed_absence_allows_only_same_episode_retry_and_unrelated_dirty_uses_clean_route(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "707")
    adapters = {stage: Adapter() for stage in STAGES}
    adapters["publication"].outcomes["publication"] = UnknownOutcome("uncertain")
    runner.run_shared_stages("707", metadata, adapters, state_path=state_path)
    adapters["publication"].receipts["publication"] = {"status": "absent", "proof": "exact-source-lookup:absent"}
    adapters["publication"].outcomes.pop("publication")
    result = runner.run_shared_stages(
        "707", metadata, adapters, state_path=state_path,
        dirty_paths=["unrelated.md"], relevant_paths=["00 Inbox/Gooaye Podcast/EP707.md"],
    )
    assert result["status"] == "done"
    assert result["publication_route"] == "use_existing_clean_publisher"
    assert len(adapters["publication"].calls) == 2
    assert adapters["publication"].calls[0][2] == adapters["publication"].calls[1][2]


def test_same_episode_concurrent_trigger_is_locked(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "708")
    adapters = {stage: Adapter() for stage in STAGES}
    entered = threading.Event()
    release = threading.Event()
    original = adapters["transcript_brief"].execute

    def blocked_execute(**kwargs):
        entered.set()
        release.wait(timeout=2)
        return original(**kwargs)

    adapters["transcript_brief"].execute = blocked_execute
    first_result = []
    thread = threading.Thread(target=lambda: first_result.append(
        runner.run_shared_stages("708", metadata, adapters, state_path=state_path)))
    thread.start()
    assert entered.wait(timeout=2)
    concurrent = runner.run_shared_stages("708", metadata, adapters, state_path=state_path)
    release.set()
    thread.join(timeout=2)
    assert concurrent["status"] == "busy"
    assert first_result[0]["status"] == "done"
    assert len(adapters["transcript_brief"].calls) == 1


def test_contract_conflict_does_not_mutate_other_episode(tmp_path):
    state_path = tmp_path / "episodes.json"
    first_metadata = setup_episode(state_path, "709")
    second_metadata = setup_episode(state_path, "710")
    adapters = {stage: Adapter() for stage in STAGES}
    wrong = copy.deepcopy(first_metadata)
    wrong["rss_guid"] = "other-guid"
    with pytest.raises(contract.IdentityConflict):
        runner.run_shared_stages("709", wrong, adapters, state_path=state_path)
    assert all(not adapter.calls for adapter in adapters.values())
    assert ledger.get_episode_record("710", path=state_path)["episode"] == "EP710"


def test_run_entrypoint_invokes_shared_transaction(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "_run_episode_transaction", lambda episode, **kwargs: (
        calls.append((episode, kwargs)) or {"status": "done", "episode": episode}))
    monkeypatch.setattr(runner.sys, "argv", ["gooaye_audio_first.py", "run", "EP711"])
    assert runner.main() == 0
    assert calls == [("EP711", {"max_chunks": None})]
    assert '"status": "done"' in capsys.readouterr().out


def test_unstructured_merge_comment_cannot_substitute_for_canonical_trusted_receipt(tmp_path, monkeypatch):
    identity = {"logical_id": "gooaye:episode-identity"}
    issue = {"number": 91, "state": "closed", "state_reason": "completed",
             "labels": ["hermes-task", "completed"], "body": "exact task body"}
    weak_comment = {"user": {"login": "davetwchiu"},
                    "body": "reason=merged_with_receipt\\nVault PR: https://github.com/davetwchiu/david-knowledge-vault/pull/92"}
    publication = {"source_path": "00 Inbox/Gooaye Podcast/EP711.md",
                   "blob_sha": "c" * 40, "head_sha": "a" * 40,
                   "base_sha": "b" * 40, "ingestion_identity": "vault-ingest:" + "a" * 40}
    monkeypatch.setattr(runner, "_find_bridge_task", lambda _identity: issue)
    monkeypatch.setattr(runner, "_bridge_issue_comments", lambda _task: [weak_comment])
    monkeypatch.setattr(runner, "_run_vault_receipt_helpers", lambda **_kwargs: {"completion": None})
    monkeypatch.setattr(runner, "_github_api", lambda *args, **kwargs: pytest.fail("weak comment reached remote verification"))

    with pytest.raises(runner.StageBlocked, match="trusted owner completion result is not yet present"):
        runner._verified_trusted_receipt("711", identity, publication)


def test_production_adapter_factory_covers_every_post_audio_stage():
    adapters = runner.production_stage_adapters()
    assert set(adapters) == set(STAGES)
    assert all(callable(adapter.execute) and callable(adapter.reconcile) for adapter in adapters.values())


def test_retryable_stage_block_is_not_misreported_as_unknown_external_write(tmp_path):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "712")
    adapters = {stage: Adapter() for stage in STAGES}
    adapters["transcript_brief"].outcomes["transcript_brief"] = runner.StageBlocked(
        "model brief artifact missing", retryable=True)

    result = runner.run_shared_stages("712", metadata, adapters, state_path=state_path)
    assert result["status"] == "retryable"
    workflow = ledger.get_episode_record("712", path=state_path)["gooaye_workflow"]
    assert workflow["stages"]["transcript_brief"]["status"] == "failed"
    assert workflow["stages"]["transcript_brief"]["retryable"] is True


def test_known_pending_trusted_receipt_keeps_typed_blocker_evidence(tmp_path, monkeypatch):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "722")
    workflow = ledger.get_episode_record("722", path=state_path)["gooaye_workflow"]
    publication = {
        "artifact_hash": "d" * 64,
        "external_id": "h" * 40,
        "base_sha": "b" * 40,
        "head_sha": "h" * 40,
        "blob_sha": "c" * 40,
        "source_path": "00 Inbox/Gooaye Podcast/EP722.md",
        "branch": "hermes/vault-ingest-gooaye-EP722",
        "ingestion_identity": "vault-ingest:" + "h" * 40,
    }
    for stage, result in (("transcript_brief", {"artifact_hash": "a" * 64}),
                          ("publication", publication),
                          ("dispatch", {"artifact_hash": "c" * 40, "external_id": "317"})):
        operation_id = f"{stage}-op-722"
        workflow = contract.start_stage(workflow, stage, operation_id, operation_id=operation_id)
        workflow = contract.complete_stage(
            workflow, stage, artifact_hash=result["artifact_hash"],
            external_id=result.get("external_id"))
        workflow["stages"][stage]["result"] = result
    workflow = contract.start_stage(workflow, "trusted_receipt", "trusted-op-722",
                                    operation_id="trusted-op-722")
    workflow = contract.mark_outcome_unknown(workflow, "trusted_receipt")
    ledger.save_gooaye_workflow("722", workflow, path=state_path)

    issue = {"number": 317, "state": "open", "state_reason": "",
             "labels": ["hermes-task", "queued"]}
    monkeypatch.setattr(runner, "_find_bridge_task", lambda _identity: issue)

    result = runner.run_shared_stages(
        "722", metadata, {"trusted_receipt": runner.production_stage_adapters()["trusted_receipt"]},
        state_path=state_path)

    assert result["status"] == "outcome_pending"
    assert result["reconciliation"] == "pending"
    assert result["stage"] == "trusted_receipt"
    assert result["reason"] == "exact Bridge task remains queued/running"
    assert result["evidence"] == {
        "kind": "bridge_task_pending", "task_id": 317, "state": "open",
        "state_reason": "", "labels": ["hermes-task", "queued"],
        "ingestion_identity": publication["ingestion_identity"],
        "stage": "trusted_receipt", "operation_id": "trusted-op-722",
    }


def test_edited_brief_uses_full_transcript_and_reuses_verified_artifact(tmp_path, monkeypatch):
    transcript = tmp_path / "transcript.md"
    transcript.write_text("# EP713\n" + "完整逐字稿。" * 300, encoding="utf-8")
    metadata = {"transcript_path": str(transcript),
                "transcript_sha256": runner.hashlib.sha256(transcript.read_bytes()).hexdigest()}
    calls = []
    monkeypatch.setattr(runner, "_original_job", lambda: {"provider": "original", "model": "original"})
    monkeypatch.setattr(runner, "_generate_edited_brief", lambda text: (calls.append(text) or "edited brief"))
    brief = runner._brief_artifact("713", metadata)
    assert runner._brief_artifact("713", metadata) == brief
    assert calls == [transcript.read_text()]
    assert brief["kind"] == "edited_brief"
    transcript.write_text("changed", encoding="utf-8")
    with pytest.raises(runner.StageBlocked, match="hash differs"):
        runner._brief_artifact("713", metadata)


def test_cycle_drains_durable_due_and_candidate_handoff_without_repeat_asr(tmp_path, monkeypatch):
    state_path = tmp_path / "episodes.json"
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for number in (714, 715):
        ledger.observe_messages([{"id": str(number + 6000), "datetime": now.isoformat(),
                                  "text": f"Gooaye EP{number}"}], now=now, path=state_path)

    def save_transcript(number, phase="transcript_ready"):
        audio_dir = tmp_path / "gooaye-audio" / f"EP{number}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript = audio_dir / "transcript.md"
        transcript.write_text(f"# EP{number}\n\n測試逐字稿內容。\n", encoding="utf-8")
        digest = runner.hashlib.sha256(transcript.read_bytes()).hexdigest()
        source = {"episode": f"EP{number}", "guid": f"guid-{number}",
                  "feed_title": "Gooaye 股癌", "feed_url": "https://feeds.soundon.fm/test.xml",
                  "audio_url": f"https://rss.soundon.fm/{number}.mp3", "title": f"EP{number}",
                  "duration_seconds": 240}
        pipeline = {"phase": phase, "source": source, "source_audio_sha256": "a" * 64,
                    "transcript_path": str(transcript), "transcript_sha256": digest,
                    "cost_usd": 0.01, "model": "openai/whisper-large-v3", "provider": "OpenRouter",
                    "chunks": [{"index": i, "state": "completed", "audio_sha256": f"{i:064x}",
                                "transcript_sha256": digest, "cost_usd": 0.001,
                                "generation_id": f"generation-{i}", "offset_seconds": i * 240,
                                "duration_seconds": 240} for i in range(13 if number == 715 else 1)]}
        ledger.save_audio_pipeline(str(number), pipeline, now=now, path=state_path)
        return source, transcript, digest

    handoff_source, handoff_transcript, handoff_digest = save_transcript(715)
    accepted = {"kind": "asr_generated_transcript", "transcription_state": "completed",
                "episode_match": True, "model": "openai/whisper-large-v3", "provider": "OpenRouter",
                "rss_guid": handoff_source["guid"], "audio_url": handoff_source["audio_url"],
                "source_audio_sha256": "a" * 64, "transcript_path": str(handoff_transcript),
                "transcript_sha256": handoff_digest}
    ledger.accept_audio_source("EP715", accepted, now=now, path=state_path)

    transcribe_calls, workflow_calls = [], []
    def fake_transcribe(episode, max_chunks=None, **kwargs):
        transcribe_calls.append(episode)
        if episode == "714":
            save_transcript(714)
        return {"status": "transcript_ready", "episode": f"EP{episode}"}
    monkeypatch.setattr(runner, "transcribe", fake_transcribe)
    monkeypatch.setattr(runner, "source_metadata", lambda episode, **kwargs: {"episode_number": f"EP{episode}"})
    monkeypatch.setattr(runner, "production_stage_adapters", lambda: {stage: object() for stage in STAGES})
    monkeypatch.setattr(runner, "run_shared_stages", lambda episode, metadata, adapters, **kwargs: (
        workflow_calls.append(episode) or {"status": "done", "episode": f"EP{episode}"}
    ))

    first = runner.run_cycle(state_path=state_path, now=now)
    second = runner.run_cycle(state_path=state_path, now=now)

    assert first["status"] == second["status"] == "cycle_complete"
    assert transcribe_calls == ["714"]
    assert workflow_calls == ["714", "715", "714", "715"]
    assert [x["episode"] for x in first["audio_attention"]] == []
    assert first["discovered"] == ["EP714", "EP715"]


def test_cycle_cli_entrypoint_invokes_shared_cycle(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "run_cycle", lambda **kwargs: (
        calls.append(kwargs) or {"status": "cycle_complete", "discovered": []}
    ))
    monkeypatch.setattr(runner.sys, "argv", ["gooaye_audio_first.py", "cycle"])

    assert runner.main() == 0
    assert calls == [{"max_chunks": None}]
    assert '"status": "cycle_complete"' in capsys.readouterr().out


def completed_audio_fixture(state_path, number):
    now = datetime.now(timezone.utc)
    ledger.observe_messages([{"id": str(6400 + int(number)), "datetime": now.isoformat(),
                              "text": f"Gooaye EP{number}"}], now=now, path=state_path)
    folder = state_path.parent / "gooaye-audio" / f"EP{number}"
    folder.mkdir(parents=True)
    transcript = folder / "transcript.md"
    transcript.write_text(f"# EP{number}\n\n完整逐字稿內容。", encoding="utf-8")
    digest = runner.hashlib.sha256(transcript.read_bytes()).hexdigest()
    source = {"episode": f"EP{number}", "guid": f"guid-{number}", "title": f"EP{number}",
              "audio_url": f"https://rss.soundon.fm/{number}.mp3", "feed_url": "https://example.org/feed.xml",
              "duration_seconds": 240, "feed_title": "Gooaye 股癌"}
    ledger.save_audio_pipeline(number, {
        "phase": "transcript_ready", "source": source, "source_audio_sha256": "a" * 64,
        "transcript_path": str(transcript), "transcript_sha256": digest, "cost_usd": 0.01,
        "provider": "OpenRouter", "model": runner.MODEL,
        "chunks": [{"index": 0, "state": "completed", "transcript_sha256": digest,
                    "generation_id": f"paid-{number}", "cost_usd": 0.01}],
    }, path=state_path)


class FakeGitHub:
    def __init__(self, lost=None):
        self.refs = {}; self.files = {}; self.tasks = []; self.calls = []; self.lost = lost
        self.parents = {}; self.counter = 20

    def api(self, method, endpoint, payload=None, **kwargs):
        self.calls.append((method, endpoint))
        if method == "GET" and "hermes-bridge/issues?" in endpoint:
            return self.tasks
        if method == "GET" and "git/ref/heads/main" in endpoint:
            return {"object": {"sha": "b" * 40}}
        if method == "GET" and "git/ref/heads/" in endpoint:
            branch = endpoint.split("git/ref/heads/")[1]
            return {"object": {"sha": self.refs[branch]}} if branch in self.refs else None
        if method == "GET" and "contents/" in endpoint:
            from urllib.parse import unquote
            path, ref = endpoint.split("contents/")[1].split("?ref=")
            return self.files.get((unquote(path), unquote(ref)))
        if method == "GET" and "/commits/" in endpoint:
            return {"parents": [{"sha": self.parents[endpoint.rsplit('/', 1)[1]]}]}
        if method == "POST" and endpoint.endswith("git/refs"):
            self.refs[payload["ref"].removeprefix("refs/heads/")] = payload["sha"]
            if self.lost == "branch":
                self.lost = None; raise UnknownOutcome("lost branch response")
            return {}
        if method == "PUT" and "contents/" in endpoint:
            from urllib.parse import unquote
            path = unquote(endpoint.split("contents/")[1]); branch = payload["branch"]
            self.counter += 1; head = f"{self.counter:040x}"
            raw = runner.base64.b64decode(payload["content"])
            blob = runner.hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            self.parents[head] = self.refs[branch]; self.refs[branch] = head
            value = {"sha": blob, "encoding": "base64", "content": payload["content"]}
            self.files[(path, branch)] = value; self.files[(path, head)] = value
            if self.lost == "publication":
                self.lost = None; raise UnknownOutcome("lost publication response")
            return {"commit": {"sha": head}, "content": {"sha": blob}}
        if method == "POST" and "dispatches" in endpoint:
            head = payload["inputs"]["head_sha"]
            self.tasks.append({"number": 500 + len(self.tasks), "body":
                f"<!-- vault-ingest:v1 identity=vault-ingest:{head} -->"})
            if self.lost == "dispatch":
                self.lost = None; raise UnknownOutcome("lost dispatch response")
            return None
        raise AssertionError((method, endpoint))


def trusted_fixture(episode, identity, publication):
    task, pr = 500, 600
    fields = {"identity": publication["ingestion_identity"], "source_path": publication["source_path"],
              "source_blob": publication["blob_sha"], "source_commit": publication["head_sha"],
              "bridge_issue": task, "vault_pr": pr, "terminal_status": "completed",
              "labels": ["hermes-task", "completed"], "receipt_result": "processed", "receipt_verified": True,
              "receipt_source_path": publication["source_path"], "receipt_source_blob": publication["blob_sha"],
              "receipt_source_commit": publication["head_sha"], "receipt_bridge_issue": task, "receipt_vault_pr": pr,
              "workflow_name": "Auto-merge and reconcile Hermes curation", "workflow_conclusion": "success",
              "workflow_url": "https://github.com/davetwchiu/david-knowledge-vault/actions/runs/700",
              "pull_request_url": "https://github.com/davetwchiu/david-knowledge-vault/pull/600",
              "pull_request_state": "MERGED"}
    return {"artifact_hash": "d" * 64, "content_hash": "d" * 64, "external_id": str(task),
            "receipt_id": "trusted-receipt", "receipt_verified": True, "handoff": fields,
            "bridge_issue": task, "pull_request_url": fields["pull_request_url"]}


@pytest.mark.parametrize("number,lost", [("701", "branch"), ("702", "publication"), ("703", "dispatch")])
def test_production_adapters_resume_unknown_effects_and_deliver_once(tmp_path, monkeypatch, number, lost):
    state_path = tmp_path / "episodes.json"
    completed_audio_fixture(state_path, number)
    github = FakeGitHub(lost)
    monkeypatch.setattr(runner, "_github_api", github.api)
    monkeypatch.setattr(runner, "_original_job", lambda: {"id": "950814801e2b", "provider": "original",
        "model": "original", "deliver": "origin", "origin": {"platform": "telegram", "chat_id": "test"}})
    monkeypatch.setattr(runner, "_generate_edited_brief", lambda text: "完整的編輯摘要")
    monkeypatch.setattr(runner, "transcribe", lambda *a, **k: pytest.fail("completed ASR repeated"))
    from cron import delivery_queue
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    ready = []
    def trusted(*args):
        if not ready:
            raise runner.StageBlocked("receipt pending", retryable=True)
        return trusted_fixture(*args)
    monkeypatch.setattr(runner, "_verified_trusted_receipt", trusted)
    first = runner.run_cycle(state_path=state_path)
    assert first["results"][0]["status"] == "outcome_unknown"
    second = runner.run_cycle(state_path=state_path)
    assert second["results"][0]["status"] == "retryable"
    for _ in range(4):
        assert runner.run_cycle(state_path=state_path)["results"][0]["status"] == "outcome_unknown"
    ready.append(True)
    pending = runner.run_cycle(state_path=state_path)
    assert pending["results"][0]["status"] == "outcome_unknown"
    assert ledger.get_episode_record(number, path=state_path)["status"] == "handoff_pending"
    sends = []
    assert delivery_queue.drain(lambda job, text, failure: sends.append((job["origin"], text)) or None) == 1
    # Simulate death after delivered ledger update but before workflow checkpoint completion.
    original_save = ledger.save_gooaye_workflow
    def crash_after_ledger(episode, workflow, **kwargs):
        if workflow["stages"]["terminal_delivery"]["status"] == "completed":
            raise OSError("death before final checkpoint")
        return original_save(episode, workflow, **kwargs)
    monkeypatch.setattr(ledger, "save_gooaye_workflow", crash_after_ledger)
    runner.run_cycle(state_path=state_path)
    assert ledger.get_episode_record(number, path=state_path)["status"] == "processed"
    monkeypatch.setattr(ledger, "save_gooaye_workflow", original_save)
    assert runner.run_cycle(state_path=state_path)["results"][0]["status"] == "done"
    assert runner.run_cycle(state_path=state_path)["discovered"] == []
    assert delivery_queue.drain(lambda *args: pytest.fail("duplicate delivery")) == 0
    assert len(sends) == 1
    assert len([c for c in github.calls if c[0] == "PUT"]) == 1
    assert len([c for c in github.calls if c[0] == "POST" and "dispatches" in c[1]]) == 1
    entry = ledger.get_episode_record(number, path=state_path)
    assert entry["audio_pipeline"]["chunks"][0]["generation_id"] == f"paid-{number}"
    assert entry["gooaye_workflow"]["stages"]["terminal_delivery"]["result"]["delivery_status"] == "delivered"


def test_deleted_source_branch_recovers_existing_exact_bridge_material(tmp_path, monkeypatch):
    state_path = tmp_path / "episodes.json"
    metadata = setup_episode(state_path, "720")
    github = FakeGitHub()
    raw = b"immutable source"
    material = {"path": "00 Inbox/Gooaye Podcast/EP720.md", "base_sha": "b" * 40,
                "head_sha": "c" * 40, "blob_sha": "d" * 40}
    github.tasks = [{"number": 500, "body": "```json\n" + runner.json.dumps([material]) + "\n```"}]
    github.files[(material["path"], material["head_sha"])] = {
        "sha": material["blob_sha"], "encoding": "base64", "content": runner.base64.b64encode(raw).decode()}
    monkeypatch.setattr(runner, "_github_api", github.api)
    monkeypatch.setattr(runner, "_source_markdown", lambda *args: raw)
    result = runner._publish_raw_source("720", metadata, contract.derive_identity(metadata), {}, state_path=state_path)
    assert result["head_sha"] == material["head_sha"]
    assert all(method == "GET" for method, _ in github.calls)
    monkeypatch.setattr(runner, "_source_markdown", lambda *args: b"conflicting bytes")
    with pytest.raises(runner.StageBlocked, match="conflicts"):
        runner._publish_raw_source("720", metadata, contract.derive_identity(metadata), {}, state_path=state_path)


def test_real_receipt_validator_checks_matching_ledger_and_curated_target(monkeypatch):
    identity = {"logical_id": "gooaye:exact"}
    material = {"path": "00 Inbox/Gooaye Podcast/EP721.md", "base_sha": "b" * 40,
                "head_sha": "c" * 40, "blob_sha": "d" * 40}
    publication = {"source_path": material["path"], "base_sha": material["base_sha"],
                   "head_sha": material["head_sha"], "blob_sha": material["blob_sha"],
                   "ingestion_identity": "vault-ingest:" + material["head_sha"]}
    issue = {"number": 500, "state": "open", "state_reason": "",
             "labels": ["hermes-task", "completed"],
             "body": "### Exact changed material\n```json\n" + runner.json.dumps([material]) + "\n```"}
    comment = {"created_at": "2026-10-01T00:00:00Z", "user": {"login": "davetwchiu"}, "body":
        "<!-- vault-e2e:pr=600 reason=merged_with_receipt -->\n<!-- hermes-result:v1 -->\n"
        "Status: completed\nTask ID: 500\n"
        "- Vault PR: https://github.com/davetwchiu/david-knowledge-vault/pull/600\n"
        "- The curation PR is merged into `main`.\n"
        "- Every exact immutable source path/blob from the bridge task has a matching processed receipt on `main`.\n"
        "- The receipt records Hermes task #500 and Vault PR #600.\nLimitations:\n- None."}
    target = "30 Research/Gooaye EP721.md"
    receipt = {"path": material["path"], "source_commit": material["head_sha"],
               "blob_sha": material["blob_sha"], "status": "processed", "result": "created",
               "task_id": 500, "pr": 600, "outputs": [target]}
    vault_state = {"items": {material["path"] + "#" + material["blob_sha"]: receipt}}
    def content(raw, sha):
        return {"sha": sha, "encoding": "base64", "content": runner.base64.b64encode(raw).decode()}
    source = content(b"Episode identity: `gooaye:exact`", material["blob_sha"])
    target_content = content(b"curated note", "e" * 40)
    def api(method, endpoint, **kwargs):
        assert method == "GET"
        if "/pulls/600" in endpoint:
            return {"state": "closed", "merged": True, "merge_commit_sha": "f" * 40,
                    "created_at": "2026-09-30T00:00:00Z", "merged_at": "2026-10-02T00:00:00Z",
                    "head": {"sha": "a" * 40}, "body": "davetwchiu/hermes-bridge#500"}
        if "/commits/" in endpoint:
            return {"parents": [{"sha": "a" * 40}]}
        if "INGESTION_STATE.json" in endpoint:
            return content(runner.json.dumps(vault_state).encode(), "1" * 40)
        if "hermes-curation-auto-merge.yml/runs" in endpoint:
            return {"workflow_runs": [{"id": 700, "name": "Auto-merge and reconcile Hermes curation",
                    "status": "completed", "conclusion": "success", "head_sha": "a" * 40,
                    "event": "workflow_dispatch", "head_branch": "main",
                    "created_at": "2026-10-01T00:00:00Z", "pull_requests": []}]}
        return target_content
    monkeypatch.setattr(runner, "_github_api", api)
    monkeypatch.setattr(runner, "_find_bridge_task", lambda _: issue)
    monkeypatch.setattr(runner, "_bridge_issue_comments", lambda _: [comment])
    monkeypatch.setattr(runner, "_read_source_at_ref", lambda *args: source)
    result = runner._verified_trusted_receipt("721", identity, publication)
    assert result["receipt_result"] == "processed"
    assert result["curation_result"] == "created"
    assert result["target_readback"] == {target: "e" * 40}
    receipt["blob_sha"] = "9" * 40
    with pytest.raises(runner.StageBlocked, match="does not match|mismatch"):
        runner._verified_trusted_receipt("721", identity, publication)
