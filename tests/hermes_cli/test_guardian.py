from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from hermes_cli import guardian


def test_pretool_blocks_live_checkout_edit_and_allows_clean_worktree_edit(tmp_path):
    live = tmp_path / "live" / "hermes-agent"
    work = tmp_path / "work" / "hermes-agent"
    live.mkdir(parents=True)
    work.mkdir(parents=True)
    dirty = live / "untracked.py"
    dirty.write_text("dirty", encoding="utf-8")

    blocked = guardian.pretool_refusal({
        "tool_name": "apply_patch",
        "cwd": str(tmp_path),
        "tool_input": {"command": f"*** Begin Patch\n*** Update File: {dirty}\n+bad\n*** End Patch"},
    }, live_root=live)
    allowed = guardian.pretool_refusal({
        "tool_name": "apply_patch",
        "cwd": str(tmp_path),
        "tool_input": {"command": f"*** Begin Patch\n*** Update File: {work / 'agent.py'}\n+good\n*** End Patch"},
    }, live_root=live)

    assert blocked and "live Hermes checkout" in blocked
    assert allowed is None


def test_dirty_live_checkout_is_refused_before_update_side_effects(tmp_path):
    live = tmp_path / "live" / "hermes-agent"
    live.mkdir(parents=True)
    (live / "untracked.py").write_text("dirty", encoding="utf-8")
    assert "refusing in-place update" in guardian.in_place_update_refusal(live, live_root=live)

    source = Path("hermes_cli/update_cmd.py").read_text(encoding="utf-8")
    update_body = source.split("def _cmd_update_impl(args, gateway_mode: bool):", 1)[1]
    assert update_body.index("in_place_update_refusal(_m().PROJECT_ROOT)") < update_body.index(
        "_begin_update_receipt_and_plan(args)"
    )
    assert "sys.exit(2)" in update_body.split("opts = _resolve_update_options", 1)[0]


def test_clean_isolated_worktree_is_not_the_live_update_target(tmp_path):
    live = tmp_path / "live" / "hermes-agent"
    clean = tmp_path / "releases" / "abc123"
    live.mkdir(parents=True)
    clean.mkdir(parents=True)
    assert guardian.in_place_update_refusal(clean, live_root=live) is None


def test_pretool_blocks_activation_but_allows_read_only_update_plan(tmp_path):
    cwd = str(tmp_path / "work")
    deploy = guardian.pretool_refusal({
        "tool_name": "Bash", "cwd": cwd, "tool_input": {"command": "hermes update"},
    })
    plan = guardian.pretool_refusal({
        "tool_name": "Bash", "cwd": cwd, "tool_input": {"command": "hermes update --plan"},
    })
    assert deploy and "activation is blocked" in deploy
    assert plan is None


def test_hook_cli_emits_codex_deny_response(monkeypatch, tmp_path, capsys):
    live = tmp_path / "live" / "hermes-agent"
    live.mkdir(parents=True)
    monkeypatch.setattr(guardian, "_live_root", lambda _root=None: live)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": str(live),
        "tool_input": {"command": "git reset --hard"},
    })))

    assert guardian.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pr_cli_accepts_slurped_github_api_files(tmp_path, capsys):
    event_path = tmp_path / "event.json"
    files_path = tmp_path / "files.json"
    event_path.write_text(json.dumps(_pr(
        "## Documentation impact\nUpdated: `website/docs/cli.md`\n"
    )), encoding="utf-8")
    files_path.write_text(json.dumps([[
        {"filename": "hermes_cli/guardian.py", "patch": "@@ -1,0 +1 @@\n+code"},
        {"filename": "website/docs/cli.md", "patch": "@@ -1,0 +1 @@\n+doc"},
    ]]), encoding="utf-8")

    assert guardian.main(["--check-pr", str(event_path), str(files_path)]) == 0
    assert "source-control identity" in capsys.readouterr().out


def _pr(body: str, *, ref: str = "main", files: int = 2):
    return {
        "repository": {"full_name": "davetwchiu/hermes-agent"},
        "pull_request": {
            "base": {"ref": ref, "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "changed_files": files,
            "body": body,
        },
    }


def test_pr_check_accepts_matching_docs_and_full_commit_ids():
    files = [
        {"filename": "hermes_cli/guardian.py", "patch": "@@ -1,0 +1,2 @@\n+one\n+two"},
        {"filename": "website/docs/developer-guide/cli-internals.md", "patch": "@@ -1,0 +1 @@\n+doc"},
    ]
    body = "## Documentation impact\nUpdated: `website/docs/developer-guide/cli-internals.md`\n"
    assert guardian.pull_request_issues(_pr(body), files) == []


def test_pr_check_rejects_missing_docs_or_weak_no_change_rationale():
    files = [{"filename": "hermes_cli/update_cmd.py", "patch": "@@ -1,0 +1 @@\n+guard"}]
    missing = _pr("", files=1)
    weak = _pr("## Documentation impact\nNo update: not needed.\n", files=1)
    assert guardian.pull_request_issues(missing, files)
    assert guardian.pull_request_issues(weak, files)


def test_pr_check_requires_no_change_evidence_on_a_changed_line():
    files = [{"filename": "tests/hermes_cli/test_guardian.py", "patch": "@@ -1,0 +40,2 @@\n+test\n+assert"}]
    body = "## Documentation impact\nNo update: tests only; runtime behavior is unchanged. Evidence: `tests/hermes_cli/test_guardian.py#L41`\n"
    assert guardian.pull_request_issues(_pr(body, files=1), files) == []
    bad = body.replace("#L41", "#L99")
    assert guardian.pull_request_issues(_pr(bad, files=1), files)


def test_pr_check_requires_canonical_main_and_matching_file_count():
    files = [{"filename": "README.md", "patch": "@@ -1,0 +1 @@\n+doc"}]
    body = "## Documentation impact\nUpdated: `README.md`\n"
    assert guardian.pull_request_issues(_pr(body, ref="release", files=1), files)
    assert guardian.pull_request_issues(_pr(body, files=2), files)
