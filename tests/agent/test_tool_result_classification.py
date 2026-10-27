"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    file_mutation_result_landed,
    trusted_gooaye_cycle_outcome,
)


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True






def test_side_effect_classification_keeps_session_mutations():
    from agent.tool_result_classification import tool_may_have_side_effect

    assert tool_may_have_side_effect("todo") is True
    assert tool_may_have_side_effect("memory") is True
    assert tool_may_have_side_effect("write_file") is True
    assert tool_may_have_side_effect("mcp_unknown") is True
    assert tool_may_have_side_effect("read_file") is False
    assert tool_may_have_side_effect("web_search") is False


def test_trusted_gooaye_cycle_requires_exact_command_and_versioned_terminal_result(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    command = (f"{tmp_path}/current/venv/bin/python "
               f"{tmp_path}/current/scripts/gooaye/gooaye_audio_first.py cycle")
    cycle = {
        "outcome_contract": "gooaye-cycle/v1",
        "status": "cycle_attention",
        "discovered": ["EP708"],
        "results": [{"episode": "EP708", "status": "blocked", "stage": "trusted_receipt"}],
        "audio_attention": [],
    }
    envelope = json.dumps({"output": json.dumps(cycle), "exit_code": 1, "error": None})

    outcome = trusted_gooaye_cycle_outcome("terminal", {"command": command}, envelope)

    assert outcome == {
        "status": "cycle_attention",
        "final_response": "Gooaye cycle completed with item(s) requiring attention: EP708 (trusted_receipt).",
    }


def test_trusted_gooaye_cycle_ignores_untrusted_output_nonzero_shells_and_other_commands(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    command = (f"{tmp_path}/current/venv/bin/python "
               f"{tmp_path}/current/scripts/gooaye/gooaye_audio_first.py cycle")
    arbitrary = json.dumps({"output": '{"status":"cycle_attention"}', "exit_code": 1, "error": None})
    valid_cycle = json.dumps({
        "output": json.dumps({"outcome_contract": "gooaye-cycle/v1", "status": "cycle_complete",
                              "discovered": [], "results": [], "audio_attention": []}),
        "exit_code": 1, "error": None,
    })

    assert trusted_gooaye_cycle_outcome("terminal", {"command": command}, arbitrary) is None
    assert trusted_gooaye_cycle_outcome(
        "terminal", {"command": command},
        json.dumps({"output": json.loads(valid_cycle)["output"], "exit_code": 1, "error": None,
                    "output_total_chars": 1000, "full_output_path": "/tmp/spill"}),
    ) is None
    assert trusted_gooaye_cycle_outcome("terminal", {"command": "echo done"}, valid_cycle) is None
    assert trusted_gooaye_cycle_outcome("terminal", {"command": command + "; echo done"}, valid_cycle) is None


def test_trusted_gooaye_cycle_reports_retryable_outcome(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    command = (f"{tmp_path}/current/venv/bin/python "
               f"{tmp_path}/current/scripts/gooaye/gooaye_audio_first.py cycle")
    cycle = {
        "outcome_contract": "gooaye-cycle/v1",
        "status": "cycle_attention",
        "discovered": ["EP708"],
        "results": [{"episode": "EP708", "status": "retryable"}],
        "audio_attention": [],
    }
    envelope = json.dumps({"output": json.dumps(cycle), "exit_code": 1, "error": None})

    assert trusted_gooaye_cycle_outcome("terminal", {"command": command}, envelope) == {
        "status": "cycle_attention",
        "final_response": "Gooaye cycle completed with item(s) requiring attention: EP708 (retryable).",
    }
