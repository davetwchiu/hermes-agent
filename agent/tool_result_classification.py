"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hermes session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


# Set by a tool that REFUSED a call the harness judged redundant (repeated identical
# read/search). The body still carries ``"error"`` so the model reads it as a stop
# signal, but nothing failed: failure classifiers must not count it, or the cheap
# refusal feeds the streak that fires ``repeated_exact_failure_block``.
GUARDRAIL_REFUSAL_KEY = "guardrail_refusal"
_GOOAYE_CYCLE_CONTRACT = "gooaye-cycle/v1"
_GOOAYE_EPISODE_ID = re.compile(r"^EP[0-9]{3,}$")
_GOOAYE_RESULT_STATUSES = frozenset({
    "blocked", "busy", "done", "expired", "failed", "handoff_pending", "missing",
    "outcome_pending", "outcome_unknown", "processed", "retryable", "review_required",
})
_GOOAYE_ATTENTION_STATUSES = frozenset({
    "blocked", "failed", "missing", "outcome_pending", "outcome_unknown", "retryable", "review_required",
})


def is_guardrail_refusal(result: Any) -> bool:
    """Return True when ``result`` (JSON string or parsed dict) is a harness refusal."""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result.strip())
        except Exception:
            return False
    return isinstance(data, dict) and data.get(GUARDRAIL_REFUSAL_KEY) is True


def trusted_gooaye_cycle_outcome(tool_name: str, args: Any, result: Any) -> dict[str, str] | None:
    """Recognize only the exact deterministic cycle command and its versioned JSON result."""
    if tool_name != "terminal" or not isinstance(args, dict):
        return None
    command = args.get("command")
    if not isinstance(command, str):
        return None
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        if shlex.split(command) != [
            str(home / "current/venv/bin/python"),
            str(home / "current/scripts/gooaye/gooaye_audio_first.py"),
            "cycle",
        ]:
            return None
        envelope = json.loads(result) if isinstance(result, str) else None
        if (not isinstance(envelope, dict) or "error" not in envelope or "output" not in envelope
                or envelope.get("error") is not None or envelope.get("stdout_truncated") is True
                or any(key in envelope for key in ("output_total_chars", "full_output_path", "truncation_note"))
                or type(envelope.get("exit_code")) is not int or not isinstance(envelope.get("output"), str)):
            return None
        cycle = json.loads(envelope["output"])
    except (ImportError, TypeError, ValueError):
        return None
    if not isinstance(cycle, dict) or cycle.get("outcome_contract") != _GOOAYE_CYCLE_CONTRACT:
        return None
    if cycle.get("status") not in {"busy", "cycle_complete", "cycle_attention"}:
        return None
    results = cycle.get("results")
    attention = cycle.get("audio_attention")
    discovered = cycle.get("discovered")
    if not (isinstance(results, list) and isinstance(attention, list) and isinstance(discovered, list)):
        return None
    if any(not isinstance(item, dict) or item.get("status") not in _GOOAYE_RESULT_STATUSES
           or not _GOOAYE_EPISODE_ID.fullmatch(str(item.get("episode") or "")) for item in results):
        return None
    if any(not isinstance(item, dict) or not _GOOAYE_EPISODE_ID.fullmatch(str(item.get("episode") or ""))
           for item in attention):
        return None
    if any(not isinstance(episode, str) or not _GOOAYE_EPISODE_ID.fullmatch(episode) for episode in discovered):
        return None

    result_episodes = [item["episode"] for item in results]
    if len(result_episodes) != len(set(result_episodes)) or set(result_episodes) != set(discovered):
        return None
    hard_failure = bool(attention) or any(item["status"] in _GOOAYE_ATTENTION_STATUSES for item in results)
    if (cycle["status"] == "cycle_attention") != hard_failure:
        return None
    if envelope["exit_code"] != int(hard_failure):
        return None

    blockers = [
        f"{item['episode']} ({item.get('stage') or item['status']})"
        for item in results if item["status"] in _GOOAYE_ATTENTION_STATUSES
    ]
    blockers.extend(
        f"{item['episode']} (audio review required)" for item in attention
    )
    message = (
        "Gooaye cycle completed with item(s) requiring attention: " + ", ".join(blockers) + "."
        if blockers else "[SILENT]"
    )
    return {"status": str(cycle["status"]), "final_response": message}


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
