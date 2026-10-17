"""Small fail-closed guards for David's Hermes production source-control policy."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath


def _live_root(live_root: Path | None = None) -> Path:
    return (live_root or Path.home() / ".hermes" / "hermes-agent").expanduser().resolve()


def in_place_update_refusal(project_root: Path | None, live_root: Path | None = None) -> str | None:
    """Refuse the existing updater's in-place mutation of the live editable checkout."""
    if project_root is None:
        return "Hermes Guardian: the updater cannot resolve its source checkout; refusing to update."
    try:
        is_live = Path(project_root).expanduser().resolve() == _live_root(live_root)
    except (OSError, RuntimeError):
        return "Hermes Guardian: source paths cannot be verified; refusing to update."
    if is_live:
        return (
            "Hermes Guardian: refusing in-place update of the live editable checkout. "
            "The existing updater cannot stage and activate an immutable release."
        )
    return None


def _is_live_path(path: str, cwd: str | None, live_root: Path) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        if not cwd:
            return False
        candidate = Path(cwd).expanduser() / candidate
    candidate = candidate.resolve()
    return candidate == live_root or live_root in candidate.parents


def _live_reference(event: dict, live_root: Path) -> bool:
    cwd = event.get("cwd")
    try:
        if cwd and _is_live_path(str(cwd), None, live_root):
            return True
    except (OSError, RuntimeError):
        return True

    # Hook inputs are strings and paths, not shell ASTs. Literal aliases are covered;
    # shortcut: symlink/variable indirection is outside this hook, use OS-level ACLs to stop other writers.
    aliases = (
        str(live_root), "~/.hermes/hermes-agent", "$HOME/.hermes/hermes-agent",
        "${HOME}/.hermes/hermes-agent",
    )
    try:
        text = json.dumps(event.get("tool_input", {}), ensure_ascii=False, default=str)
    except Exception:
        return True
    return any(alias in text for alias in aliases)


def _readonly_shell(command: str) -> bool:
    if any(mark in command for mark in (";", "|", "&", ">", "<", "`", "$", "\n")):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    if executable in {"ls", "cat", "head", "tail", "stat", "file", "rg", "grep"}:
        return True
    if executable != "git":
        return False
    if any(
        token == "--output" or token.startswith(("--output=", "--ext-diff", "--textconv"))
        for token in tokens
    ):
        return False
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-C":
            index += 2
        elif token == "--no-pager":
            index += 1
        elif token.startswith("-"):
            return False
        else:
            return token in {"status", "diff", "log", "show", "rev-parse", "ls-files"}
    return False


def _apply_patch_targets_live(command: str, cwd: str | None, live_root: Path) -> bool:
    targets = re.findall(r"(?m)^\*\*\* (?:Update|Delete|Add) File:\s*(.+?)\s*$", command)
    targets += [p for p in re.findall(r"(?m)^\+\+\+ [ab]/(.+)$", command) if p != "/dev/null"]
    targets += [p for p in re.findall(r"(?m)^--- [ab]/(.+)$", command) if p != "/dev/null"]
    try:
        return any(_is_live_path(path, cwd, live_root) for path in targets)
    except (OSError, RuntimeError):
        return True


def _update_command(command: str) -> bool:
    return bool(re.search(
        r"(?i)(?:^|\s)(?:[^\s;&|]*/)?hermes(?:\.exe)?\s+(?:(?:-p|--profile)\s+\S+\s+)?update\b"
        r"|(?:^|\s)(?:[^\s;&|]*/)?python(?:\d(?:\.\d+)*)?\s+-m\s+hermes_cli(?:\.main)?\s+update\b",
        command,
    ))


def pretool_refusal(event: dict, live_root: Path | None = None) -> str | None:
    """Return a denial for supported local tool calls that can mutate the live checkout."""
    live = _live_root(live_root)
    tool_name = str(event.get("tool_name", ""))
    tool_input = event.get("tool_input", {})
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if not isinstance(command, str):
        command = ""

    if tool_name == "apply_patch":
        return "Hermes Guardian: edits to the live Hermes checkout are blocked." if _apply_patch_targets_live(
            command, event.get("cwd"), live
        ) else None

    if tool_name in {"Bash", "exec_command"}:
        update = _update_command(command)
        if update:
            if re.search(r"--(?:check|plan)\b", command) and not any(
                mark in command for mark in (";", "|", "&", ">", "<", "`", "$", "\n")
            ):
                return None
            return "Hermes Guardian: direct `hermes update` activation is blocked; no immutable release gate is available."
        if not _live_reference(event, live):
            return None
        if _readonly_shell(command):
            return None
        return "Hermes Guardian: local tool call may modify the live Hermes checkout; use a clean development checkout."

    if not _live_reference(event, live):
        return None
    if re.search(r"(?:^|__)(?:read_[a-z0-9_]+|list_[a-z0-9_]+|search_[a-z0-9_]+|get_file_info)$", tool_name.lower()):
        return None
    return "Hermes Guardian: local tool call targets the live Hermes checkout; use a clean development checkout."


_SHA = re.compile(r"^[0-9a-f]{40}$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _changed_new_lines(patch: str | None) -> set[int]:
    changed: set[int] = set()
    line: int | None = None
    for row in (patch or "").splitlines():
        hunk = _HUNK.match(row)
        if hunk:
            line = int(hunk.group(1))
        elif line is not None:
            if row.startswith("+") and not row.startswith("+++"):
                changed.add(line)
                line += 1
            elif row.startswith("-") and not row.startswith("---"):
                continue
            elif row.startswith("\\"):
                continue
            else:
                line += 1
    return changed


def _documentation_section(body: str) -> str | None:
    match = re.search(
        r"(?ims)^##\s+Documentation impact\s*$([\s\S]*?)(?=^##\s+|\Z)", body
    )
    return match.group(1) if match else None


def pull_request_issues(event: dict, api_files: list) -> list[str]:
    """Check trusted PR metadata and the required documentation-impact declaration."""
    issues: list[str] = []
    pr = event.get("pull_request") or {}
    base = pr.get("base") or {}
    head = pr.get("head") or {}
    if (event.get("repository") or {}).get("full_name") != "davetwchiu/hermes-agent":
        issues.append("This check only accepts PRs targeting davetwchiu/hermes-agent.")
    if base.get("ref") != "main":
        issues.append("Production changes must target main.")
    if not _SHA.fullmatch(str(base.get("sha", ""))) or not _SHA.fullmatch(str(head.get("sha", ""))):
        issues.append("The PR must identify full immutable base and head commit SHAs.")

    if not isinstance(api_files, list):
        return issues + ["GitHub did not return a complete PR file list."]
    files = [row for page in api_files for row in page] if api_files and isinstance(api_files[0], list) else api_files
    paths = [row.get("filename") for row in files if isinstance(row, dict)]
    expected_count = pr.get("changed_files")
    if len(paths) != len(files) or not paths or len(set(paths)) != len(paths):
        issues.append("The PR file list is empty, incomplete, or contains duplicate paths.")
    if not isinstance(expected_count, int) or len(paths) != expected_count:
        issues.append("The PR file list does not match GitHub's changed-file count.")
    if any(
        not isinstance(path, str) or path.startswith("/") or "\\" in path
        or ".." in PurePosixPath(path).parts
        for path in paths
    ):
        issues.append("The PR contains a non-repository or unsafe file path.")
    if issues:
        return issues

    body = pr.get("body") or ""
    section = _documentation_section(body)
    if section is None:
        return ["Add a `## Documentation impact` section to the PR description."]

    docs = {p for p in paths if p.lower().endswith((".md", ".mdx"))}
    behavior = {p for p in paths if p not in docs and not p.startswith("tests/")}
    updated_line = re.search(r"(?im)^\s*(?:[-*]\s*)?Updated:\s*(.+)$", section)
    if docs:
        listed = set(re.findall(r"`([^`]+)`", updated_line.group(1))) if updated_line else set()
        if not docs.issubset(listed):
            issues.append("List every changed Markdown file after `Updated:` in Documentation impact.")
    elif behavior:
        no_update = re.search(r"(?im)^\s*(?:[-*]\s*)?No update:\s*(.+)$", section)
        evidence = re.findall(r"`([^`#]+)#L([1-9]\d*)`", section)
        changed = {row["filename"]: _changed_new_lines(row.get("patch")) for row in files}
        rationale = no_update.group(1) if no_update else ""
        if len(rationale.strip()) < 20 or not any(
            path in behavior and int(line) in changed.get(path, set()) for path, line in evidence
        ):
            issues.append("Without a Markdown update, give a no-change rationale and a changed-line evidence reference such as `path/file.py#L42`.")
    else:
        no_update = re.search(r"(?im)^\s*(?:[-*]\s*)?No update:\s*(.+)$", section)
        evidence = re.findall(r"`([^`#]+)#L([1-9]\d*)`", section)
        changed = {row["filename"]: _changed_new_lines(row.get("patch")) for row in files}
        rationale = no_update.group(1) if no_update else ""
        if len(rationale.strip()) < 20 or not any(
            path.startswith("tests/") and int(line) in changed.get(path, set()) for path, line in evidence
        ):
            issues.append("For tests-only changes, explain why no runtime documentation changes and cite a changed test line.")
    return issues


def _read_pr_files(path: str) -> list:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--check-pr":
        if len(args) != 3:
            print("::error::Hermes Guardian PR checker needs the event and PR-files JSON paths.")
            return 2
        event = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        issues = pull_request_issues(event, _read_pr_files(args[2]))
        if issues:
            for issue in issues:
                print(f"::error::{issue}")
            return 1
        print("Hermes Guardian: source-control identity and documentation impact are complete.")
        return 0
    if args:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Hermes Guardian: invalid hook invocation; refusing the tool request.",
        }}))
        return 0
    try:
        event = json.load(sys.stdin)
        reason = pretool_refusal(event)
    except Exception:
        reason = "Hermes Guardian: could not validate the local tool request; refusing it."
    if reason:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
