"""Tests for tools/cronjob_job_args.py::_validate_cron_script_path (issue #105761).

Regression coverage: the validator used to accept a script path that pointed
at a file that doesn't exist, only failing later at every cron fire with a
generic "Script not found" error from cron/scheduler_script.py. It also
hardcoded "~/.hermes/scripts/" in its messages even though resolution goes
through get_hermes_home(), which is per-profile.
"""

from hermes_constants import get_hermes_home
from tools.cronjob_job_args import _validate_cron_script_path


class TestValidateCronScriptPath:
    def test_missing_script_file_is_rejected_at_creation_time(self):
        error = _validate_cron_script_path("does_not_exist.sh")
        assert error is not None
        assert "not found" in error.lower()

    def test_existing_script_file_passes(self):
        scripts_dir = get_hermes_home() / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "real.sh").write_text("#!/bin/sh\necho hi\n")
        assert _validate_cron_script_path("real.sh") is None

    def test_missing_file_error_names_the_resolved_scripts_dir(self):
        # The resolved dir must appear literally so the message stays correct
        # under profiles, where get_hermes_home() is not the global ~/.hermes.
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("missing.py")
        assert str(scripts_dir) in error

    def test_absolute_path_error_names_the_resolved_scripts_dir_not_global_literal(self):
        scripts_dir = get_hermes_home() / "scripts"
        error = _validate_cron_script_path("/etc/passwd")
        assert str(scripts_dir) in error

    def test_active_immutable_gooaye_release_script_is_allowed(self, monkeypatch, tmp_path):
        root = tmp_path / ".hermes"
        release = root / "deployments" / f"hermes-agent-{'a' * 40}"
        script = release / "scripts" / "gooaye" / "monitor.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/bash\necho ok\n")
        (root / "current").symlink_to(release, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(root))

        path = root / "current" / "scripts" / "gooaye" / "monitor.sh"
        assert _validate_cron_script_path(str(path)) is None

    def test_stale_release_script_is_not_allowed(self, monkeypatch, tmp_path):
        root = tmp_path / ".hermes"
        active = root / "deployments" / f"hermes-agent-{'a' * 40}"
        stale = root / "deployments" / f"hermes-agent-{'b' * 40}"
        active.mkdir(parents=True)
        script = stale / "scripts" / "gooaye" / "monitor.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/bash\necho stale\n")
        (root / "current").symlink_to(active, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(root))

        assert _validate_cron_script_path(str(script)) is not None

    def test_gooaye_release_script_is_not_shared_with_named_profiles(self, monkeypatch, tmp_path):
        root = tmp_path / ".hermes"
        release = root / "deployments" / f"hermes-agent-{'a' * 40}"
        script = release / "scripts" / "gooaye" / "monitor.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/bash\necho ok\n")
        (root / "current").symlink_to(release, target_is_directory=True)
        profile = root / "profiles" / "finance"
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))

        assert _validate_cron_script_path(str(script)) is not None
