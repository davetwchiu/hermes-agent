"""Shared path boundary for cron scripts."""

from __future__ import annotations

import re
from pathlib import Path

from hermes_constants import get_default_hermes_root, get_hermes_home

_RELEASE_NAME = re.compile(r"hermes-agent-[0-9a-f]{40}")


def active_gooaye_release_scripts_dir() -> Path | None:
    """The Gooaye script directory in the active immutable deployment, if well formed."""
    try:
        root = get_default_hermes_root()
        if get_hermes_home().resolve(strict=False) != root.resolve(strict=False):
            return None
        current = root / "current"
        deployments = (root / "deployments").resolve(strict=True)
        if not current.is_symlink():
            return None
        release = current.resolve(strict=True)
        if release.parent != deployments or not _RELEASE_NAME.fullmatch(release.name):
            return None
        return release / "scripts" / "gooaye"
    except (OSError, RuntimeError, ValueError):
        return None


def is_active_gooaye_release_script(path: Path) -> bool:
    """Allow only resolved files from ``current``'s full-SHA Hermes deployment."""
    release_scripts = active_gooaye_release_scripts_dir()
    if release_scripts is None:
        return False
    try:
        Path(path).resolve(strict=False).relative_to(release_scripts.resolve(strict=False))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def is_allowed_cron_script(path: Path, scripts_dir: Path) -> bool:
    """Keep profile scripts contained; narrowly permit the active release's Gooaye scripts."""
    try:
        resolved = Path(path).resolve(strict=False)
        resolved.relative_to(Path(scripts_dir).resolve(strict=False))
        return True
    except ValueError:
        return is_active_gooaye_release_script(path)
    except (OSError, RuntimeError):
        return False
