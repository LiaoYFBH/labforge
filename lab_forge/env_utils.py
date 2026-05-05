"""
Environment file loading helpers.

AutoDL may forbid creating `.env`, so we support `env.txt` as a first-class
alternative while remaining compatible with the existing `.env` workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ENV_CANDIDATE_FILENAMES = ("env.txt", ".env")
ENV_PATH_VARS = ("SCIPRM_ENV_FILE", "ENV_FILE")


def load_env_file(path: str | Path, override: bool = False) -> Path:
    """Load simple KEY=VALUE pairs from a file into ``os.environ``."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return env_path


def load_project_env(
    project_root: str | Path | None = None,
    extra_search_dirs: Iterable[str | Path] | None = None,
    override: bool = False,
) -> Path | None:
    """
    Load the first available environment file.

    Resolution order:
    1. Explicit path via ``SCIPRM_ENV_FILE`` or ``ENV_FILE``
    2. ``env.txt`` in search roots
    3. ``.env`` in search roots
    """
    candidates: list[Path] = []

    for env_var in ENV_PATH_VARS:
        explicit = os.environ.get(env_var, "").strip()
        if explicit:
            candidates.append(Path(explicit).expanduser())

    search_roots: list[Path] = []
    if project_root is not None:
        search_roots.append(Path(project_root).expanduser())
    if extra_search_dirs:
        search_roots.extend(Path(p).expanduser() for p in extra_search_dirs)

    seen: set[Path] = set()
    for root in search_roots:
        try:
            resolved_root = root.resolve()
        except FileNotFoundError:
            resolved_root = root
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        for filename in ENV_CANDIDATE_FILENAMES:
            candidates.append(resolved_root / filename)

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.exists():
            return load_env_file(resolved, override=override)

    return None
