from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists() or not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def candidate_env_paths(start: str | Path | None = None) -> Iterable[Path]:
    seen: set[Path] = set()
    starts = []
    if start is not None:
        starts.append(Path(start).resolve())
    starts.append(Path.cwd().resolve())
    starts.append(Path(__file__).resolve().parents[3])
    for base in starts:
        if base.is_file():
            base = base.parent
        for parent in [base, *base.parents]:
            path = parent / ".env"
            if path not in seen:
                seen.add(path)
                yield path


def load_dotenv_if_present(start: str | Path | None = None, *, override: bool = False) -> list[Path]:
    loaded_paths: list[Path] = []
    for path in candidate_env_paths(start):
        loaded = load_env_file(path, override=override)
        if loaded or path.exists():
            loaded_paths.append(path)
            break
    return loaded_paths
