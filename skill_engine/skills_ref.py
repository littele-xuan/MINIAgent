from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SkillsRefResult:
    ok: bool
    stdout: str = ''
    stderr: str = ''


class SkillsRefUnavailableError(RuntimeError):
    pass


class SkillsRefValidator:
    """Thin wrapper around the official Agent Skills reference validator.

    This keeps our framework stable while preferring the reference implementation
    whenever `skills-ref` is installed in the runtime environment.
    """

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which('skills-ref') or 'skills-ref'

    def is_available(self) -> bool:
        resolved = shutil.which(self.executable) if self.executable == 'skills-ref' else self.executable
        return bool(resolved)

    def validate(self, skill_dir: str | Path) -> SkillsRefResult:
        if not self.is_available():
            raise SkillsRefUnavailableError('skills-ref is not installed')
        path = str(Path(skill_dir).resolve())
        proc = subprocess.run(
            [self.executable, 'validate', path],
            check=False,
            capture_output=True,
            text=True,
        )
        return SkillsRefResult(ok=proc.returncode == 0, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())

    def validate_or_raise(self, skill_dir: str | Path) -> None:
        result = self.validate(skill_dir)
        if not result.ok:
            detail = '\n'.join(part for part in (result.stdout, result.stderr) if part).strip()
            raise ValueError(detail or 'skills-ref validation failed')
