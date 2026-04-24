from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SkillFrontmatter:
    name: str
    description: str
    when_to_use: str | None = None
    argument_hint: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    context: str | None = None
    agent: str | None = None
    hooks: dict[str, Any] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    shell: str | None = None
    output_modes: list[str] = field(default_factory=lambda: ['text/plain', 'application/json'])
    accepted_output_modes: list[str] = field(default_factory=lambda: ['text/plain', 'application/json'])
    mcp: dict[str, Any] = field(default_factory=dict)
    a2a: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillLocalToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler_file: Path
    handler_symbol: str = 'handler'
    version: str = '1.0.0'
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillCatalogEntry:
    name: str
    description: str
    when_to_use: str | None
    path: Path
    allowed_tools: list[str] = field(default_factory=list)
    output_modes: list[str] = field(default_factory=lambda: ['text/plain', 'application/json'])


@dataclass(slots=True)
class SkillActivation:
    entry: SkillCatalogEntry
    reason: str
    score: float


@dataclass(slots=True)
class SkillBundle:
    frontmatter: SkillFrontmatter
    root_path: Path
    skill_md_path: Path
    body: str
    resources: list[Path] = field(default_factory=list)
    local_tools: list[SkillLocalToolSpec] = field(default_factory=list)
    rendered_body: str | None = None
    render_arguments: str = ''
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def when_to_use(self) -> str | None:
        return self.frontmatter.when_to_use

    @property
    def allowed_tools(self) -> list[str]:
        return self.frontmatter.allowed_tools

    @property
    def output_modes(self) -> list[str]:
        return self.frontmatter.output_modes

    @property
    def accepted_output_modes(self) -> list[str]:
        return self.frontmatter.accepted_output_modes

    def catalog_entry(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'when_to_use': self.when_to_use,
            'allowed_tools': list(self.allowed_tools),
            'local_tools': [tool.name for tool in self.local_tools],
            'output_modes': list(self.output_modes),
            'accepted_output_modes': list(self.accepted_output_modes),
            'examples': list(self.frontmatter.examples),
            'mcp': dict(self.frontmatter.mcp or {}),
            'a2a': dict(self.frontmatter.a2a or {}),
            'license': self.frontmatter.license,
            'compatibility': self.frontmatter.compatibility,
            'metadata': dict(self.frontmatter.metadata or {}),
        }

    def clone_for_arguments(self, arguments: str) -> 'SkillBundle':
        bundle = replace(self)
        bundle.render_arguments = arguments
        bundle.session_id = str(uuid.uuid4())
        bundle.rendered_body = self.render(arguments=arguments, session_id=bundle.session_id)
        return bundle

    def render(self, *, arguments: str = '', session_id: str | None = None) -> str:
        rendered = self.body
        parts = shlex.split(arguments) if arguments else []
        rendered = rendered.replace('${CLAUDE_SKILL_DIR}', str(self.root_path))
        rendered = rendered.replace('${CLAUDE_SESSION_ID}', session_id or self.session_id)
        rendered = rendered.replace('$ARGUMENTS', arguments)
        for idx, value in enumerate(parts):
            rendered = rendered.replace(f'$ARGUMENTS[{idx}]', value)
            rendered = rendered.replace(f'${idx}', value)
        has_positional_tokens = any(
            f'$ARGUMENTS[{idx}]' in self.body or f'${idx}' in self.body
            for idx in range(len(parts))
        )
        if arguments and '$ARGUMENTS' not in self.body and not has_positional_tokens:
            rendered = rendered.rstrip() + f'\n\nARGUMENTS: {arguments}\n'
        return rendered
