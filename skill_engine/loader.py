from __future__ import annotations

from pathlib import Path

import yaml

from .frontmatter import split_frontmatter
from .models import SkillBundle, SkillCatalogEntry, SkillFrontmatter, SkillLocalToolSpec
from .validator import (
    SkillValidationError,
    normalize_allowed_tools,
    normalize_mapping,
    normalize_paths,
    normalize_string_list,
    validate_context_agent,
    validate_description,
    validate_effort,
    validate_frontmatter_keys,
    validate_local_tool_spec,
    validate_name,
    validate_shell,
)


class AnthropicSkillLoader:
    def __init__(self, skills_root: str | Path | None):
        self.skills_root = Path(skills_root).resolve() if skills_root else None

    def discover_bundles(self) -> list[SkillBundle]:
        if self.skills_root is None or not self.skills_root.exists():
            return []
        bundles: list[SkillBundle] = []
        for skill_dir in sorted(self.skills_root.iterdir()):
            if skill_dir.is_dir() and (skill_dir / 'SKILL.md').exists():
                bundles.append(self.load_bundle(skill_dir))
        return bundles

    def discover_catalog(self) -> list[SkillCatalogEntry]:
        return [
            SkillCatalogEntry(
                name=bundle.name,
                description=bundle.description,
                when_to_use=bundle.when_to_use,
                path=bundle.root_path,
                allowed_tools=list(bundle.allowed_tools),
                output_modes=list(bundle.output_modes),
            )
            for bundle in self.discover_bundles()
        ]

    def load_bundle(self, skill_dir: str | Path) -> SkillBundle:
        skill_dir = Path(skill_dir).resolve()
        skill_md = skill_dir / 'SKILL.md'
        raw_frontmatter, body = split_frontmatter(skill_md.read_text(encoding='utf-8'))
        data = yaml.safe_load(raw_frontmatter) or {}
        if not isinstance(data, dict):
            raise SkillValidationError('SKILL.md frontmatter must be a YAML mapping')
        validate_frontmatter_keys(data)

        name = str(data.get('name') or skill_dir.name).strip()
        description = str(data.get('description') or '').strip()
        when_to_use = str(data.get('when_to_use')).strip() if data.get('when_to_use') is not None else None
        argument_hint = str(data.get('argument-hint')).strip() if data.get('argument-hint') is not None else None
        disable_model_invocation = bool(data.get('disable-model-invocation', False))
        user_invocable = bool(data.get('user-invocable', True))
        allowed_tools = normalize_allowed_tools(data.get('allowed-tools'))
        model = str(data.get('model')).strip() if data.get('model') is not None else None
        effort = str(data.get('effort')).strip() if data.get('effort') is not None else None
        context = str(data.get('context')).strip() if data.get('context') is not None else None
        agent = str(data.get('agent')).strip() if data.get('agent') is not None else None
        hooks = normalize_mapping(data.get('hooks'), 'hooks')
        paths = normalize_paths(data.get('paths'))
        shell = str(data.get('shell')).strip() if data.get('shell') is not None else None
        output_modes = normalize_string_list(data.get('output-modes'), 'output-modes') or ['text/plain', 'application/json']
        accepted_output_modes = normalize_string_list(data.get('accepted-output-modes'), 'accepted-output-modes') or ['text/plain', 'application/json']
        mcp = normalize_mapping(data.get('mcp'), 'mcp')
        a2a = normalize_mapping(data.get('a2a'), 'a2a')
        examples = normalize_string_list(data.get('examples'), 'examples')

        validate_name(name, skill_dir.name)
        validate_description(description)
        validate_effort(effort)
        validate_context_agent(context, agent)
        validate_shell(shell)

        frontmatter = SkillFrontmatter(
            name=name,
            description=description,
            when_to_use=when_to_use,
            argument_hint=argument_hint,
            disable_model_invocation=disable_model_invocation,
            user_invocable=user_invocable,
            allowed_tools=allowed_tools,
            model=model,
            effort=effort,
            context=context,
            agent=agent,
            hooks=hooks,
            paths=paths,
            shell=shell,
            output_modes=output_modes,
            accepted_output_modes=accepted_output_modes,
            mcp=mcp,
            a2a=a2a,
            examples=examples,
        )

        resources = [path for path in skill_dir.rglob('*') if path.is_file() and path.name != 'SKILL.md' and 'mcp_tools' not in path.parts]
        local_tools = self._load_local_tools(skill_dir)
        bundle = SkillBundle(
            frontmatter=frontmatter,
            root_path=skill_dir,
            skill_md_path=skill_md,
            body=body.strip(),
            resources=sorted(resources),
            local_tools=local_tools,
        )
        bundle.rendered_body = bundle.render()
        return bundle

    def load_manifest(self, skill_dir: str | Path) -> SkillBundle:
        return self.load_bundle(skill_dir)

    def validate_against_tool_names(self, manifests, tool_names: set[str]) -> list[str]:
        warnings: list[str] = []
        for manifest in manifests:
            referenced = set(manifest.allowed_tools) | {tool.name for tool in manifest.local_tools}
            missing = sorted(name for name in referenced if name not in tool_names)
            if missing:
                warnings.append(f"Skill '{manifest.name}' references missing tools: {', '.join(missing)}")
        return warnings

    def _load_local_tools(self, skill_dir: Path) -> list[SkillLocalToolSpec]:
        root = skill_dir / 'mcp_tools'
        if not root.exists():
            return []
        tools: list[SkillLocalToolSpec] = []
        for tool_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest_path = tool_dir / 'tool.yaml'
            if not manifest_path.exists():
                continue
            data = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
            if not isinstance(data, dict):
                raise SkillValidationError(f'local tool manifest must be a YAML mapping: {manifest_path}')
            name = str(data.get('name') or tool_dir.name).strip()
            description = str(data.get('description') or '').strip()
            handler_ref = str(data.get('handler') or 'handler.py:handler').strip()
            file_name, _, symbol = handler_ref.partition(':')
            handler_file = tool_dir / (file_name or 'handler.py')
            validate_local_tool_spec(name, description, handler_file)
            input_schema = data.get('input_schema') or {
                'type': 'object',
                'properties': data.get('properties') or {},
                'required': data.get('required') or [],
            }
            tools.append(SkillLocalToolSpec(
                name=name,
                description=description,
                input_schema=input_schema,
                handler_file=handler_file,
                handler_symbol=symbol or 'handler',
                version=str(data.get('version') or '1.0.0'),
                examples=[str(item) for item in data.get('examples') or []],
                tags=[str(item) for item in data.get('tags') or []],
                aliases=[str(item) for item in data.get('aliases') or []],
                metadata=data.get('metadata') or {},
            ))
        return tools
