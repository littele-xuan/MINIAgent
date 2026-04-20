from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from skill_engine import AnthropicSkillLoader

from .models import A2A_PROTOCOL_VERSION, AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill


@dataclass(slots=True)
class PeerAgent:
    name: str
    description: str
    agent_card_url: str
    tags: list[str] = field(default_factory=list)


class AgentCardBuilder:
    def __init__(
        self,
        *,
        agent_name: str,
        description: str,
        provider_org: str,
        provider_url: str,
        agent_version: str = '1.0.0',
        documentation_url: str | None = None,
        skills_root: str | Path | None = None,
        icon_url: str | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.description = description
        self.provider_org = provider_org
        self.provider_url = provider_url
        self.agent_version = agent_version
        self.documentation_url = documentation_url
        self.skills_root = Path(skills_root).resolve() if skills_root else None
        self.icon_url = icon_url

    def build(self, *, base_url: str, tool_names: Iterable[str]) -> AgentCard:
        root = base_url.rstrip('/')
        interfaces = [
            AgentInterface(url=f'{root}/a2a/v1', protocolBinding='HTTP+JSON', protocolVersion=A2A_PROTOCOL_VERSION),
            AgentInterface(url=f'{root}/a2a/v1/rpc', protocolBinding='JSONRPC', protocolVersion=A2A_PROTOCOL_VERSION),
        ]
        return AgentCard(
            name=self.agent_name,
            description=self.description,
            version=self.agent_version,
            supportedInterfaces=interfaces,
            capabilities=AgentCapabilities(streaming=False, pushNotifications=False, extendedAgentCard=False, extensions=[]),
            defaultInputModes=['text/plain', 'application/json'],
            defaultOutputModes=['text/plain', 'application/json'],
            skills=self._load_skills(tool_names),
            provider=AgentProvider(organization=self.provider_org, url=self.provider_url),
            documentationUrl=self.documentation_url,
            securitySchemes={},
            security=[],
            iconUrl=self.icon_url,
        )

    def compute_etag(self, card: AgentCard) -> str:
        payload = json.dumps(card.model_dump(by_alias=True, exclude_none=True), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _load_skills(self, tool_names: Iterable[str]) -> list[AgentSkill]:
        fallback = [AgentSkill(
            id='default-agent',
            name='Default Agent',
            description='Answer questions, invoke MCP tools, and optionally collaborate with peer agents.',
            tags=sorted(set(tool_names)) or ['agent'],
            examples=['请计算 18*(7+2)', '列出当前系统可用工具'],
            inputModes=['text/plain', 'application/json'],
            outputModes=['text/plain', 'application/json'],
            security=[],
        )]
        if self.skills_root is None or not self.skills_root.exists():
            return fallback
        loader = AnthropicSkillLoader(self.skills_root)
        bundles = loader.discover_bundles()
        if not bundles:
            return fallback
        skills: list[AgentSkill] = []
        for bundle in bundles:
            tags = list(dict.fromkeys(bundle.allowed_tools + [tool.name for tool in bundle.local_tools]))
            examples = list(bundle.frontmatter.examples)
            if not examples:
                for tool in bundle.local_tools:
                    examples.extend(tool.examples)
            skills.append(AgentSkill(
                id=bundle.name,
                name=bundle.name,
                description=bundle.description,
                tags=tags or [bundle.name],
                examples=examples[:8],
                inputModes=['text/plain', 'application/json'],
                outputModes=list(bundle.output_modes) or ['text/plain', 'application/json'],
                security=[],
            ))
        return skills or fallback
