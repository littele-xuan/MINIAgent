from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

A2A_PROTOCOL_VERSION = '1.0'


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class A2ABaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel, extra='forbid', use_enum_values=False)


class JsonRpcRequest(A2ABaseModel):
    jsonrpc: str = '2.0'
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcErrorObject(A2ABaseModel):
    code: int
    message: str
    data: list[dict[str, Any]] | None = None


class JsonRpcResponse(A2ABaseModel):
    jsonrpc: str = '2.0'
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcErrorObject | None = None


class Role(str, Enum):
    ROLE_UNSPECIFIED = 'ROLE_UNSPECIFIED'
    ROLE_USER = 'ROLE_USER'
    ROLE_AGENT = 'ROLE_AGENT'


class TaskState(str, Enum):
    TASK_STATE_UNSPECIFIED = 'TASK_STATE_UNSPECIFIED'
    TASK_STATE_SUBMITTED = 'TASK_STATE_SUBMITTED'
    TASK_STATE_WORKING = 'TASK_STATE_WORKING'
    TASK_STATE_INPUT_REQUIRED = 'TASK_STATE_INPUT_REQUIRED'
    TASK_STATE_COMPLETED = 'TASK_STATE_COMPLETED'
    TASK_STATE_CANCELED = 'TASK_STATE_CANCELED'
    TASK_STATE_FAILED = 'TASK_STATE_FAILED'
    TASK_STATE_REJECTED = 'TASK_STATE_REJECTED'
    TASK_STATE_AUTH_REQUIRED = 'TASK_STATE_AUTH_REQUIRED'


TERMINAL_TASK_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
}

CANCELABLE_TASK_STATES = {
    TaskState.TASK_STATE_SUBMITTED,
    TaskState.TASK_STATE_WORKING,
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
}


class Part(A2ABaseModel):
    text: str | None = None
    raw: str | None = None
    url: str | None = None
    data: Any | None = None
    metadata: dict[str, Any] | None = None
    filename: str | None = None
    media_type: str | None = Field(default=None, alias='mediaType')

    @model_validator(mode='after')
    def validate_payload(self) -> 'Part':
        present = [self.text is not None, self.raw is not None, self.url is not None, self.data is not None]
        if sum(present) != 1:
            raise ValueError('Part must contain exactly one of text/raw/url/data')
        return self


class Message(A2ABaseModel):
    message_id: str = Field(alias='messageId')
    role: Role
    parts: list[Part] = Field(min_length=1)
    context_id: str | None = Field(default=None, alias='contextId')
    task_id: str | None = Field(default=None, alias='taskId')
    metadata: dict[str, Any] | None = None
    extensions: list[str] = Field(default_factory=list)
    reference_task_ids: list[str] = Field(default_factory=list, alias='referenceTaskIds')


class Artifact(A2ABaseModel):
    artifact_id: str = Field(alias='artifactId')
    parts: list[Part] = Field(min_length=1)
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] = Field(default_factory=list)


class TaskStatus(A2ABaseModel):
    state: TaskState
    timestamp: str
    message: Message | None = None


class Task(A2ABaseModel):
    id: str
    context_id: str = Field(alias='contextId')
    status: TaskStatus
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = Field(alias='createdAt')
    last_modified: str = Field(alias='lastModified')


class AgentInterface(A2ABaseModel):
    url: str
    protocol_binding: str = Field(alias='protocolBinding')
    protocol_version: str = Field(alias='protocolVersion')
    tenant: str | None = None


class AgentProvider(A2ABaseModel):
    organization: str
    url: str


class AgentExtension(A2ABaseModel):
    uri: str
    description: str
    required: bool = False
    params: dict[str, Any] | None = None


class AgentCapabilities(A2ABaseModel):
    streaming: bool | None = None
    push_notifications: bool | None = Field(default=None, alias='pushNotifications')
    extensions: list[AgentExtension] = Field(default_factory=list)
    extended_agent_card: bool | None = Field(default=None, alias='extendedAgentCard')


class AgentSkill(A2ABaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list, alias='inputModes')
    output_modes: list[str] = Field(default_factory=list, alias='outputModes')
    security: list[dict[str, list[str]]] = Field(default_factory=list)


class AgentCard(A2ABaseModel):
    name: str
    description: str
    version: str
    supported_interfaces: list[AgentInterface] = Field(alias='supportedInterfaces', min_length=1)
    capabilities: AgentCapabilities
    default_input_modes: list[str] = Field(alias='defaultInputModes', min_length=1)
    default_output_modes: list[str] = Field(alias='defaultOutputModes', min_length=1)
    skills: list[AgentSkill] = Field(default_factory=list)
    provider: AgentProvider | None = None
    documentation_url: str | None = Field(default=None, alias='documentationUrl')
    security_schemes: dict[str, Any] = Field(default_factory=dict, alias='securitySchemes')
    security: list[dict[str, list[str]]] = Field(default_factory=list)
    icon_url: str | None = Field(default=None, alias='iconUrl')


class SendMessageConfiguration(A2ABaseModel):
    accepted_output_modes: list[str] = Field(default_factory=list, alias='acceptedOutputModes')
    history_length: int | None = Field(default=None, alias='historyLength')
    return_immediately: bool | None = Field(default=None, alias='returnImmediately')
    task_push_notification_config: dict[str, Any] | None = Field(default=None, alias='taskPushNotificationConfig')


class SendMessageRequest(A2ABaseModel):
    tenant: str | None = None
    message: Message
    configuration: SendMessageConfiguration | None = None
    metadata: dict[str, Any] | None = None


class SendMessageResponse(A2ABaseModel):
    task: Task | None = None
    message: Message | None = None

    @model_validator(mode='after')
    def validate_union(self) -> 'SendMessageResponse':
        present = int(self.task is not None) + int(self.message is not None)
        if present != 1:
            raise ValueError('SendMessageResponse must contain exactly one of task/message')
        return self


class GetTaskRequest(A2ABaseModel):
    tenant: str | None = None
    id: str
    history_length: int | None = Field(default=None, alias='historyLength')


class ListTasksRequest(A2ABaseModel):
    tenant: str | None = None
    context_id: str | None = Field(default=None, alias='contextId')
    status: TaskState | None = None
    page_size: int | None = Field(default=None, alias='pageSize')
    page_token: str = Field(default='', alias='pageToken')
    history_length: int | None = Field(default=None, alias='historyLength')
    status_timestamp_after: str | None = Field(default=None, alias='statusTimestampAfter')
    include_artifacts: bool | None = Field(default=False, alias='includeArtifacts')


class ListTasksResponse(A2ABaseModel):
    tasks: list[Task]
    next_page_token: str = Field(alias='nextPageToken')
    page_size: int = Field(alias='pageSize')
    total_size: int = Field(alias='totalSize')


class CancelTaskRequest(A2ABaseModel):
    tenant: str | None = None
    id: str
    metadata: dict[str, Any] | None = None


class GetExtendedAgentCardRequest(A2ABaseModel):
    tenant: str | None = None
