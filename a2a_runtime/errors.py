from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus


@dataclass(frozen=True)
class ErrorSpec:
    code: int
    reason: str
    http_status: int
    default_message: str


ERROR_SPECS = {
    'TaskNotFoundError': ErrorSpec(-32001, 'TASK_NOT_FOUND', HTTPStatus.NOT_FOUND, 'Task not found.'),
    'TaskNotCancelableError': ErrorSpec(-32002, 'TASK_NOT_CANCELABLE', HTTPStatus.BAD_REQUEST, 'Task is not cancelable.'),
    'PushNotificationNotSupportedError': ErrorSpec(-32003, 'PUSH_NOTIFICATION_NOT_SUPPORTED', HTTPStatus.BAD_REQUEST, 'Push notifications are not supported.'),
    'UnsupportedOperationError': ErrorSpec(-32004, 'UNSUPPORTED_OPERATION', HTTPStatus.BAD_REQUEST, 'Requested operation is not supported.'),
    'ContentTypeNotSupportedError': ErrorSpec(-32005, 'CONTENT_TYPE_NOT_SUPPORTED', HTTPStatus.BAD_REQUEST, 'Content type is not supported.'),
    'InvalidAgentResponseError': ErrorSpec(-32006, 'INVALID_AGENT_RESPONSE', HTTPStatus.INTERNAL_SERVER_ERROR, 'Agent returned an invalid response.'),
    'ExtendedAgentCardNotConfiguredError': ErrorSpec(-32007, 'EXTENDED_AGENT_CARD_NOT_CONFIGURED', HTTPStatus.BAD_REQUEST, 'Extended Agent Card is not configured.'),
    'ExtensionSupportRequiredError': ErrorSpec(-32008, 'EXTENSION_SUPPORT_REQUIRED', HTTPStatus.BAD_REQUEST, 'Required extensions are not supported by the client.'),
    'VersionNotSupportedError': ErrorSpec(-32009, 'VERSION_NOT_SUPPORTED', HTTPStatus.BAD_REQUEST, 'Requested A2A version is not supported.'),
}


class A2AError(Exception):
    error_name = 'A2AError'

    def __init__(self, message: str | None = None, *, metadata: dict[str, str] | None = None):
        self.spec = ERROR_SPECS[self.error_name]
        self.message = message or self.spec.default_message
        self.metadata = metadata or {}
        super().__init__(self.message)


class TaskNotFoundError(A2AError):
    error_name = 'TaskNotFoundError'


class TaskNotCancelableError(A2AError):
    error_name = 'TaskNotCancelableError'


class PushNotificationNotSupportedError(A2AError):
    error_name = 'PushNotificationNotSupportedError'


class UnsupportedOperationError(A2AError):
    error_name = 'UnsupportedOperationError'


class ContentTypeNotSupportedError(A2AError):
    error_name = 'ContentTypeNotSupportedError'


class InvalidAgentResponseError(A2AError):
    error_name = 'InvalidAgentResponseError'


class ExtendedAgentCardNotConfiguredError(A2AError):
    error_name = 'ExtendedAgentCardNotConfiguredError'


class ExtensionSupportRequiredError(A2AError):
    error_name = 'ExtensionSupportRequiredError'


class VersionNotSupportedError(A2AError):
    error_name = 'VersionNotSupportedError'
