from __future__ import annotations

import uuid
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse

from .errors import A2AError, ExtendedAgentCardNotConfiguredError, PushNotificationNotSupportedError, UnsupportedOperationError, VersionNotSupportedError, TaskNotCancelableError, TaskNotFoundError
from .models import (
    A2A_PROTOCOL_VERSION,
    Artifact,
    CancelTaskRequest,
    GetExtendedAgentCardRequest,
    GetTaskRequest,
    JsonRpcErrorObject,
    JsonRpcRequest,
    JsonRpcResponse,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskState,
    TaskStatus,
    TERMINAL_TASK_STATES,
    CANCELABLE_TASK_STATES,
    utc_now_iso,
)


class JsonRpcHttpError(Exception):
    def __init__(self, code: int, message: str, http_status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class InMemoryTaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def save(self, task: Task) -> None:
        self.tasks[task.id] = task

    def get(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise TaskNotFoundError()
        return self.tasks[task_id]

    def list(self, request: ListTasksRequest) -> list[Task]:
        tasks = list(self.tasks.values())
        if request.context_id:
            tasks = [task for task in tasks if task.context_id == request.context_id]
        if request.status:
            tasks = [task for task in tasks if task.status.state == request.status]
        return tasks


async def _execute_send_message(agent: Any, request: SendMessageRequest, store: InMemoryTaskStore) -> SendMessageResponse:
    run_result = await agent.handle_a2a_request(request)
    task_id = request.message.task_id or str(uuid.uuid4())
    context_id = request.message.context_id or str(uuid.uuid4())
    now = utc_now_iso()
    if run_result.output_mode == 'application/json':
        payload_part = Part(data=run_result.payload, mediaType='application/json')
    else:
        payload_part = Part(text=run_result.answer, mediaType='text/plain')
    task = Task(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(
            state=TaskState.TASK_STATE_COMPLETED,
            timestamp=now,
            message=Message(
                messageId=str(uuid.uuid4()),
                contextId=context_id,
                taskId=task_id,
                role=Role.ROLE_AGENT,
                parts=[payload_part],
            ),
        ),
        history=[request.message],
        artifacts=[Artifact(artifactId=str(uuid.uuid4()), parts=[payload_part], name='response')],
        metadata={'trace': run_result.trace, 'selected_skill': run_result.selected_skill, 'output_mode': run_result.output_mode},
        createdAt=now,
        lastModified=now,
    )
    store.save(task)
    return SendMessageResponse(task=task)


def build_a2a_app(*, agent: Any, card_builder: Any, base_url: str):
    """Prefer the official a2a-sdk server implementation, with a compatibility fallback."""
    try:
        return _build_sdk_app(agent=agent, card_builder=card_builder, base_url=base_url)
    except Exception:
        return _build_compat_app(agent=agent, card_builder=card_builder, base_url=base_url)


def _build_sdk_app(*, agent: Any, card_builder: Any, base_url: str):
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore as SdkInMemoryTaskStore
    from a2a.server.tasks import TaskUpdater
    from a2a.types import Part as SdkPart
    from a2a.types import TaskState as SdkTaskState
    from a2a.types import TextPart as SdkTextPart
    from a2a.types import UnsupportedOperationError as SdkUnsupportedOperationError
    from a2a.utils import new_agent_text_message, new_task
    from a2a.utils.errors import ServerError
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse as StarletteJSONResponse
    from starlette.routing import Route

    sdk_card = card_builder.build_sdk_card(base_url=base_url.rstrip('/'), tool_names=[tool.name for tool in getattr(agent, '_tools', [])])

    class MiniAgentExecutor(AgentExecutor):
        async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
            query = context.get_user_input()
            task = context.current_task
            if not task:
                task = new_task(context.message)  # type: ignore[arg-type]
                await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            run_result = await agent.run_detailed(
                query,
                accepted_output_modes=list(context.message.metadata.get('accepted_output_modes', [])) if getattr(context, 'message', None) and getattr(context.message, 'metadata', None) else None,
            )
            if run_result.output_mode == 'application/json':
                await updater.add_artifact([SdkPart(root=SdkTextPart(text=run_result.answer or ''))], name='response')
            else:
                await updater.add_artifact([SdkPart(root=SdkTextPart(text=run_result.answer))], name='response')
            await updater.complete()

        async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
            raise ServerError(error=SdkUnsupportedOperationError())

    request_handler = DefaultRequestHandler(
        agent_executor=MiniAgentExecutor(),
        task_store=SdkInMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(agent_card=sdk_card, http_handler=request_handler)

    async def health(_request):
        return StarletteJSONResponse({'ok': True, 'agent': agent.config.name})

    routes = list(a2a_app.routes())
    routes.append(Route('/health', methods=['GET'], endpoint=health))
    return Starlette(routes=routes)


def _build_compat_app(*, agent: Any, card_builder: Any, base_url: str) -> FastAPI:
    app = FastAPI(title=f'{agent.config.name} A2A Agent', version=A2A_PROTOCOL_VERSION)
    store = InMemoryTaskStore()
    agent_card = card_builder.build(base_url=base_url.rstrip('/'), tool_names=[tool.name for tool in getattr(agent, '_tools', [])])
    etag = card_builder.compute_etag(agent_card)

    async def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == 'SendMessage':
            request = SendMessageRequest.model_validate(params)
            response = await _execute_send_message(agent, request, store)
            return response.model_dump(by_alias=True, exclude_none=True)
        if method == 'GetTask':
            request = GetTaskRequest.model_validate(params)
            task = store.get(request.id)
            return task.model_dump(by_alias=True, exclude_none=True)
        if method == 'ListTasks':
            request = ListTasksRequest.model_validate(params)
            tasks = store.list(request)
            response = ListTasksResponse(tasks=tasks, nextPageToken='', pageSize=len(tasks), totalSize=len(tasks))
            return response.model_dump(by_alias=True, exclude_none=True)
        if method == 'CancelTask':
            request = CancelTaskRequest.model_validate(params)
            task = store.get(request.id)
            if task.status.state not in CANCELABLE_TASK_STATES:
                raise TaskNotCancelableError()
            task.status.state = TaskState.TASK_STATE_CANCELED
            task.last_modified = utc_now_iso()
            store.save(task)
            return task.model_dump(by_alias=True, exclude_none=True)
        if method == 'GetExtendedAgentCard':
            _ = GetExtendedAgentCardRequest.model_validate(params)
            raise ExtendedAgentCardNotConfiguredError()
        if method in {'SendStreamingMessage', 'SubscribeToTask'}:
            raise UnsupportedOperationError('This A2A server does not support streaming.')
        if method in {'CreateTaskPushNotificationConfig', 'GetTaskPushNotificationConfig', 'ListTaskPushNotificationConfigs', 'DeleteTaskPushNotificationConfig'}:
            raise PushNotificationNotSupportedError()
        raise JsonRpcHttpError(-32601, f'Method not found: {method}', HTTPStatus.NOT_FOUND)

    @app.get('/.well-known/agent-card.json')
    async def agent_card_endpoint() -> JSONResponse:
        return JSONResponse(content=agent_card.model_dump(by_alias=True, exclude_none=True), headers={'Cache-Control': 'public, max-age=300', 'ETag': etag})

    @app.get('/health')
    async def health() -> JSONResponse:
        return JSONResponse({'ok': True, 'agent': agent.config.name})

    @app.post('/a2a/v1/message:send')
    async def http_send_message(request: Request, a2a_version: str | None = Header(default=None, alias='A2A-Version'), a2a_version_query: str | None = Query(default=None, alias='A2A-Version')) -> JSONResponse:
        validate_version(a2a_version or a2a_version_query or '0.3')
        body = SendMessageRequest.model_validate(await request.json())
        response = await _execute_send_message(agent, body, store)
        return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))

    @app.get('/a2a/v1/tasks/{task_id}')
    async def http_get_task(task_id: str, a2a_version: str | None = Header(default=None, alias='A2A-Version'), a2a_version_query: str | None = Query(default=None, alias='A2A-Version')) -> JSONResponse:
        validate_version(a2a_version or a2a_version_query or '0.3')
        task = store.get(task_id)
        return JSONResponse(content=task.model_dump(by_alias=True, exclude_none=True))

    @app.get('/a2a/v1/tasks')
    async def http_list_tasks(a2a_version: str | None = Header(default=None, alias='A2A-Version'), a2a_version_query: str | None = Query(default=None, alias='A2A-Version')) -> JSONResponse:
        validate_version(a2a_version or a2a_version_query or '0.3')
        tasks = store.list(ListTasksRequest())
        response = ListTasksResponse(tasks=tasks, nextPageToken='', pageSize=len(tasks), totalSize=len(tasks))
        return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))

    @app.post('/a2a/v1/tasks/{task_id}:cancel')
    async def http_cancel_task(task_id: str, a2a_version: str | None = Header(default=None, alias='A2A-Version'), a2a_version_query: str | None = Query(default=None, alias='A2A-Version')) -> JSONResponse:
        validate_version(a2a_version or a2a_version_query or '0.3')
        task = store.get(task_id)
        if task.status.state not in CANCELABLE_TASK_STATES:
            raise TaskNotCancelableError()
        task.status.state = TaskState.TASK_STATE_CANCELED
        task.last_modified = utc_now_iso()
        store.save(task)
        return JSONResponse(content=task.model_dump(by_alias=True, exclude_none=True))

    @app.post('/a2a/v1/rpc')
    async def rpc_endpoint(request: Request, a2a_version: str | None = Header(default=None, alias='A2A-Version'), a2a_version_query: str | None = Query(default=None, alias='A2A-Version')) -> JSONResponse:
        request_id: str | int | None = None
        try:
            validate_version(a2a_version or a2a_version_query or '0.3')
            body = JsonRpcRequest.model_validate(await request.json())
            request_id = body.id
            if body.jsonrpc != '2.0':
                raise JsonRpcHttpError(-32600, "Invalid Request: jsonrpc must equal '2.0'", HTTPStatus.BAD_REQUEST)
            result = await dispatch(body.method, body.params)
            response = JsonRpcResponse(id=body.id, result=result)
            return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))
        except A2AError as exc:
            return jsonrpc_error_response(code=exc.spec.code, message=exc.message, error_name=exc.error_name, metadata=exc.metadata, request_id=request_id, http_status=exc.spec.http_status)
        except JsonRpcHttpError as exc:
            return jsonrpc_error_response(code=exc.code, message=exc.message, error_name=None, metadata=None, request_id=request_id, http_status=exc.http_status)
        except Exception as exc:
            return jsonrpc_error_response(code=-32603, message=f'Internal error: {exc}', error_name=None, metadata=None, request_id=request_id, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return app


def validate_version(version: str) -> None:
    if version not in {A2A_PROTOCOL_VERSION, '0.3'}:
        raise VersionNotSupportedError(metadata={'requestedVersion': version, 'supportedVersion': A2A_PROTOCOL_VERSION})


def jsonrpc_error_response(*, code: int, message: str, error_name: str | None, metadata: dict[str, str] | None, request_id: str | int | None, http_status: int) -> JSONResponse:
    data = None
    if error_name:
        data = [{
            '@type': 'type.googleapis.com/google.rpc.ErrorInfo',
            'reason': error_name.replace('Error', '').upper(),
            'domain': 'a2a-protocol.org',
            'metadata': metadata or {},
        }]
    response = JsonRpcResponse(id=request_id, error=JsonRpcErrorObject(code=code, message=message, data=data))
    return JSONResponse(status_code=int(http_status), content=response.model_dump(by_alias=True, exclude_none=True))
