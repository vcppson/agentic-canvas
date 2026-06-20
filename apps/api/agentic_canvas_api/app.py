from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentic_canvas.kernel.context import utc_now
from agentic_canvas.kernel.run import Kernel
from agentic_canvas.kernel.trigger import Trigger


TERMINAL_EVENTS = {"run_completed", "run_aborted", "run_stopped", "server_error"}
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class RunCreateRequest(BaseModel):
    input: str = Field(min_length=1)


class RunStream:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.start_error: str | None = None
        self.ready = threading.Event()
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._terminal = False
        self._lock = threading.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            if event["type"] in TERMINAL_EVENTS:
                self._terminal = True
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            subscriber.put(event)
            if event["type"] in TERMINAL_EVENTS:
                subscriber.put(None)

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        subscriber: queue.Queue[dict[str, Any] | None] = queue.Queue()
        with self._lock:
            for event in self._events:
                subscriber.put(event)
            if self._terminal:
                subscriber.put(None)
            else:
                self._subscribers.append(subscriber)
        return subscriber

    def fail_before_start(self, message: str) -> None:
        self.start_error = message
        self.ready.set()


class RunManager:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._streams: dict[str, RunStream] = {}
        self._lock = threading.Lock()

    def start_run(self, user_input: str) -> str:
        stream = RunStream()
        worker = threading.Thread(
            target=self._run_kernel,
            args=(stream, user_input),
            daemon=True,
        )
        worker.start()

        if not stream.ready.wait(timeout=5):
            raise RuntimeError("Timed out while starting run.")
        if stream.start_error is not None:
            raise RuntimeError(stream.start_error)
        if stream.run_id is None:
            raise RuntimeError("Run started without a run_id.")
        return stream.run_id

    def subscribe(self, run_id: str) -> queue.Queue[dict[str, Any] | None]:
        with self._lock:
            stream = self._streams.get(run_id)
        if stream is None:
            raise KeyError(run_id)
        return stream.subscribe()

    def _run_kernel(self, stream: RunStream, user_input: str) -> None:
        def handle_event(event: dict[str, Any]) -> None:
            run_id = event.get("run_id")
            if isinstance(run_id, str) and stream.run_id is None:
                stream.run_id = run_id
                with self._lock:
                    self._streams[run_id] = stream
                stream.ready.set()
            stream.publish(event)

        try:
            Kernel(self.workspace_root, event_handler=handle_event).start(
                Trigger(user_input=user_input)
            )
        except Exception as exc:
            event = {
                "type": "server_error",
                "timestamp": utc_now(),
                "run_id": stream.run_id,
                "message": str(exc),
            }
            if stream.run_id is None:
                stream.fail_before_start(str(exc))
                return
            stream.publish(event)


def create_app(workspace_root: str | Path) -> FastAPI:
    workspace = Path(workspace_root).resolve()
    manager = RunManager(workspace)
    app = FastAPI(title="Agentic Canvas API")
    app.state.run_manager = manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/runs")
    def create_run(request: RunCreateRequest) -> dict[str, str]:
        if not request.input.strip():
            raise HTTPException(status_code=400, detail="Input cannot be blank.")
        try:
            run_id = manager.start_run(request.input)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        try:
            events = manager.subscribe(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return StreamingResponse(
            _sse_events(events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _cors_origins() -> list[str]:
    configured = os.getenv("AGENTIC_CANVAS_CORS_ORIGINS")
    if configured is None:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


async def _sse_events(events: queue.Queue[dict[str, Any] | None]):
    while True:
        event = await asyncio.to_thread(events.get)
        if event is None:
            break
        event_type = str(event.get("type") or "message")
        data = json.dumps(event, ensure_ascii=False, default=str)
        yield f"event: {event_type}\ndata: {data}\n\n"
