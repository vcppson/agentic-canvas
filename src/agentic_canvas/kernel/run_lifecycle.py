from __future__ import annotations

from typing import Any

from agentic_canvas.kernel.context import RunContext, utc_now
from agentic_canvas.kernel.decision import Decision, RunStatus, parse_decision
from agentic_canvas.kernel.plugin_runner import (
    PluginCallRequest,
    PluginExecutionResult,
    PluginInputRequest,
)
from agentic_canvas.orchestrator.orchestrator import Orchestrator, compose_system_prompt


class RunLifecycleMixin:
    """Run execution behavior shared by the Kernel facade."""

    def _run_from_start(self, context: RunContext) -> RunContext:
        if not self._execute_stage(context, "pre_orchestrator", start_index=0):
            return context
        if not self._execute_orchestrator(context):
            return context
        if not self._execute_stage(context, "post_orchestrator", start_index=0):
            return context
        return self._complete(context)

    def _execute_stage(
        self,
        context: RunContext,
        stage_name: str,
        *,
        start_index: int,
    ) -> bool:
        context.stage = stage_name
        context.decision = Decision.CONTINUE.value
        context.record_event("stage_started", stage=stage_name, start_index=start_index)
        self.trace_store.append(context, "stage_started", stage_name=stage_name, start_index=start_index)
        self.store.save(context)

        entries = self.workspace.stage_plugins(stage_name)
        for index, entry in enumerate(entries[start_index:], start=start_index):
            manifest = self.plugin_registry.get(entry.name)
            params = entry.params
            context.record_event(
                "plugin_started",
                plugin=entry.name,
                mode="stage",
                stage=stage_name,
                index=index,
            )
            self.store.save(context)

            result = self.plugin_runner.run(
                workspace=self.workspace,
                manifest=manifest,
                context=context,
                mode="stage",
                params=params,
                input_handler=lambda request, stage=stage_name, plugin_index=index: self._handle_plugin_input_request(
                    context,
                    request,
                    stage=stage,
                    index=plugin_index,
                ),
                plugin_call_handler=lambda request, stage=stage_name, plugin_index=index: self._handle_plugin_call_request(
                    context,
                    request,
                    stage=stage,
                    index=plugin_index,
                ),
            )
            self.trace_store.append(
                context,
                "plugin_execution",
                plugin=entry.name,
                mode="stage",
                stage_name=stage_name,
                index=index,
                params=params,
                result=result.to_trace_dict(),
            )
            context.record_event(
                "plugin_finished",
                plugin=entry.name,
                mode="stage",
                stage=stage_name,
                index=index,
                kind=result.kind,
                ok=result.ok,
            )

            if result.kind == "error":
                self._abort(context, f"Plugin {entry.name!r} failed: {result.message}")
                return False

            if result.is_run_control:
                self._handle_run_control(
                    context,
                    result,
                    source_plugin=entry.name,
                    stage=stage_name,
                )
                return False

            if result.patch:
                context.apply_patch(result.patch)

            stage_result = result.result
            if "decision" not in stage_result:
                self._abort(
                    context,
                    f"Stage plugin {entry.name!r} returned no decision.",
                )
                return False

            if stage_result.get("patch"):
                context.apply_patch(stage_result["patch"])

            try:
                decision = parse_decision(stage_result["decision"])
            except ValueError as exc:
                self._abort(context, str(exc))
                return False

            context.decision = decision.value
            if decision == Decision.CONTINUE:
                self.store.save(context)
                continue

            self._handle_stage_decision(
                context,
                decision,
                result=stage_result,
                source_plugin=entry.name,
                stage=stage_name,
            )
            return False

        context.record_event("stage_finished", stage=stage_name)
        self.trace_store.append(context, "stage_finished", stage_name=stage_name)
        self.store.save(context)
        return True

    def _execute_orchestrator(self, context: RunContext) -> bool:
        context.stage = "orchestrator"
        context.decision = Decision.CONTINUE.value
        context.record_event("orchestrator_started")

        orchestrator = Orchestrator(
            workspace=self.workspace,
            plugin_runner=self.plugin_runner,
            provider=self.provider,
            trace_store=self.trace_store,
            input_handler=lambda request: self._handle_plugin_input_request(
                context,
                request,
                stage="orchestrator",
                index=None,
            ),
            plugin_call_handler=lambda request: self._handle_plugin_call_request(
                context,
                request,
                stage="orchestrator",
                index=None,
            ),
            max_turns=int(self.workspace.provider_config.get("max_turns", 20)),
        )
        tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in orchestrator.tool_definitions(context)
        ]
        self.trace_store.append(
            context,
            "orchestrator_started",
            input=context.input,
            orchestrator_system_prompt=compose_system_prompt(context),
            base_orchestrator_system_prompt=context.orchestrator_system_prompt,
            additional_prompts=context.additional_prompts,
            tools=tool_schemas,
        )
        self.store.save(context)
        try:
            result = orchestrator.invoke(context)
        except Exception as exc:
            self.trace_store.append(context, "orchestrator_error", error=str(exc))
            self._abort(context, f"Orchestrator failed: {exc}")
            return False

        if result.control:
            self._handle_run_control(
                context,
                result.control,
                source_plugin=result.control.plugin_name,
                stage="orchestrator",
            )
            return False

        context.orchestrator_response = result.response or ""
        if context.final_response is None:
            context.final_response = context.orchestrator_response
        context.record_event("orchestrator_finished")
        self.trace_store.append(
            context,
            "orchestrator_finished",
            response=context.orchestrator_response,
        )
        self.store.save(context)
        return True

    def _handle_run_control(
        self,
        context: RunContext,
        result: PluginExecutionResult,
        *,
        source_plugin: str,
        stage: str,
    ) -> None:
        if result.patch:
            context.apply_patch(result.patch)

        try:
            decision = parse_decision(result.decision or Decision.ABORT.value)
        except ValueError as exc:
            self._abort(context, str(exc))
            return

        self._handle_stage_decision(
            context,
            decision,
            result={
                "message": result.message,
                "reason": result.reason,
                "response": result.response,
            },
            source_plugin=source_plugin,
            stage=stage,
        )

    def _handle_stage_decision(
        self,
        context: RunContext,
        decision: Decision,
        *,
        result: dict[str, Any],
        source_plugin: str,
        stage: str,
    ) -> None:
        context.decision = decision.value
        if decision == Decision.STOP:
            response = result.get("response") or result.get("reason") or result.get("message") or ""
            context.final_response = str(response)
            context.status = RunStatus.COMPLETED.value
            context.record_event("run_stopped", plugin=source_plugin, stage=stage)
            self.trace_store.append(
                context,
                "run_stopped",
                plugin=source_plugin,
                stage_name=stage,
                response=context.final_response,
            )
            self.store.save(context)
            return

        if decision == Decision.AWAIT_USER:
            self._abort(
                context,
                (
                    "Returned decision 'await_user' cannot continue a finished plugin "
                    "invocation. Use libs.run_control.await_user(...) so the answer "
                    "returns to the live plugin call."
                ),
                source_plugin=source_plugin,
                stage=stage,
            )
            return

        if decision == Decision.ABORT:
            reason = result.get("reason") or result.get("message") or result.get("response") or ""
            self._abort(context, str(reason), source_plugin=source_plugin, stage=stage)
            return

        self.store.save(context)

    def _complete(self, context: RunContext) -> RunContext:
        context.status = RunStatus.COMPLETED.value
        if context.final_response is None:
            context.final_response = context.orchestrator_response or ""
        context.record_event("run_completed")
        self.trace_store.append(context, "run_completed", final_response=context.final_response)
        self.store.save(context)
        return context

    def _abort(
        self,
        context: RunContext,
        reason: str,
        *,
        source_plugin: str | None = None,
        stage: str | None = None,
    ) -> RunContext:
        context.status = RunStatus.ABORTED.value
        context.decision = Decision.ABORT.value
        context.final_response = reason
        context.record_event("run_aborted", reason=reason, plugin=source_plugin, stage=stage)
        self.trace_store.append(
            context,
            "run_aborted",
            reason=reason,
            plugin=source_plugin,
            stage_name=stage,
        )
        self.store.save(context)
        return context

    def _handle_plugin_input_request(
        self,
        context: RunContext,
        request: PluginInputRequest,
        *,
        stage: str,
        index: int | None,
    ) -> str:
        if request.patch:
            context.apply_patch(request.patch)

        context.status = RunStatus.AWAITING_USER_INPUT.value
        context.decision = Decision.AWAIT_USER.value
        context.awaiting = {
            "message": request.message,
            "source_plugin": request.plugin_name,
            "stage": stage,
            "request_id": request.request_id,
        }
        request_record = {
            "request_id": request.request_id,
            "plugin": request.plugin_name,
            "mode": request.mode,
            "stage": stage,
            "index": index,
            "message": request.message,
            "params": request.params,
            "patch": request.patch,
            "requested_at": utc_now(),
        }
        context.user_input_requests.append(request_record)
        context.record_event(
            "user_input_requested",
            plugin=request.plugin_name,
            mode=request.mode,
            stage=stage,
            request_id=request.request_id,
            message=request.message,
        )
        self.trace_store.append(
            context,
            "user_input_requested",
            plugin=request.plugin_name,
            mode=request.mode,
            stage_name=stage,
            index=index,
            request_id=request.request_id,
            message=request.message,
            params=request.params,
            patch=request.patch,
            awaiting=context.awaiting,
        )
        self.store.save(context)

        if self.input_provider is None:
            raise RuntimeError("No input provider is configured for plugin await_user().")

        answer = self.input_provider(request)
        request_record["response"] = answer
        request_record["answered_at"] = utc_now()
        context.user_inputs.append(
            {
                "input": answer,
                "metadata": {
                    "source": "await_user",
                    "plugin": request.plugin_name,
                    "request_id": request.request_id,
                },
                "timestamp": request_record["answered_at"],
            }
        )
        context.status = RunStatus.RUNNING.value
        context.decision = Decision.CONTINUE.value
        context.awaiting = None
        context.record_event(
            "user_input_received",
            plugin=request.plugin_name,
            mode=request.mode,
            stage=stage,
            request_id=request.request_id,
        )
        self.trace_store.append(
            context,
            "user_input_received",
            plugin=request.plugin_name,
            mode=request.mode,
            stage_name=stage,
            index=index,
            request_id=request.request_id,
            user_input=answer,
        )
        self.store.save(context)
        return answer

    def _handle_plugin_call_request(
        self,
        context: RunContext,
        request: PluginCallRequest,
        *,
        stage: str,
        index: int | None,
    ) -> dict[str, Any]:
        manifest = self.plugin_registry.get(request.plugin_name)
        result = self.plugin_runner.run(
            workspace=self.workspace,
            manifest=manifest,
            context=context,
            mode="command",
            params=request.params,
            input_handler=lambda input_request: self._handle_plugin_input_request(
                context,
                input_request,
                stage=stage,
                index=index,
            ),
            plugin_call_handler=lambda call_request: self._handle_plugin_call_request(
                context,
                call_request,
                stage=stage,
                index=index,
            ),
        )
        context.record_event(
            "plugin_called",
            plugin=request.plugin_name,
            mode="command",
            source_plugin=request.source_plugin_name,
            kind=result.kind,
            ok=result.ok,
        )
        self.trace_store.append(
            context,
            "plugin_execution",
            plugin=request.plugin_name,
            mode="command",
            source_plugin=request.source_plugin_name,
            source_mode=request.source_mode,
            stage_name=stage,
            index=index,
            request_id=request.request_id,
            params=request.params,
            command=request.metadata,
            result=result.to_trace_dict(),
        )

        if result.kind == "result" and result.patch:
            context.apply_patch(result.patch)

        self.store.save(context)
        return result.to_trace_dict()
