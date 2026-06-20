import { FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Blocks,
  CheckCircle2,
  ChevronDown,
  CircleStop,
  Clock3,
  GitBranch,
  Layers3,
  Loader2,
  MessagesSquare,
  PanelRightOpen,
  Play,
  SendHorizontal,
  Sparkles,
  SquareTerminal,
  Trash2,
  UserRound,
} from "lucide-react";

import { startRun } from "./api";
import type { RunEvent } from "./types";

const STREAM_EVENTS = [
  "run_started",
  "stage_started",
  "stage_finished",
  "plugin_started",
  "plugin_finished",
  "orchestrator_started",
  "orchestrator_finished",
  "user_input_requested",
  "user_input_received",
  "run_stopped",
  "run_completed",
  "run_aborted",
  "server_error",
];

const TERMINAL_EVENTS = new Set([
  "run_stopped",
  "run_completed",
  "run_aborted",
  "server_error",
]);

type RunStatus = "starting" | "running" | "complete" | "error" | "stopped" | "aborted";

type ConversationRun = {
  clientId: string;
  createdAt: number;
  error: string;
  events: RunEvent[];
  finalResponse: string;
  prompt: string;
  runId: string | null;
  status: RunStatus;
};

type StageLayer = {
  events: RunEvent[];
  looseEvents: RunEvent[];
  name: string;
  plugins: PluginLayer[];
};

type PluginLayer = {
  events: RunEvent[];
  mode?: string;
  name: string;
};

type RunLayers = {
  looseEvents: RunEvent[];
  loosePlugins: PluginLayer[];
  orchestratorEvents: RunEvent[];
  stages: StageLayer[];
};

export function App() {
  const [input, setInput] = useState("");
  const [runs, setRuns] = useState<ConversationRun[]>([]);
  const [activeClientId, setActiveClientId] = useState<string | null>(null);
  const [isTimelineOpen, setIsTimelineOpen] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);
  const activeClientIdRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  const activeRun = useMemo(
    () => runs.find((run) => run.clientId === activeClientId) ?? null,
    [activeClientId, runs],
  );
  const latestRun = runs.length ? runs[runs.length - 1] : null;
  const isRunning = Boolean(activeRun && (activeRun.status === "starting" || activeRun.status === "running"));
  const totalEvents = runs.reduce((count, run) => count + run.events.length, 0);
  const latestEvent = latestRun?.events.length ? latestRun.events[latestRun.events.length - 1] : undefined;

  const statusLabel = useMemo(() => {
    if (!latestRun) return "idle";
    if (latestRun.status === "error") return "error";
    if (latestRun.status === "starting" || latestRun.status === "running") return "running";
    if (latestRun.status === "complete") return "complete";
    return latestRun.status;
  }, [latestRun]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [runs]);

  useEffect(() => {
    return () => sourceRef.current?.close();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submitPrompt();
  }

  async function submitPrompt() {
    const trimmed = input.trim();
    if (!trimmed || isRunning) return;

    const clientId = createClientId();
    const nextRun: ConversationRun = {
      clientId,
      createdAt: Date.now(),
      error: "",
      events: [],
      finalResponse: "",
      prompt: trimmed,
      runId: null,
      status: "starting",
    };

    sourceRef.current?.close();
    activeClientIdRef.current = clientId;
    setActiveClientId(clientId);
    setRuns((current) => [...current, nextRun]);
    setInput("");
    setIsTimelineOpen(false);

    try {
      const run = await startRun(trimmed);
      updateRun(clientId, (current) => ({
        ...current,
        runId: run.run_id,
        status: "running",
      }));
      connectRunStream(run.run_id, clientId);
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : "Unable to start run.";
      updateRun(clientId, (current) => ({
        ...current,
        error: message,
        finalResponse: "",
        status: "error",
      }));
      activeClientIdRef.current = null;
      setActiveClientId(null);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    submitPrompt();
  }

  function clearConversation() {
    sourceRef.current?.close();
    sourceRef.current = null;
    activeClientIdRef.current = null;
    setActiveClientId(null);
    setRuns([]);
    setInput("");
    setIsTimelineOpen(false);
  }

  function updateRun(clientId: string, updater: (run: ConversationRun) => ConversationRun) {
    setRuns((current) => current.map((run) => (run.clientId === clientId ? updater(run) : run)));
  }

  function connectRunStream(nextRunId: string, clientId: string) {
    const source = new EventSource(`/api/runs/${nextRunId}/events`);
    sourceRef.current = source;

    const handleMessage = (message: MessageEvent<string>) => {
      if (activeClientIdRef.current !== clientId) return;

      const runEvent = JSON.parse(message.data) as RunEvent;
      updateRun(clientId, (current) => {
        const nextStatus = getStatusFromEvent(runEvent, current.status);
        const serverError = runEvent.type === "server_error" && typeof runEvent.message === "string";
        return {
          ...current,
          error: serverError ? runEvent.message ?? "" : current.error,
          events: [...current.events, runEvent],
          finalResponse:
            typeof runEvent.final_response === "string"
              ? runEvent.final_response
              : serverError
                ? runEvent.message ?? ""
                : current.finalResponse,
          status: nextStatus,
        };
      });

      if (TERMINAL_EVENTS.has(runEvent.type)) {
        activeClientIdRef.current = null;
        setActiveClientId(null);
        source.close();
      }
    };

    for (const eventName of STREAM_EVENTS) {
      source.addEventListener(eventName, handleMessage as EventListener);
    }

    source.onerror = () => {
      if (sourceRef.current === source && activeClientIdRef.current === clientId) {
        updateRun(clientId, (current) => ({
          ...current,
          error: "The run stream disconnected.",
          status: "error",
        }));
        activeClientIdRef.current = null;
        setActiveClientId(null);
      }
      source.close();
    };
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <Sparkles size={19} strokeWidth={2.4} />
          </div>
          <div>
            <h1>Agentic Canvas</h1>
            <p>
              {latestRun?.runId
                ? `run ${latestRun.runId}`
                : latestEvent
                  ? describeEvent(latestEvent)
                  : runs.length
                    ? `${runs.length} displayed runs`
                    : "Ready for a new run"}
            </p>
          </div>
        </div>

        <div className="topbar-actions">
          <button
            className="ghost-button"
            type="button"
            onClick={() => setIsTimelineOpen((current) => !current)}
            aria-expanded={isTimelineOpen}
            aria-controls="timeline-panel"
          >
            <PanelRightOpen size={16} aria-hidden="true" />
            Activity
            <span>{totalEvents}</span>
          </button>
          <button className="ghost-button clear-button" type="button" onClick={clearConversation} disabled={!runs.length}>
            <Trash2 size={16} aria-hidden="true" />
            Clear
          </button>
          <div className={`status-pill status-${statusLabel}`}>
            <span aria-hidden="true" />
            {statusLabel}
          </div>
        </div>
      </header>

      <section className={`workspace ${isTimelineOpen ? "timeline-open" : ""}`}>
        <section className="conversation-panel" aria-label="Conversation">
          <div className="transcript">
            {runs.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon" aria-hidden="true">
                  <MessagesSquare size={28} />
                </div>
                <h2>Start a workspace run</h2>
                <p>Send prompts continuously. Each run stays visible until you clear the displayed conversation.</p>
              </div>
            )}

            {runs.map((run) => (
              <RunConversation key={run.clientId} run={run} />
            ))}

            <div ref={transcriptEndRef} />
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="run-input">
              Prompt
            </label>
            <textarea
              id="run-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder={isRunning ? "Current run is still working..." : "Message Agentic Canvas..."}
              rows={1}
              disabled={isRunning}
            />
            <button className="send-button" type="submit" disabled={!input.trim() || isRunning} aria-label="Run prompt">
              {isRunning ? <Loader2 size={18} aria-hidden="true" /> : <SendHorizontal size={18} aria-hidden="true" />}
            </button>
          </form>

          <div className="composer-status">
            <span>
              {runs.length ? `${runs.length} runs displayed / ${totalEvents} events streamed` : "Idle"}
            </span>
            {latestRun?.error && <strong>{latestRun.error}</strong>}
          </div>
        </section>

        <aside className="timeline-panel" id="timeline-panel" aria-label="Run activity">
          <div className="timeline-heading">
            <div>
              <h2>Activity Layers</h2>
              <p>{runs.length ? `${runs.length} displayed runs` : "No active run"}</p>
            </div>
            <button className="icon-button" type="button" onClick={() => setIsTimelineOpen(false)} aria-label="Collapse activity">
              <ChevronDown size={18} aria-hidden="true" />
            </button>
          </div>

          <div className="timeline-list">
            {runs.length === 0 && (
              <div className="timeline-empty">
                <SquareTerminal size={18} aria-hidden="true" />
                Run, stage, and plugin layers will appear here.
              </div>
            )}
            {runs.map((run) => (
              <RunActivityStack key={run.clientId} run={run} compact />
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function RunConversation({ run }: { run: ConversationRun }) {
  return (
    <section className="run-thread" aria-label={`Run for ${run.prompt}`}>
      <article className="message message-user">
        <div className="message-avatar" aria-hidden="true">
          <UserRound size={17} />
        </div>
        <div className="message-content">
          <div className="message-meta">You</div>
          <p>{run.prompt}</p>
        </div>
      </article>

      <article className="message message-activity">
        <div className="message-avatar" aria-hidden="true">
          {run.status === "running" || run.status === "starting" ? <Loader2 size={18} /> : <Layers3 size={18} />}
        </div>
        <div className="message-content">
          <RunActivityStack run={run} />
        </div>
      </article>

      {run.finalResponse && (
        <article className={`message ${run.status === "error" ? "message-error" : "message-assistant"}`}>
          <div className="message-avatar" aria-hidden="true">
            {run.status === "error" ? <AlertCircle size={18} /> : <Sparkles size={18} />}
          </div>
          <div className="message-content">
            <div className="message-meta">{run.status === "error" ? "Error" : "Final response"}</div>
            <pre>{run.finalResponse}</pre>
          </div>
        </article>
      )}

      {run.error && !run.finalResponse && (
        <article className="message message-error">
          <div className="message-avatar" aria-hidden="true">
            <AlertCircle size={18} />
          </div>
          <div className="message-content">
            <div className="message-meta">Error</div>
            <p>{run.error}</p>
          </div>
        </article>
      )}
    </section>
  );
}

function RunActivityStack({ compact = false, run }: { compact?: boolean; run: ConversationRun }) {
  const layers = buildRunLayers(run.events);
  const eventCountLabel = `${run.events.length} ${run.events.length === 1 ? "event" : "events"}`;

  return (
    <div className={`activity-stack ${compact ? "activity-compact" : ""}`}>
      <div className={`activity-root event-${getRunTone(run.status)}`}>
        <div className="activity-line">
          <div className="activity-icon" aria-hidden="true">
            {renderRunIcon(run.status)}
          </div>
          <div className="activity-copy">
            <div className="activity-title">
              <strong>Run</strong>
              <span>{run.runId ? run.runId.slice(0, 8) : "starting"}</span>
            </div>
            <p>{getRunSummary(run)}</p>
          </div>
          <span className="activity-count">{eventCountLabel}</span>
        </div>

        <div className="activity-children">
          {orderStages(layers.stages, "pre").map((stage) => (
            <StageActivityNode stage={stage} key={stage.name} />
          ))}

          {layers.orchestratorEvents.map((event, index) => (
            <ActivityEvent event={event} key={`${event.type}-orchestrator-${index}`} />
          ))}

          {orderStages(layers.stages, "post").map((stage) => (
            <StageActivityNode stage={stage} key={stage.name} />
          ))}

          {layers.loosePlugins.map((plugin) => (
            <div className="activity-node plugin-node" key={`run-${plugin.name}-${plugin.mode ?? "plugin"}`}>
              <div className="activity-line">
                <div className="activity-icon" aria-hidden="true">
                  <Blocks size={15} />
                </div>
                <div className="activity-copy">
                  <div className="activity-title">
                    <strong>Plugin</strong>
                    <span>{plugin.name}</span>
                  </div>
                  <p>{plugin.mode ? `${plugin.mode} / ${plugin.events.length} events` : `${plugin.events.length} events`}</p>
                </div>
              </div>
              <div className="activity-children">
                {plugin.events.map((event, index) => (
                  <ActivityEvent event={event} key={`${event.type}-${plugin.name}-root-${index}`} />
                ))}
              </div>
            </div>
          ))}

          {layers.looseEvents.map((event, index) => (
            <ActivityEvent event={event} key={`${event.type}-loose-${index}`} />
          ))}
        </div>
      </div>
    </div>
  );
}

function StageActivityNode({ stage }: { stage: StageLayer }) {
  const boundaryEvents = splitBoundaryEvents(stage.looseEvents);

  return (
    <div className="activity-node stage-node">
      <div className="activity-line">
        <div className="activity-icon" aria-hidden="true">
          <GitBranch size={15} />
        </div>
        <div className="activity-copy">
          <div className="activity-title">
            <strong>Stage</strong>
            <span>{stage.name}</span>
          </div>
          <p>{summarizeLayer(stage.events, stage.looseEvents)}</p>
        </div>
      </div>

      <div className="activity-children">
        {boundaryEvents.opening.map((event, index) => (
          <ActivityEvent event={event} key={`${event.type}-${stage.name}-opening-${index}`} />
        ))}

        {stage.plugins.map((plugin) => (
          <div className="activity-node plugin-node" key={`${stage.name}-${plugin.name}-${plugin.mode ?? "plugin"}`}>
            <div className="activity-line">
              <div className="activity-icon" aria-hidden="true">
                <Blocks size={15} />
              </div>
              <div className="activity-copy">
                <div className="activity-title">
                  <strong>Plugin</strong>
                  <span>{plugin.name}</span>
                </div>
                <p>{plugin.mode ? `${plugin.mode} / ${plugin.events.length} events` : `${plugin.events.length} events`}</p>
              </div>
            </div>
            <div className="activity-children">
              {plugin.events.map((event, index) => (
                <ActivityEvent event={event} key={`${event.type}-${plugin.name}-${index}`} />
              ))}
            </div>
          </div>
        ))}

        {boundaryEvents.closing.map((event, index) => (
          <ActivityEvent event={event} key={`${event.type}-${stage.name}-closing-${index}`} />
        ))}
      </div>
    </div>
  );
}

function ActivityEvent({ event }: { event: RunEvent }) {
  return (
    <div className={`activity-node event-node event-${getEventTone(event.type)}`}>
      <div className="activity-line">
        <div className="activity-icon" aria-hidden="true">
          {renderEventIcon(event.type, 14)}
        </div>
        <div className="activity-copy">
          <div className="activity-title">
            <strong>{formatEventType(event.type)}</strong>
            <time>{formatTime(event.timestamp)}</time>
          </div>
          <p>{describeEvent(event)}</p>
        </div>
      </div>
    </div>
  );
}

function buildRunLayers(events: RunEvent[]): RunLayers {
  const layers: RunLayers = {
    looseEvents: [],
    loosePlugins: [],
    orchestratorEvents: [],
    stages: [],
  };
  const stageMap = new Map<string, StageLayer>();
  const loosePluginMap = new Map<string, PluginLayer>();

  for (const event of events) {
    if (isRunEvent(event)) {
      continue;
    }

    if (isOrchestratorEvent(event)) {
      layers.orchestratorEvents.push(event);
      continue;
    }

    const stageName = event.stage ? String(event.stage) : "";
    const pluginName = event.plugin ? String(event.plugin) : "";

    if (stageName) {
      const stage = getOrCreateStage(layers, stageMap, stageName);
      stage.events.push(event);

      if (pluginName) {
        getOrCreatePlugin(stage.plugins, `${pluginName}:${event.mode ?? ""}`, pluginName, event.mode).events.push(event);
      } else {
        stage.looseEvents.push(event);
      }
      continue;
    }

    if (pluginName) {
      const key = `${pluginName}:${event.mode ?? ""}`;
      if (!loosePluginMap.has(key)) {
        loosePluginMap.set(key, {
          events: [],
          mode: event.mode,
          name: pluginName,
        });
      }
      loosePluginMap.get(key)?.events.push(event);
      continue;
    }

    layers.looseEvents.push(event);
  }

  layers.loosePlugins = [...loosePluginMap.values()];

  return layers;
}

function getOrCreateStage(layers: RunLayers, stageMap: Map<string, StageLayer>, name: string) {
  let stage = stageMap.get(name);
  if (!stage) {
    stage = {
      events: [],
      looseEvents: [],
      name,
      plugins: [],
    };
    stageMap.set(name, stage);
    layers.stages.push(stage);
  }
  return stage;
}

function getOrCreatePlugin(plugins: PluginLayer[], key: string, name: string, mode?: string) {
  let plugin = plugins.find((item) => `${item.name}:${item.mode ?? ""}` === key);
  if (!plugin) {
    plugin = {
      events: [],
      mode,
      name,
    };
    plugins.push(plugin);
  }
  return plugin;
}

function isRunEvent(event: RunEvent) {
  return event.type.startsWith("run_") || event.type === "server_error";
}

function isOrchestratorEvent(event: RunEvent) {
  return event.type.startsWith("orchestrator_");
}

function orderStages(stages: StageLayer[], position: "pre" | "post") {
  if (position === "pre") {
    return stages.filter((stage) => !stage.name.toLowerCase().startsWith("post_"));
  }
  return stages.filter((stage) => stage.name.toLowerCase().startsWith("post_"));
}

function splitBoundaryEvents(events: RunEvent[]) {
  return {
    closing: events.filter((event) => event.type.endsWith("_finished")),
    opening: events.filter((event) => !event.type.endsWith("_finished")),
  };
}

function createClientId() {
  return `run-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getStatusFromEvent(event: RunEvent, current: RunStatus): RunStatus {
  if (event.type === "run_completed") return "complete";
  if (event.type === "run_aborted") return "aborted";
  if (event.type === "run_stopped") return "stopped";
  if (event.type === "server_error") return "error";
  if (event.type === "run_started") return "running";
  return current === "starting" ? "running" : current;
}

function getRunSummary(run: ConversationRun) {
  const latestEvent = run.events.length ? run.events[run.events.length - 1] : undefined;
  if (run.error) return run.error;
  if (run.finalResponse && run.status === "complete") return "Completed with a final response.";
  if (latestEvent) return describeEvent(latestEvent);
  if (run.status === "starting") return "Creating the run...";
  return "Waiting for activity.";
}

function summarizeLayer(events: RunEvent[], looseEvents: RunEvent[]) {
  const latest = events.length ? events[events.length - 1] : undefined;
  if (!latest) return "Waiting for stage activity.";
  return `${events.length} events / ${looseEvents.length} stage updates / latest ${formatEventType(latest.type)}`;
}

function renderRunIcon(status: RunStatus) {
  if (status === "complete") return <CheckCircle2 size={16} />;
  if (status === "error" || status === "aborted") return <AlertCircle size={16} />;
  if (status === "stopped") return <CircleStop size={16} />;
  if (status === "starting" || status === "running") return <Loader2 size={16} />;
  return <Play size={16} />;
}

function renderEventIcon(type: string, size = 16): ReactNode {
  if (type === "run_completed") return <CheckCircle2 size={size} />;
  if (type === "run_aborted" || type === "server_error") return <AlertCircle size={size} />;
  if (type === "run_stopped") return <CircleStop size={size} />;
  if (type.endsWith("_started")) return <Activity size={size} />;
  if (type.endsWith("_finished")) return <CheckCircle2 size={size} />;
  return <Clock3 size={size} />;
}

function getRunTone(status: RunStatus) {
  if (status === "complete") return "success";
  if (status === "error" || status === "aborted") return "danger";
  if (status === "starting" || status === "running") return "active";
  return "neutral";
}

function getEventTone(type: string) {
  if (type === "run_completed" || type.endsWith("_finished")) return "success";
  if (type === "server_error" || type === "run_aborted") return "danger";
  if (type.endsWith("_started") || type === "run_started") return "active";
  if (type === "user_input_requested") return "waiting";
  return "neutral";
}

function formatEventType(type: string) {
  return type.replace(/_/g, " ");
}

function formatTime(timestamp: string) {
  if (!timestamp) return "";
  return new Date(timestamp).toLocaleTimeString();
}

function describeEvent(event: RunEvent) {
  if (event.plugin) {
    return [event.mode, event.plugin, event.stage].filter(Boolean).join(" / ");
  }
  if (event.stage) return String(event.stage);
  if (event.message) return String(event.message);
  if (event.reason) return String(event.reason);
  if (event.status) return String(event.status);
  if (event.response) return String(event.response);
  return event.run_id ? event.run_id : "event received";
}
