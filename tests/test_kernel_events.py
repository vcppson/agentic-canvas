from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_canvas.kernel.run import Kernel
from agentic_canvas.kernel.trigger import Trigger
from agentic_canvas.kernel.workspace import init_workspace
from tests.support import RaisingProvider, StaticProvider, set_stages, write_plugin


class KernelEventsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = init_workspace(Path(self.tmp.name) / "workspace")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_event_handler_receives_completed_run_events_in_order(self) -> None:
        events: list[dict] = []

        context = Kernel(
            self.workspace,
            provider=StaticProvider("done"),
            event_handler=events.append,
        ).start(Trigger("hello"))

        event_types = [event["type"] for event in events]
        self.assertEqual(context.status, "completed")
        self.assertEqual(event_types[0], "run_started")
        self.assertLess(event_types.index("orchestrator_started"), event_types.index("orchestrator_finished"))
        self.assertEqual(event_types[-1], "run_completed")
        self.assertEqual(events[-1]["final_response"], "done")
        self.assertTrue(all(event.get("run_id") == context.run_id for event in events))

    def test_event_handler_receives_run_stopped_terminal_event(self) -> None:
        write_plugin(
            self.workspace,
            "stopper",
            """
def plugin_main(request):
    return {"decision": "stop", "response": "stopped"}
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["stopper"])
        events: list[dict] = []

        context = Kernel(
            self.workspace,
            provider=RaisingProvider(),
            event_handler=events.append,
        ).start(Trigger("stop"))

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.final_response, "stopped")
        self.assertEqual(events[-1]["type"], "run_stopped")
        self.assertEqual(events[-1]["final_response"], "stopped")

    def test_event_handler_receives_run_aborted_terminal_event(self) -> None:
        write_plugin(
            self.workspace,
            "bad_stage",
            """
def plugin_main(request):
    return {"data": "no decision"}
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["bad_stage"])
        events: list[dict] = []

        context = Kernel(
            self.workspace,
            provider=StaticProvider("unused"),
            event_handler=events.append,
        ).start(Trigger("abort"))

        self.assertEqual(context.status, "aborted")
        self.assertEqual(events[-1]["type"], "run_aborted")
        self.assertIn("returned no decision", events[-1]["final_response"])


if __name__ == "__main__":
    unittest.main()
