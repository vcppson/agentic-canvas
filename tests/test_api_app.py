from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from fastapi.testclient import TestClient

from agentic_canvas.kernel.workspace import init_workspace
from agentic_canvas_api.app import create_app


class ApiAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = init_workspace(Path(self.tmp.name) / "workspace")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_post_run_returns_id_and_sse_replays_terminal_event(self) -> None:
        client = TestClient(create_app(self.workspace))

        created = client.post("/api/runs", json={"input": "hello"})

        self.assertEqual(created.status_code, 200)
        run_id = created.json()["run_id"]
        self.assertTrue(run_id)

        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            self.assertEqual(response.status_code, 200)
            events = _read_sse_events(response.iter_lines())

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "run_started")
        self.assertIn("orchestrator_finished", event_types)
        self.assertEqual(event_types[-1], "run_completed")
        self.assertTrue(events[-1]["final_response"])

    def test_unknown_run_stream_returns_404(self) -> None:
        client = TestClient(create_app(self.workspace))

        response = client.get("/api/runs/missing/events")

        self.assertEqual(response.status_code, 404)

    def test_blank_input_is_rejected(self) -> None:
        client = TestClient(create_app(self.workspace))

        response = client.post("/api/runs", json={"input": "   "})

        self.assertEqual(response.status_code, 400)

    def test_cors_origins_can_be_configured(self) -> None:
        with patch.dict(
            os.environ,
            {"AGENTIC_CANVAS_CORS_ORIGINS": "https://app.example.com, https://admin.example.com"},
        ):
            client = TestClient(create_app(self.workspace))

        response = client.options(
            "/api/runs",
            headers={
                "Access-Control-Request-Method": "POST",
                "Origin": "https://admin.example.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://admin.example.com",
        )


def _read_sse_events(lines) -> list[dict]:
    events = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line.removeprefix("data: ")))
    return events


if __name__ == "__main__":
    unittest.main()
