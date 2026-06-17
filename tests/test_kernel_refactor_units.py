from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_canvas.kernel.manifest_registry import load_manifest_directory
from agentic_canvas.kernel.plugin_protocol import PluginExecutionResult
from agentic_canvas.kernel.storage import FileSystemWorkspaceStorage
from agentic_canvas.plugins.manifest import ManifestError, PluginManifest


class PluginProtocolResultTest(unittest.TestCase):
    def test_transport_error_and_result_payloads_normalize_to_trace_shape(self) -> None:
        failed = PluginExecutionResult.transport_error(
            plugin_name="broken",
            mode="call",
            message="boom",
            stderr="trace",
        )
        self.assertFalse(failed.ok)
        self.assertEqual(failed.kind, "error")
        self.assertEqual(failed.to_trace_dict()["message"], "boom")

        scalar = PluginExecutionResult.from_transport(
            plugin_name="value_plugin",
            mode="call",
            payload={"ok": True, "result": "done"},
            stderr="",
            duration_seconds=0.1,
        )
        self.assertTrue(scalar.ok)
        self.assertEqual(scalar.result, {"value": "done"})

        control = PluginExecutionResult.from_transport(
            plugin_name="stopper",
            mode="stage",
            payload={"ok": True, "kind": "run_control", "decision": "stop"},
            stderr="",
            duration_seconds=0.1,
        )
        self.assertTrue(control.is_run_control)
        self.assertEqual(control.decision, "stop")


class ManifestRegistryLoaderTest(unittest.TestCase):
    def test_load_manifest_directory_validates_names_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / "plugins" / "alpha"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "manifest.json").write_text(
                """
{
  "name": "alpha",
  "version": "0.1.0",
  "summary": "Alpha.",
  "documentation": "Alpha.",
  "entry_point": "plugin.py:plugin_main",
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "requirements": [],
  "libraries": [],
  "compatibility": {"agentic_canvas": ">=0.1.0"}
}
""".lstrip(),
                encoding="utf-8",
            )
            storage = FileSystemWorkspaceStorage(root)

            manifests = load_manifest_directory(
                storage=storage,
                directory="plugins",
                kind="plugin",
                manifest_factory=lambda data, path: PluginManifest.from_dict(data, path=path),
                manifest_name=lambda manifest: manifest.name,
                error_type=ManifestError,
            )

            self.assertEqual(list(manifests), ["alpha"])

            beta = root / "plugins" / "beta"
            beta.mkdir()
            (beta / "manifest.json").write_text(
                (plugin_dir / "manifest.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "does not match directory"):
                load_manifest_directory(
                    storage=storage,
                    directory="plugins",
                    kind="plugin",
                    manifest_factory=lambda data, path: PluginManifest.from_dict(data, path=path),
                    manifest_name=lambda manifest: manifest.name,
                    error_type=ManifestError,
                )


if __name__ == "__main__":
    unittest.main()
