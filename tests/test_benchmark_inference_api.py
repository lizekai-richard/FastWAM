from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "experiments" / "benchmark" / "inference_api.py"


def _load_api_module():
    spec = importlib.util.spec_from_file_location("fastwam_benchmark_inference_api", API_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkInferenceAPITest(unittest.TestCase):
    def test_streaming_config_is_opt_in_and_covers_action_horizon(self) -> None:
        text = (ROOT / "configs" / "model" / "fastwam.yaml").read_text(encoding="utf-8")
        block = text.split("\nstreaming_action:\n", maxsplit=1)[1].split(
            "\nloss:\n", maxsplit=1
        )[0]
        self.assertIn("enabled: false", block)
        self.assertIn("num_slots: 8", block)
        self.assertIn("chunk_size: 4", block)
        self.assertEqual(8 * 4, 32)

    def test_benchmark_cli_declares_explicit_inference_mode(self) -> None:
        source = (
            ROOT / "experiments" / "benchmark" / "benchmark_inference_latency.py"
        ).read_text(encoding="utf-8")
        constants = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("--inference-mode", constants)

    def test_streaming_api_is_required_without_legacy_fallback(self) -> None:
        api = _load_api_module()

        class LegacyOnly:
            def infer_action(self):
                return {"action": object()}

        with self.assertRaisesRegex(RuntimeError, "infer_action_streaming"):
            api.resolve_inference_api(LegacyOnly(), "streaming", require_profiled=False)

    def test_streaming_result_keeps_state_external(self) -> None:
        api = _load_api_module()
        state = object()
        action = object()

        unpacked_action, unpacked_state = api.unpack_inference_result(
            {"action": action, "streaming_state": state}, "streaming"
        )
        self.assertIs(unpacked_action, action)
        self.assertIs(unpacked_state, state)

        unpacked_action, unpacked_state = api.unpack_inference_result(
            ({"action": action}, state), "streaming"
        )
        self.assertIs(unpacked_action, action)
        self.assertIs(unpacked_state, state)

        with self.assertRaisesRegex(RuntimeError, "caller-owned"):
            api.unpack_inference_result({"action": action}, "streaming")

    def test_profiled_streaming_api_is_not_synthesized(self) -> None:
        api = _load_api_module()

        class StreamingWithoutProfiler:
            def infer_action_streaming(self, *, streaming_state, action_generator):
                return {"action": None, "streaming_state": streaming_state}

        with self.assertRaisesRegex(RuntimeError, "infer_action_streaming_profiled"):
            api.resolve_inference_api(
                StreamingWithoutProfiler(), "streaming", require_profiled=True
            )


if __name__ == "__main__":
    unittest.main()
