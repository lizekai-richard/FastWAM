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
        self.assertIn("num_slots: 4", block)
        self.assertIn("chunk_size: 16", block)
        self.assertEqual(4 * 16, 64)
        self.assertIn("torch_compile_infer_action: false", text)
        self.assertIn("torch_compile_mode: max-autotune", text)
        self.assertIn("torch_compile_dynamic: null", text)
        self.assertIn("torch_compile_disable_cudagraphs: null", text)

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
        self.assertIn("--torch-compile", constants)

    def test_compile_execution_reports_streaming_core_scope(self) -> None:
        api = _load_api_module()

        class Model:
            torch_compile_infer_action = True
            torch_compile_scope = "streaming_inference_kernels"

        execution = api.describe_compile_execution(Model(), "streaming")
        self.assertEqual(
            execution.canonical_execution_mode,
            "eager_wrapper_with_compiled_streaming_kernels",
        )
        self.assertEqual(execution.public_wrapper_execution_mode, "eager")
        self.assertEqual(execution.video_prefill_core_execution_mode, "compiled")
        self.assertEqual(execution.action_core_execution_mode, "compiled")
        self.assertTrue(execution.profiled_matches_canonical)

    def test_compile_execution_distinguishes_legacy_profiled_companion(self) -> None:
        api = _load_api_module()

        class LegacyCompiled:
            torch_compile_infer_action = True
            torch_compile_scope = "legacy_infer_action"

        compiled = api.describe_compile_execution(LegacyCompiled(), "legacy")
        self.assertEqual(compiled.public_wrapper_execution_mode, "compiled")
        self.assertFalse(compiled.profiled_matches_canonical)

        class Eager:
            torch_compile_infer_action = False
            torch_compile_scope = "none"

        eager = api.describe_compile_execution(Eager(), "streaming")
        self.assertEqual(eager.canonical_execution_mode, "eager")
        self.assertEqual(eager.video_prefill_core_execution_mode, "eager")
        self.assertEqual(eager.action_core_execution_mode, "eager")
        self.assertTrue(eager.profiled_matches_canonical)

    def test_compile_execution_rejects_scope_mode_mismatch(self) -> None:
        api = _load_api_module()

        class WrongScope:
            torch_compile_infer_action = True
            torch_compile_scope = "legacy_infer_action"

        with self.assertRaisesRegex(RuntimeError, "Streaming inference requires"):
            api.describe_compile_execution(WrongScope(), "streaming")

    def test_robotwin_forwards_compile_settings_with_unsupported_model_guard(self) -> None:
        deploy_yml = (
            ROOT / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.yml"
        ).read_text(encoding="utf-8")
        launcher = (
            ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
        ).read_text(encoding="utf-8")
        policy = (
            ROOT
            / "experiments"
            / "robotwin"
            / "fastwam_policy"
            / "deploy_policy.py"
        ).read_text(encoding="utf-8")
        for key in (
            "torch_compile_infer_action",
            "torch_compile_mode",
            "torch_compile_dynamic",
            "torch_compile_disable_cudagraphs",
        ):
            self.assertIn(key, deploy_yml)
            self.assertIn(key, launcher)
            self.assertIn(key, policy)
        self.assertIn('compile_supported = "torch_compile_infer_action" in cfg.model', policy)

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
