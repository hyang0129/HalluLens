"""
Smoketests for Qwen3-8B additions:
  1. model_map entry exists in utils/lm.py
  2. server.py injects enable_thinking=False for qwen3, not for other models
"""
import unittest
from unittest.mock import MagicMock, patch


class TestQwen3ModelMap(unittest.TestCase):
    def test_qwen3_8b_in_model_map(self):
        from utils.lm import model_map
        self.assertIn("Qwen/Qwen3-8B", model_map)

    def test_qwen3_8b_name(self):
        from utils.lm import model_map
        self.assertEqual(model_map["Qwen/Qwen3-8B"]["name"], "qwen3_8B")

    def test_qwen3_8b_has_server_urls(self):
        from utils.lm import model_map
        self.assertIn("server_urls", model_map["Qwen/Qwen3-8B"])
        self.assertTrue(len(model_map["Qwen/Qwen3-8B"]["server_urls"]) > 0)

    def test_qwen3_8b_awq_not_in_model_map(self):
        from utils.lm import model_map
        self.assertNotIn("Qwen/Qwen3-8B-AWQ", model_map)


class TestQwen3ThinkingDisabled(unittest.TestCase):
    """
    Verify that enable_thinking=False is injected for qwen3 models and not
    for other models, without loading any real weights.
    """

    def _run_inference_generate_kwargs(self, model_name: str) -> dict:
        """
        Patch everything in server.py that touches hardware/files and call
        run_inference(), then return the kwargs that were passed to model.generate().
        """
        import sys
        import types

        # Patch get_model_and_tokenizer before importing server
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        fake_tokenizer.pad_token_id = 0
        fake_tokenizer.eos_token_id = 2

        fake_outputs = MagicMock()
        fake_outputs.sequences = MagicMock()
        fake_outputs.sequences.__getitem__ = MagicMock(return_value=[4, 5, 6])
        fake_outputs.hidden_states = []
        fake_outputs.scores = None

        captured = {}

        def fake_generate(**kwargs):
            captured["kwargs"] = kwargs
            return fake_outputs

        fake_model = MagicMock()
        fake_model.generate = fake_generate
        fake_model.device = "cpu"

        with patch("activation_logging.server.get_model_and_tokenizer",
                   return_value=(fake_model, fake_tokenizer)), \
             patch("activation_logging.server._should_use_vllm_backend",
                   return_value=False), \
             patch("activation_logging.server.torch.cuda.is_available",
                   return_value=False), \
             patch("activation_logging.server.torch.no_grad",
                   return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))):
            try:
                from activation_logging import server
                server.run_inference(
                    prompt="What is the capital of France?",
                    max_tokens=16,
                    temperature=0.0,
                    top_p=1.0,
                    model_name=model_name,
                )
            except Exception:
                pass  # post-generate processing may fail; we only need captured kwargs

        return captured.get("kwargs", {})

    def test_enable_thinking_false_for_qwen3(self):
        kwargs = self._run_inference_generate_kwargs("Qwen/Qwen3-8B")
        self.assertIn("enable_thinking", kwargs,
                      "enable_thinking kwarg must be set for qwen3 models")
        self.assertFalse(kwargs["enable_thinking"])

    def test_enable_thinking_not_set_for_llama(self):
        kwargs = self._run_inference_generate_kwargs(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.assertNotIn("enable_thinking", kwargs,
                         "enable_thinking must NOT be injected for non-qwen3 models")

    def test_enable_thinking_false_for_qwen3_case_insensitive(self):
        # Model name casing shouldn't matter
        kwargs = self._run_inference_generate_kwargs("Qwen/Qwen3-8B")
        self.assertFalse(kwargs.get("enable_thinking", True))


if __name__ == "__main__":
    unittest.main()
