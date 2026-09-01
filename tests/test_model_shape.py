"""Offline architecture smoke test."""

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM


class ModelShapeTests(unittest.TestCase):
    def test_local_config_supports_a_forward_pass(self):
        config = LlamaConfig.from_json_file("configs/tiny_llama.json")
        model = LlamaForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        self.assertEqual(tuple(output.logits.shape), (1, 8, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
