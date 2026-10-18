"""
tests/test_checkpoint.py - 单元测试: train_state.pt 完整状态存储与恢复
"""

import os
import tempfile
import unittest
import torch

from train_selfplay import save_checkpoint


class DummyTokenizer:
    def save_pretrained(self, save_dir: str):
        pass


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def save_pretrained(self, save_dir: str, selected_adapters=None):
        pass


class TestCheckpointState(unittest.TestCase):

    def test_save_and_load_train_state(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        opt_mutator = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_breaker = torch.optim.AdamW(model.parameters(), lr=5e-5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                save_dir=tmp_dir,
                optimizer_mutator=opt_mutator,
                optimizer_breaker=opt_breaker,
                step=42,
                round_idx=3,
                best_val_pass1=0.75
            )

            state_file = os.path.join(tmp_dir, "train_state.pt")
            self.assertTrue(os.path.isfile(state_file), "train_state.pt must exist in checkpoint directory")

            state = torch.load(state_file, map_location="cpu")
            self.assertEqual(state["step"], 42)
            self.assertEqual(state["round"], 3)
            self.assertEqual(state["best_val_pass1"], 0.75)
            self.assertIn("optimizer_mutator", state)
            self.assertIn("optimizer_breaker", state)
            self.assertIn("torch_rng_state", state)
            self.assertIn("python_rng_state", state)


if __name__ == "__main__":
    unittest.main()
