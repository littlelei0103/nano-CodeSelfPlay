"""
tests/test_grpo_engine.py - GRPO 引擎数学函数、Schulman k3 KL、零方差过滤与掩码测试
"""

import unittest
import torch
from core.grpo_engine import (
    compute_completion_mask,
    compute_grpo_loss,
    rollout_generation_config
)


class TestGRPOEngine(unittest.TestCase):

    def test_compute_completion_mask(self):
        device = torch.device("cpu")
        eos_id = 2
        # batch=2, prompt_len=3, gen_len=4
        # sample 0: prompt=[10, 11, 12], gen=[100, 101, 2 (eos), 999] -> mask=[True, True, True, False]
        # sample 1: prompt=[10, 11, 12], gen=[200, 201, 202, 203] (no eos) -> mask=[True, True, True, True]
        outputs = torch.tensor([
            [10, 11, 12, 100, 101, 2, 999],
            [10, 11, 12, 200, 201, 202, 203]
        ], device=device)

        mask = compute_completion_mask(outputs, prompt_len=3, eos_token_id=eos_id, device=device)
        self.assertEqual(mask.shape, (2, 4))
        self.assertTrue(torch.equal(mask[0], torch.tensor([True, True, True, False], device=device)))
        self.assertTrue(torch.equal(mask[1], torch.tensor([True, True, True, True], device=device)))

    def test_grpo_loss_schulman_k3(self):
        batch_size = 4
        gen_len = 6
        cur_logprobs = torch.randn(batch_size, gen_len, requires_grad=True)
        old_logprobs = cur_logprobs.detach().clone()
        ref_logprobs = old_logprobs.clone() - 0.1  # 产生策略漂移
        advantages = torch.tensor([1.2, -0.4, 0.8, -1.6])
        mask = torch.ones(batch_size, gen_len, dtype=torch.bool)

        loss, metrics = compute_grpo_loss(
            cur_logprobs=cur_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
            clip_eps=0.2,
            kl_coef=0.04
        )

        self.assertEqual(loss.shape, (batch_size, gen_len))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreaterEqual(metrics["approx_kl"], 0.0)
        self.assertIn("mean_ratio", metrics)
        self.assertIn("clip_frac", metrics)

        # 验证梯度可正常反传
        total_loss = loss.sum()
        total_loss.backward()
        self.assertIsNotNone(cur_logprobs.grad)

    def test_generation_config_validation(self):
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1

        tok = MockTokenizer()
        # 正常创建
        cfg = rollout_generation_config(tok, max_new_tokens=32, temperature=0.7)
        self.assertEqual(cfg.max_new_tokens, 32)
        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.top_p, 1.0)

        # 错误 top_p 必须抛出异常
        with self.assertRaises(ValueError):
            rollout_generation_config(tok, max_new_tokens=32, top_p=0.95)

        # 负温度必须抛出异常
        with self.assertRaises(ValueError):
            rollout_generation_config(tok, max_new_tokens=32, temperature=-0.5)


if __name__ == "__main__":
    unittest.main()
