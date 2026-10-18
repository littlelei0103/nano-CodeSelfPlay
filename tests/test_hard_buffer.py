"""
tests/test_hard_buffer.py - 单元测试: HardBuffer 对抗题库增删改查与加权采样
"""

import os
import tempfile
import unittest

from core.hard_buffer import HardBuffer


class TestHardBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = HardBuffer(capacity=10)
        self.f_ref = "def f_ref(x):\n    return x > 10\n"
        self.f_mut = "def f_ref(x):\n    return x >= 10\n"

    def test_add_and_deduplication(self):
        added1 = self.buffer.add(
            f_ref_src=self.f_ref,
            f_mut_src=self.f_mut,
            domain=(0, 50),
            seed_id="test_01",
            sparsity=0.02,
            witnesses=[10],
            solve_rate=0.25,
            step=1
        )
        self.assertTrue(added1)
        self.assertEqual(len(self.buffer), 1)

        # 重复添加应返回 False，但更新信息
        added2 = self.buffer.add(
            f_ref_src=self.f_ref,
            f_mut_src=self.f_mut,
            domain=(0, 50),
            seed_id="test_01",
            sparsity=0.02,
            witnesses=[10],
            solve_rate=0.5,
            step=2
        )
        self.assertFalse(added2)
        self.assertEqual(len(self.buffer), 1)

    def test_priority_sampling(self):
        # 添加一道简单题与一道极难题
        self.buffer.add(
            f_ref_src="def f_ref(x): return x < 5",
            f_mut_src="def f_ref(x): return x < 10",
            domain=(0, 50),
            seed_id="easy",
            sparsity=0.5,
            solve_rate=1.0,
            step=1
        )
        self.buffer.add(
            f_ref_src=self.f_ref,
            f_mut_src=self.f_mut,
            domain=(0, 50),
            seed_id="hard",
            sparsity=0.02,
            solve_rate=0.0,
            step=1
        )
        self.assertEqual(len(self.buffer), 2)

        sampled = self.buffer.sample(k=1)
        self.assertEqual(len(sampled), 1)
        self.assertIn("key", sampled[0])
        self.assertEqual(sampled[0]["times_sampled"], 1)

    def test_save_and_load(self):
        self.buffer.add(
            f_ref_src=self.f_ref,
            f_mut_src=self.f_mut,
            domain=(0, 50),
            seed_id="test_io",
            sparsity=0.05,
            witnesses=[10],
            solve_rate=0.0,
            step=1
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.buffer.save_jsonl(tmp_path)
            new_buf = HardBuffer()
            new_buf.load_jsonl(tmp_path)
            self.assertEqual(len(new_buf), 1)
            item = list(new_buf.items.values())[0]
            self.assertEqual(item["seed_id"], "test_io")
            self.assertEqual(item["witnesses"], [10])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_candidate_level_solve_rate_tracking(self):
        # 初始插入：4个候选中有1个破解，初始 solve_rate = 0.25
        self.buffer.add(
            f_ref_src=self.f_ref,
            f_mut_src=self.f_mut,
            domain=(0, 50),
            seed_id="candidate_tracking",
            sparsity=0.02,
            candidate_hits=1,
            total_candidates=4,
            step=1
        )
        item = list(self.buffer.items.values())[0]
        self.assertEqual(item["total_hits"], 1)
        self.assertEqual(item["total_trials"], 4)
        self.assertAlmostEqual(item["solve_rate"], 0.25)

        # 第一次回放测试：4个候选中有1个破解
        # total_hits -> 2, total_trials -> 8, solve_rate 仍为 2/8 = 0.25 (而不是跳变到 1.0)
        self.buffer.update_result(item["key"], candidate_hits=1, total_candidates=4)
        self.assertEqual(item["total_hits"], 2)
        self.assertEqual(item["total_trials"], 8)
        self.assertAlmostEqual(item["solve_rate"], 0.25)
        self.assertEqual(item["times_sampled"], 0)
        self.assertEqual(item["groups_evaluated"], 1)
        self.assertEqual(item["times_solved"], 1)

        # 第二次回放测试：4个候选全部未破解
        # total_hits -> 2, total_trials -> 12, solve_rate 平滑下降至 2/12 = 0.1667
        self.buffer.update_result(item["key"], candidate_hits=0, total_candidates=4)
        self.assertEqual(item["total_hits"], 2)
        self.assertEqual(item["total_trials"], 12)
        self.assertAlmostEqual(item["solve_rate"], 2.0 / 12.0)
        self.assertEqual(item["times_sampled"], 0)
        self.assertEqual(item["groups_evaluated"], 2)
        self.assertEqual(item["times_solved"], 1)


if __name__ == "__main__":
    unittest.main()
