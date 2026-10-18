"""
tests/test_roles.py - Mutator 与 Breaker 角色提示词、输入输出解析及奖励函数测试
"""

import unittest
from roles.mutator import (
    format_mutator_prompt,
    extract_mutator_code,
    compute_mutator_reward
)
from roles.breaker import (
    format_breaker_prompt,
    extract_breaker_input,
    compute_breaker_reward
)


class TestRoles(unittest.TestCase):

    def test_mutator_prompt_formatting(self):
        seed = {
            "id": "test_seed",
            "domain": (0, 50),
            "sanity_inputs": [5, 10, 45],
            "description": "Test check x < 30",
            "func_src": "def f_ref(x: int) -> bool:\n    return x < 30\n",
        }
        prompt = format_mutator_prompt(seed)
        self.assertIn("Sanity inputs", prompt)
        self.assertIn("[5, 10, 45]", prompt)
        self.assertIn("def f_ref", prompt)
        self.assertIn("<answer>", prompt)
        self.assertIn("OLD:", prompt)

    def test_mutator_sweet_spot_curriculum_reward(self):
        # 1. 完全失败 -> 0.0
        r0, d0 = compute_mutator_reward()
        self.assertEqual(r0, 0.0)

        # 2. 格式提取成功但语法错误 -> 0.10
        r1, d1 = compute_mutator_reward(is_valid_format=True)
        self.assertEqual(r1, 0.10)

        # 3. 语法 AST 通过但 sanity 失败 -> 0.25 (0.10 + 0.15)
        r2, d2 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True)
        self.assertEqual(r2, 0.25)

        # 4. sanity 通过但全域等价 (非有效变异) -> 0.45 (0.10 + 0.15 + 0.20)
        r3, d3 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True)
        self.assertEqual(r3, 0.45)

        # 5. 黄金带 (Golden Teaching Zone, 0 < p_solve <= 0.5): 提供最大方差与梯度 -> 满分 1.00
        r4, d4 = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.25
        )
        self.assertEqual(r4, 1.00)

        # 6. 盲区暂存 (Hard Buffer Zone, p_solve == 0.0) -> 0.80
        r5, d5 = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.0
        )
        self.assertEqual(r5, 0.80)

        # 7. 过易惩罚 (Too Easy Zone, p_solve == 1.0) -> 仅剩基础分 0.45
        r6, d6 = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=1.0
        )
        self.assertEqual(r6, 0.45)

    def test_mutator_coevolution_reward(self):
        # 协进化期 (is_warmup=False) 严惩无效与等价变异
        # 1. 语法/sanity 失败 -> 0.0 或微量语法分 0.05
        r0, _ = compute_mutator_reward(is_valid_format=True, is_warmup=False)
        self.assertEqual(r0, 0.0)

        r_ast, _ = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True, is_warmup=False)
        self.assertEqual(r_ast, 0.05)

        # 2. 黄金带 (solve_rate = 0.5) -> 满分 1.00
        r_gold, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.5, is_warmup=False
        )
        self.assertEqual(r_gold, 1.00)

        # 3. 盲区暂存 (solve_rate = 0.0) -> 0.80
        r_hard, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.0, is_warmup=False
        )
        self.assertEqual(r_hard, 0.80)

        # 4. 过易衰减 (solve_rate = 0.75) -> 0.45 + 1.1*0.25 = 0.725
        r_easy, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.75, is_warmup=False
        )
        self.assertAlmostEqual(r_easy, 0.725)

    def test_mutator_tightened_curriculum_reward(self):
        # 中后期深水区课程收紧测试 (is_tightened=True)
        # 1. 黄金带 (0 < p_solve <= 0.35) -> 0.25 + 0.75 = 1.00
        r_gold, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.25, is_warmup=False, is_tightened=True
        )
        self.assertEqual(r_gold, 1.00)

        # 2. 盲区暂存 (solve_rate = 0.0) -> 0.25 + 0.45 = 0.70
        r_hard, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=0.0, is_warmup=False, is_tightened=True
        )
        self.assertEqual(r_hard, 0.70)

        # 3. 过易惩罚 (solve_rate = 1.0) -> 仅剩保底分 0.25
        r_easy, _ = compute_mutator_reward(
            is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True,
            breaker_solve_rate=1.0, is_warmup=False, is_tightened=True
        )
        self.assertEqual(r_easy, 0.25)

    def test_breaker_prompt_formatting(self):

        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        f_mut = "def f_mut(x: int) -> bool:\n    return x <= 30\n"
        prompt = format_breaker_prompt(f_ref, f_mut, (0, 50), "Test spec")
        self.assertIn("f_a", prompt)
        self.assertIn("f_b", prompt)
        self.assertIn("Domain of x: [0, 50]", prompt)
        self.assertIn("<input>INTEGER</input>", prompt)

    def test_breaker_input_parsing(self):
        domain = (0, 50)

        # 标签完整闭合
        val, cat, _ = extract_breaker_input("Analysis done. <input>38</input>", domain)
        self.assertEqual(val, 38)
        self.assertEqual(cat, "valid_extracted")

        # 未闭合标签
        val, cat, _ = extract_breaker_input("Thought: <input> 42", domain)
        self.assertEqual(val, 42)
        self.assertEqual(cat, "valid_extracted")

        # 赋值语句
        val, cat, _ = extract_breaker_input("Therefore x = 25 is the counterexample", domain)
        self.assertEqual(val, 25)
        self.assertEqual(cat, "valid_extracted")

        # 越界输入
        val, cat, _ = extract_breaker_input("<input>99</input>", domain)
        self.assertEqual(val, 99)
        self.assertEqual(cat, "out_of_domain")

        # 无整数
        val, cat, _ = extract_breaker_input("No answer here.", domain)
        self.assertIsNone(val)
        self.assertEqual(cat, "format_malformed")

    def test_breaker_reward(self):
        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        f_mut = "def f_mut(x: int) -> bool:\n    return x <= 30\n"
        domain = (0, 50)

        # 命中反例 30 -> 1.0
        r_hit, d_hit = compute_breaker_reward(f_ref, f_mut, 30, domain)
        self.assertEqual(r_hit, 1.0)
        self.assertTrue(d_hit["success"])

        # 未命中反例 20 -> 0.0
        r_miss, d_miss = compute_breaker_reward(f_ref, f_mut, 20, domain)
        self.assertEqual(r_miss, 0.0)
        self.assertFalse(d_miss["success"])

        # None 输入 -> 0.0
        r_none, d_none = compute_breaker_reward(f_ref, f_mut, None, domain)
        self.assertEqual(r_none, 0.0)


if __name__ == "__main__":
    unittest.main()
