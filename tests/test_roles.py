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
        self.assertIn("Sanity Inputs", prompt)
        self.assertIn("[5, 10, 45]", prompt)
        self.assertIn("def f_ref", prompt)
        self.assertIn("Assistant: <code>", prompt)

    def test_mutator_curriculum_reward(self):
        # 1. 不合法突变 -> 0.0
        r_fail, d_fail = compute_mutator_reward(is_valid_mutation=False, breaker_solve_rate=0.0, validation_reason="syntax_error")
        self.assertEqual(r_fail, 0.0)
        self.assertFalse(d_fail["valid"])

        # 2. 合法且彻底难倒 Breaker (solve_rate=0.0) -> 满分 1.0
        r_hard, d_hard = compute_mutator_reward(is_valid_mutation=True, breaker_solve_rate=0.0)
        self.assertEqual(r_hard, 1.0)
        self.assertTrue(d_hard["valid"])

        # 3. 合法但被 Breaker 轻易识破 (solve_rate=1.0) -> 0.0
        r_easy, d_easy = compute_mutator_reward(is_valid_mutation=True, breaker_solve_rate=1.0)
        self.assertEqual(r_easy, 0.0)

        # 4. 合法且部分被识破 (solve_rate=0.25) -> 0.75
        r_mid, d_mid = compute_mutator_reward(is_valid_mutation=True, breaker_solve_rate=0.25)
        self.assertAlmostEqual(r_mid, 0.75)

    def test_breaker_prompt_formatting(self):
        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        f_mut = "def f_mut(x: int) -> bool:\n    return x <= 30\n"
        prompt = format_breaker_prompt(f_ref, f_mut, (0, 50), "Test spec")
        self.assertIn("f_ref", prompt)
        self.assertIn("f_mut", prompt)
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
