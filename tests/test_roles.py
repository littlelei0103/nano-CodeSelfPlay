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

    def test_mutator_dense_shaping_reward(self):
        # 1. 完全失败 -> 0.0
        r0, d0 = compute_mutator_reward()
        self.assertEqual(r0, 0.0)

        # 2. 格式提取成功但语法错误 -> 0.1
        r1, d1 = compute_mutator_reward(is_valid_format=True)
        self.assertEqual(r1, 0.1)

        # 3. 语法 AST 通过但 sanity 失败 -> 0.3
        r2, d2 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True)
        self.assertEqual(r2, 0.3)

        # 4. sanity 通过但全域等价 -> 0.6
        r3, d3 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True)
        self.assertEqual(r3, 0.6)

        # 5. 非等价且难倒 Breaker (solve_rate=0.0) -> 满分 1.0
        r4, d4 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True, breaker_solve_rate=0.0)
        self.assertEqual(r4, 1.0)

        # 6. 非等价但 Breaker 破解率 0.5 -> 0.6 + 0.4*0.5 = 0.8
        r5, d5 = compute_mutator_reward(is_valid_format=True, is_valid_syntax=True, is_valid_sanity=True, is_non_equivalent=True, breaker_solve_rate=0.5)
        self.assertAlmostEqual(r5, 0.8)

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
