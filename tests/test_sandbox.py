"""
tests/test_sandbox.py - 沙箱代码执行、突变三层校验与反例判定测试
"""

import unittest
from core.sandbox import (
    execute_function_safely,
    extract_python_function,
    validate_mutation,
    evaluate_breaker_counterexample
)


class TestSandbox(unittest.TestCase):

    def test_extract_python_function_from_code_tags(self):
        text = (
            "I will modify the boundary condition.\n"
            "<code>\n"
            "def f_ref(x: int) -> bool:\n"
            "    return x <= 30\n"
            "</code>\n"
            "This introduces an off-by-one error."
        )
        src, name, err = extract_python_function(text, target_name="f_ref")
        self.assertIsNone(err)
        self.assertEqual(name, "f_ref")
        self.assertIn("return x <= 30", src)

    def test_extract_python_function_from_markdown(self):
        text = (
            "Here is the function:\n"
            "```python\n"
            "def f_ref(x: int) -> int:\n"
            "    return min(x, 26)\n"
            "```"
        )
        src, name, err = extract_python_function(text)
        self.assertIsNone(err)
        self.assertEqual(name, "f_ref")
        self.assertIn("min(x, 26)", src)

    def test_execute_function_safely(self):
        src = (
            "def test_fn(x: int) -> int:\n"
            "    return x * 2 + 1\n"
        )
        ok, res, err = execute_function_safely(src, "test_fn", 5)
        self.assertTrue(ok)
        self.assertEqual(res, 11)
        self.assertIsNone(err)

    def test_execute_function_crash_caught(self):
        src = (
            "def test_div(x: int) -> int:\n"
            "    return 10 // x\n"
        )
        ok, res, err = execute_function_safely(src, "test_div", 0)
        self.assertFalse(ok)
        self.assertIn("ZeroDivisionError", str(err))

    def test_validate_mutation_legit_subtle_defect(self):
        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        # 突变为 <= 30: 在常规输入 5, 10, 45 上两边均为 True, True, False，完全一致！
        # 但在边界 30 上产生分流 (False vs True)
        f_mut = "def f_ref(x: int) -> bool:\n    return x <= 30\n"
        sanity = [5, 10, 45]
        domain = (0, 50)

        is_valid, reason, details = validate_mutation(f_ref, f_mut, sanity, domain)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "valid_mutation")
        self.assertIn(30, details["witnesses"])
        self.assertTrue(details["sanity_passed"])

    def test_validate_mutation_crude_hack_rejected(self):
        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        # 粗糙的作弊改动: 直接 return False
        f_mut = "def f_ref(x: int) -> bool:\n    return False\n"
        sanity = [5, 10, 45]  # f_ref(5)=True, 但 f_mut(5)=False
        domain = (0, 50)

        is_valid, reason, details = validate_mutation(f_ref, f_mut, sanity, domain)
        self.assertFalse(is_valid)
        self.assertIn("Sanity invariant violated", reason)
        self.assertFalse(details["sanity_passed"])

    def test_validate_mutation_equivalent_rejected(self):
        f_ref = "def f_ref(x: int) -> int:\n    return x + 1\n"
        # 无效改动: 逻辑完全等价
        f_mut = "def f_ref(x: int) -> int:\n    y = x\n    return y + 1\n"
        sanity = [2, 5, 10]
        domain = (0, 20)

        is_valid, reason, details = validate_mutation(f_ref, f_mut, sanity, domain)
        self.assertFalse(is_valid)
        self.assertIn("Equivalent function", reason)

    def test_evaluate_breaker_counterexample(self):
        f_ref = "def f_ref(x: int) -> bool:\n    return x < 30\n"
        f_mut = "def f_mut(x: int) -> bool:\n    return x <= 30\n"
        domain = (0, 50)

        # 1. 命中反例 30
        div, r_ref, r_mut, err = evaluate_breaker_counterexample(f_ref, f_mut, 30, domain)
        self.assertTrue(div)
        self.assertFalse(r_ref)
        self.assertTrue(r_mut)

        # 2. 未命中反例 15 (两边均为 True)
        div, r_ref, r_mut, err = evaluate_breaker_counterexample(f_ref, f_mut, 15, domain)
        self.assertFalse(div)
        self.assertEqual(r_ref, r_mut)

        # 3. 越界输入 99
        div, _, _, err = evaluate_breaker_counterexample(f_ref, f_mut, 99, domain)
        self.assertFalse(div)
        self.assertIn("out of domain", str(err))


if __name__ == "__main__":
    unittest.main()
