"""
tests/test_data.py - 种子函数集与独立外部基准评测集的完备性与契约测试
"""

import unittest
from core.sandbox import execute_function_safely, evaluate_breaker_counterexample
from data.seed_functions import get_seed_functions, get_seed_by_id
from data.heldout_eval import get_heldout_benchmark


class TestDataIntegrity(unittest.TestCase):

    def test_all_seed_functions_integrity(self):
        seeds = get_seed_functions()
        self.assertGreaterEqual(len(seeds), 30, "Seed functions should contain at least 30 problems")

        for seed in seeds:
            s_id = seed["id"]
            domain = seed["domain"]
            sanity = seed["sanity_inputs"]
            func_src = seed["func_src"]

            self.assertIsInstance(domain, (tuple, list))
            self.assertEqual(len(domain), 2)
            self.assertLess(domain[0], domain[1], f"Domain for {s_id} is invalid: {domain}")

            self.assertTrue(len(sanity) >= 2, f"Seed {s_id} should have at least 2 sanity inputs")
            for s_in in sanity:
                self.assertGreaterEqual(s_in, domain[0], f"Sanity input {s_in} out of lower bound for {s_id}")
                self.assertLessEqual(s_in, domain[1], f"Sanity input {s_in} out of upper bound for {s_id}")

                ok, out, err = execute_function_safely(func_src, "f_ref", s_in)
                self.assertTrue(ok, f"Seed function {s_id} crashed on sanity input {s_in}: {err}")

    def test_all_heldout_problems_integrity(self):
        problems = get_heldout_benchmark()
        self.assertEqual(len(problems), 20, "Held-out benchmark must contain exactly 20 test problems")

        for prob in problems:
            p_id = prob["id"]
            domain = tuple(prob["domain"])
            f_ref = prob["f_ref_src"]
            f_mut = prob["f_mut_src"]
            ces = prob["counterexamples"]

            self.assertTrue(len(ces) > 0, f"Held-out problem {p_id} has no counterexamples!")
            for ce in ces:
                self.assertGreaterEqual(ce, domain[0], f"Counterexample {ce} out of domain for {p_id}")
                self.assertLessEqual(ce, domain[1], f"Counterexample {ce} out of domain for {p_id}")

                diverged, r_ref, r_mut, err = evaluate_breaker_counterexample(f_ref, f_mut, ce, domain)
                self.assertTrue(diverged, f"Held-out problem {p_id} ground-truth counterexample {ce} failed to diverge: f_ref={r_ref}, f_mut={r_mut}, err={err}")


if __name__ == "__main__":
    unittest.main()
