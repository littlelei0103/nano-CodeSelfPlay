"""
tests/test_spin_e2e.py - nano-SPIN 端到端自博弈微闭环逻辑集成测试
"""

import unittest
from core.sandbox import validate_mutation, evaluate_breaker_counterexample
from data.seed_functions import get_seed_by_id
from roles.mutator import compute_mutator_reward
from roles.breaker import compute_breaker_reward


class TestSPINEndToEnd(unittest.TestCase):

    def test_adversarial_self_play_round_flow(self):
        # 1. 出题阶段 (Mutator Phase)
        seed = get_seed_by_id("cmp_lt_30")
        f_ref = seed["func_src"]
        domain = seed["domain"]
        sanity = seed["sanity_inputs"]

        # 模拟 Mutator 生成了一个合格的高质量突变: x <= 30
        simulated_mut_code = (
            "def f_ref(x: int) -> bool:\n"
            "    return x <= 30\n"
        )

        is_valid, reason, details = validate_mutation(f_ref, simulated_mut_code, sanity, domain)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "valid_mutation")
        self.assertIn(30, details["witnesses"])

        # 模拟当前弱 Breaker 进行 4 次探测: 猜了 [10, 20, 25, 40] -> 全部 miss
        breaker_simulated_probes = [10, 20, 25, 40]
        hits = 0
        for p in breaker_simulated_probes:
            diverged, _, _, _ = evaluate_breaker_counterexample(f_ref, simulated_mut_code, p, domain)
            if diverged:
                hits += 1

        solve_rate = hits / 4.0
        self.assertEqual(solve_rate, 0.0)  # Breaker 彻底被难倒

        # Mutator 获得满分对抗奖励
        r_mut, d_mut = compute_mutator_reward(is_valid=True, breaker_solve_rate=solve_rate)
        self.assertEqual(r_mut, 1.0)
        self.assertTrue(d_mut["valid"])

        # 2. 存入对抗题库 (Adversarial Bug Pool)
        adversarial_bug_pool = []
        if solve_rate < 1.0:
            adversarial_bug_pool.append({
                "seed_id": seed["id"],
                "f_ref_src": details["clean_f_ref_src"],
                "f_mut_src": details["clean_f_mut_src"],
                "domain": domain,
                "witnesses": details["witnesses"]
            })

        self.assertEqual(len(adversarial_bug_pool), 1)

        # 3. 破题阶段 (Breaker Phase)
        sampled_pair = adversarial_bug_pool[0]
        # 经过对抗训练后的 Breaker 进阶，精准命中了边界点 30
        trained_breaker_input = 30
        r_brk, d_brk = compute_breaker_reward(
            f_ref_src=sampled_pair["f_ref_src"],
            f_mut_src=sampled_pair["f_mut_src"],
            candidate_input=trained_breaker_input,
            domain=tuple(sampled_pair["domain"])
        )
        self.assertEqual(r_brk, 1.0)
        self.assertTrue(d_brk["success"])


if __name__ == "__main__":
    unittest.main()
