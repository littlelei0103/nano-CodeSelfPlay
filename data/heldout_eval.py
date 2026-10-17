"""
data/heldout_eval.py - nano-SPIN 独立外部基准评测集 (Held-out Benchmark Suite)

包含 20 道严格隔离的外部差分测试基准题目，覆盖比较、区间、截断与复合逻辑。
在对抗自博弈训练期间，该测试集仅用于客观评估 Breaker 的跨步泛化能力（Pass@1 与 Pass@4），
零参与训练与样本生成，确保评估指标无污染。
"""

from typing import Any, Dict, List, Tuple


HELDOUT_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "heldout_01_cmp_lt_lte",
        "level": "level_1",
        "domain": (0, 50),
        "spec": "Check if x is strictly less than 38 vs at most 38",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x < 38\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x <= 38\n"
        ),
        "counterexamples": [38],
    },
    {
        "id": "heldout_02_cmp_gt_gte",
        "level": "level_1",
        "domain": (0, 50),
        "spec": "Check if x is strictly greater than 46 vs at least 46",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x > 46\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x >= 46\n"
        ),
        "counterexamples": [46],
    },
    {
        "id": "heldout_03_cmp_equality_flip",
        "level": "level_1",
        "domain": (0, 50),
        "spec": "Equality check: x == 22 vs x == 23",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x == 22\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x == 23\n"
        ),
        "counterexamples": [22, 23],
    },
    {
        "id": "heldout_04_cmp_neg_boundary",
        "level": "level_1",
        "domain": (-30, 30),
        "spec": "Negative boundary: x < -12 vs x <= -12",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x < -12\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x <= -12\n"
        ),
        "counterexamples": [-12],
    },
    {
        "id": "heldout_05_interval_right_open",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Interval inclusion: [33, 42] vs [33, 42)",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 33 <= x <= 42\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return 33 <= x < 42\n"
        ),
        "counterexamples": [42],
    },
    {
        "id": "heldout_06_interval_left_open",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Interval inclusion: [12, 30] vs (12, 30]",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 12 <= x <= 30\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return 12 < x <= 30\n"
        ),
        "counterexamples": [12],
    },
    {
        "id": "heldout_07_interval_endpoint_shift",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Interval right endpoint shifted by 1: [15, 28] vs [15, 29]",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 15 <= x <= 28\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return 15 <= x <= 29\n"
        ),
        "counterexamples": [29],
    },
    {
        "id": "heldout_08_interval_symmetric_neg",
        "level": "level_2",
        "domain": (-25, 25),
        "spec": "Symmetric negative interval: -10 < x < 10 vs -10 <= x <= 10",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return -10 < x < 10\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return -10 <= x <= 10\n"
        ),
        "counterexamples": [-10, 10],
    },
    {
        "id": "heldout_09_clamp_min_clip",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Upper clamp threshold: min(x, 25) vs min(x, 26)",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    return min(x, 25)\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    return min(x, 26)\n"
        ),
        "counterexamples": [26, 27, 28, 29, 30, 35, 40, 50],
    },
    {
        "id": "heldout_10_clamp_max_clip",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Lower clamp threshold: max(x, 18) vs max(x, 17)",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    return max(x, 18)\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    return max(x, 17)\n"
        ),
        "counterexamples": [0, 5, 10, 15, 17],
    },
    {
        "id": "heldout_11_clamp_relu_off1",
        "level": "level_2",
        "domain": (-20, 20),
        "spec": "ReLU zero clamp: max(0, x) vs max(1, x)",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    return max(0, x)\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    return max(1, x)\n"
        ),
        "counterexamples": [-10, -5, 0],
    },
    {
        "id": "heldout_12_arith_even_odd_flip",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Parity check: x % 2 == 0 vs (x + 1) % 2 == 0",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 2 == 0\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return (x + 1) % 2 == 0\n"
        ),
        "counterexamples": list(range(0, 51)),
    },
    {
        "id": "heldout_13_arith_mod_divisor_shift",
        "level": "level_3",
        "domain": (0, 50),
        "spec": "Modulo divisor: x % 4 == 0 vs x % 5 == 0",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 4 == 0\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x % 5 == 0\n"
        ),
        "counterexamples": [4, 5, 8, 10, 12, 15, 16],
    },
    {
        "id": "heldout_14_arith_abs_threshold",
        "level": "level_2",
        "domain": (0, 50),
        "spec": "Distance threshold: abs(x - 30) < 6 vs abs(x - 30) <= 6",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return abs(x - 30) < 6\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return abs(x - 30) <= 6\n"
        ),
        "counterexamples": [24, 36],
    },
    {
        "id": "heldout_15_arith_integer_div_rounding",
        "level": "level_2",
        "domain": (0, 40),
        "spec": "Division rounding: x // 3 vs (x + 1) // 3",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    return x // 3\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    return (x + 1) // 3\n"
        ),
        "counterexamples": [2, 5, 8, 11, 14, 17, 20, 23],
    },
    {
        "id": "heldout_16_piecewise_boundary_inclusion",
        "level": "level_3",
        "domain": (0, 50),
        "spec": "Branch condition: x <= 20 vs x < 20 in piecewise assignment",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x <= 20:\n"
            "        return 1\n"
            "    return 2\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    if x < 20:\n"
            "        return 1\n"
            "    return 2\n"
        ),
        "counterexamples": [20],
    },
    {
        "id": "heldout_17_piecewise_three_tier",
        "level": "level_3",
        "domain": (0, 50),
        "spec": "Tier threshold shift: middle tier boundary 35 vs 36",
        "f_ref_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x < 15:\n"
            "        return 10\n"
            "    elif x <= 35:\n"
            "        return 20\n"
            "    return 30\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> int:\n"
            "    if x < 15:\n"
            "        return 10\n"
            "    elif x <= 36:\n"
            "        return 20\n"
            "    return 30\n"
        ),
        "counterexamples": [36],
    },
    {
        "id": "heldout_18_bool_and_or_precedence",
        "level": "level_3",
        "domain": (0, 50),
        "spec": "Boolean compound: (x > 10 and x < 30) vs (x >= 10 and x < 30)",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x > 10 and x < 30\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x >= 10 and x < 30\n"
        ),
        "counterexamples": [10],
    },
    {
        "id": "heldout_19_bool_negation_demorgan",
        "level": "level_3",
        "domain": (0, 50),
        "spec": "Negation range: not (20 <= x <= 40) vs not (20 < x <= 40)",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return not (20 <= x <= 40)\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return not (20 < x <= 40)\n"
        ),
        "counterexamples": [20],
    },
    {
        "id": "heldout_20_bool_compound_leap",
        "level": "level_3",
        "domain": (0, 60),
        "spec": "Leap year style condition: (x % 4 == 0 and x % 10 != 0) vs (x % 4 == 0)",
        "f_ref_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 4 == 0 and x % 10 != 0\n"
        ),
        "f_mut_src": (
            "def f_mut(x: int) -> bool:\n"
            "    return x % 4 == 0\n"
        ),
        "counterexamples": [0, 20, 40, 60],
    },
]


def get_heldout_benchmark() -> List[Dict[str, Any]]:
    """获取固定的 20 道外部基准测试集"""
    return list(HELDOUT_PROBLEMS)
