"""
data/seed_functions.py - nano-CodeSelfPlay 种子函数库 (Seed Reference Functions)

包含 40 个逻辑清晰、类型多样的标准 Python 参考函数 f_ref。
每个种子条目提供:
  - id: 唯一标识
  - family: 算子家族 (comparison, interval, clamping, arithmetic, piecewise, boolean)
  - func_src: 规范函数源码 (函数名固定为 f_ref)
  - domain: 定义域 (low, high)
  - sanity_inputs: 常规基础测试集 X_sanity (Mutator 突变后的函数在这些常规输入上必须与 f_ref 完全一致)
  - description: 规格说明
"""

from typing import Any, Dict, List, Tuple

SEED_FUNCTIONS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------
    # Family 1: Comparisons & Thresholds (单点比较与边界阈值)
    # -------------------------------------------------------------
    {
        "id": "cmp_lt_30",
        "family": "comparison",
        "domain": (0, 50),
        "sanity_inputs": [5, 10, 45],
        "description": "Check if x is strictly less than 30",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x < 30\n"
        ),
    },
    {
        "id": "cmp_lte_25",
        "family": "comparison",
        "domain": (0, 50),
        "sanity_inputs": [5, 12, 40],
        "description": "Check if x is at most 25",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x <= 25\n"
        ),
    },
    {
        "id": "cmp_gt_20",
        "family": "comparison",
        "domain": (0, 50),
        "sanity_inputs": [5, 35, 45],
        "description": "Check if x is strictly greater than 20",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x > 20\n"
        ),
    },
    {
        "id": "cmp_gte_35",
        "family": "comparison",
        "domain": (0, 50),
        "sanity_inputs": [10, 20, 48],
        "description": "Check if x is at least 35",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x >= 35\n"
        ),
    },
    {
        "id": "cmp_eq_15",
        "family": "comparison",
        "domain": (0, 50),
        "sanity_inputs": [0, 25, 40],
        "description": "Check if x equals 15",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x == 15\n"
        ),
    },
    {
        "id": "cmp_neq_0",
        "family": "comparison",
        "domain": (-20, 20),
        "sanity_inputs": [-10, 5, 18],
        "description": "Check if x is non-zero",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x != 0\n"
        ),
    },
    {
        "id": "cmp_lt_neg5",
        "family": "comparison",
        "domain": (-25, 25),
        "sanity_inputs": [-20, 0, 15],
        "description": "Check if x is strictly less than -5",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x < -5\n"
        ),
    },
    {
        "id": "cmp_gte_neg10",
        "family": "comparison",
        "domain": (-30, 20),
        "sanity_inputs": [-25, 0, 15],
        "description": "Check if x is at least -10",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x >= -10\n"
        ),
    },

    # -------------------------------------------------------------
    # Family 2: Intervals & Inclusion (区间包含与开闭边界)
    # -------------------------------------------------------------
    {
        "id": "interval_closed_15_35",
        "family": "interval",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Check if x is in closed interval [15, 35]",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 15 <= x <= 35\n"
        ),
    },
    {
        "id": "interval_open_10_40",
        "family": "interval",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Check if x is in open interval (10, 40)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 10 < x < 40\n"
        ),
    },
    {
        "id": "interval_left_open_20_45",
        "family": "interval",
        "domain": (0, 50),
        "sanity_inputs": [5, 30, 48],
        "description": "Check if x is in half-open interval (20, 45]",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 20 < x <= 45\n"
        ),
    },
    {
        "id": "interval_right_open_5_30",
        "family": "interval",
        "domain": (0, 50),
        "sanity_inputs": [2, 18, 40],
        "description": "Check if x is in half-open interval [5, 30)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return 5 <= x < 30\n"
        ),
    },
    {
        "id": "interval_neg_symm_12",
        "family": "interval",
        "domain": (-30, 30),
        "sanity_inputs": [-25, 0, 25],
        "description": "Check if x is between -12 and 12 inclusive",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return -12 <= x <= 12\n"
        ),
    },
    {
        "id": "interval_outside_10_30",
        "family": "interval",
        "domain": (0, 50),
        "sanity_inputs": [2, 20, 45],
        "description": "Check if x is outside interval [10, 30]",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x < 10 or x > 30\n"
        ),
    },

    # -------------------------------------------------------------
    # Family 3: Clamping & Min / Max (上下界截断与极值约束)
    # -------------------------------------------------------------
    {
        "id": "clamp_min_upper_30",
        "family": "clamping",
        "domain": (0, 50),
        "sanity_inputs": [5, 12, 20],
        "description": "Clamp upper bound to 30 using min(x, 30)",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return min(x, 30)\n"
        ),
    },
    {
        "id": "clamp_max_lower_15",
        "family": "clamping",
        "domain": (0, 50),
        "sanity_inputs": [20, 30, 45],
        "description": "Clamp lower bound to 15 using max(x, 15)",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return max(x, 15)\n"
        ),
    },
    {
        "id": "clamp_both_10_40",
        "family": "clamping",
        "domain": (0, 50),
        "sanity_inputs": [15, 25, 35],
        "description": "Clamp x to range [10, 40] using max(10, min(x, 40))",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return max(10, min(x, 40))\n"
        ),
    },
    {
        "id": "clamp_neg_max_zero",
        "family": "clamping",
        "domain": (-25, 25),
        "sanity_inputs": [5, 10, 20],
        "description": "ReLU-style activation: max(0, x)",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return max(0, x)\n"
        ),
    },
    {
        "id": "clamp_neg_min_zero",
        "family": "clamping",
        "domain": (-25, 25),
        "sanity_inputs": [-20, -15, -8],
        "description": "Clamp upper bound to zero: min(0, x)",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return min(0, x)\n"
        ),
    },
    {
        "id": "clamp_box_neg15_15",
        "family": "clamping",
        "domain": (-30, 30),
        "sanity_inputs": [-8, 0, 8],
        "description": "Symmetric clamp between -15 and 15",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    return min(15, max(-15, x))\n"
        ),
    },

    # -------------------------------------------------------------
    # Family 4: Arithmetic & Modulo (算术、模运算与绝对值)
    # -------------------------------------------------------------
    {
        "id": "arith_is_even",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [4, 7, 26],
        "description": "Check if x is even",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 2 == 0\n"
        ),
    },
    {
        "id": "arith_mod3_zero",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [6, 11, 27],
        "description": "Check if x is divisible by 3",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 3 == 0\n"
        ),
    },
    {
        "id": "arith_mod5_nonzero",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [10, 13, 35],
        "description": "Check if x is not divisible by 5",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 5 != 0\n"
        ),
    },
    {
        "id": "arith_abs_diff_lte_5",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Check if distance to 25 is at most 5: abs(x - 25) <= 5",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return abs(x - 25) <= 5\n"
        ),
    },
    {
        "id": "arith_abs_diff_gt_10",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [5, 20, 42],
        "description": "Check if distance to 20 is strictly greater than 10",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return abs(x - 20) > 10\n"
        ),
    },
    {
        "id": "arith_abs_center_diff",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [5, 30, 45],
        "description": "Check if distance to 30 is strictly less than 12",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return abs(x - 30) < 12\n"
        ),
    },
    {
        "id": "arith_mod4_remainder",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [6, 17, 33],
        "description": "Check if remainder modulo 4 is at least 2",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x % 4 >= 2\n"
        ),
    },
    {
        "id": "arith_sum_parity",
        "family": "arithmetic",
        "domain": (0, 50),
        "sanity_inputs": [4, 9, 28],
        "description": "Check if (x + 3) is divisible by 4",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return (x + 3) % 4 == 0\n"
        ),
    },

    # -------------------------------------------------------------
    # Family 5: Piecewise & Multi-Branch (分段函数与多重分支逻辑)
    # -------------------------------------------------------------
    {
        "id": "piecewise_tax_tier",
        "family": "piecewise",
        "domain": (0, 50),
        "sanity_inputs": [10, 25, 45],
        "description": "Piecewise tax slab: 0 if x <= 15, 1 if x <= 35, else 2",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x <= 15:\n"
            "        return 0\n"
            "    elif x <= 35:\n"
            "        return 1\n"
            "    else:\n"
            "        return 2\n"
        ),
    },
    {
        "id": "piecewise_discount",
        "family": "piecewise",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Tiered discount: 5 if x < 20, 10 if x < 40, else 20",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x < 20:\n"
            "        return 5\n"
            "    elif x < 40:\n"
            "        return 10\n"
            "    else:\n"
            "        return 20\n"
        ),
    },
    {
        "id": "piecewise_sign",
        "family": "piecewise",
        "domain": (-30, 30),
        "sanity_inputs": [-20, 0, 20],
        "description": "Signum function: -1 if x < 0, 0 if x == 0, else 1",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x < 0:\n"
            "        return -1\n"
            "    elif x == 0:\n"
            "        return 0\n"
            "    else:\n"
            "        return 1\n"
        ),
    },
    {
        "id": "piecewise_step_zero",
        "family": "piecewise",
        "domain": (-20, 30),
        "sanity_inputs": [-10, 10, 25],
        "description": "Heaviside step: 0 if x <= 0, else 10",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x <= 0:\n"
            "        return 0\n"
            "    else:\n"
            "        return 10\n"
        ),
    },
    {
        "id": "piecewise_odd_bonus",
        "family": "piecewise",
        "domain": (0, 50),
        "sanity_inputs": [10, 15, 30],
        "description": "Bonus on even/odd: x * 2 if x % 2 == 0 else x + 5",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x % 2 == 0:\n"
            "        return x * 2\n"
            "    else:\n"
            "        return x + 5\n"
        ),
    },
    {
        "id": "piecewise_bracket_grade",
        "family": "piecewise",
        "domain": (0, 50),
        "sanity_inputs": [10, 30, 48],
        "description": "Letter grade code: 3 if x >= 40, 2 if x >= 25, else 1",
        "func_src": (
            "def f_ref(x: int) -> int:\n"
            "    if x >= 40:\n"
            "        return 3\n"
            "    elif x >= 25:\n"
            "        return 2\n"
            "    else:\n"
            "        return 1\n"
        ),
    },

    # -------------------------------------------------------------
    # Family 6: Compound Boolean & Logic (复合布尔与布尔短路)
    # -------------------------------------------------------------
    {
        "id": "bool_and_or_mix",
        "family": "boolean",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Check if (x > 10 and x < 20) or x > 40",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return (x > 10 and x < 20) or x > 40\n"
        ),
    },
    {
        "id": "bool_not_range",
        "family": "boolean",
        "domain": (0, 50),
        "sanity_inputs": [5, 25, 45],
        "description": "Check if not (15 <= x <= 35)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return not (15 <= x <= 35)\n"
        ),
    },
    {
        "id": "bool_xor_parity",
        "family": "boolean",
        "domain": (0, 50),
        "sanity_inputs": [5, 22, 45],
        "description": "XOR condition: (x > 20) != (x % 2 == 0)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return (x > 20) != (x % 2 == 0)\n"
        ),
    },
    {
        "id": "bool_ends_or_mid",
        "family": "boolean",
        "domain": (0, 50),
        "sanity_inputs": [2, 25, 48],
        "description": "Check if x <= 5 or (20 <= x <= 30) or x >= 45",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return x <= 5 or (20 <= x <= 30) or x >= 45\n"
        ),
    },
    {
        "id": "bool_leap_style",
        "family": "boolean",
        "domain": (0, 60),
        "sanity_inputs": [10, 24, 40],
        "description": "Leap-like rule: (x % 4 == 0 and x % 10 != 0) or (x % 20 == 0)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return (x % 4 == 0 and x % 10 != 0) or (x % 20 == 0)\n"
        ),
    },
    {
        "id": "bool_double_threshold",
        "family": "boolean",
        "domain": (-30, 30),
        "sanity_inputs": [-20, 0, 20],
        "description": "Check if (x < -10) or (x > 10 and x % 2 == 0)",
        "func_src": (
            "def f_ref(x: int) -> bool:\n"
            "    return (x < -10) or (x > 10 and x % 2 == 0)\n"
        ),
    },
]


def get_seed_functions() -> List[Dict[str, Any]]:
    """获取全量种子参考函数列表 (40 道)"""
    return list(SEED_FUNCTIONS)


def get_seed_by_id(seed_id: str) -> Dict[str, Any]:
    """按 ID 查找种子参考函数"""
    for item in SEED_FUNCTIONS:
        if item["id"] == seed_id:
            return dict(item)
    raise KeyError(f"Seed function with id '{seed_id}' not found")
