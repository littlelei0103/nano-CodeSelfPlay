"""
roles/breaker.py - nano-SPIN 找茬者角色 (Breaker / Counterexample Finder)

职责:
  1. format_breaker_prompt: 生成找茬者专用的系统与用户提示词 (引导模型在 <think> 中分析逻辑差异，在 <input> 中给出反例)。
  2. extract_breaker_input: 从模型输出文本中抽取整数反例并进行定义域边界检查。
  3. evaluate_breaker_candidate: 调度安全沙箱执行 f_ref 与 f_mut，返回布尔奖励 R_breaker (+1.0 或 0.0)。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from core.sandbox import evaluate_breaker_counterexample


BREAKER_SYSTEM_PROMPT = (
    "You are an expert software test engineer (Breaker).\n"
    "Given a clean reference function f_ref and a subtly mutated function f_mut, "
    "your goal is to find an integer input x in the specified domain where their return values differ: f_ref(x) != f_mut(x).\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Analyze the control flow, edge cases, and boundary conditions step-by-step inside <think>...</think>.\n"
    "2. Locate a specific counterexample integer value x within the domain.\n"
    "3. Output ONLY the integer inside <input>...</input> tags (e.g., <input>38</input>)."
)


def format_breaker_prompt(
    f_ref_src: str,
    f_mut_src: str,
    domain: Tuple[int, int],
    spec_desc: str = ""
) -> str:
    """格式化找茬者提示词"""
    low, high = domain
    desc_str = f"Specification: {spec_desc}\n" if spec_desc else ""

    prompt = (
        f"{BREAKER_SYSTEM_PROMPT}\n\n"
        f"Domain of x: [{low}, {high}]\n"
        f"{desc_str}\n"
        f"Reference Function:\n```python\n{f_ref_src.strip()}\n```\n\n"
        f"Mutated Function:\n```python\n{f_mut_src.strip()}\n```\n\n"
        f"Output format: <input>INTEGER</input>\n\n"
        f"Assistant: <input>"
    )
    return prompt


def extract_breaker_input(
    completion_text: str,
    domain: Tuple[int, int]
) -> Tuple[Optional[int], str, str]:
    """
    从 Breaker 输出文本中抽取目标整数反例。
    
    Returns:
        (value_or_none, category, reason)
        category: "valid_extracted", "out_of_domain", "format_malformed"
    """
    low, high = domain

    # 1. 匹配 <input>...</input> 标签
    match = re.search(r"<input>\s*([+-]?\d+)\s*</input>", completion_text, flags=re.IGNORECASE)
    if not match:
        # 2. 匹配未闭合的 <input> 标签
        match = re.search(r"<input>\s*([+-]?\d+)", completion_text, flags=re.IGNORECASE)
    if not match:
        # 3. 匹配 x = 12 或 ans = 12 形式
        match = re.search(r"(?:x|input|ans)\s*=\s*([+-]?\d+)", completion_text, flags=re.IGNORECASE)
    if not match:
        # 4. 匹配最后的独立整数
        ints = re.findall(r"\b([+-]?\d+)\b", completion_text.strip())
        if ints:
            val_str = ints[-1]
            try:
                val = int(val_str)
                if val < low or val > high:
                    return val, "out_of_domain", f"Integer {val} is outside domain [{low}, {high}]"
                return val, "valid_extracted", "Extracted trailing integer"
            except ValueError:
                pass
        return None, "format_malformed", "No integer found in completion text"

    try:
        val = int(match.group(1))
    except ValueError:
        return None, "format_malformed", f"Failed to parse '{match.group(1)}' as integer"

    if val < low or val > high:
        return val, "out_of_domain", f"Integer {val} is outside domain [{low}, {high}]"

    return val, "valid_extracted", "Valid integer within domain"


def compute_breaker_reward(
    f_ref_src: str,
    f_mut_src: str,
    candidate_input: Any,
    domain: Tuple[int, int]
) -> Tuple[float, Dict[str, Any]]:
    """
    计算 Breaker 的对抗自博弈奖励 R_breaker:
      - 成功找到反例并触发 f_ref(x) != f_mut(x): Reward = +1.0
      - 否则 (格式错误 / 越界 / 两者输出相同): Reward = 0.0
    """
    if candidate_input is None:
        return 0.0, {
            "success": False,
            "reason": "missing_input",
            "res_ref": None,
            "res_mut": None,
        }

    diverged, res_ref, res_mut, err = evaluate_breaker_counterexample(
        f_ref_src=f_ref_src,
        f_mut_src=f_mut_src,
        candidate_input=candidate_input,
        domain=domain,
    )

    reward = 1.0 if diverged else 0.0
    return reward, {
        "success": diverged,
        "reason": "divergence_confirmed" if diverged else (f"error: {err}" if err else "outputs_match"),
        "res_ref": res_ref,
        "res_mut": res_mut,
    }
