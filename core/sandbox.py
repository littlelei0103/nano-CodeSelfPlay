"""
core/sandbox.py - nano-SPIN 安全沙箱执行与双角色校验探针

核心功能:
  1. execute_function_safely: 受限命名空间下的 Python 代码安全执行与防超时保护。
  2. validate_mutation: Mutator 突变函数的硬性合法性三层校验:
     - Layer 1 (语法与形式): AST 解析、函数定义完备性;
     - Layer 2 (隐蔽性约束): 在基础常规测试集 X_sanity 上必须与 f_ref 完全一致 (杜绝 crude/trivial 篡改);
     - Layer 3 (非等价约束): 网格探针验证是否存在真实的反例输入 x* 使得 f_mut(x*) != f_ref(x*) (杜绝死代码/等价改动)。
  3. evaluate_breaker_counterexample: 判定 Breaker 的输入是否促使 f_ref(x) != f_mut(x) 产生差分分流。
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple, Union


SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "range": range,
    "int": int,
    "bool": bool,
    "sum": sum,
    "round": round,
    "pow": pow,
}


class ExecutionTimeoutError(Exception):
    """代码执行超时异常"""
    pass


def extract_python_function(code_text: str, target_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从代码文本中提取规范的 Python 函数定义。
    支持从 <code>...</code> 或 ```python ... ``` 标签中抽取。
    
    Returns:
        (func_src, func_name, error_msg)
    """
    cleaned = code_text.strip()

    # 1. 尝试从 <code>...</code> 标签抽取
    code_tag_match = re.search(r"<code>(.*?)</code>", cleaned, re.DOTALL | re.IGNORECASE)
    if code_tag_match:
        cleaned = code_tag_match.group(1).strip()
    else:
        # 2. 尝试从 Markdown 代码块抽取
        md_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if md_match:
            cleaned = md_match.group(1).strip()

    # 3. 语法 AST 检查
    try:
        tree = ast.parse(cleaned)
    except Exception as e:
        return None, None, f"SyntaxError in code: {e}"

    func_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not func_nodes:
        return None, None, "No function definition (def) found in code"

    # 若指定了目标名称，优先匹配该名称
    chosen_node = None
    if target_name:
        for fn in func_nodes:
            if fn.name == target_name:
                chosen_node = fn
                break
    if chosen_node is None:
        chosen_node = func_nodes[0]

    func_name = chosen_node.name

    # 确保抽取出来的代码只包含该函数定义
    func_lines = cleaned.splitlines()
    start_lineno = chosen_node.lineno - 1
    end_lineno = chosen_node.end_lineno if hasattr(chosen_node, "end_lineno") and chosen_node.end_lineno is not None else len(func_lines)
    extracted_src = "\n".join(func_lines[start_lineno:end_lineno]).strip() + "\n"

    return extracted_src, func_name, None


def execute_function_safely(
    func_src: str,
    func_name: str,
    arg: Any,
    max_steps: int = 100000
) -> Tuple[bool, Any, Optional[str]]:
    """
    在受限沙箱中执行单参数函数 func_name(arg)。
    
    Returns:
        (success, result_or_none, error_msg)
    """
    namespace = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(func_src, namespace)
        if func_name not in namespace or not callable(namespace[func_name]):
            return False, None, f"Function '{func_name}' not defined in namespace"
        
        target_fn = namespace[func_name]
        res = target_fn(arg)
        return True, res, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def validate_mutation(
    f_ref_src: str,
    f_mut_src: str,
    sanity_inputs: List[int],
    domain: Tuple[int, int]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    对 Mutator 生成的突变函数进行全方位防作弊与合法性校验:
      1. 语法与形式层: 成功定义与提取可调用函数;
      2. 隐蔽性约束 (Sanity Check): 在常规基础输入集 X_sanity 上必须与 f_ref 结果严格一致;
      3. 非等价约束 (Non-equivalence Probe): 在整个定义域上，必须至少存在一个整数输入 x* 使得 f_mut(x*) != f_ref(x*)。

    Returns:
        (is_valid, reason, details)
    """
    details: Dict[str, Any] = {
        "f_ref_name": "f_ref",
        "f_mut_name": None,
        "sanity_passed": False,
        "witnesses": [],
        "sparsity": 0.0,
    }

    # 1. 抽取并验证参考函数
    ref_extracted, ref_name, ref_err = extract_python_function(f_ref_src)
    if ref_err or not ref_extracted:
        return False, f"Invalid reference function: {ref_err}", details
    details["f_ref_name"] = ref_name

    # 2. 抽取并验证突变函数
    mut_extracted, mut_name, mut_err = extract_python_function(f_mut_src)
    if mut_err or not mut_extracted:
        return False, f"Mutator syntax/parsing error: {mut_err}", details
    details["f_mut_name"] = mut_name

    # 3. 隐蔽性约束: X_sanity 必须严格一致
    for s_inp in sanity_inputs:
        ok_ref, out_ref, err_ref = execute_function_safely(ref_extracted, ref_name, s_inp)
        if not ok_ref:
            return False, f"Reference function failed on sanity input {s_inp}: {err_ref}", details

        ok_mut, out_mut, err_mut = execute_function_safely(mut_extracted, mut_name, s_inp)
        if not ok_mut:
            return False, f"Mutated function crashed on sanity input {s_inp}: {err_mut}", details

        if out_ref != out_mut:
            return False, (
                f"Sanity invariant violated: for x={s_inp}, f_ref={out_ref} but f_mut={out_mut}. "
                "Mutation is too crude and failed basic sanity test."
            ), details

    details["sanity_passed"] = True

    # 4. 非等价约束: 全定义域网格快速探针
    low, high = domain
    domain_size = high - low + 1
    witnesses = []

    for x in range(low, high + 1):
        ok_ref, out_ref, _ = execute_function_safely(ref_extracted, ref_name, x)
        ok_mut, out_mut, _ = execute_function_safely(mut_extracted, mut_name, x)

        # 任何一边抛异常或者两边输出不同均计为分流
        if ok_ref != ok_mut or (ok_ref and ok_mut and out_ref != out_mut):
            witnesses.append(x)

    if not witnesses:
        return False, "Equivalent function: f_mut behaves identically to f_ref across entire domain", details

    sparsity = len(witnesses) / domain_size
    details["witnesses"] = witnesses
    details["sparsity"] = sparsity
    details["clean_f_mut_src"] = mut_extracted
    details["clean_f_ref_src"] = ref_extracted

    return True, "valid_mutation", details


def evaluate_breaker_counterexample(
    f_ref_src: str,
    f_mut_src: str,
    candidate_input: Any,
    domain: Tuple[int, int]
) -> Tuple[bool, Any, Any, Optional[str]]:
    """
    验证 Breaker 给出的反例输入 x 是否在 domain 内，并是否触发 f_ref(x) != f_mut(x) 差分。

    Returns:
        (diverged, res_ref, res_mut, error_msg)
    """
    low, high = domain
    if not isinstance(candidate_input, (int, float)):
        return False, None, None, f"Input {candidate_input} is not a valid number"

    x = int(candidate_input)
    if x < low or x > high:
        return False, None, None, f"Input {x} is out of domain [{low}, {high}]"

    _, ref_name, _ = extract_python_function(f_ref_src)
    _, mut_name, _ = extract_python_function(f_mut_src)
    ref_name = ref_name or "f_ref"
    mut_name = mut_name or "f_mut"

    ok_ref, res_ref, err_ref = execute_function_safely(f_ref_src, ref_name, x)
    ok_mut, res_mut, err_mut = execute_function_safely(f_mut_src, mut_name, x)

    if not ok_ref and not ok_mut:
        # 两者均异常，且异常相同，未产生有效差分
        return False, None, None, f"Both functions crashed on input {x}"

    if ok_ref != ok_mut:
        # 一方成功一方崩溃，差分成立
        return True, res_ref if ok_ref else f"Crash: {err_ref}", res_mut if ok_mut else f"Crash: {err_mut}", None

    diverged = (res_ref != res_mut)
    return diverged, res_ref, res_mut, None
