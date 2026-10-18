"""
core/sandbox.py - nano-CodeSelfPlay 安全沙箱执行与双角色校验探针

核心机制与加固设计:
  1. 隔离子进程执行 (Subprocess Worker Isolation):
     代码在独立的无污染 Python 子进程中运行 (-I -S)，阻断任何对全局 __builtins__ 或父进程内存的篡改。
  2. 毫秒级超时与死锁熔断 (Timeout & Infinite-Loop Guard):
     通过 subprocess timeout 机制 (默认 1.0s)，强力阻断死循环 (while True)、深度递归与进程挂起。
  3. 静态 AST 受限语法检查 (AST Safety Guard):
     严禁 import、反射与内部高危属性 (__*)、类定义与协程，限制生成代码的能力（不替代操作系统级沙盒）。
  4. 严谨客观差分判题 (Ground-Truth Differential Evaluation):
     - 要求两边代码必须均为合法的可调用 Python 函数；
     - 语法编译错误、未定义函数绝不计为有效分流。
"""

import ast
import json
import os
import re
import subprocess
import sys
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


# Each input gets a newly executed function definition, including fresh defaults.
# Infrastructure failures are never evidence of a program defect.
WORKER_PAYLOAD_CODE = r"""
import sys, json, math
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except ImportError:
    pass
SAFE_BUILTINS = {
    'abs': abs, 'min': min, 'max': max, 'len': len, 'range': range,
    'int': int, 'bool': bool, 'sum': sum, 'round': round, 'pow': pow
}
inputs = []
def failure(error, kind):
    return {'success': False, 'result': None, 'error': error, 'error_kind': kind}
try:
    payload = json.loads(sys.stdin.read(1_000_000))
    inputs = payload['inputs']
    compiled = compile(payload['code'], '<candidate>', 'exec')
    results = []
    for x in inputs:
        ns = {'__builtins__': SAFE_BUILTINS.copy()}
        try:
            exec(compiled, ns)
            fn = ns[payload['func_name']]
        except BaseException as e:
            results.append(failure(type(e).__name__, 'infrastructure'))
            continue
        try:
            value = fn(x)
            # This environment's contract is a scalar numeric function.
            if type(value) not in (int, bool, float) or (type(value) is float and not math.isfinite(value)):
                results.append(failure('Unsupported or non-finite return value', 'infrastructure'))
            else:
                results.append({'success': True, 'result': value, 'error': None, 'error_kind': None})
        except (MemoryError, RecursionError) as e:
            results.append(failure(type(e).__name__, 'infrastructure'))
        except Exception as e:
            results.append(failure(type(e).__name__ + ': ' + str(e), 'runtime'))
        except BaseException as e:
            results.append(failure(type(e).__name__, 'infrastructure'))
    print(json.dumps(results, allow_nan=False))
except BaseException as e:
    print(json.dumps([failure('Worker failure: ' + type(e).__name__, 'infrastructure') for _ in inputs]))
"""


def validate_ast_safety(tree: ast.AST) -> Tuple[bool, Optional[str]]:
    """
    静态语法与安全检查:
      1. 拒绝无限循环语法 (while 语句);
      2. 拒绝模块导入 (import / from ... import);
      3. 拒绝高危反射与系统内置名 (exec, eval, compile, open, globals, locals, vars, getattr, setattr, delattr, hasattr, type, object, breakpoint, __import__, exit, quit, SystemExit, BaseException);
      4. 拒绝私有与双下划线属性访问 (__*);
      5. 拒绝类与异步定义 (ClassDef, AsyncFunctionDef).
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Disallowed syntax: imports are not permitted"
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            return False, "Disallowed syntax: classes and coroutines are not permitted"
        if isinstance(node, ast.While):
            return False, "Disallowed syntax: while loops are not permitted to ensure termination"
        if isinstance(node, ast.Name):
            if (node.id.startswith("_") and node.id != "_") or node.id in {
                "exec", "eval", "compile", "open", "input", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "hasattr", "type", "object", "breakpoint",
                "__import__", "exit", "quit", "SystemExit", "BaseException", "memoryview"
            }:
                return False, f"Disallowed name: '{node.id}'"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in {
                "__builtins__", "__globals__", "__code__", "__subclasses__", "f_back", "f_globals"
            }:
                return False, f"Disallowed attribute access: '{node.attr}'"
    return True, None


def extract_python_function(code_text: str, target_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从代码文本中提取规范的 Python 函数定义。
    支持从 <code>...</code> 或 ```python ... ``` 标签中抽取，并过滤多余的尾部标记。
    
    Returns:
        (func_src, func_name, error_msg)
    """
    cleaned = code_text.strip()

    # 1. 优先尝试从 <code>...</code> 闭合标签抽取
    code_tag_match = re.search(r"<code>(.*?)</code>", cleaned, re.DOTALL | re.IGNORECASE)
    if code_tag_match:
        cleaned = code_tag_match.group(1).strip()
    else:
        # 尝试从 Markdown 代码块抽取
        md_match = re.search(r"```(?:python)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if md_match:
            cleaned = md_match.group(1).strip()

    # 2. 清洗可能未闭合或孤立的 html/markdown 标签
    cleaned = re.sub(r"</?code>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```(?:python)?", "", cleaned, flags=re.IGNORECASE)

    # 3. 按行扫描提取 def 函数块 (防止模型末尾输出测试用例或自然语言导致语法解析失败)
    lines = cleaned.splitlines()
    fn_lines = []
    in_fn = False
    for line in lines:
        if not in_fn:
            if line.strip().startswith("def "):
                in_fn = True
                fn_lines.append(line)
        else:
            if line.startswith(" ") or line.startswith("\t") or not line.strip():
                fn_lines.append(line)
            else:
                # 遇到顶格非空行，判定函数主体已结束
                break

    if fn_lines:
        cleaned = "\n".join(fn_lines).strip()

    # 4. 语法 AST 检查与安全过滤
    try:
        tree = ast.parse(cleaned)
    except Exception as e:
        return None, None, f"SyntaxError in code: {e}"

    is_safe, sec_err = validate_ast_safety(tree)
    if not is_safe:
        return None, None, f"SecurityPolicyViolation: {sec_err}"

    func_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not func_nodes:
        return None, None, "No function definition (def) found in code"

    chosen_node = None
    if target_name:
        for fn in func_nodes:
            if fn.name == target_name:
                chosen_node = fn
                break
    if chosen_node is None:
        chosen_node = func_nodes[0]

    func_name = chosen_node.name

    func_lines = cleaned.splitlines()
    start_lineno = chosen_node.lineno - 1
    end_lineno = chosen_node.end_lineno if hasattr(chosen_node, "end_lineno") and chosen_node.end_lineno is not None else len(func_lines)
    extracted_src = "\n".join(func_lines[start_lineno:end_lineno]).strip() + "\n"

    return extracted_src, func_name, None


def apply_line_edit_or_extract(
    f_ref_src: str,
    raw_text: str
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    支持行级补丁 (Line-Edit Mode) 与完整函数 (Full-Function Mode) 双模式解析。
    
    Returns:
        (func_src, func_name, error_msg, is_line_edit)
    """
    old_match = re.search(r"OLD:\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)
    new_match = re.search(r"NEW:\s*(.+?)(?:\n|</answer>|$)", raw_text, re.IGNORECASE)

    if old_match and new_match:
        old_line = old_match.group(1).strip()
        new_line = new_match.group(1).strip()

        lines = f_ref_src.splitlines()
        replaced = False
        new_lines = []
        for line in lines:
            if not replaced and old_line in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + new_line.lstrip())
                replaced = True
            else:
                new_lines.append(line)

        if replaced:
            candidate_src = "\n".join(new_lines) + "\n"
            try:
                tree = ast.parse(candidate_src)
                is_safe, _ = validate_ast_safety(tree)
                if is_safe:
                    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                    if funcs:
                        return candidate_src, funcs[0].name, None, True
            except Exception:
                pass

    src, name, err = extract_python_function(raw_text)
    return src, name, err, False


def execute_batch_safely(
    func_src: str,
    func_name: str,
    args_list: List[Any],
    timeout_sec: float = 1.0
) -> List[Dict[str, Any]]:
    """
    在隔离的独立子进程沙箱中批量执行函数调用 func_name(x)，保障环境隔离与超时熔断。

    Returns:
        List of dicts: [{'success': bool, 'result': Any, 'error': Optional[str]}]
    """
    if not args_list:
        return []

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", WORKER_PAYLOAD_CODE],
            input=json.dumps({"code": func_src, "func_name": func_name, "inputs": args_list}),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            cwd="/tmp"
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or f"Subprocess exited with code {proc.returncode}"
            return [{"success": False, "result": None, "error": f"SubprocessCrash: {err}", "error_kind": "infrastructure"} for _ in args_list]
        results = json.loads(proc.stdout)
        if not isinstance(results, list) or len(results) != len(args_list):
            raise ValueError("Invalid worker result count")
        for item in results:
            if (not isinstance(item, dict) or type(item.get("success")) is not bool
                    or not {"result", "error", "error_kind"}.issubset(item)
                    or item["error_kind"] not in (None, "runtime", "infrastructure")
                    or (item["success"] and (item["error_kind"] is not None
                        or type(item["result"]) not in (int, bool, float)))
                    or (not item["success"] and item["error_kind"] is None)):
                raise ValueError("Invalid worker result schema")
        return results
    except subprocess.TimeoutExpired:
        return [{"success": False, "result": None, "error": f"ExecutionTimeoutError: exceeded {timeout_sec}s", "error_kind": "infrastructure"} for _ in args_list]
    except Exception as e:
        return [{"success": False, "result": None, "error": f"{type(e).__name__}: {e}", "error_kind": "infrastructure"} for _ in args_list]


def execute_function_safely(
    func_src: str,
    func_name: str,
    arg: Any,
    timeout_sec: float = 1.0
) -> Tuple[bool, Any, Optional[str]]:
    """
    单参数受限安全执行。
    
    Returns:
        (success, result_or_none, error_msg)
    """
    results = execute_batch_safely(func_src, func_name, [arg], timeout_sec=timeout_sec)
    res = results[0]
    return res["success"], res["result"], res["error"]


def validate_mutation(
    f_ref_src: str,
    f_mut_src: str,
    sanity_inputs: List[int],
    domain: Tuple[int, int],
    timeout_sec: float = 1.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    对 Mutator 生成的突变函数进行严格的防作弊与合法性三层校验:
      1. 语法与形式层: AST 解析、函数定义完备性、受限语法检查;
      2. 隐蔽性约束 (Sanity Check): 在基础常规测试集 X_sanity 上必须与 f_ref 完全一致;
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

    # 3. 隐蔽性约束: X_sanity 批量受限执行校验
    res_ref_sanity = execute_batch_safely(ref_extracted, ref_name, sanity_inputs, timeout_sec=timeout_sec)
    res_mut_sanity = execute_batch_safely(mut_extracted, mut_name, sanity_inputs, timeout_sec=timeout_sec)

    for idx, s_inp in enumerate(sanity_inputs):
        r_ref = res_ref_sanity[idx]
        r_mut = res_mut_sanity[idx]

        if not r_ref["success"]:
            return False, f"Reference function failed on sanity input {s_inp}: {r_ref['error']}", details
        if not r_mut["success"]:
            return False, f"Mutated function crashed on sanity input {s_inp}: {r_mut['error']}", details
        if r_ref["result"] != r_mut["result"]:
            return False, (
                f"Sanity invariant violated: for x={s_inp}, f_ref={r_ref['result']} but f_mut={r_mut['result']}. "
                "Mutation is too crude and failed basic sanity test."
            ), details

    details["sanity_passed"] = True

    # 4. 非等价约束: 全定义域网格批量快速探针
    low, high = domain
    domain_size = high - low + 1
    grid_inputs = list(range(low, high + 1))

    res_ref_grid = execute_batch_safely(ref_extracted, ref_name, grid_inputs, timeout_sec=timeout_sec)
    res_mut_grid = execute_batch_safely(mut_extracted, mut_name, grid_inputs, timeout_sec=timeout_sec)

    witnesses = []
    for idx, x in enumerate(grid_inputs):
        r_ref = res_ref_grid[idx]
        r_mut = res_mut_grid[idx]

        if not r_ref["success"]:
            return False, f"Reference failed on domain input {x}: {r_ref['error']}", details
        if r_mut.get("error_kind") == "infrastructure":
            return False, f"Mutation execution unavailable on input {x}: {r_mut['error']}", details
        # A real runtime exception in the mutant is a defect; a timeout is not.
        if not r_mut["success"] or r_ref["result"] != r_mut["result"]:
            witnesses.append(x)

    if not witnesses:
        return False, "Equivalent function: f_mut behaves identically to f_ref across entire domain", details

    sparsity = len(witnesses) / max(1, domain_size)
    details["witnesses"] = witnesses
    details["sparsity"] = sparsity
    details["clean_f_mut_src"] = mut_extracted
    details["clean_f_ref_src"] = ref_extracted

    return True, "valid_mutation", details


def evaluate_breaker_counterexample(
    f_ref_src: str,
    f_mut_src: str,
    candidate_input: Any,
    domain: Tuple[int, int],
    timeout_sec: float = 1.0
) -> Tuple[bool, Any, Any, Optional[str]]:
    """
    验证 Breaker 给出的反例输入 x 是否在 domain 内，并是否触发 f_ref(x) != f_mut(x) 差分。
    注意: 两边函数必须均为可解析、可定义的合法 Python 函数，语法错误或无法编译的代码绝不计为分流！

    Returns:
        (diverged, res_ref, res_mut, error_msg)
    """
    low, high = domain
    if type(candidate_input) is not int:
        return False, None, None, "Input must be an integer (not a float or bool)"
    x = candidate_input
    if x < low or x > high:
        return False, None, None, f"Input {x} is out of domain [{low}, {high}]"

    # 严格语法校验: 两端代码必须均可解析为合法的函数定义
    ref_extracted, ref_name, ref_err = extract_python_function(f_ref_src)
    if ref_err or not ref_extracted:
        return False, None, None, f"Reference function malformed: {ref_err}"

    mut_extracted, mut_name, mut_err = extract_python_function(f_mut_src)
    if mut_err or not mut_extracted:
        return False, None, None, f"Mutated function malformed: {mut_err}"

    ref_name = ref_name or "f_ref"
    mut_name = mut_name or "f_mut"

    res_ref_list = execute_batch_safely(ref_extracted, ref_name, [x], timeout_sec=timeout_sec)
    res_mut_list = execute_batch_safely(mut_extracted, mut_name, [x], timeout_sec=timeout_sec)

    r_ref = res_ref_list[0]
    r_mut = res_mut_list[0]

    if not r_ref["success"]:
        return False, None, None, f"Reference execution failed: {r_ref['error']}"
    if r_mut.get("error_kind") == "infrastructure":
        return False, r_ref["result"], None, f"Mutation execution unavailable: {r_mut['error']}"

    # 若一方正常返回，另一方抛出合法运行时异常 (例如 ZeroDivisionError)，确认为差分成功
    if r_ref["success"] != r_mut["success"]:
        return (
            True,
            r_ref["result"] if r_ref["success"] else f"Crash: {r_ref['error']}",
            r_mut["result"] if r_mut["success"] else f"Crash: {r_mut['error']}",
            None
        )

    # 两者均正常执行，比对输出是否不相等
    diverged = (r_ref["result"] != r_mut["result"])
    return diverged, r_ref["result"], r_mut["result"], None
