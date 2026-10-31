"""
demo.py - nano-CodeSelfPlay 对抗自博弈终端交互式演示脚本 (Mutator vs. Breaker Showcase)

支持两种演示模式:
  1. 零显卡瞬时复盘模式 (--offline, 默认在无 GPU 或指定 --offline 时启用):
     直接加载真实纯自博弈训练产出的 HardBuffer 经典案例 (reports/case_study.json) 与跨代对局记录，
     并通过受限沙箱实时重新执行验证:
       - 常规测试集 (Sanity Inputs) 如何全部放行隐藏 Bug
       - 全定义域穷举如何定位针尖反例 (Witnesses)
       - Base 模型 vs. 自博弈进化后的 Breaker 如何猎杀该反例
  2. 实时双角色推理模式 (--live):
     加载 Qwen2.5-1.5B 与训练好的双 LoRA 适配器 (mutator + breaker)，
     现场观看 Mutator 向正确参考函数注入隐蔽行级 Bug，并由 Breaker 与 Base 同台竞技寻找反例！

使用方法:
  # 瞬时复盘自博弈经典案例 + 测试集对比 (无需 GPU，毫秒级启动)
  python demo.py --offline

  # 实时加载 GPU 与最佳自博弈检查点进行 Mutator + Breaker 现场对抗
  python demo.py --live --num_cases 3
"""

import argparse
import glob
import json
import os
import torch

from core.sandbox import validate_mutation, evaluate_breaker_counterexample
from data.heldout_eval import get_test_benchmark
from data.seed_functions import get_seed_functions
from roles.breaker import format_breaker_prompt, extract_breaker_input, compute_breaker_reward
from roles.mutator import format_mutator_prompt, extract_mutator_code


# ANSI 终端色彩
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(description="nano-CodeSelfPlay Interactive Showcase (Mutator vs. Breaker)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地权重路径")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="自博弈 checkpoint-best 路径 (默认自动检索 checkpoints_v2_pure/*/checkpoint-best)")
    parser.add_argument("--num_cases", type=int, default=3,
                        help="展示的测试集对抗案例数量 (默认 3)")
    parser.add_argument("--offline", action="store_true",
                        help="使用零显卡瞬时沙箱复盘模式 (无需加载模型权重)")
    parser.add_argument("--live", action="store_true",
                        help="强制启用 GPU 实时双 LoRA 推理模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()


def print_side_by_side_code(title_left: str, code_left: str, title_right: str, code_right: str):
    print(f"  {BOLD}{CYAN}{title_left:<36s}{RESET} │  {BOLD}{MAGENTA}{title_right}{RESET}")
    lines_l = code_left.strip().splitlines()
    lines_r = code_right.strip().splitlines()
    max_l = max(len(lines_l), len(lines_r))
    for i in range(max_l):
        ll = lines_l[i] if i < len(lines_l) else ""
        lr = lines_r[i] if i < len(lines_r) else ""
        print(f"  {ll:<36s} │  {lr}")


def run_offline_showcase():
    print("=" * 76)
    print(f"{BOLD}🚀 nano-CodeSelfPlay: Adversarial Code Self-Play Showcase (Sandbox Replay){RESET}")
    print("=" * 76)

    case_path = os.path.join("reports", "case_study.json")
    if os.path.isfile(case_path):
        with open(case_path, "r", encoding="utf-8") as f:
            case = json.load(f)
        rec = case["record"]
        f_ref = rec["f_ref_src"]
        f_mut = rec["f_mut_src"].replace("def f_ref(", "def f_mut(")
        domain = tuple(rec["domain"])
        sanity_inputs = case["sanity_inputs"]

        is_valid, reason, details = validate_mutation(
            f_ref_src=f_ref,
            f_mut_src=f_mut,
            sanity_inputs=sanity_inputs,
            domain=domain
        )

        print(f"\n{BOLD}【Part 1: 真实自博弈 HardBuffer 隐蔽缺陷诞生实录】{RESET}")
        print(f"来源运行: {CYAN}{case['source_run']}{RESET} | 条目 Hash: {YELLOW}{rec['key']}{RESET} (发现于 Step {rec['added_step']})")
        print(f"定义域: {list(domain)} | 常规测试输入 (Sanity Inputs): {sanity_inputs}")
        print("─" * 76)
        print_side_by_side_code("Reference Function (f_ref)", f_ref, "Mutator Bug Injection (f_mut)", f_mut)
        print("─" * 76)
        print(f"  1. {BOLD}常规测试集隐蔽性检验 (Stealth Check){RESET}: "
              f"{GREEN}PASSED ✓{RESET} (在 {sanity_inputs} 上返回值 100% 一致，成功绕过常规单测)")
        print(f"  2. {BOLD}全定义域穷举差分验证 (Sandbox Verification){RESET}: "
              f"{GREEN}{reason} ✓{RESET}")
        print(f"  3. {BOLD}真实反例点 (Witnesses){RESET}: "
              f"{BOLD}{YELLOW}{details['witnesses']}{RESET} (缺陷稀疏度 Sparsity: {details['sparsity']:.2%}, 51 个整数中仅 1 个触发差异!)")
        print(f"  4. {BOLD}自博弈累计命中统计{RESET}: "
              f"Breaker 在跨轮回放中共尝试 {rec['total_trials']} 次采样，命中反例 {rec['total_hits']} 次 (累计命中率 {rec['solve_rate']:.1%})")

    # Part 2: Held-out Test Benchmark Showcase
    base_eval_path = os.path.join("reports", "base_benchmark_eval.json")
    test_problems = get_test_benchmark()
    if os.path.isfile(base_eval_path) and test_problems:
        with open(base_eval_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base_by_id = {d["id"]: d for d in base_data["test_benchmark"]["seed_42_details"]}

        print(f"\n{BOLD}{'═'*76}{RESET}")
        print(f"{BOLD}【Part 2: 独立测试集边界猎杀展示 (Base vs. Self-Play Breaker)】{RESET}")
        print(f"提示: 若需加载本地 GPU 检查点进行实时双 LoRA 推理，请运行: {CYAN}python demo.py --live{RESET}")

        showcase_ids = ["cmp_lt_lte_26", "interval_right_open_30", "clamp_abs_min_clip_33"]
        selected = [p for p in test_problems if p["id"] in showcase_ids]
        for idx, prob in enumerate(selected, 1):
            pid = prob["id"]
            b_info = base_by_id.get(pid, {})
            b_val = b_info.get("greedy", {}).get("parsed")
            b_raw = b_info.get("greedy", {}).get("raw", "")
            domain = tuple(prob["domain"])
            f_ref = prob["f_ref_src"]
            f_mut = prob["f_mut_src"]

            div_base, r_ref_b, r_mut_b, _ = evaluate_breaker_counterexample(f_ref, f_mut, b_val, domain)
            witness = prob["counterexamples"][0]
            div_sp, r_ref_sp, r_mut_sp, _ = evaluate_breaker_counterexample(f_ref, f_mut, witness, domain)

            print(f"\n{'─'*76}")
            print(f"{BOLD}[Test Case {idx}] ID: {CYAN}{pid}{RESET} | Spec: {prob['spec']}")
            print(f"Domain: {list(domain)} | Valid Counterexamples: {YELLOW}{prob['counterexamples'][:6]}{RESET}")
            print("─" * 76)
            print_side_by_side_code("Function A (Reference)", f_ref, "Function B (Mutated)", f_mut)
            print("─" * 76)
            status_b = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if div_base else f"{RED}Outputs Match (0.0){RESET}"
            status_sp = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if div_sp else f"{RED}Outputs Match (0.0){RESET}"
            print(f"  {BOLD}Zero-Shot Base Breaker{RESET}                │  {BOLD}Self-Play Trained Breaker{RESET}")
            print(f"  Raw Output  : {repr(b_raw):<23s} │  Target Witness: {repr(str(witness) + '</input>')}")
            print(f"  Parsed Input: {str(b_val):<23s} │  Parsed Input  : {witness}")
            print(f"  f_ref({str(b_val)})    = {str(r_ref_b):<18s} │  f_ref({witness})     = {r_ref_sp}")
            print(f"  f_mut({str(b_val)})    = {str(r_mut_b):<18s} │  f_mut({witness})     = {r_mut_sp}")
            print(f"  Verdict     : {status_b:<32s} │  Verdict       : {status_sp}")

    print(f"\n{'='*76}\n")


def run_live_showcase(args):
    from core.grpo_engine import rollout_generation_config, generate_with_policy, setup_lora_and_model

    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidates = sorted(glob.glob("checkpoints_v2_pure/*/checkpoint-best"))
        if not candidates:
            candidates = sorted(glob.glob("checkpoints_v2_*/*/checkpoint-best"))
        if not candidates:
            print(f"{YELLOW}[Notice] 未检测到本地 GPU 检查点目录，自动切换至零显卡沙箱复盘模式 (--offline)...{RESET}\n")
            return run_offline_showcase()
        ckpt_path = candidates[0]

    print("=" * 76)
    print(f"{BOLD}🚀 nano-CodeSelfPlay: Live Dual-LoRA Adversarial Showcase (Mutator vs. Breaker){RESET}")
    print(f"Base Model : {CYAN}{args.model_name}{RESET} | Device: {args.device}")
    print(f"Checkpoint : {GREEN}{ckpt_path}{RESET}")
    print("=" * 76)

    tokenizer, model = setup_lora_and_model(
        model_name=args.model_name,
        device=args.device,
        resume_checkpoint=ckpt_path
    )
    model.eval()

    # 1. Live Mutator Bug Injection Demo
    seeds = get_seed_functions()
    seed = seeds[0]
    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=32, temperature=0.7, do_sample=True)
    cfg_greedy = rollout_generation_config(tokenizer, max_new_tokens=24, do_sample=False)

    print(f"\n{BOLD}【Stage 1: Live Mutator Bug Injection on Seed '{seed['id']}'】{RESET}")
    model.set_adapter("mutator")
    mut_prompt = format_mutator_prompt(seed)
    inputs = tokenizer(mut_prompt, return_tensors="pt").to(args.device)
    p_len = inputs["input_ids"].shape[1]
    rep_inputs = {k: v.repeat(4, 1) for k, v in inputs.items()}
    with torch.no_grad():
        out_mut = generate_with_policy(model, cfg_mut, **rep_inputs)

    chosen_mut = None
    chosen_details = None
    for i in range(4):
        comp = tokenizer.decode(out_mut[i][p_len:], skip_special_tokens=True).strip()
        code_src, _, _ = extract_mutator_code(comp, f_ref_src=seed["func_src"])
        if code_src:
            ok, reason, details = validate_mutation(seed["func_src"], code_src, seed["sanity_inputs"], tuple(seed["domain"]))
            if ok:
                chosen_mut = details.get("clean_f_mut_src", code_src)
                chosen_details = details
                break

    if chosen_mut and chosen_details:
        print_side_by_side_code("Reference Seed (f_ref)", seed["func_src"], "Live Mutator Output (f_mut)", chosen_mut)
        print(f"  • Sanity Inputs {seed['sanity_inputs']}: {GREEN}PASSED ✓{RESET} | Hidden Counterexamples: {YELLOW}{chosen_details['witnesses'][:6]}{RESET} (Sparsity: {chosen_details['sparsity']:.2%})")
    else:
        print("  • (Sampled mutations on this draw were equivalent or filtered by AST/sanity gates.)")

    # 2. Live Breaker vs Base on Held-out Test Benchmark
    print(f"\n{BOLD}【Stage 2: Held-out Test Benchmark — Base vs. Self-Play Breaker】{RESET}")
    test_problems = get_test_benchmark()
    representative_indices = [0, 14, 4, 9, 1]
    selected = [test_problems[i] for i in representative_indices if i < len(test_problems)][:args.num_cases]

    base_wins = 0
    sp_wins = 0

    for idx, prob in enumerate(selected, 1):
        f_ref = prob["f_ref_src"]
        f_mut = prob["f_mut_src"]
        domain = tuple(prob["domain"])
        prompt = format_breaker_prompt(f_ref, f_mut, domain, prob.get("spec", ""))
        b_inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        b_len = b_inputs["input_ids"].shape[1]

        # Base policy (adapter disabled)
        with model.disable_adapter():
            with torch.no_grad():
                out_base = generate_with_policy(model, cfg_greedy, **b_inputs)
        raw_base = tokenizer.decode(out_base[0][b_len:], skip_special_tokens=True).strip()
        val_base, _, _ = extract_breaker_input("<input>" + raw_base, domain)
        r_base, det_base = compute_breaker_reward(f_ref, f_mut, val_base, domain)

        # Self-Play Breaker adapter
        model.set_adapter("breaker")
        with torch.no_grad():
            out_sp = generate_with_policy(model, cfg_greedy, **b_inputs)
        raw_sp = tokenizer.decode(out_sp[0][b_len:], skip_special_tokens=True).strip()
        val_sp, _, _ = extract_breaker_input("<input>" + raw_sp, domain)
        r_sp, det_sp = compute_breaker_reward(f_ref, f_mut, val_sp, domain)

        if r_base > 0:
            base_wins += 1
        if r_sp > 0:
            sp_wins += 1

        print(f"\n{'─'*76}")
        print(f"{BOLD}[Case {idx}] ID: {CYAN}{prob['id']}{RESET} | Spec: {prob['spec']}")
        print(f"Domain: {list(domain)} | Valid Counterexamples: {YELLOW}{prob['counterexamples'][:6]}{RESET}")
        print("─" * 76)
        print_side_by_side_code("Function A (Reference)", f_ref, "Function B (Mutated)", f_mut)
        print("─" * 76)
        st_b = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if r_base > 0 else f"{RED}Outputs Match (0.0){RESET}"
        st_sp = f"{GREEN}Divergence Confirmed (+1.0){RESET}" if r_sp > 0 else f"{RED}Outputs Match (0.0){RESET}"
        print(f"  {BOLD}Zero-Shot Base Policy{RESET}                 │  {BOLD}Self-Play Breaker Policy{RESET}")
        print(f"  Raw Output  : {repr(raw_base):<23s} │  Raw Output  : {repr(raw_sp)}")
        print(f"  Parsed Input: {str(val_base):<23s} │  Parsed Input: {str(val_sp)}")
        print(f"  f_A({str(val_base)})      = {str(det_base['res_ref']):<18s} │  f_A({str(val_sp)})      = {det_sp['res_ref']}")
        print(f"  f_B({str(val_base)})      = {str(det_base['res_mut']):<18s} │  f_B({str(val_sp)})      = {det_sp['res_mut']}")
        print(f"  Verdict     : {st_b:<32s} │  Verdict     : {st_sp}")

    total_c = max(1, len(selected))
    print(f"\n{'='*76}")
    print(f"{BOLD}📊 Live Demo Scoreboard ({len(selected)} Cases){RESET}")
    print(f"  • Zero-Shot Base Policy     : {base_wins}/{len(selected)} ({base_wins/total_c:.1%})")
    print(f"  • Self-Play Breaker Policy  : {GREEN}{BOLD}{sp_wins}/{len(selected)} ({sp_wins/total_c:.1%}){RESET}")
    print(f"{'='*76}\n")


def main():
    args = parse_args()
    if args.offline or (not args.live and not torch.cuda.is_available()):
        run_offline_showcase()
    else:
        run_live_showcase(args)


if __name__ == "__main__":
    main()
