"""
scripts/verify_dual_lora.py - 双独立 LoRA 适配器物理隔离与 Mutator 冷启动合法率验证脚本

核心校验项:
  1. Base 权重全冻结校验: 验证 base_model 中非 LoRA 参数 requires_grad 均为 False
  2. 梯度与权重更新绝对物理隔离 (Zero Parameter Bleed):
     - 更新 mutator 时，breaker 所有参数权重绝对不变 (delta == 0.0)
     - 更新 breaker 时，mutator 所有参数权重绝对不变 (delta == 0.0)
  3. 初始 Mutator 冷启动合法率校验:
     - 采样种子池并生成行级补丁突变，验证有效突变率 (VMR) > 20%
"""

import argparse
import os
import sys
import torch

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grpo_engine import generate_with_policy, rollout_generation_config, setup_lora_and_model
from core.sandbox import validate_mutation
from data.seed_functions import get_seed_functions
from roles.mutator import extract_mutator_code, format_mutator_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Dual-LoRA Isolation and Initial Mutator VMR")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地路径")
    parser.add_argument("--init_adapter", type=str,
                        default="",
                        help="nano-GRPO 训练好的 breaker 适配器路径")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="用于测试 Mutator 初始合法率的种子函数数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()


def verify_base_freezing(model: torch.nn.Module):
    """校验 1: 验证 base 骨干权重全部冻结"""
    print("\n[Check 1/3] Verifying Base Backbone Parameter Freezing...")
    trainable_base_params = []
    for name, param in model.named_parameters():
        if "lora" not in name.lower() and param.requires_grad:
            trainable_base_params.append(name)

    if trainable_base_params:
        print(f"❌ FAIL: Found {len(trainable_base_params)} unfrozen base parameters!")
        for n in trainable_base_params[:5]:
            print(f"   * {n}")
        raise RuntimeError("Base backbone parameters are not properly frozen!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ PASS: Base weights 100% frozen! Total: {total_params:,} | Currently active trainable: {trainable_params:,}")


def verify_weight_isolation(model: torch.nn.Module, tokenizer: any, device: str):
    """校验 2: 验证双优化器与双 LoRA 物理更新隔离 (Zero Bleed)"""
    print("\n[Check 2/3] Verifying Adapter Weight Invariance (Zero Parameter Bleed)...")

    # 1. 实例化双角色专属独立优化器
    model.set_adapter("mutator")
    mut_params = [p for n, p in model.named_parameters() if "mutator" in n and p.requires_grad]
    opt_mut = torch.optim.AdamW(mut_params, lr=1e-4)

    model.set_adapter("breaker")
    brk_params = [p for n, p in model.named_parameters() if "breaker" in n and p.requires_grad]
    opt_brk = torch.optim.AdamW(brk_params, lr=1e-4)

    # 记录更新前 Breaker 所有参数的精确镜像
    breaker_before = {n: p.detach().clone() for n, p in model.named_parameters() if "breaker" in n}
    mutator_before = {n: p.detach().clone() for n, p in model.named_parameters() if "mutator" in n}

    # 2. 激活 Mutator，执行前向、反传与优化更新
    model.set_adapter("mutator")
    model.train()
    dummy_input = tokenizer("def test(): pass", return_tensors="pt").to(device)
    out = model(**dummy_input, labels=dummy_input["input_ids"])
    loss_mut = out.loss
    loss_mut.backward()
    opt_mut.step()
    opt_mut.zero_grad()

    # 3. 校验 Breaker 参数是否发生了任何变化
    breaker_after_mut_step = {n: p.detach().clone() for n, p in model.named_parameters() if "breaker" in n}
    mutator_after_mut_step = {n: p.detach().clone() for n, p in model.named_parameters() if "mutator" in n}

    delta_breaker = sum((breaker_after_mut_step[n] - breaker_before[n]).abs().sum().item() for n in breaker_before)
    delta_mutator = sum((mutator_after_mut_step[n] - mutator_before[n]).abs().sum().item() for n in mutator_before)

    print(f"   * Mutator Step: delta(Mutator weights) = {delta_mutator:.6f} (Should be > 0)")
    print(f"   * Mutator Step: delta(Breaker weights) = {delta_breaker:.10f} (Must be == 0.0)")

    if delta_breaker != 0.0:
        raise RuntimeError(f"❌ Isolation breached! Breaker weights changed by {delta_breaker} during mutator step!")
    if delta_mutator == 0.0:
        raise RuntimeError("❌ Mutator weights failed to update during mutator step!")

    # 4. 激活 Breaker，执行前向、反传与优化更新
    model.set_adapter("breaker")
    model.train()
    out = model(**dummy_input, labels=dummy_input["input_ids"])
    loss_brk = out.loss
    loss_brk.backward()
    opt_brk.step()
    opt_brk.zero_grad()

    # 5. 校验 Mutator 参数是否发生了任何变化
    mutator_after_brk_step = {n: p.detach().clone() for n, p in model.named_parameters() if "mutator" in n}
    delta_mutator_on_brk_step = sum((mutator_after_brk_step[n] - mutator_after_mut_step[n]).abs().sum().item() for n in mutator_before)
    delta_breaker_on_brk_step = sum((model.state_dict()[n].detach() - breaker_after_mut_step[n]).abs().sum().item() for n in breaker_before if n in model.state_dict())

    print(f"   * Breaker Step: delta(Breaker weights) = {delta_breaker_on_brk_step:.6f} (Should be > 0)")
    print(f"   * Breaker Step: delta(Mutator weights) = {delta_mutator_on_brk_step:.10f} (Must be == 0.0)")

    if delta_mutator_on_brk_step != 0.0:
        raise RuntimeError(f"❌ Isolation breached! Mutator weights changed by {delta_mutator_on_brk_step} during breaker step!")

    print("✅ PASS: Perfect Dual-LoRA & Dual-Optimizer Physical Isolation! (delta_cross == 0.0000000000)")


def verify_initial_mutator_vmr(model: torch.nn.Module, tokenizer: any, num_seeds: int, device: str):
    """校验 3: 验证全新初始化的 Mutator 结合行级补丁首发合法率 (VMR) > 20%"""
    print(f"\n[Check 3/3] Verifying Initial Mutator Cold-Start Valid Mutation Rate (VMR > 20%)...")
    model.set_adapter("mutator")
    model.eval()

    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=32, temperature=0.7, do_sample=True)
    all_seeds = get_seed_functions()[:num_seeds]

    valid_count = 0
    total_rollouts = 0
    passed_syntax = 0
    passed_sanity = 0

    for idx, seed in enumerate(all_seeds, 1):
        prompt = format_mutator_prompt(seed)
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        p_len = inp["input_ids"].shape[1]
        rep_inp = {k: v.repeat(4, 1) for k, v in inp.items()}

        with torch.no_grad():
            out = generate_with_policy(model, cfg_mut, **rep_inp)

        for c_i in range(4):
            total_rollouts += 1
            comp = tokenizer.decode(out[c_i][p_len:], skip_special_tokens=True).strip()
            code_src, _, _ = extract_mutator_code(comp, f_ref_src=seed["func_src"])

            if code_src and code_src.strip() != seed["func_src"].strip():
                try:
                    import ast
                    ast.parse(code_src)
                    passed_syntax += 1
                    is_valid, _, details = validate_mutation(
                        f_ref_src=seed["func_src"],
                        f_mut_src=code_src,
                        sanity_inputs=seed["sanity_inputs"],
                        domain=tuple(seed["domain"])
                    )
                    if details.get("sanity_passed", False):
                        passed_sanity += 1
                    if is_valid:
                        valid_count += 1
                except Exception:
                    pass

    vmr = valid_count / max(1, total_rollouts)
    syntax_rate = passed_syntax / max(1, total_rollouts)
    sanity_rate = passed_sanity / max(1, total_rollouts)

    print(f"   * Total Rollouts Tested: {total_rollouts} (from {num_seeds} seed functions)")
    print(f"   * Syntax Parse Rate:     {syntax_rate:.1%} ({passed_syntax}/{total_rollouts})")
    print(f"   * X_sanity Invariant:    {sanity_rate:.1%} ({passed_sanity}/{total_rollouts})")
    print(f"   * Valid Mutation Rate:   {vmr:.1%} ({valid_count}/{total_rollouts}) (Target: > 20.0%)")

    if vmr <= 0.20:
        raise AssertionError(f"❌ FAIL: Cold-start Mutator VMR is {vmr:.1%}, which does not exceed 20.0% requirement!")

    print(f"✅ PASS: Mutator cold-start VMR is {vmr:.1%} (> 20.0% threshold fulfilled)!")


def main():
    args = parse_args()
    print("=" * 72)
    print("🔍 Running nano-CodeSelfPlay Dual-LoRA & Mutator Cold-Start Verification Suite")
    print("=" * 72)

    # 1. 加载双 LoRA 模型
    tokenizer, model = setup_lora_and_model(
        model_name=args.model_name,
        device=args.device,
        init_adapter=args.init_adapter
    )

    # 2. 校验 Base 权重冻结
    verify_base_freezing(model)

    # 3. 校验双 LoRA 物理更新隔离
    verify_weight_isolation(model, tokenizer, args.device)

    # 4. 校验 Mutator 冷启动合法率
    verify_initial_mutator_vmr(model, tokenizer, args.num_seeds, args.device)

    print("\n" + "=" * 72)
    print("🎉 ALL DUAL-LORA ARCHITECTURAL & VALIDITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
