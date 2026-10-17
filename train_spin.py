"""
train_spin.py - nano-SPIN 双角色代码对抗自博弈训练主循环 (Adversarial Self-Play Loop)

设计理念:
  1. 零新增外部数据: 仅依赖 data/seed_functions.py 中的轻量种子函数池。
  2. 单模型共享权重: 通过 Prompt 切换分饰 Mutator 与 Breaker 两角。
  3. 互为环境的共演飞轮 (Co-Evolution):
     - Phase A (出题): Mutator 针对种子注入隐蔽缺陷，通过 X_sanity 验证并对抗当前 Breaker，更新 Mutator 策略；
     - Phase B (破题): 将生成的合法缺陷对注入对抗缓冲池，Breaker 尝试挖掘反例，更新 Breaker 策略；
     - Phase C (大考): 每轮结束后在固定的 held-out 测试集上评估 Breaker 的真实泛化演进。
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from core.grpo_engine import (
    SPINGRPOTrainer,
    generate_with_policy,
    rollout_generation_config,
    setup_lora_and_model
)
from core.sandbox import (
    evaluate_breaker_counterexample,
    extract_python_function,
    validate_mutation
)
from data.heldout_eval import get_heldout_benchmark
from data.seed_functions import get_seed_functions
from roles.breaker import (
    compute_breaker_reward,
    extract_breaker_input,
    format_breaker_prompt
)
from roles.mutator import (
    compute_mutator_reward,
    extract_mutator_code,
    format_mutator_prompt
)


def parse_args():
    parser = argparse.ArgumentParser(description="nano-SPIN Adversarial Self-Play Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地路径")
    parser.add_argument("--init_adapter", type=str, default="",
                        help="可选预训练 LoRA 适配器路径 (可热启动 nano-GRPO 训练好的权重)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="自博弈总轮次 (默认 5 轮)")
    parser.add_argument("--seeds_per_round", type=int, default=8,
                        help="每轮选取的种子函数数量")
    parser.add_argument("--mut_rollouts", type=int, default=4,
                        help="Mutator 每题采样的候选突变数 (G_mut)")
    parser.add_argument("--breaker_rollouts", type=int, default=4,
                        help="Breaker 每题采样的候选反例数 (G_brk)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="优化器学习率")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="基座参考策略 KL 阻尼系数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样探索温度")
    parser.add_argument("--output_dir", type=str, default="checkpoints_spin",
                        help="检查点与日志输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()


def evaluate_on_heldout(
    model: torch.nn.Module,
    tokenizer: Any,
    heldout_problems: List[Dict[str, Any]],
    device: str
) -> Tuple[float, float, float]:
    """
    在独立外部基准上评估当前策略的 Greedy Pass@1 与 Sample Pass@4。
    
    Returns:
        (greedy_pass1, sample_pass4, hit_rate)
    """
    model.eval()
    cfg_greedy = rollout_generation_config(tokenizer, max_new_tokens=24, do_sample=False)
    cfg_sample = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=0.7, do_sample=True)

    greedy_hits = 0
    sample_hits = 0
    total_candidates = 0
    total_candidate_hits = 0

    for prob in heldout_problems:
        f_ref = prob["f_ref_src"]
        f_mut = prob["f_mut_src"]
        domain = tuple(prob["domain"])
        prompt = format_breaker_prompt(f_ref, f_mut, domain, prob.get("spec", ""))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        p_len = inputs["input_ids"].shape[1]

        # 1. Greedy Pass@1
        with torch.no_grad():
            out_g = generate_with_policy(model, cfg_greedy, **inputs)
        raw_g = tokenizer.decode(out_g[0][p_len:], skip_special_tokens=True).strip()
        val_g, _, _ = extract_breaker_input("<input>" + raw_g, domain)
        r_g, _ = compute_breaker_reward(f_ref, f_mut, val_g, domain)
        if r_g > 0.0:
            greedy_hits += 1

        # 2. Sample Pass@4
        rep_inputs = {k: v.repeat(4, 1) for k, v in inputs.items()}
        with torch.no_grad():
            out_s = generate_with_policy(model, cfg_sample, **rep_inputs)
        solved_any = False
        for i in range(4):
            total_candidates += 1
            raw_s = tokenizer.decode(out_s[i][p_len:], skip_special_tokens=True).strip()
            val_s, _, _ = extract_breaker_input("<input>" + raw_s, domain)
            r_s, _ = compute_breaker_reward(f_ref, f_mut, val_s, domain)
            if r_s > 0.0:
                total_candidate_hits += 1
                solved_any = True
        if solved_any:
            sample_hits += 1

    total_probs = len(heldout_problems)
    greedy_acc = greedy_hits / max(1, total_probs)
    sample_p4 = sample_hits / max(1, total_probs)
    hit_rate = total_candidate_hits / max(1, total_candidates)
    model.train()
    return greedy_acc, sample_p4, hit_rate


def main():
    args = parse_args()
    run_timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = os.path.join(args.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print("🚀 nano-SPIN: Code Mutation & Counterexample Adversarial Self-Play")
    print(f"Base Model: {args.model_name} | Device: {args.device}")
    print(f"Self-Play Rounds: {args.rounds} | Output Dir: {run_dir}")
    print("=" * 72)

    # 1. 加载模型与 LoRA
    tokenizer, model = setup_lora_and_model(
        model_name=args.model_name,
        device=args.device,
        init_adapter=args.init_adapter
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    trainer = SPINGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=args.device,
        kl_coef=args.kl_coef
    )

    all_seeds = get_seed_functions()
    heldout_problems = get_heldout_benchmark()
    print(f"[Data Setup] Loaded {len(all_seeds)} seed functions and {len(heldout_problems)} heldout evaluation problems.")

    # 对抗题目缓冲池 (Adversarial Bug Pool)
    adversarial_bug_pool: List[Dict[str, Any]] = []

    # 初始评估基座模型在 Held-out 测试集上的表现
    print("\n[Baseline Eval] Evaluating initial model on held-out benchmark...")
    base_greedy, base_p4, base_hit = evaluate_on_heldout(model, tokenizer, heldout_problems, args.device)
    print(f"⭐ [Round 00 / Baseline] Held-out Greedy Pass@1: {base_greedy:.2%} | Sample Pass@4: {base_p4:.2%} | Hit Density: {base_hit:.2%}\n")

    best_heldout_pass1 = base_greedy
    metrics_log_path = os.path.join(run_dir, "metrics_spin.jsonl")

    # 预设配置
    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=96, temperature=args.temperature, do_sample=True)
    cfg_brk = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=args.temperature, do_sample=True)

    for round_idx in range(1, args.rounds + 1):
        round_start_time = time.time()
        print(f"\n{'='*72}")
        print(f"🔄 【Self-Play Round {round_idx:02d} / {args.rounds:02d}】")
        print(f"{'='*72}")

        # -------------------------------------------------------------
        # Phase A: 出题阶段 (Mutator Rollout & Policy Update)
        # -------------------------------------------------------------
        print("\n--- [Phase A: Mutator Exploration & Injection] ---")
        sampled_seeds = random.sample(all_seeds, min(args.seeds_per_round, len(all_seeds)))
        mut_valid_count = 0
        mut_total_candidates = 0
        mut_rewards_list = []
        mut_effective_groups = 0

        for seed_i, seed in enumerate(sampled_seeds, 1):
            mut_prompt = format_mutator_prompt(seed)
            inputs = tokenizer(mut_prompt, return_tensors="pt").to(args.device)
            p_len = inputs["input_ids"].shape[1]
            rep_inputs = {k: v.repeat(args.mut_rollouts, 1) for k, v in inputs.items()}

            # 生成突变代码
            with torch.no_grad():
                out_mut = generate_with_policy(model, cfg_mut, **rep_inputs)

            candidate_rewards = []
            for c_i in range(args.mut_rollouts):
                mut_total_candidates += 1
                completion = tokenizer.decode(out_mut[c_i][p_len:], skip_special_tokens=True).strip()
                raw_candidate = completion if "def " in completion else ("def f_mut(x: int)" + completion)
                code_src, _, _ = extract_mutator_code(raw_candidate)
                if not code_src:
                    candidate_rewards.append(0.0)
                    continue

                # 沙箱三层校验
                is_valid, reason, details = validate_mutation(
                    f_ref_src=seed["func_src"],
                    f_mut_src=code_src,
                    sanity_inputs=seed["sanity_inputs"],
                    domain=tuple(seed["domain"])
                )

                if not is_valid:
                    candidate_rewards.append(0.0)
                else:
                    mut_valid_count += 1
                    # 让当前 Breaker 尝试破解以确定突变难度
                    clean_ref = details.get("clean_f_ref_src", seed["func_src"])
                    clean_mut = details.get("clean_f_mut_src", code_src)
                    brk_prompt = format_breaker_prompt(clean_ref, clean_mut, tuple(seed["domain"]))
                    brk_inputs = tokenizer(brk_prompt, return_tensors="pt").to(args.device)
                    brk_p_len = brk_inputs["input_ids"].shape[1]
                    brk_rep = {k: v.repeat(4, 1) for k, v in brk_inputs.items()}

                    with torch.no_grad():
                        out_brk_probe = generate_with_policy(model, cfg_brk, **brk_rep)

                    brk_hits = 0
                    for probe_j in range(4):
                        probe_comp = tokenizer.decode(out_brk_probe[probe_j][brk_p_len:], skip_special_tokens=True).strip()
                        p_val, _, _ = extract_breaker_input("<input>" + probe_comp, tuple(seed["domain"]))
                        r_probe, _ = compute_breaker_reward(clean_ref, clean_mut, p_val, tuple(seed["domain"]))
                        if r_probe > 0.0:
                            brk_hits += 1

                    solve_rate = brk_hits / 4.0
                    r_mut, _ = compute_mutator_reward(is_valid=True, breaker_solve_rate=solve_rate)
                    candidate_rewards.append(r_mut)

                    # 若难倒了 Breaker (p_solve < 1.0)，存入对抗题库供 Breaker 针对性训练
                    if solve_rate < 1.0:
                        adversarial_bug_pool.append({
                            "seed_id": seed["id"],
                            "domain": seed["domain"],
                            "f_ref_src": clean_ref,
                            "f_mut_src": clean_mut,
                            "sparsity": details.get("sparsity", 0.0),
                            "witnesses": details.get("witnesses", []),
                        })

            mut_rewards_list.extend(candidate_rewards)
            # 执行单题 GRPO 反向传播
            is_eff, _, _ = trainer.backward_grpo_group(
                prompt_ids=inputs["input_ids"],
                generated_ids=out_mut,
                rewards=candidate_rewards,
                temperature=args.temperature,
                num_groups_in_update=len(sampled_seeds)
            )
            if is_eff:
                mut_effective_groups += 1

        grad_norm_mut = trainer.step_optimizer()
        mut_validity_rate = mut_valid_count / max(1, mut_total_candidates)
        avg_mut_reward = sum(mut_rewards_list) / max(1, len(mut_rewards_list))
        print(f"  [Mutator Stats] Valid Rate: {mut_validity_rate:.1%} ({mut_valid_count}/{mut_total_candidates}) | Avg Reward: {avg_mut_reward:.3f} | Eff Groups: {mut_effective_groups}/{len(sampled_seeds)} | GradNorm: {grad_norm_mut:.3f}")
        print(f"  [Pool Size] Total adversarial bug pairs in pool: {len(adversarial_bug_pool)}")

        # -------------------------------------------------------------
        # Phase B: 破题阶段 (Breaker Rollout & Policy Update)
        # -------------------------------------------------------------
        print("\n--- [Phase B: Breaker Defense & Counterexample Mining] ---")
        if not adversarial_bug_pool:
            # 缓冲池为空时，从 held-out 或种子构造基础对
            print("  [Notice] Adversarial bug pool empty, synthesizing from default seed mutations...")
            active_pairs = heldout_problems[:min(args.seeds_per_round, len(heldout_problems))]
        else:
            sample_k = min(args.seeds_per_round, len(adversarial_bug_pool))
            active_pairs = random.sample(adversarial_bug_pool, sample_k)

        brk_solved_count = 0
        brk_total_candidates = 0
        brk_rewards_list = []
        brk_effective_groups = 0

        for pair_i, pair in enumerate(active_pairs, 1):
            f_ref = pair.get("f_ref_src")
            f_mut = pair.get("f_mut_src")
            domain = tuple(pair.get("domain", (0, 50)))

            brk_prompt = format_breaker_prompt(f_ref, f_mut, domain)
            inputs = tokenizer(brk_prompt, return_tensors="pt").to(args.device)
            p_len = inputs["input_ids"].shape[1]
            rep_inputs = {k: v.repeat(args.breaker_rollouts, 1) for k, v in inputs.items()}

            with torch.no_grad():
                out_brk = generate_with_policy(model, cfg_brk, **rep_inputs)

            candidate_rewards = []
            for c_i in range(args.breaker_rollouts):
                brk_total_candidates += 1
                comp = tokenizer.decode(out_brk[c_i][p_len:], skip_special_tokens=True).strip()
                val, _, _ = extract_breaker_input("<input>" + comp, domain)
                r_brk, _ = compute_breaker_reward(f_ref, f_mut, val, domain)
                candidate_rewards.append(r_brk)
                if r_brk > 0.0:
                    brk_solved_count += 1

            brk_rewards_list.extend(candidate_rewards)
            is_eff, _, _ = trainer.backward_grpo_group(
                prompt_ids=inputs["input_ids"],
                generated_ids=out_brk,
                rewards=candidate_rewards,
                temperature=args.temperature,
                num_groups_in_update=len(active_pairs)
            )
            if is_eff:
                brk_effective_groups += 1

        grad_norm_brk = trainer.step_optimizer()
        brk_win_rate = brk_solved_count / max(1, brk_total_candidates)
        avg_brk_reward = sum(brk_rewards_list) / max(1, len(brk_rewards_list))
        print(f"  [Breaker Stats] Solve Rate: {brk_win_rate:.1%} ({brk_solved_count}/{brk_total_candidates}) | Avg Reward: {avg_brk_reward:.3f} | Eff Groups: {brk_effective_groups}/{len(active_pairs)} | GradNorm: {grad_norm_brk:.3f}")

        # -------------------------------------------------------------
        # Phase C: 独立外部大考 (Held-out Benchmark Evaluation)
        # -------------------------------------------------------------
        print("\n--- [Phase C: Objective Held-out Benchmark Audit] ---")
        ho_greedy, ho_p4, ho_hit = evaluate_on_heldout(model, tokenizer, heldout_problems, args.device)
        round_duration = time.time() - round_start_time

        print(f"📊 【Round {round_idx:02d} Held-out Benchmark】")
        print(f"   * Greedy Pass@1: {ho_greedy:.2%} (vs Base: {base_greedy:.2%})")
        print(f"   * Sample Pass@4: {ho_p4:.2%} (vs Base: {base_p4:.2%})")
        print(f"   * Candidate Hit: {ho_hit:.2%} (vs Base: {base_hit:.2%})")
        print(f"   * Round Duration: {round_duration:.1f}s")

        # 记录指标
        log_entry = {
            "round": round_idx,
            "mut_validity_rate": mut_validity_rate,
            "mut_avg_reward": avg_mut_reward,
            "mut_effective_groups": mut_effective_groups,
            "brk_solve_rate": brk_win_rate,
            "brk_avg_reward": avg_brk_reward,
            "brk_effective_groups": brk_effective_groups,
            "pool_size": len(adversarial_bug_pool),
            "heldout_greedy_pass1": ho_greedy,
            "heldout_sample_pass4": ho_p4,
            "heldout_hit_rate": ho_hit,
            "duration_sec": round_duration
        }
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # 保存最新与最佳模型权重
        last_ckpt_dir = os.path.join(run_dir, "checkpoint-last")
        model.save_pretrained(last_ckpt_dir)
        tokenizer.save_pretrained(last_ckpt_dir)

        if ho_greedy >= best_heldout_pass1:
            best_heldout_pass1 = ho_greedy
            best_ckpt_dir = os.path.join(run_dir, "checkpoint-best")
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"  🏆 New Best Held-out Greedy Pass@1: {best_heldout_pass1:.2%} -> Saved to {best_ckpt_dir}")

    print("\n" + "=" * 72)
    print(f"✅ nano-SPIN Self-Play Training Completed! Logs saved to: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
