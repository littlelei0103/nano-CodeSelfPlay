"""
train_spin.py - nano-SPIN 双角色代码对抗自博弈训练主循环 (Adversarial Self-Play Loop)

核心设计与机制:
  1. 零外部新增数据 (Zero-Data RSI): 仅依赖轻量种子函数池 data/seed_functions.py。
  2. 单底座双独立 LoRA (Dual-LoRA Hot-Swapping):
     - 共享冻结基座: Qwen2.5-1.5B (bfloat16, 梯度全冻结)
     - 找茬者 (breaker): 热加载 nano-GRPO 78.33% 最佳权重 (或全新初始化)
     - 出题者 (mutator): 干净 Base 重新初始化，通过行级补丁与阶梯塑形奖励彻底消除初始 0% 合法率死锁
  3. 双独立优化器 (Dual Independent Optimizers):
     - optimizer_mutator 与 optimizer_breaker 优化器状态物理隔离，零交叉参数漂移
  4. 单 Step 实时交替训练流 + Mutator 预热轮次 (Mutator Warm-up):
     - Step 1 ~ mutator_warmup_steps: 仅更新 Mutator 策略，Breaker 仅在 torch.no_grad() 下充当裁判员，
       未破解隐蔽难题 (p_solve < 1.0) 存入 HardBuffer 对抗题库；
     - Step > mutator_warmup_steps: 正式开启双向共演 (Co-Evolution)，Phase 1 更新 Mutator，
       Phase 2 从 HardBuffer 按难度与稀疏度加权采样题目更新 Breaker。
  5. 周期性独立外部基准大考 (Held-out Benchmark):
     - 周期性评估 Breaker 在 20 道固定测试集上的 Greedy Pass@1 与 Sample Pass@4，自动保存跨代检查点。
"""

import argparse
import ast
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
from core.hard_buffer import HardBuffer
from core.sandbox import (
    evaluate_breaker_counterexample,
    extract_python_function,
    validate_mutation
)
from data.anchor_tasks import get_anchor_tasks
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


DEFAULT_INIT_ADAPTER = (
    "../nano-GRPO/checkpoints_counterexample/20261014T102327Z-d1694cb8/checkpoint-best"
    if os.path.exists("../nano-GRPO/checkpoints_counterexample/20261014T102327Z-d1694cb8/checkpoint-best")
    else ""
)


def parse_args():
    parser = argparse.ArgumentParser(description="nano-SPIN Adversarial Self-Play Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地路径")
    parser.add_argument("--init_adapter", type=str, default=DEFAULT_INIT_ADAPTER,
                        help="预训练 Breaker LoRA 适配器路径 (热启动 nano-GRPO 训练好的最佳权重)")
    parser.add_argument("--total_steps", type=int, default=60,
                        help="自博弈总训练步数 (每步包含独立的更新流)")
    parser.add_argument("--mutator_warmup_steps", type=int, default=15,
                        help="Mutator 独占预热步数 (此期间仅更新 Mutator，Breaker 冻结并建立 HardBuffer)")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="独立外部基准评估与检查点保存间隔步数")
    parser.add_argument("--rounds", type=int, default=0,
                        help="若指定 rounds > 0，则总步数自动设置为 rounds * seeds_per_round (保持与轮次模式兼容)")
    parser.add_argument("--seeds_per_round", type=int, default=8,
                        help="轮次兼容模式下每轮包含的题目步数")
    parser.add_argument("--mut_rollouts", type=int, default=4,
                        help="Mutator 每步采样的候选突变数 (G_mut)")
    parser.add_argument("--breaker_rollouts", type=int, default=4,
                        help="Breaker 每步采样的候选反例数 (G_brk)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="通用优化器学习率")
    parser.add_argument("--lr_mutator", type=float, default=2e-5,
                        help="Mutator 出题模型学习率 (默认 2e-5)")
    parser.add_argument("--lr_breaker", type=float, default=5e-6,
                        help="Breaker 做题模型学习率 (默认 5e-6，保守更新防遗忘)")
    parser.add_argument("--anchor_ratio", type=float, default=0.5,
                        help="Breaker 训练阶段抽取原始静态锚定题目的比例 (默认 0.5)")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="基座参考策略 KL 散度惩罚系数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样探索温度")
    parser.add_argument("--output_dir", type=str, default="checkpoints_spin",
                        help="检查点与日志输出根目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()



def save_spin_checkpoint(model: torch.nn.Module, tokenizer: Any, save_dir: str):
    """
    精确保存双 LoRA 适配器与分词器，确保 checkpoint-round-X/breaker 与 mutator 均完整就绪。
    """
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    # 显式确保 breaker 与 mutator 子目录均包含独立适配器权重
    for adapter_name in ["breaker", "mutator"]:
        sub_dir = os.path.join(save_dir, adapter_name)
        if not os.path.exists(os.path.join(sub_dir, "adapter_config.json")):
            try:
                model.save_pretrained(sub_dir, selected_adapters=[adapter_name])
            except Exception:
                pass


def evaluate_on_heldout(
    model: torch.nn.Module,
    tokenizer: Any,
    heldout_problems: List[Dict[str, Any]],
    device: str
) -> Tuple[float, float, float]:
    """
    在独立外部基准上评估当前策略的 Greedy Pass@1 与 Sample Pass@4。
    自动激活 breaker 适配器并评估。
    """
    model.set_adapter("breaker")
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

    # 兼容性处理: 若指定 --rounds，则步数自动换算
    if args.rounds > 0:
        total_steps = args.rounds * args.seeds_per_round
        eval_interval = args.seeds_per_round
    else:
        total_steps = args.total_steps
        eval_interval = args.eval_interval

    run_timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = os.path.join(args.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print("🚀 nano-SPIN: Code Mutation & Counterexample Adversarial Self-Play")
    print(f"Base Model: {args.model_name} | Device: {args.device}")
    print(f"Total Steps: {total_steps} | Mutator Warm-up Steps: {args.mutator_warmup_steps}")
    print(f"Eval Interval: {eval_interval} | Output Dir: {run_dir}")
    if args.init_adapter:
        print(f"Pretrained Breaker Warm-start: {args.init_adapter}")
    print("=" * 72)

    # 1. 加载模型与双独立 LoRA (breaker & mutator)
    tokenizer, model = setup_lora_and_model(
        model_name=args.model_name,
        device=args.device,
        init_adapter=args.init_adapter
    )

    # 2. 实例化双角色专属独立优化器 (Breaker 采用保守低学习率防遗忘)
    lr_mut = args.lr_mutator if args.lr_mutator is not None else args.lr
    lr_brk = args.lr_breaker if args.lr_breaker is not None else args.lr

    model.set_adapter("mutator")
    opt_mutator = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "mutator" in n and p.requires_grad],
        lr=lr_mut,
        weight_decay=0.01
    )

    model.set_adapter("breaker")
    opt_breaker = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "breaker" in n and p.requires_grad],
        lr=lr_brk,
        weight_decay=0.01
    )

    trainer = SPINGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer_mutator=opt_mutator,
        optimizer_breaker=opt_breaker,
        device=args.device,
        kl_coef=args.kl_coef
    )

    all_seeds = get_seed_functions()
    anchor_tasks = get_anchor_tasks()
    heldout_problems = get_heldout_benchmark()
    print(f"[Data Setup] Loaded {len(all_seeds)} seeds, {len(anchor_tasks)} anchor replay tasks, and {len(heldout_problems)} heldout problems.")
    print(f"[Hyperparams] LR Mutator: {lr_mut} | LR Breaker: {lr_brk} (Conservative) | Anchor Replay Ratio: {args.anchor_ratio:.0%}")


    # 3. 实例化动态对抗难题缓冲池 (HardBuffer)
    hard_buffer = HardBuffer(capacity=500)

    # 4. 初始基线评估 (Round 0)
    print("\n[Baseline Eval] Evaluating initial breaker adapter on held-out benchmark...")
    base_greedy, base_p4, base_hit = evaluate_on_heldout(model, tokenizer, heldout_problems, args.device)
    print(f"⭐ [Round 00 / Baseline] Held-out Greedy Pass@1: {base_greedy:.2%} | Sample Pass@4: {base_p4:.2%} | Hit Density: {base_hit:.2%}\n")

    best_heldout_pass1 = base_greedy
    metrics_log_path = os.path.join(run_dir, "metrics_spin.jsonl")

    # 保存初始基线作为 Round 0 检查点
    round0_dir = os.path.join(run_dir, "checkpoint-round-0")
    save_spin_checkpoint(model, tokenizer, round0_dir)

    r0_entry = {
        "step": 0,
        "round": 0,
        "is_warmup": True,
        "mut_validity_rate": 0.0,
        "mut_avg_reward": 0.0,
        "mut_grad_norm": 0.0,
        "brk_solve_rate": 0.0,
        "brk_avg_reward": 0.0,
        "brk_grad_norm": 0.0,
        "pool_size": 0,
        "heldout_greedy_pass1": base_greedy,
        "heldout_sample_pass4": base_p4,
        "heldout_hit_rate": base_hit,
        "duration_sec": 0.0
    }
    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r0_entry) + "\n")

    # 预设计算效率配置 (行级突变 32 tokens, 反例 24 tokens)
    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=32, temperature=args.temperature, do_sample=True)
    cfg_brk = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=args.temperature, do_sample=True)

    # -------------------------------------------------------------
    # 主训练循环: 单 Step 实时交替自博弈 (Step-Level Alternating Flow)
    # -------------------------------------------------------------
    for step in range(1, total_steps + 1):
        step_start_time = time.time()
        is_warmup = (step <= args.mutator_warmup_steps)
        current_round = (step - 1) // eval_interval + 1

        mode_desc = f"MUTATOR WARM-UP ({step}/{args.mutator_warmup_steps})" if is_warmup else "CO-EVOLUTION SELF-PLAY"

        # ---------------------------------------------------------
        # Phase 1: 训练出题模型 (Mutator Policy Update)
        # ---------------------------------------------------------
        model.set_adapter("mutator")
        seed = all_seeds[(step - 1) % len(all_seeds)]
        mut_prompt = format_mutator_prompt(seed)
        inputs = tokenizer(mut_prompt, return_tensors="pt").to(args.device)
        p_len = inputs["input_ids"].shape[1]
        rep_inputs = {k: v.repeat(args.mut_rollouts, 1) for k, v in inputs.items()}

        with torch.no_grad():
            out_mut = generate_with_policy(model, cfg_mut, **rep_inputs)

        candidate_rewards = []
        valid_mutations_in_step = 0

        for c_i in range(args.mut_rollouts):
            completion = tokenizer.decode(out_mut[c_i][p_len:], skip_special_tokens=True).strip()
            code_src, _, _ = extract_mutator_code(completion, f_ref_src=seed["func_src"])

            is_format = (code_src is not None) and (code_src.strip() != seed["func_src"].strip())
            is_syntax = False
            is_sanity = False
            is_diff = False
            solve_rate = 1.0
            witnesses = []
            sparsity = 0.0

            if is_format:
                try:
                    ast.parse(code_src)
                    is_syntax = True
                except Exception:
                    is_syntax = False

            if is_syntax:
                is_valid, reason, details = validate_mutation(
                    f_ref_src=seed["func_src"],
                    f_mut_src=code_src,
                    sanity_inputs=seed["sanity_inputs"],
                    domain=tuple(seed["domain"])
                )
                is_sanity = details.get("sanity_passed", False)
                is_diff = is_valid
                sparsity = details.get("sparsity", 0.0)
                witnesses = details.get("witnesses", [])

            if is_diff:
                valid_mutations_in_step += 1
                clean_ref = details.get("clean_f_ref_src", seed["func_src"])
                clean_mut = details.get("clean_f_mut_src", code_src)

                # 探针 Breaker 防守能力 (临时切 adapter 并关闭梯度)
                model.set_adapter("breaker")
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
                model.set_adapter("mutator")

                # 若难倒了当前 Breaker (p_solve < 1.0)，存入动态对抗题库 HardBuffer
                if solve_rate < 1.0:
                    hard_buffer.add(
                        f_ref_src=clean_ref,
                        f_mut_src=clean_mut,
                        domain=tuple(seed["domain"]),
                        seed_id=seed["id"],
                        sparsity=sparsity,
                        witnesses=witnesses,
                        solve_rate=solve_rate,
                        step=step
                    )

            # 阶梯塑形奖励 (Warm-up 期间密集阶梯，Co-evolution 期间高对抗消减保底)
            r_mut, _ = compute_mutator_reward(
                is_valid_format=is_format,
                is_valid_syntax=is_syntax,
                is_valid_sanity=is_sanity,
                is_non_equivalent=is_diff,
                breaker_solve_rate=solve_rate,
                is_warmup=is_warmup
            )
            candidate_rewards.append(r_mut)

        # Mutator GRPO 反向传播与优化器更新
        is_eff_mut, loss_mut, _ = trainer.backward_grpo_group(
            prompt_ids=inputs["input_ids"],
            generated_ids=out_mut,
            rewards=candidate_rewards,
            temperature=args.temperature,
            num_groups_in_update=1
        )
        grad_norm_mut = trainer.step_optimizer(role="mutator")
        mut_vmr = valid_mutations_in_step / max(1, args.mut_rollouts)
        avg_mut_reward = sum(candidate_rewards) / max(1, len(candidate_rewards))

        # ---------------------------------------------------------
        # Phase 2: 训练做题模型 (Breaker Policy Update + 50% Anchor Replay)
        # ---------------------------------------------------------
        brk_win_rate = 0.0
        avg_brk_reward = 0.0
        grad_norm_brk = 0.0

        if is_warmup:
            # 预热期: Breaker 保持完全冻结，不计算梯度与更新
            brk_desc = "FROZEN (Warm-up)"
        else:
            # 共演期: 50% 概率从 HardBuffer 抽取高难对抗题 + 50% 概率从 Anchor Tasks 回放锚定题 (防分布偏移)
            model.set_adapter("breaker")
            use_anchor = (random.random() < args.anchor_ratio) or (len(hard_buffer) == 0)
            if not use_anchor and len(hard_buffer) > 0:
                sampled_pairs = hard_buffer.sample(k=1)
                prob = sampled_pairs[0]
                source_tag = "HardBuffer"
            else:
                prob = random.choice(anchor_tasks)
                source_tag = "Anchor"

            f_ref = prob["f_ref_src"]
            f_mut = prob["f_mut_src"]
            domain = tuple(prob["domain"])

            brk_prompt = format_breaker_prompt(f_ref, f_mut, domain, prob.get("spec", ""))
            brk_inputs = tokenizer(brk_prompt, return_tensors="pt").to(args.device)
            brk_p_len = brk_inputs["input_ids"].shape[1]
            brk_rep = {k: v.repeat(args.breaker_rollouts, 1) for k, v in brk_inputs.items()}

            with torch.no_grad():
                out_brk = generate_with_policy(model, cfg_brk, **brk_rep)

            brk_candidate_rewards = []
            solved_count = 0
            for c_j in range(args.breaker_rollouts):
                comp = tokenizer.decode(out_brk[c_j][brk_p_len:], skip_special_tokens=True).strip()
                val, _, _ = extract_breaker_input("<input>" + comp, domain)
                r_brk, _ = compute_breaker_reward(f_ref, f_mut, val, domain)
                brk_candidate_rewards.append(r_brk)
                if r_brk > 0.0:
                    solved_count += 1

            # 仅对 HardBuffer 中的自博弈题目更新统计
            if source_tag == "HardBuffer" and "key" in prob:
                hard_buffer.update_result(prob["key"], solved=(solved_count > 0))

            # Breaker GRPO 反向传播与优化器更新
            is_eff_brk, loss_brk, _ = trainer.backward_grpo_group(
                prompt_ids=brk_inputs["input_ids"],
                generated_ids=out_brk,
                rewards=brk_candidate_rewards,
                temperature=args.temperature,
                num_groups_in_update=1
            )
            grad_norm_brk = trainer.step_optimizer(role="breaker")
            brk_win_rate = solved_count / max(1, args.breaker_rollouts)
            avg_brk_reward = sum(brk_candidate_rewards) / max(1, len(brk_candidate_rewards))
            brk_desc = f"[{source_tag}] Solve: {brk_win_rate:.0%} | R: {avg_brk_reward:.2f} | Grad: {grad_norm_brk:.3f}"


        step_duration = time.time() - step_start_time
        print(f"[{step:03d}/{total_steps:03d} | {mode_desc}] Mutator VMR: {mut_vmr:.0%} | Mut R: {avg_mut_reward:.2f} | Breaker: {brk_desc} | HardBuffer: {len(hard_buffer)} ({step_duration:.1f}s)")

        # ---------------------------------------------------------
        # Phase 3: 周期性独立外部大考与跨代检查点保存
        # ---------------------------------------------------------
        is_eval_step = (step % eval_interval == 0) or (step == total_steps)
        if is_eval_step:
            round_idx = max(1, step // eval_interval)
            print(f"\n{'='*72}")
            print(f"📊 【Auditing Held-out Benchmark @ Step {step:03d} / Round {round_idx:02d}】")
            print(f"{'='*72}")

            ho_greedy, ho_p4, ho_hit = evaluate_on_heldout(model, tokenizer, heldout_problems, args.device)
            hb_stats = hard_buffer.get_stats()

            print(f"   * Held-out Greedy Pass@1: {ho_greedy:.2%} (vs Base: {base_greedy:.2%})")
            print(f"   * Held-out Sample Pass@4: {ho_p4:.2%} (vs Base: {base_p4:.2%})")
            print(f"   * Candidate Hit Density: {ho_hit:.2%} (vs Base: {base_hit:.2%})")
            print(f"   * HardBuffer Pool Size: {hb_stats['total_problems']} (Avg Solve Rate: {hb_stats['avg_solve_rate']:.1%})")

            # 记录指标日志
            log_entry = {
                "step": step,
                "round": round_idx,
                "is_warmup": is_warmup,
                "mut_validity_rate": mut_vmr,
                "mut_avg_reward": avg_mut_reward,
                "mut_grad_norm": grad_norm_mut,
                "brk_solve_rate": brk_win_rate,
                "brk_avg_reward": avg_brk_reward,
                "brk_grad_norm": grad_norm_brk,
                "pool_size": len(hard_buffer),
                "heldout_greedy_pass1": ho_greedy,
                "heldout_sample_pass4": ho_p4,
                "heldout_hit_rate": ho_hit,
                "duration_sec": step_duration
            }
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # 保存轮次检查点 (支持跨代矩阵评测)
            round_ckpt_dir = os.path.join(run_dir, f"checkpoint-round-{round_idx}")
            save_spin_checkpoint(model, tokenizer, round_ckpt_dir)
            print(f"   * Checkpoint saved to: {round_ckpt_dir}")

            # 保存 last 检查点
            last_ckpt_dir = os.path.join(run_dir, "checkpoint-last")
            save_spin_checkpoint(model, tokenizer, last_ckpt_dir)

            # 保存 hard_buffer
            hard_buffer.save_jsonl(os.path.join(run_dir, "hard_buffer.jsonl"))

            # 刷新最佳检查点
            if ho_greedy >= best_heldout_pass1:
                best_heldout_pass1 = ho_greedy
                best_ckpt_dir = os.path.join(run_dir, "checkpoint-best")
                save_spin_checkpoint(model, tokenizer, best_ckpt_dir)
                print(f"   🏆 New Best Held-out Greedy Pass@1: {best_heldout_pass1:.2%} -> Saved to {best_ckpt_dir}")

            print(f"{'='*72}\n")

    print("\n" + "=" * 72)
    print(f"✅ nano-SPIN Self-Play Training Completed! Logs and checkpoints saved to: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
