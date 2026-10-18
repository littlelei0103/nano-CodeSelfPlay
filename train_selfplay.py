"""Dual-LoRA adversarial self-play with validation selection and versioned state."""

import argparse
import ast
import json
import math
import os
import random
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from core.grpo_engine import (
    SelfPlayGRPOTrainer,
    generate_with_policy,
    rollout_generation_config,
    setup_lora_and_model
)
from core.hard_buffer import HardBuffer
from core.run_files import training_metrics_path
from core.experiment import (
    PROTOCOL_VERSION, discover_checkpoint, experiment_manifest, read_training_state,
    restore_training_state, select_validation_best, validate_resume,
)
from core.sandbox import (
    evaluate_breaker_counterexample,
    extract_python_function,
    validate_mutation
)
from data.anchor_tasks import get_anchor_tasks
from data.heldout_eval import get_heldout_benchmark, get_validation_benchmark, get_test_benchmark
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


DEFAULT_INIT_ADAPTER = ""


def parse_args():
    parser = argparse.ArgumentParser(description="nano-CodeSelfPlay Adversarial Self-Play Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型名称或本地路径")
    parser.add_argument("--init_adapter", type=str, default="",
                        help="可选 Breaker 权重热启动路径（新实验，不恢复优化器）")
    parser.add_argument("--total_steps", type=int, default=75,
                        help="自博弈总训练步数 (默认 75 步，划分为 5 轮完整自博弈)")
    parser.add_argument("--mutator_warmup_steps", type=int, default=5,
                        help="Mutator 独占预热步数 (此期间仅更新 Mutator，Breaker 冻结并建立 HardBuffer)")
    parser.add_argument("--eval_interval", type=int, default=15,
                        help="验证集评估与检查点保存间隔（默认 15 步）")
    parser.add_argument("--rounds", type=int, default=0,
                        help="若指定 rounds > 0，则总步数自动设置为 rounds * seeds_per_round (保持与轮次模式兼容)")
    parser.add_argument("--seeds_per_round", type=int, default=15,
                        help="轮次兼容模式下每轮包含的题目步数")
    parser.add_argument("--mut_rollouts", type=int, default=4,
                        help="Mutator 每步采样的候选突变数 (G_mut)")
    parser.add_argument("--breaker_rollouts", type=int, default=8,
                        help="Breaker 每步采样的候选反例数（默认 8）")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="通用优化器学习率")
    parser.add_argument("--lr_mutator", type=float, default=2e-5,
                        help="Mutator 出题模型学习率 (默认 2e-5)")
    parser.add_argument("--lr_breaker", type=float, default=2e-5,
                        help="Breaker 优化器学习率（默认 2e-5）")
    parser.add_argument("--lr_breaker_min", type=float, default=5e-6,
                        help="Breaker 后期余弦衰减的最低学习率 (默认 5e-6)")
    parser.add_argument("--cosine_decay_start", type=int, default=25,
                        help="Breaker 开始余弦学习率衰减的步数 (默认 25 步)")
    parser.add_argument("--curriculum_tighten_step", type=int, default=30,
                        help="甜蜜点课程收紧步数 (默认 30 步，即从 Round 3 开始收紧黄金带至 0.35)")
    parser.add_argument("--anchor_ratio", type=float, default=0.0,
                        help="静态锚定题回放比例；默认 0.0，不载入外部 Bug 题")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="基座参考策略 KL 散度惩罚系数 (对齐 nano-GRPO 0.01)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样探索温度")
    parser.add_argument("--output_dir", type=str, default="checkpoints_selfplay",
                        help="检查点与日志输出根目录")
    parser.add_argument("--resume_dir", type=str, default="",
                        help="续跑检查点目录 (若指定，则从中恢复最新的 round 检查点与 HardBuffer 状态，并继续向前训练)")
    parser.add_argument("--freeze_mutator_after_warmup", action="store_true", default=False,
                        help="预热步数结束后完全冻结 Mutator 权重更新，仅作为固定出题策略（对照组消融实验）")
    parser.add_argument("--breaker_only_static", action="store_true", default=False,
                        help="仅训练找茬者 Breaker（不运行 Mutator，纯静态题库 GRPO 基线）")
    parser.add_argument("--seed", type=int, default=42, help="Python / PyTorch random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    return parser.parse_args()



def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    save_dir: str,
    optimizer_mutator: Optional[torch.optim.Optimizer] = None,
    optimizer_breaker: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    round_idx: int = 0,
    best_val_pass1: float = 0.0,
    extra_state: Optional[Dict[str, Any]] = None
):
    """
    精确保存双 LoRA 适配器、分词器与完整训练状态 (优化器状态、RNG 种子、步数调度)，
    确保断点续训 (Resume) 达到精确状态恢复。
    """
    parent = os.path.abspath(os.path.dirname(save_dir) or ".")
    os.makedirs(parent, exist_ok=True)
    staged = tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=parent)
    backup = save_dir + ".previous"
    try:
        tokenizer.save_pretrained(staged)
        # PEFT writes non-default named adapters to <directory>/<adapter_name>.
        model.save_pretrained(staged)
        if hasattr(model, "peft_config"):
            for role in ("breaker", "mutator"):
                if not os.path.isfile(os.path.join(staged, role, "adapter_config.json")):
                    raise RuntimeError(f"Missing saved {role} adapter")
        train_state = {
            "step": step, "round": round_idx, "best_val_pass1": best_val_pass1,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
        }
        if optimizer_mutator is not None:
            train_state["optimizer_mutator"] = optimizer_mutator.state_dict()
        if optimizer_breaker is not None:
            train_state["optimizer_breaker"] = optimizer_breaker.state_dict()
        if extra_state:
            train_state.update(extra_state)
        torch.save(train_state, os.path.join(staged, "train_state.pt"))
        if extra_state and "manifest" in extra_state:
            with open(os.path.join(staged, "experiment.json"), "w", encoding="utf-8") as f:
                json.dump(extra_state["manifest"], f, ensure_ascii=False, indent=2)
        if os.path.exists(backup):
            shutil.rmtree(backup)
        if os.path.exists(save_dir):
            os.replace(save_dir, backup)
        try:
            os.replace(staged, save_dir)
        except BaseException:
            if os.path.exists(backup):
                os.replace(backup, save_dir)
            raise
        if os.path.exists(backup):
            shutil.rmtree(backup)
    finally:
        if os.path.exists(staged):
            shutil.rmtree(staged)


def evaluate_on_heldout(
    model: torch.nn.Module,
    tokenizer: Any,
    heldout_problems: List[Dict[str, Any]],
    device: str,
    adapter_name: str = "breaker",
) -> Tuple[float, float, float]:
    """
    在独立外部基准上评估当前策略的 Greedy Pass@1 与 Sample Pass@4。
    自动激活 breaker 适配器并评估。
    """
    model.set_adapter(adapter_name)
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

    if total_steps < 0 or eval_interval <= 0 or min(args.mut_rollouts, args.breaker_rollouts) < 2:
        raise ValueError("Require nonnegative steps, positive eval interval, and rollout groups >= 2")
    if not 0 <= args.anchor_ratio <= 1 or args.temperature <= 0:
        raise ValueError("Require anchor_ratio in [0,1] and positive temperature")
    if args.breaker_only_static and args.freeze_mutator_after_warmup:
        raise ValueError("Static-only and frozen-Mutator modes are mutually exclusive")
    if args.resume_dir and args.init_adapter:
        raise ValueError("Choose exact resume or weights-only initialization, not both")
    if args.mutator_warmup_steps < 0:
        raise ValueError("mutator_warmup_steps must be nonnegative")
    if args.init_adapter and not os.path.isdir(args.init_adapter):
        raise FileNotFoundError(args.init_adapter)
    if args.breaker_only_static:
        args.mutator_warmup_steps = 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = experiment_manifest(args, eval_interval)
    resume_state = None
    if args.resume_dir:
        run_dir = args.resume_dir
        resume_ckpt, resume_state = discover_checkpoint(run_dir)
        validate_resume(resume_state, manifest)
        if total_steps < resume_state["step"]:
            raise ValueError("total_steps precedes the saved training step")
        manifest = resume_state["manifest"]
        print(f"[Resume] Loading complete state at step {resume_state['step']}: {resume_ckpt}")
    else:
        run_timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = os.path.join(args.output_dir, f"{run_timestamp}-seed{args.seed}")
        os.makedirs(run_dir, exist_ok=False)
        resume_ckpt = ""
    with open(os.path.join(run_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("🚀 nano-CodeSelfPlay: Code Mutation & Counterexample Adversarial Self-Play")
    print(f"Base Model: {args.model_name} | Device: {args.device}")
    print(f"Total Steps: {total_steps} | Mutator Warm-up Steps: {args.mutator_warmup_steps}")
    print(f"Eval Interval: {eval_interval} | Output Dir: {run_dir}")
    if args.init_adapter:
        print(f"Pretrained Breaker Warm-start: {args.init_adapter}")
    if resume_ckpt:
        print(f"Resumed Adapters: {resume_ckpt}")
    print("=" * 72)

    # 1. 加载模型与双独立 LoRA (breaker & mutator)
    tokenizer, model = setup_lora_and_model(
        model_name=args.model_name,
        device=args.device,
        init_adapter=args.init_adapter,
        resume_checkpoint=resume_ckpt
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

    trainer = SelfPlayGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer_mutator=opt_mutator,
        optimizer_breaker=opt_breaker,
        device=args.device,
        kl_coef=args.kl_coef
    )

    all_seeds = get_seed_functions()
    if args.breaker_only_static:
        anchor_tasks = get_anchor_tasks()
        rng_data = random.Random(args.seed)
        shuffled_anchors = list(anchor_tasks)
        rng_data.shuffle(shuffled_anchors)
        print(f"[Data Setup] Breaker-Only Static Mode: Loaded and permuted {len(anchor_tasks)} static anchor tasks.")
    elif args.anchor_ratio > 0.0:
        anchor_tasks = get_anchor_tasks()
        shuffled_anchors = anchor_tasks
        print(f"[Data Setup] Loaded {len(all_seeds)} seeds, {len(anchor_tasks)} anchor replay tasks (Ratio: {args.anchor_ratio:.0%}).")
    else:
        anchor_tasks = []
        shuffled_anchors = []
        print(f"[Data Setup] Self-play mode (anchor_ratio={args.anchor_ratio}): no static bug-pair tasks loaded.")

    # 训练监控与最佳模型选择专用验证集 (Validation Set, 20题)
    val_problems = get_validation_benchmark()
    print(f"[Data Setup] Loaded {len(val_problems)} validation audit problems (strictly isolated from final heldout test set).")
    print(f"[Hyperparams] LR Mutator: {lr_mut} | LR Breaker: {lr_brk} | Cosine Decay Start: {args.cosine_decay_start}")

    hard_buffer = HardBuffer(capacity=500)
    metrics_log_path = training_metrics_path(run_dir)
    if resume_state is not None:
        restore_training_state(resume_state, opt_mutator, opt_breaker)
        hard_buffer.items = resume_state["hard_buffer_items"]
        start_step = resume_state["step"] + 1
        best_val_pass1 = resume_state["best_val_pass1"]
        best_step = resume_state["best_step"]
        baseline = resume_state["baseline"]
        schedule_total_steps = resume_state["schedule_total_steps"]
        best_state = read_training_state(os.path.join(run_dir, "checkpoint-best"))
        if best_state["step"] != best_step or best_state["best_val_pass1"] != best_val_pass1:
            raise ValueError("checkpoint-best disagrees with the resumed state; cannot safely resume")
        # Logs can be ahead of a checkpoint after interruption; checkpoint state is authoritative.
        records = []
        if os.path.isfile(metrics_log_path):
            with open(metrics_log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        records = [r for r in records if r["step"] <= resume_state["step"]]
        with open(metrics_log_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    else:
        start_step = 1
        best_step = 0
        schedule_total_steps = total_steps
        print("\n[Baseline Eval] Evaluating initial breaker on validation problems...")
        base_greedy, base_p4, base_hit = evaluate_on_heldout(model, tokenizer, val_problems, args.device)
        baseline = {"greedy_pass1": base_greedy, "sample_pass4": base_p4, "hit_rate": base_hit}
        best_val_pass1 = base_greedy

    # Cumulative telemetry is checkpoint state, just like optimizer/RNG state.
    saved_telemetry = resume_state.get("training_telemetry") if resume_state else None
    def optimizer_steps(optimizer):
        counts = {int(v["step"]) for v in optimizer.state.values() if "step" in v}
        if len(counts) > 1:
            raise ValueError("Nonuniform optimizer step counters; cannot infer update count")
        return next(iter(counts), 0)
    if resume_state is None:
        cum_breaker_updates = cum_mutator_updates = 0
        cum_breaker_candidates = cum_breaker_probe_candidates = cum_mutator_candidates = 0
        prior_walltime = 0.0
    elif saved_telemetry is not None:
        cum_breaker_updates = saved_telemetry["actual_breaker_updates"]
        cum_mutator_updates = saved_telemetry["actual_mutator_updates"]
        cum_breaker_candidates = saved_telemetry["breaker_training_candidates"]
        cum_breaker_probe_candidates = saved_telemetry["breaker_probe_candidates"]
        cum_mutator_candidates = saved_telemetry["total_mutator_generations"]
        prior_walltime = saved_telemetry["total_training_walltime_sec"]
    else:
        # Older states cannot recover all discarded probes/rescues or elapsed time.
        cum_breaker_updates = optimizer_steps(opt_breaker)
        cum_mutator_updates = optimizer_steps(opt_mutator)
        cum_breaker_candidates = cum_breaker_probe_candidates = cum_mutator_candidates = None
        prior_walltime = None
    train_start_walltime = None

    def telemetry_snapshot():
        elapsed = 0.0 if train_start_walltime is None else time.time() - train_start_walltime
        total = (None if cum_breaker_candidates is None or cum_breaker_probe_candidates is None
                 else cum_breaker_candidates + cum_breaker_probe_candidates)
        return {"schema_version": 1, "actual_breaker_updates": cum_breaker_updates,
                "actual_mutator_updates": cum_mutator_updates,
                "breaker_training_candidates": cum_breaker_candidates,
                "breaker_probe_candidates": cum_breaker_probe_candidates,
                "total_breaker_generations": total,
                "total_mutator_generations": cum_mutator_candidates,
                "total_training_walltime_sec": None if prior_walltime is None else prior_walltime + elapsed,
                "scope": "Training rollouts including probes/rescue; wall time includes periodic validation and saves, excludes initial/final audits and model loading."}

    def checkpoint_extra():
        return {"manifest": manifest, "best_step": best_step, "baseline": baseline,
                "schedule_total_steps": schedule_total_steps, "hard_buffer_items": hard_buffer.items,
                "training_telemetry": telemetry_snapshot()}

    def save_snapshot(name, step, round_idx):
        save_checkpoint(model, tokenizer, os.path.join(run_dir, name), opt_mutator, opt_breaker,
                             step, round_idx, best_val_pass1, checkpoint_extra())

    base_greedy, base_p4, base_hit = (baseline[k] for k in ("greedy_pass1", "sample_pass4", "hit_rate"))
    if resume_state is None:
        # The untrained baseline is a real candidate, even if every update makes validation worse.
        for name in ("checkpoint-round-0", "checkpoint-best", "checkpoint-last"):
            save_snapshot(name, 0, 0)
        with open(metrics_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"protocol_version": PROTOCOL_VERSION, "step": 0, "round": 0,
                                "val_greedy_pass1": base_greedy, "val_sample_pass4": base_p4,
                                "val_hit_rate": base_hit, "pool_size": 0}) + "\n")

    # 预设计算效率配置 (行级突变 32 tokens, 反例 24 tokens)
    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=32, temperature=args.temperature, do_sample=True)
    cfg_brk = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=args.temperature, do_sample=True)

    train_start_walltime = time.time()

    # -------------------------------------------------------------
    # 主训练循环: 单 Step 实时交替自博弈 (Step-Level Alternating Flow)
    # -------------------------------------------------------------
    for step in range(start_step, total_steps + 1):
        step_start_time = time.time()
        is_warmup = (step <= args.mutator_warmup_steps) if not args.breaker_only_static else False
        current_round = (step - 1) // eval_interval + 1

        is_tightened = (step > args.curriculum_tighten_step)

        # 针对 Breaker 的余弦学习率衰减 (Cosine Annealing)
        if step <= args.cosine_decay_start:
            current_lr_brk = lr_brk
        else:
            progress = min(1.0, (step - args.cosine_decay_start) / max(1, schedule_total_steps - args.cosine_decay_start))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr_brk = args.lr_breaker_min + (lr_brk - args.lr_breaker_min) * cosine_factor

        for param_group in opt_breaker.param_groups:
            param_group["lr"] = current_lr_brk

        tight_tag = " [TIGHT]" if is_tightened else ""
        if is_warmup:
            mode_desc = f"MUTATOR WARM-UP ({step}/{args.mutator_warmup_steps})"
        elif args.breaker_only_static:
            mode_desc = f"BREAKER-ONLY STATIC GRPO"
        elif args.freeze_mutator_after_warmup:
            mode_desc = f"FROZEN-MUTATOR ABLATION{tight_tag}"
        else:
            mode_desc = f"CO-EVOLUTION SELF-PLAY{tight_tag}"

        # ---------------------------------------------------------
        # Phase 1: 训练出题模型 (Mutator Policy Update)
        # ---------------------------------------------------------
        if args.breaker_only_static:
            mut_vmr = 0.0
            avg_mut_reward = 0.0
            grad_norm_mut = 0.0
            is_eff_mut = False
        else:
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
                    cum_breaker_probe_candidates = (None if cum_breaker_probe_candidates is None
                                                    else cum_breaker_probe_candidates + 4)

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
                            candidate_hits=brk_hits,
                            total_candidates=4,
                            step=step
                        )

                # 阶梯塑形奖励 (支持前后期渐进式甜蜜点课程收紧)
                r_mut, _ = compute_mutator_reward(
                    is_valid_format=is_format,
                    is_valid_syntax=is_syntax,
                    is_valid_sanity=is_sanity,
                    is_non_equivalent=is_diff,
                    breaker_solve_rate=solve_rate,
                    is_warmup=is_warmup,
                    is_tightened=is_tightened
                )
                candidate_rewards.append(r_mut)

            # Mutator GRPO 反向传播与优化器更新
            if args.freeze_mutator_after_warmup and not is_warmup:
                is_eff_mut, loss_mut = False, 0.0
                grad_norm_mut = 0.0
            else:
                is_eff_mut, loss_mut, _ = trainer.backward_grpo_group(
                    prompt_ids=inputs["input_ids"],
                    generated_ids=out_mut,
                    rewards=candidate_rewards,
                    temperature=args.temperature,
                    num_groups_in_update=1
                )
                if is_eff_mut:
                    grad_norm_mut = trainer.step_optimizer(role="mutator")
                    cum_mutator_updates += 1
                else:
                    grad_norm_mut = 0.0
            cum_mutator_candidates = None if cum_mutator_candidates is None else cum_mutator_candidates + args.mut_rollouts
            mut_vmr = valid_mutations_in_step / max(1, args.mut_rollouts)
            avg_mut_reward = sum(candidate_rewards) / max(1, len(candidate_rewards))

        # ---------------------------------------------------------
        # Phase 2: Breaker update with optional anchor replay
        # ---------------------------------------------------------
        brk_win_rate = 0.0
        brk_evaluated = False
        brk_attempts = 0
        brk_total_hits = 0
        source_tag = "Skipped"
        avg_brk_reward = 0.0
        grad_norm_brk = 0.0

        if is_warmup:
            # 预热期: Breaker 保持完全冻结，不计算梯度与更新
            brk_desc = "FROZEN (Warm-up)"
        else:
            # Optional anchors are loaded only when anchor_ratio > 0.
            model.set_adapter("breaker")
            if args.breaker_only_static:
                prob = shuffled_anchors[(step - 1) % len(shuffled_anchors)]
                source_tag = "StaticAnchor"
            elif args.anchor_ratio > 0.0 and len(anchor_tasks) > 0:
                use_anchor = (random.random() < args.anchor_ratio) or (len(hard_buffer) == 0)
                if use_anchor:
                    prob = random.choice(anchor_tasks)
                    source_tag = "Anchor"
                elif len(hard_buffer) > 0:
                    prob = hard_buffer.sample(k=1, is_tightened=is_tightened)[0]
                    source_tag = "HardBuffer"
                else:
                    prob = None
                    source_tag = "EmptyBuffer"
            elif len(hard_buffer) > 0:
                prob = hard_buffer.sample(k=1, is_tightened=is_tightened)[0]
                source_tag = "HardBuffer"
            else:
                prob = None
                source_tag = "EmptyBuffer"

            if prob is not None:
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

                brk_evaluated = True
                brk_attempts += args.breaker_rollouts
                brk_total_hits += solved_count
                cum_breaker_candidates = None if cum_breaker_candidates is None else cum_breaker_candidates + args.breaker_rollouts
                # 仅对 HardBuffer 中的自博弈题目更新统计 (严格按 candidate_hits 累加)
                if source_tag == "HardBuffer" and "key" in prob:
                    hard_buffer.update_result(
                        prob["key"],
                        candidate_hits=solved_count,
                        total_candidates=args.breaker_rollouts,
                        solved=(solved_count > 0)
                    )

                # Breaker GRPO 反向传播与优化器更新
                is_eff_brk, loss_brk, _ = trainer.backward_grpo_group(
                    prompt_ids=brk_inputs["input_ids"],
                    generated_ids=out_brk,
                    rewards=brk_candidate_rewards,
                    temperature=args.temperature,
                    num_groups_in_update=1
                )

                # 动态零方差救援 (Dynamic Variance Rescue):
                # 若 HardBuffer 题目因过难导致候选反例全错 (方差为 0)，尝试通过备用题提供有效梯度
                if (not is_eff_brk) and (source_tag == "HardBuffer"):
                    rescue_prob = None
                    rescue_tag = ""
                    if args.anchor_ratio > 0.0 and len(anchor_tasks) > 0:
                        rescue_prob = random.choice(anchor_tasks)
                        rescue_tag = "Rescue[Anchor]"
                    elif len(hard_buffer) > 1:
                        # 纯自博弈模式: 仅从自博弈 HardBuffer 中抽取已知有解的题目进行方差补充，绝不碰外部人工题
                        easier_tasks = [it for it in hard_buffer.items.values() if it.get("solve_rate", 0) > 0]
                        if easier_tasks:
                            rescue_prob = random.choice(easier_tasks)
                            rescue_tag = "Rescue[Buffer]"

                    if rescue_prob is not None:
                        r_ref = rescue_prob["f_ref_src"]
                        r_mut = rescue_prob["f_mut_src"]
                        r_dom = tuple(rescue_prob["domain"])
                        r_prompt = format_breaker_prompt(r_ref, r_mut, r_dom, rescue_prob.get("spec", ""))
                        r_inputs = tokenizer(r_prompt, return_tensors="pt").to(args.device)
                        r_p_len = r_inputs["input_ids"].shape[1]
                        r_rep = {k: v.repeat(args.breaker_rollouts, 1) for k, v in r_inputs.items()}

                        with torch.no_grad():
                            out_rescue = generate_with_policy(model, cfg_brk, **r_rep)

                        rescue_rewards = []
                        rescue_hits = 0
                        for c_k in range(args.breaker_rollouts):
                            r_comp = tokenizer.decode(out_rescue[c_k][r_p_len:], skip_special_tokens=True).strip()
                            r_val, _, _ = extract_breaker_input("<input>" + r_comp, r_dom)
                            r_reward, _ = compute_breaker_reward(r_ref, r_mut, r_val, r_dom)
                            rescue_rewards.append(r_reward)
                            if r_reward > 0.0:
                                rescue_hits += 1

                        brk_attempts += args.breaker_rollouts
                        brk_total_hits += rescue_hits
                        cum_breaker_candidates = None if cum_breaker_candidates is None else cum_breaker_candidates + args.breaker_rollouts
                        if rescue_tag == "Rescue[Buffer]":
                            rescue_prob["times_sampled"] += 1
                            hard_buffer.update_result(rescue_prob["key"], candidate_hits=rescue_hits,
                                                      total_candidates=args.breaker_rollouts)
                        is_rescued, _, _ = trainer.backward_grpo_group(
                            prompt_ids=r_inputs["input_ids"],
                            generated_ids=out_rescue,
                            rewards=rescue_rewards,
                            temperature=args.temperature,
                            num_groups_in_update=1
                        )
                        if is_rescued:
                            is_eff_brk = True
                            source_tag = rescue_tag
                            brk_candidate_rewards = rescue_rewards
                            solved_count = rescue_hits

                if is_eff_brk:
                    grad_norm_brk = trainer.step_optimizer(role="breaker")
                    cum_breaker_updates += 1
                else:
                    trainer.opt_breaker.zero_grad()

                brk_win_rate = solved_count / max(1, args.breaker_rollouts)
                avg_brk_reward = sum(brk_candidate_rewards) / max(1, len(brk_candidate_rewards))
                brk_desc = f"[{source_tag}] Solve: {brk_win_rate:.0%} | R: {avg_brk_reward:.2f} | Grad: {grad_norm_brk:.3f}"
            else:
                brk_desc = "SKIPPED (EmptyBuffer)"


        step_duration = time.time() - step_start_time
        lr_tag = f" | LR_b: {current_lr_brk:.1e}" if current_lr_brk != lr_brk else ""
        print(f"[{step:03d}/{total_steps:03d} | {mode_desc}] Mutator VMR: {mut_vmr:.0%} | Mut R: {avg_mut_reward:.2f} | Breaker: {brk_desc}{lr_tag} | HardBuffer: {len(hard_buffer)} ({step_duration:.1f}s)")

        # ---------------------------------------------------------
        # Phase 3: 周期性独立外部大考与跨代检查点保存
        # ---------------------------------------------------------
        is_eval_step = (step % eval_interval == 0) or (step == total_steps)
        if is_eval_step:
            round_idx = max(1, math.ceil(step / eval_interval))
            print(f"\n{'='*72}")
            print(f"📊 【Auditing Validation Benchmark @ Step {step:03d} / Round {round_idx:02d}】")
            print(f"{'='*72}")

            val_greedy, val_p4, val_hit = evaluate_on_heldout(model, tokenizer, val_problems, args.device)
            hb_stats = hard_buffer.get_stats()

            print(f"   * Validation Greedy Pass@1: {val_greedy:.2%} (vs Initial: {base_greedy:.2%})")
            print(f"   * Validation Sample Pass@4: {val_p4:.2%} (vs Initial: {base_p4:.2%})")
            print(f"   * Candidate Hit Density: {val_hit:.2%} (vs Initial: {base_hit:.2%})")
            print(f"   * HardBuffer Pool Size: {hb_stats['total_problems']} (Avg Solve Rate: {hb_stats['avg_solve_rate']:.1%})")

            # Validation telemetry; final test scores live in a separate report.
            log_entry = {
                "step": step,
                "round": round_idx,
                "is_warmup": is_warmup,
                "mut_validity_rate": mut_vmr,
                "mut_avg_reward": avg_mut_reward,
                "mut_grad_norm": grad_norm_mut,
                "brk_solve_rate": brk_win_rate if brk_evaluated else None,
                "brk_source": source_tag,
                "brk_total_candidates": brk_attempts,
                "brk_total_hits": brk_total_hits,
                "brk_overall_hit_rate": brk_total_hits / brk_attempts if brk_attempts else None,
                "brk_effective_update": bool(is_eff_brk) if brk_evaluated else False,
                "brk_avg_reward": avg_brk_reward,
                "brk_grad_norm": grad_norm_brk,
                "cum_breaker_updates": cum_breaker_updates,
                "cum_breaker_candidates": cum_breaker_candidates,
                "cum_breaker_probe_candidates": cum_breaker_probe_candidates,
                "cum_mutator_updates": cum_mutator_updates,
                "cum_mutator_candidates": cum_mutator_candidates,
                "pool_size": len(hard_buffer),
                "val_greedy_pass1": val_greedy,
                "val_sample_pass4": val_p4,
                "val_hit_rate": val_hit,
                "protocol_version": PROTOCOL_VERSION,
                "duration_sec": step_duration
            }
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            best_val_pass1, best_step, improved = select_validation_best(
                val_greedy, step, best_val_pass1, best_step)
            # All snapshots contain the updated best metadata and the same buffer state.
            if improved:
                save_snapshot("checkpoint-best", step, round_idx)
                print(f"   New validation best: {best_val_pass1:.2%} at step {best_step}")
            save_snapshot(f"checkpoint-round-{round_idx}", step, round_idx)
            save_snapshot("checkpoint-last", step, round_idx)
            hard_buffer.save_jsonl(os.path.join(run_dir, "hard_buffer.jsonl"))

            print(f"{'='*72}\n")

    # Final test is a post-selection audit. Loading failures must be visible.
    final_telemetry = telemetry_snapshot()
    selected_dir = os.path.join(run_dir, "checkpoint-best")
    selected_state = read_training_state(selected_dir)
    if selected_state["step"] != best_step or selected_state["best_val_pass1"] != best_val_pass1:
        raise ValueError("Selected adapter metadata disagrees with validation selection")
    model.load_adapter(os.path.join(selected_dir, "breaker"), adapter_name="breaker_final_audit")
    model.set_adapter("breaker_final_audit")
    test_problems = get_test_benchmark()
    test_greedy, test_p4, test_hit = evaluate_on_heldout(
        model, tokenizer, test_problems, args.device, adapter_name="breaker_final_audit")
    report = {"protocol_version": PROTOCOL_VERSION, "manifest": manifest,
              "selection_split": "validation", "selected_step": best_step,
              "selected_checkpoint": selected_dir, "best_val_greedy_pass1": best_val_pass1,
              "test_size": len(test_problems), "test_greedy_pass1": test_greedy,
              "test_sample_pass4": test_p4, "test_hit_rate": test_hit,
              "test_decoding": {"temperature": 0.7, "top_p": 1.0, "top_k": 0,
                                "max_new_tokens": 24, "sample_candidates": 4},
              "training_telemetry": final_telemetry}
    report_path = os.path.join(run_dir, f"final_test_step_{total_steps}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Final test: greedy={test_greedy:.2%}, sampled Pass@4={test_p4:.2%}, hit rate={test_hit:.2%}")
    print(f"Training telemetry: {json.dumps(final_telemetry)}")
    print(f"Selected validation step {best_step}. Audit saved to {report_path}")


if __name__ == "__main__":
    main()
