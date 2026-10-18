"""Post-training benchmark and cross-play audit. Cross-play uses training seed functions."""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.experiment import PROTOCOL_VERSION, finite_json, source_fingerprints
from core.run_files import training_metrics_path
from core.grpo_engine import generate_with_policy, rollout_generation_config
from core.sandbox import evaluate_breaker_counterexample, validate_mutation
from data.heldout_eval import get_heldout_benchmark, get_test_benchmark, get_validation_benchmark
from data.seed_functions import get_seed_functions
from roles.breaker import compute_breaker_reward, extract_breaker_input, format_breaker_prompt
from roles.mutator import compute_mutator_reward, extract_mutator_code, format_mutator_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="nano-CodeSelfPlay 4-Dimensional & Cross-Play Evaluation")
    parser.add_argument("--run_dir", type=str, default="",
                        help="自博弈检查点目录（默认自动检索 checkpoints*/ 下的运行目录）")
    parser.add_argument("--output_dir", type=str, default="eval_outputs",
                        help="图表与评估产物输出目录")
    parser.add_argument("--split", type=str, choices=["test", "val", "both"], default="test",
                        help="外部评测集划分: 'test' (固定历史测试基准, 20题；不可用于选模), 'val' (验证集, 20题), 或 'both'")
    parser.add_argument("--num_eval_seeds", type=int, default=12,
                        help="用于跨代矩阵采样的种子函数数量")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基座模型路径或 HuggingFace ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="评估随机种子")
    return parser.parse_args()


def find_latest_run_dir() -> str:
    """自动查找 checkpoints*/ 下包含 checkpoint-round 检查点的运行目录。"""
    candidates = sorted(glob.glob("checkpoints*/20*"), key=os.path.getmtime)
    for c in reversed(candidates):
        if os.path.isdir(os.path.join(c, "checkpoint-round-0")) and os.path.isdir(os.path.join(c, "checkpoint-round-5")):
            return c
    for c in reversed(candidates):
        if os.path.isdir(os.path.join(c, "checkpoint-round-0")) and os.path.isdir(os.path.join(c, "checkpoint-round-3")):
            return c
    for c in reversed(candidates):
        if os.path.isdir(os.path.join(c, "checkpoint-round-0")):
            return c
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("未在 checkpoints*/ 下找到可评估的检查点目录！")


def load_multi_round_model(
    base_model_name: str,
    run_dir: str,
    rounds: List[int],
    device: str
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """单基座挂载多轮双角色 LoRA 适配器，零显存冗余"""
    print(f"[Model Loading] Loading tokenizer & base model: {base_model_name} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if str(device).startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    base_model.eval()

    # 首先加载 breaker_0
    b0_dir = os.path.join(run_dir, f"checkpoint-round-{rounds[0]}", "breaker")
    model = PeftModel.from_pretrained(base_model, b0_dir, adapter_name=f"breaker_{rounds[0]}")

    # 依次挂载其它轮次的 breaker 和 mutator
    for r in rounds:
        if r != rounds[0]:
            b_path = os.path.join(run_dir, f"checkpoint-round-{r}", "breaker")
            if os.path.isdir(b_path):
                model.load_adapter(b_path, adapter_name=f"breaker_{r}")

        m_path = os.path.join(run_dir, f"checkpoint-round-{r}", "mutator")
        if os.path.isdir(m_path):
            model.load_adapter(m_path, adapter_name=f"mutator_{r}")

    print(f"[Model Loading] Successfully mounted adapters: {list(model.peft_config.keys())}\n", flush=True)
    return tokenizer, model


def evaluate_heldout_suite(
    model: torch.nn.Module,
    tokenizer: Any,
    heldout_problems: List[Dict[str, Any]],
    device: str
) -> Tuple[float, float, float]:
    """外部基准评估: Greedy Pass@1, Sample Pass@4, Hit Density"""
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
    return (greedy_hits / total_probs, sample_hits / total_probs, total_candidate_hits / total_candidates)


def run_cross_play_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    rounds: List[int],
    eval_seeds: List[Dict[str, Any]],
    device: str,
    audit: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, float]]:
    """
    运行跨代博弈矩阵 (Cross-Play Matrix):
      对于每个 Mutator_i (i in rounds):
        生成一组经沙箱严格校验的非等价 Bug 题库 (通过 X_sanity)
      分别交由 Breaker_j (j in rounds) 进行测试:
        计算 Breaker_j 胜率 S[i, j] 与 Mutator_i 欺骗率 D[i, j] = 1 - S[i, j]
    """
    if not rounds or not eval_seeds:
        raise ValueError("Cross-play requires at least one round and one seed")
    if audit is None:
        audit = {}
    audit.update({"seed_scope": "training_seed_functions", "seed_ids": [s["id"] for s in eval_seeds],
                  "seed_sha256": hashlib.sha256(json.dumps(eval_seeds, sort_keys=True).encode()).hexdigest(),
                  "suite_sizes": {}, "valid_candidates": {}, "total_candidates": {},
                  "suites": {}, "mutation_trials": [], "breaker_trials": []})
    cfg_mut = rollout_generation_config(tokenizer, max_new_tokens=32, temperature=0.7, do_sample=True)
    cfg_brk = rollout_generation_config(tokenizer, max_new_tokens=24, do_sample=False)

    num_r = len(rounds)
    round_to_idx = {r: idx for idx, r in enumerate(rounds)}

    # 1. 为每个版本的 Mutator 生成一组检验合格的测试题
    print("=" * 72)
    print("🎯 [Cross-Play Step 1] Generating Verified Mutations per Mutator Version...")
    print("=" * 72)

    mutator_test_suites: Dict[int, List[Dict[str, Any]]] = {}
    mutator_vmr: Dict[int, float] = {}

    for r in rounds:
        adapter_name = f"mutator_{r}"
        model.set_adapter(adapter_name)
        model.eval()

        valid_pairs: List[Dict[str, Any]] = []
        total_rollouts = 0
        valid_count = 0

        for seed in eval_seeds:
            prompt = format_mutator_prompt(seed)
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            p_len = inp["input_ids"].shape[1]
            rep_inp = {k: v.repeat(4, 1) for k, v in inp.items()}

            with torch.no_grad():
                out = generate_with_policy(model, cfg_mut, **rep_inp)

            seed_valid_found = False
            for c_i in range(4):
                total_rollouts += 1
                comp = tokenizer.decode(out[c_i][p_len:], skip_special_tokens=True).strip()
                code_src, _, parse_error = extract_mutator_code(comp, f_ref_src=seed["func_src"])
                is_valid, reason, details = False, parse_error or "No edit", {}
                if code_src and code_src.strip() != seed["func_src"].strip():
                    is_valid, reason, details = validate_mutation(
                        f_ref_src=seed["func_src"], f_mut_src=code_src,
                        sanity_inputs=seed["sanity_inputs"], domain=tuple(seed["domain"]))
                audit["mutation_trials"].append({"mutator_round": r, "seed_id": seed["id"],
                    "completion": comp, "valid": is_valid, "reason": reason})
                if is_valid and details.get("witnesses"):
                    valid_count += 1
                    if not seed_valid_found:
                        valid_pairs.append({
                            "seed_id": seed["id"],
                            "f_ref_src": details["clean_f_ref_src"],
                            "f_mut_src": details["clean_f_mut_src"],
                            "domain": seed["domain"], "witnesses": details["witnesses"],
                        })
                        seed_valid_found = True

        audit["suite_sizes"][r] = len(valid_pairs)
        audit["valid_candidates"][r] = valid_count
        audit["total_candidates"][r] = total_rollouts
        audit["suites"][r] = valid_pairs
        vmr = valid_count / max(1, total_rollouts)
        mutator_vmr[r] = vmr
        mutator_test_suites[r] = valid_pairs
        print(f"  * Mutator_{r}: Valid Mutation Rate (VMR) = {vmr:.1%} ({valid_count}/{total_rollouts}) | Verified Problems Generated = {len(valid_pairs)}/{len(eval_seeds)}")

    # 2. 攻防矩阵对决 (Cross-Play Evaluation)
    print("\n" + "=" * 72)
    print("⚔️ [Cross-Play Step 2] Evaluating Full Cross-Play Matrix (Breaker_j vs Mutator_i)...")
    print("=" * 72)

    solve_matrix = np.full((num_r, num_r), np.nan, dtype=float)
    deception_matrix = np.full((num_r, num_r), np.nan, dtype=float)

    for i_idx, r_mut in enumerate(rounds):
        test_suite = mutator_test_suites[r_mut]
        num_pairs = len(test_suite)

        for j_idx, r_brk in enumerate(rounds):
            model.set_adapter(f"breaker_{r_brk}")
            model.eval()

            solved_hits = 0
            for pair in test_suite:
                f_ref = pair["f_ref_src"]
                f_mut = pair["f_mut_src"]
                domain = tuple(pair["domain"])

                prompt = format_breaker_prompt(f_ref, f_mut, domain)
                inp = tokenizer(prompt, return_tensors="pt").to(device)
                p_len = inp["input_ids"].shape[1]

                with torch.no_grad():
                    out = generate_with_policy(model, cfg_brk, **inp)
                comp = tokenizer.decode(out[0][p_len:], skip_special_tokens=True).strip()
                val, _, _ = extract_breaker_input("<input>" + comp, domain)
                r_hit, _ = compute_breaker_reward(f_ref, f_mut, val, domain)
                audit["breaker_trials"].append({"mutator_round": r_mut, "breaker_round": r_brk,
                    "seed_id": pair["seed_id"], "completion": comp, "input": val, "reward": r_hit})
                if r_hit > 0.0:
                    solved_hits += 1

            if num_pairs > 0:
                solve_rate = solved_hits / num_pairs
                deception_rate = 1.0 - solve_rate
            else:
                solve_rate = float("nan")
                deception_rate = float("nan")
            solve_matrix[i_idx, j_idx] = solve_rate
            deception_matrix[i_idx, j_idx] = deception_rate

    # 计算针对初始冻结 Breaker (V_0) 的欺骗率
    deception_vs_v0: Dict[int, float] = {}
    for i_idx, r_mut in enumerate(rounds):
        deception_vs_v0[r_mut] = deception_matrix[i_idx, 0]

    return solve_matrix, deception_matrix, mutator_vmr, deception_vs_v0


def plot_crossplay_heatmap(
    matrix: np.ndarray,
    rounds: List[int],
    output_png: str,
    output_svg: str,
    metric_name: str = "Breaker Solve Rate"
):
    """绘制高分辨率出版级跨代对抗热力图 (PNG & SVG)"""
    fig, ax = plt.subplots(figsize=(8, 6.8), dpi=300)

    # 选择感知均匀渐变色
    cmap = plt.cm.YlGnBu if "Solve" in metric_name else plt.cm.OrRd
    cmap = cmap.with_extremes(bad="#e5e7eb")
    im = ax.imshow(np.ma.masked_invalid(matrix * 100), cmap=cmap, vmin=0, vmax=100, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(f"{metric_name} (%)", rotation=-90, va="bottom", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    labels = [f"Round {r}" for r in rounds]
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_yticks(np.arange(len(rounds)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")

    ax.set_xlabel("Breaker Version $\\mathcal{B}_j$  \nVersion index", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mutator Version $\\mathcal{M}_i$  \nVersion index", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"nano-CodeSelfPlay Cross-Play Evaluation Matrix\n({metric_name})", fontsize=14, fontweight="bold", pad=14)

    # 绘制格内百分比数值
    font_size = 11 if len(rounds) <= 6 else 8.5
    for i in range(len(rounds)):
        for j in range(len(rounds)):
            val = matrix[i, j] * 100
            text_color = "white" if val > 55 else "black"
            ax.text(j, i, "N/A" if np.isnan(val) else f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=font_size, fontweight="bold")

    ax.set_xticks(np.arange(len(rounds) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rounds) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved Cross-Play Heatmap to:\n   - {output_png}\n   - {output_svg}")


def plot_four_dimension_dashboard(
    rounds: List[int],
    heldout_pass1: List[float],
    heldout_pass4: List[float],
    mut_vmr: List[float],
    deception_v0: List[float],
    solve_matrix: np.ndarray,
    pool_sizes: List[int],
    output_png: str,
    output_svg: str,
    benchmark_label: str = "Fixed benchmark",
):
    """绘制 nano-CodeSelfPlay 四维核心评估出版级综合看板 (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.26)

    # -------------------------------------------------------------
    # Panel 1: 外部基准评估 (Held-out Benchmark vs Base & nano-GRPO)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    methods = [f"Round {r}" for r in rounds]
    p1_scores = [s * 100 for s in heldout_pass1]
    p4_scores = [s * 100 for s in heldout_pass4]

    x = np.arange(len(methods))
    width = 0.35
    rects1 = ax1.bar(x - width/2, p1_scores, width, label="Greedy Pass@1", color="#1f77b4", alpha=0.9, edgecolor="black")
    rects2 = ax1.bar(x + width/2, p4_scores, width, label="Sample Pass@4", color="#2ca02c", alpha=0.85, edgecolor="black")

    ax1.set_title(f"1. {benchmark_label}\nMeasured checkpoints only", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Accuracy / Pass Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    x_fontsize = 7.5 if len(methods) > 8 else 9
    ax1.set_xticklabels(methods, fontsize=x_fontsize, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # 在柱状图上方标注数值
    bar_fs = 6.5 if len(methods) > 8 else 7.5
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=bar_fs, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=bar_fs, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: Mutator 进化指标监控 (VMR & Deception Rate vs V0)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(rounds, [v * 100 for v in mut_vmr], color="#ff7f0e", marker="o", linewidth=2.4, label="Valid Mutation Rate (VMR)")
    ax2.plot(rounds, [d * 100 for d in deception_v0], color="#d62728", marker="s", linewidth=2.2, linestyle="--", label="Deception Rate vs $\\mathcal{B}_0$ ($V_0$)")
    ax2.set_title("2. Mutator Evolution Dynamics\nVMR & Deception Rate against Frozen $\\mathcal{B}_0$", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Self-Play Round", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(rounds)
    ax2.set_xticklabels([f"R{r}" for r in rounds], fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center right", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: 跨代对抗矩阵 (Cross-Play Matrix Heatmap)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    cmap = plt.cm.YlGnBu.with_extremes(bad="#e5e7eb")
    im = ax3.imshow(np.ma.masked_invalid(solve_matrix * 100), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    cbar3 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.set_ylabel("Solve Rate (%)", rotation=-90, va="bottom", fontsize=9, fontweight="bold")
    cbar3.ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    r_labels = [f"R{r}" for r in rounds]
    ax3.set_xticks(np.arange(len(rounds)))
    ax3.set_yticks(np.arange(len(rounds)))
    ax3.set_xticklabels(r_labels, fontsize=10, fontweight="bold")
    ax3.set_yticklabels(r_labels, fontsize=10, fontweight="bold")
    ax3.set_xlabel("Breaker $\\mathcal{B}_j$ (version)", fontsize=10, fontweight="bold", labelpad=6)
    ax3.set_ylabel("Mutator $\\mathcal{M}_i$ (version)", fontsize=10, fontweight="bold", labelpad=6)
    ax3.set_title("3. Cross-Play Matrix Heatmap\nBreaker $\\mathcal{B}_j$ Solve Rate on Mutator $\\mathcal{M}_i$ Bugs", fontsize=12, fontweight="bold", pad=8)

    cell_fs = 8.0 if len(rounds) > 6 else 10.0
    for i in range(len(rounds)):
        for j in range(len(rounds)):
            val = solve_matrix[i, j] * 100
            tc = "white" if val > 55 else "black"
            ax3.text(j, i, "N/A" if np.isnan(val) else f"{val:.1f}%", ha="center", va="center", color=tc, fontsize=cell_fs, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: 对抗题库扩张与自博弈平衡
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    color_pool = "#9467bd"
    ax4.plot(rounds, pool_sizes, color=color_pool, marker="^", linewidth=2.5, label="Verified Adversarial Bug Pool Size")
    ax4.set_title("4. Synthesized Bug Pool Scaling\nAccumulation of Subtle Boundary Counterexamples", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Self-Play Round", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Total Verified Bug Pairs in Pool", fontsize=10, fontweight="bold", color=color_pool)
    ax4.set_xticks(rounds)
    ax4.set_xticklabels([f"R{r}" for r in rounds], fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved 4-Dimension Dashboard to:\n   - {output_png}\n   - {output_svg}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_dir
    if not run_dir:
        run_dir = find_latest_run_dir()
    print(f"\n========================================================================")
    print(f"🚀 Starting nano-CodeSelfPlay Four-Dimension & Cross-Play Evaluation")
    print(f"Run Dir: {run_dir}")
    print(f"Device: {args.device} | Base Model: {args.base_model}")
    print(f"========================================================================\n")

    # 检查可用轮次检查点 (checkpoint-round-0 .. 5+)
    available_rounds = []
    for path in glob.glob(os.path.join(run_dir, "checkpoint-round-*")):
        suffix = os.path.basename(path).removeprefix("checkpoint-round-")
        if suffix.isdigit() and os.path.isdir(path):
            available_rounds.append(int(suffix))
    available_rounds.sort()
    if args.num_eval_seeds <= 0:
        raise ValueError("num_eval_seeds must be positive")

    if 0 not in available_rounds:
        raise ValueError("Round 0 is required as the initial Breaker comparison")
    if len(available_rounds) < 2:
        raise ValueError(f"在 {run_dir} 中未找到足够的 round-specific 检查点 (至少需要 2 轮进行跨代对比，当前找到: {available_rounds})")

    print(f"🔍 Discovered rounds for Cross-Play evaluation: {available_rounds}\n")

    # 读取流水日志指标
    metrics_path = training_metrics_path(run_dir)
    metrics_by_round: Dict[int, Dict[str, Any]] = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    metrics_by_round[item["round"]] = item

    # 加载模型与多轮双角色适配器
    tokenizer, model = load_multi_round_model(args.base_model, run_dir, available_rounds, args.device)

    # 1. 外部基准评估 (Held-out Benchmark)
    splits_to_eval = []
    if args.split in ("val", "both"):
        splits_to_eval.append(("Validation Set", get_validation_benchmark()))
    if args.split in ("test", "both"):
        splits_to_eval.append(("Held-out Test Set", get_test_benchmark()))

    heldout_eval_data = {}
    for split_name, suite_probs in splits_to_eval:
        print("=" * 72)
        print(f"📚 [Dimension 1] {split_name} Audit ({len(suite_probs)} Problems)")
        print("=" * 72)
        p1_list, p4_list, hit_list = [], [], []
        for r in available_rounds:
            model.set_adapter(f"breaker_{r}")
            g_p1, s_p4, hit = evaluate_heldout_suite(model, tokenizer, suite_probs, args.device)
            p1_list.append(g_p1)
            p4_list.append(s_p4)
            hit_list.append(hit)
            print(f"  * Breaker_{r} (Round {r}): Greedy Pass@1 = {g_p1:.2%} | Sample Pass@4 = {s_p4:.2%} | Hit Density = {hit:.2%}")
        heldout_eval_data[split_name] = {
            "pass1": p1_list,
            "pass4": p4_list,
            "hit": hit_list,
            "size": len(suite_probs),
            "problem_ids": [p["id"] for p in suite_probs],
            "problem_sha256": hashlib.sha256(json.dumps(suite_probs, sort_keys=True).encode()).hexdigest(),
        }

    # 主图表以测试集优先，若无则使用验证集
    primary_split_name = "Held-out Test Set" if "Held-out Test Set" in heldout_eval_data else "Validation Set"
    heldout_pass1 = heldout_eval_data[primary_split_name]["pass1"]
    heldout_pass4 = heldout_eval_data[primary_split_name]["pass4"]

    # 2 & 3. 跨代对抗博弈矩阵与 Mutator 进化评估
    all_seeds = get_seed_functions()
    eval_seeds = all_seeds[:min(args.num_eval_seeds, len(all_seeds))]

    crossplay_audit = {}
    solve_matrix, deception_matrix, mut_vmr, deception_v0 = run_cross_play_evaluation(
        model=model,
        tokenizer=tokenizer,
        rounds=available_rounds,
        eval_seeds=eval_seeds,
        device=args.device,
        audit=crossplay_audit,
    )

    pool_sizes = [metrics_by_round.get(r, {}).get("pool_size", float("nan")) for r in available_rounds]

    # 打印格式化 Markdown 表格报告
    print("\n" + "=" * 72)
    print("📊 [Evaluation Report Summary]")
    print("=" * 72)

    for split_name, s_data in heldout_eval_data.items():
        print(f"\n### 外部基准对照表 ({split_name}, {s_data['size']}题)")
        print("| 模型架构 / 训练方法 | 贪心 Pass@1 | 采样 Pass@4 | 训练数据来源 |")
        print("| :--- | :---: | :---: | :--- |")
        for idx, r in enumerate(available_rounds):
            print(f"| **nano-CodeSelfPlay (Round {r})** | **{s_data['pass1'][idx]:.2%}** | **{s_data['pass4'][idx]:.2%}** | 当前运行检查点（训练数据来源见 experiment.json） |")

    print("\n### Mutator 进化指标")
    print("| 轮次 | 有效突变率 (VMR) | 难倒初始 Breaker 欺骗率 (vs V_0) | 有效题数 | 累计对抗题库 (Pool Size) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for r in available_rounds:
        print(f"| Round {r} | {mut_vmr[r]:.2%} | {('N/A' if np.isnan(deception_v0[r]) else format(deception_v0[r], '.2%'))} | {crossplay_audit['suite_sizes'][r]} | {metrics_by_round.get(r, {}).get('pool_size', 'N/A')} |")

    print("\n### 跨代博弈矩阵: Breaker 破解胜率 S[i, j] (%)")
    header = "| Mutator \\ Breaker | " + " | ".join([f"Breaker_{r}" for r in available_rounds]) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in available_rounds]) + " |"
    print(header)
    print(sep)
    for i_idx, r_mut in enumerate(available_rounds):
        row = [f"**Mutator_{r_mut}**"] + ["N/A" if np.isnan(solve_matrix[i_idx, j_idx]) else f"{solve_matrix[i_idx, j_idx]*100:.1f}%" for j_idx in range(len(available_rounds))]
        print("| " + " | ".join(row) + " |")

    # 4. 生成图表与保存原始 JSON 报告
    os.makedirs(args.output_dir, exist_ok=True)
    heatmap_png = os.path.join(args.output_dir, "crossplay_heatmap.png")
    heatmap_svg = os.path.join(args.output_dir, "crossplay_heatmap.svg")
    dashboard_png = os.path.join(args.output_dir, "four_dimension_eval.png")
    dashboard_svg = os.path.join(args.output_dir, "four_dimension_eval.svg")

    plot_crossplay_heatmap(solve_matrix, available_rounds, heatmap_png, heatmap_svg, metric_name="Breaker Solve Rate")
    plot_four_dimension_dashboard(
        rounds=available_rounds,
        heldout_pass1=heldout_pass1,
        heldout_pass4=heldout_pass4,
        mut_vmr=[mut_vmr[r] for r in available_rounds],
        deception_v0=[deception_v0[r] for r in available_rounds],
        solve_matrix=solve_matrix,
        pool_sizes=pool_sizes,
        output_png=dashboard_png,
        output_svg=dashboard_svg,
        benchmark_label=primary_split_name,
    )

    eval_json_path = os.path.join(args.output_dir, "eval_results.json")
    manifest_path = os.path.join(run_dir, "experiment.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            training_manifest = json.load(f)
    else:
        training_manifest = {"protocol_version": "legacy-unrecorded", "config": None}
    report_dict = {
        "evaluation_protocol": PROTOCOL_VERSION,
        "evaluation_source_sha256": source_fingerprints([
            "eval_selfplay.py", "core/experiment.py", "core/run_files.py", "core/sandbox.py", "core/grpo_engine.py",
            "roles/mutator.py", "roles/breaker.py", "data/heldout_eval.py", "data/seed_functions.py"]),
        "training_manifest": training_manifest,
        "seed": args.seed,
        "base_model": args.base_model,
        "decoding": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "sample_candidates": 4,
                     "mutator_max_new_tokens": 32, "breaker_max_new_tokens": 24},
        "crossplay_audit": crossplay_audit,
        "run_dir": run_dir,
        "available_rounds": available_rounds,
        "heldout_eval": heldout_eval_data,
        "mutator_vmr": mut_vmr,
        "deception_vs_v0": deception_v0,
        "solve_matrix": solve_matrix.tolist(),
        "deception_matrix": deception_matrix.tolist(),
        "pool_sizes": pool_sizes,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(finite_json(report_dict), f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"💾 Raw evaluation report saved to: {eval_json_path}")

    print("\n" + "=" * 72)
    print("✅ All Four-Dimension Evaluations & Cross-Play Visualizations Completed!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
