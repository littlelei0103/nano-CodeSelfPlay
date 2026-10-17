"""
plot_evolution.py - 绘制 nano-SPIN 对抗自博弈共演化曲线 (Co-Evolution Dynamics)

呈现 4 栏出版级指标看板:
  1. 独立外部大考演化 (Held-out Greedy Pass@1 & Sample Pass@4);
  2. 左右互搏胜率平衡 (Mutator 欺骗成功率 vs Breaker 破解成功率);
  3. 突变合法性与隐蔽率 (Mutator Validity Rate & X_sanity 通关率);
  4. 对抗 Bug 缓冲池扩张轨迹 (Adversarial Bug Pool Scaling).
"""

import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_args():
    parser = argparse.ArgumentParser(description="Plot nano-SPIN Self-Play Evolution Dynamics")
    parser.add_argument("--run_dir", type=str, default="",
                        help="指定自博弈运行目录 (默认自动查找 checkpoints_spin 下最新目录)")
    parser.add_argument("--output", type=str, default="assets/spin_evolution_curves.png",
                        help="图表输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir:
        candidates = sorted(glob.glob("checkpoints_spin/20*"))
        if not candidates:
            # 兼容当前目录下查找
            candidates = sorted(glob.glob("checkpoints_spin/*"))
        if not candidates:
            raise FileNotFoundError("未在 checkpoints_spin 下找到任何运行目录！")
        run_dir = candidates[-1]

    metrics_path = os.path.join(run_dir, "metrics_spin.jsonl")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"未找到自博弈流水日志: {metrics_path}")

    print(f"📊 正在从自博弈实验目录提取共演化数据: {run_dir}")

    rounds = []
    ho_pass1 = []
    ho_pass4 = []
    mut_validity = []
    mut_rewards = []
    brk_solve = []
    pool_sizes = []
    durations = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            rounds.append(entry["round"])
            ho_pass1.append(entry.get("heldout_greedy_pass1", 0.0) * 100)
            ho_pass4.append(entry.get("heldout_sample_pass4", 0.0) * 100)
            mut_validity.append(entry.get("mut_validity_rate", 0.0) * 100)
            mut_rewards.append(entry.get("mut_avg_reward", 0.0))
            brk_solve.append(entry.get("brk_solve_rate", 0.0) * 100)
            pool_sizes.append(entry.get("pool_size", 0))
            durations.append(entry.get("duration_sec", 0.0))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. 独立外部大考进化曲线
    ax1 = axes[0, 0]
    ax1.plot(rounds, ho_pass1, color="#1f77b4", marker="o", linewidth=2.4, label="Held-out Greedy Pass@1")
    ax1.plot(rounds, ho_pass4, color="#2ca02c", marker="s", linewidth=2.2, linestyle="--", label="Held-out Sample Pass@4")
    ax1.set_title("Breaker Generalization on Held-out Benchmark", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Self-Play Round", fontsize=10)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_ylim(min(20, min(ho_pass1) - 5 if ho_pass1 else 20), 100)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. 左右互搏对抗平衡
    ax2 = axes[0, 1]
    ax2.plot(rounds, brk_solve, color="#2ca02c", marker="^", linewidth=2.2, label="Breaker Solve Rate on Injected Bugs")
    mut_deception = [(100.0 - s) for s in brk_solve]
    ax2.plot(rounds, mut_deception, color="#d62728", marker="v", linewidth=2.2, linestyle="--", label="Mutator Deception Rate (1 - p_solve)")
    ax2.set_title("Adversarial Self-Play Balance (Zero-Sum)", fontsize=13, fontweight="bold", pad=8)
    ax2.set_xlabel("Self-Play Round", fontsize=10)
    ax2.set_ylabel("Rate (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylim(0, 100)
    ax2.legend(loc="center right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. 出题者合法性与隐蔽率
    ax3 = axes[1, 0]
    ax3.plot(rounds, mut_validity, color="#ff7f0e", marker="o", linewidth=2.2, label="Mutator Validity Rate (Passed X_sanity & Non-Eq)")
    ax3.set_title("Mutator Code Invariant & Sanity Rate", fontsize=13, fontweight="bold", pad=8)
    ax3.set_xlabel("Self-Play Round", fontsize=10)
    ax3.set_ylabel("Validity Rate (%)", fontsize=10)
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax3.set_ylim(0, 100)
    ax3.legend(loc="lower right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. 对抗 Bug 缓冲池规模
    ax4 = axes[1, 1]
    ax4.plot(rounds, pool_sizes, color="#9467bd", marker="d", linewidth=2.2, label="Adversarial Bug Pool Size")
    ax4.set_title("Synthesized Tricky Bug Pool Accumulation", fontsize=13, fontweight="bold", pad=8)
    ax4.set_xlabel("Self-Play Round", fontsize=10)
    ax4.set_ylabel("Total Active Verified Bug Pairs", fontsize=10)
    ax4.legend(loc="upper left", frameon=True, fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"nano-SPIN: Code Mutation & Counterexample Adversarial Self-Play Co-Evolution\nQwen2.5-1.5B (Shared Weights) on NVIDIA RTX 5070 (8GB VRAM)",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"✅ 自博弈共演化图表已成功保存至: {args.output}")


if __name__ == "__main__":
    main()
