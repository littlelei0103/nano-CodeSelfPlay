"""
scripts/plot_training.py - 绘制 nano-CodeSelfPlay 单次运行的训练遥测。

只读训练日志，无需加载模型。展示验证分数、Breaker 候选命中率、
当前种子函数上的 Mutator 有效率，以及 HardBuffer 大小。
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.run_files import training_metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training telemetry for one nano-CodeSelfPlay run; no model required")
    parser.add_argument("--run_dir", type=str, default="",
                        help="指定运行目录（默认自动查找 checkpoints*/ 下最新目录）")
    parser.add_argument("--output", type=str, default="eval_outputs/training_telemetry.png",
                        help="图表输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir:
        candidates = sorted(glob.glob("checkpoints*/20*"), key=os.path.getmtime)
        if not candidates:
            # 兼容当前目录下查找
            candidates = sorted(glob.glob("checkpoints*/*"), key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError("未在 checkpoints*/ 下找到任何运行目录！")
        run_dir = candidates[-1]

    metrics_path = training_metrics_path(run_dir)
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"未找到自博弈流水日志: {metrics_path}")

    print(f"📊 正在从实验目录读取训练遥测: {run_dir}")

    rounds = []
    ho_pass1 = []
    ho_pass4 = []
    mut_validity = []
    mut_rewards = []
    brk_solve = []
    pool_sizes = []
    durations = []

    validation_protocol = False
    def percent(value):
        return float("nan") if value is None else value * 100

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            rounds.append(entry["round"])
            validation_protocol = validation_protocol or "val_greedy_pass1" in entry
            ho_pass1.append(percent(entry.get("val_greedy_pass1", entry.get("heldout_greedy_pass1"))))
            ho_pass4.append(percent(entry.get("val_sample_pass4", entry.get("heldout_sample_pass4"))))
            mut_validity.append(percent(entry.get("mut_validity_rate")))
            mut_rewards.append(entry.get("mut_avg_reward", 0.0))
            brk_solve.append(percent(entry.get("brk_overall_hit_rate", entry.get("brk_solve_rate"))))
            pool_sizes.append(entry.get("pool_size", 0))
            durations.append(entry.get("duration_sec", 0.0))

    if not rounds:
        raise ValueError("Metrics log is empty")
    benchmark = "Validation" if validation_protocol else "Legacy benchmark (protocol unverified)"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # 1. 独立外部大考进化曲线
    ax1 = axes[0, 0]
    ax1.plot(rounds, ho_pass1, color="#1f77b4", marker="o", linewidth=2.4, label=f"{benchmark} Greedy Pass@1")
    ax1.plot(rounds, ho_pass4, color="#2ca02c", marker="s", linewidth=2.2, linestyle="--", label=f"{benchmark} Sample Pass@4")
    ax1.set_title(f"Breaker on {benchmark}", fontsize=13, fontweight="bold", pad=8)
    ax1.set_xlabel("Self-Play Round", fontsize=10)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. 左右互搏对抗平衡
    ax2 = axes[0, 1]
    ax2.plot(rounds, brk_solve, color="#2ca02c", marker="^", linewidth=2.2, label="Breaker candidate hit rate (all attempts)")
    ax2.set_title("Breaker Rollout Outcomes (not a fixed test set)", fontsize=13, fontweight="bold", pad=8)
    ax2.set_xlabel("Self-Play Round", fontsize=10)
    ax2.set_ylabel("Rate (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylim(0, 100)
    ax2.legend(loc="center right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. 出题者合法性与隐蔽率
    ax3 = axes[1, 0]
    ax3.plot(rounds, mut_validity, color="#ff7f0e", marker="o", linewidth=2.2, label="Mutator Validity Rate (Passed X_sanity & Non-Eq)")
    ax3.set_title("Mutator Validity on the Current Training Seed", fontsize=13, fontweight="bold", pad=8)
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

    fig.suptitle(f"nano-CodeSelfPlay: Training Telemetry\nRun: {os.path.basename(run_dir)}",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training telemetry to: {args.output}")


if __name__ == "__main__":
    main()
