# nano-SPIN: Code Mutation & Counterexample Adversarial Self-Play

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware: 8GB VRAM](https://img.shields.io/badge/Hardware-8GB%20VRAM%20Compatible-brightgreen.svg)]()

> **基于单模型共享权重的代码突变与差分反例挖掘对抗自博弈（Adversarial Self-Play）框架**  
> 突破静态题目上限，通过零外部数据（Zero-Data RSI）的“左右互搏”，实现模型推理能力的递归自我进化。

---

## 🌟 核心愿景与架构演进 (Motivation)

在前期项目 [`nano-GRPO`](../nano-GRPO) 中，找茬模型（Breaker）在 60 道固定的静态合成题目上通过可验证强化学习（RLVR），将独立测试集上的贪心通过率从基座的 40.0% 提升至 78.33%。然而，静态题库的覆盖面终有上限。

**nano-SPIN** 引入双角色对抗自博弈闭环（Adversarial Self-Play Fine-Tuning）：
* **零外部新增数据**：仅依赖轻量级干净种子参考函数库（30~40 个基础数学与逻辑函数 $f_{\text{ref}}$ 及常规测试输入 $X_{\text{sanity}}$）；
* **单模型共享权重（Shared-Weight Dual Roles）**：无需加载双份大模型。基于单个 Qwen2.5-1.5B 权重（bfloat16 + 梯度检查点 + 轻量 LoRA，显存稳定在 ~3.0 GB），通过切换 Prompt 分别扮演**出题者（Mutator）**与**找茬者（Breaker）**；
* **客观对抗难度平衡（Sweet-Spot Curriculum Reward）**：Mutator 只有造出既满足常规输入隐蔽性约束、又能难倒当前 Breaker 的微妙 Bug 时才能获得高奖励，形成互相驱动、水涨船高的认知演化飞轮。

```mermaid
flowchart TD
    A["种子函数库 (f_ref + X_sanity)"] --> B["出题者角色: Mutator<br/>(System Prompt 引导思考注入缺陷)"]
    B --> C["候选突变函数: f_mut"]
    
    subgraph S1 ["安全沙箱三层防作弊校验 (Sandbox Guard)"]
        C --> D1["1. 语法编译与单函数 AST 解析"]
        D1 -->|通过| D2["2. 隐蔽性约束: X_sanity 上 f_mut == f_ref<br/>(杜绝死值作弊与低幼改动)"]
        D2 -->|通过| D3["3. 非等价约束: 全域网格探针验证存在反例<br/>(杜绝无效果改动与死代码)"]
    end
    
    D3 -->|合法突变| E["当前 Breaker 尝试破解 (K=4 探测采样)"]
    D3 -->|校验失败| F["奖励 R_mutator = 0.0"]
    
    E --> G["计算破解率 p_solve<br/>对抗奖励 R_mutator = 1.0 - p_solve"]
    G --> H["Mutator GRPO 策略参数更新"]
    
    G -->|成功难倒 Breaker (p_solve < 1.0)| I["注入动态对抗题库<br/>(Adversarial Bug Pool)"]
    
    I --> J["找茬者角色: Breaker<br/>(自博弈对抗采样 G=4)"]
    J --> K["安全沙箱差分执行: f_ref(x) != f_mut(x)"]
    K -->|命中反例| L["奖励 R_breaker = +1.0"]
    K -->|未命中| M["奖励 R_breaker = 0.0"]
    
    L & M --> N["Breaker GRPO 策略参数更新"]
    N -.->|权重同步演化| B
    
    N --> O["每轮独立外部大考 (Held-out Benchmark 20 题)<br/>客观监控无偏泛化进阶"]
```

---

## 🎭 双角色自博弈机制与对抗奖励

### 1. 角色 A：出题者（Mutator / Bug Injector）
* **任务目标**：针对标准函数 $f_{\text{ref}}$，在 `<think>` 中构思细微的逻辑边界偏差（如 off-by-one、区间开闭混淆、极值截断越界、取模余数偏置等），输出修改后的 Python 函数代码。
* **三层防作弊硬性校验（Hard Sanity Guard）**：
  1. **语法与形式**：必须为合法、可调用的 Python 函数；
  2. **隐蔽性约束（Sanity Invariant）**：在常规基础输入 $X_{\text{sanity}}$（如常规正整数 `[5, 10, 45]`）上的执行输出必须与 $f_{\text{ref}}$ **完全一致**。若模型直接写 `return 0` 或破坏基础功能，直接判定失败；
  3. **非等价探针（Non-Equivalence Probe）**：全定义域快速网格探针验证，确保在定义域内至少存在一个整数 $x^*$ 使得 $f_{\text{mut}}(x^*) \neq f_{\text{ref}}(x^*)$。若修改为等价无用代码（如重命名局部变量），直接判定失败。
* **对抗奖励（Sweet-Spot Curriculum）**：
  $$R_{\text{mutator}} = \begin{cases} 0.0, & \text{若未通过三层校验} \\ 1.0 - p_{\text{solve}}, & \text{若合法突变} \end{cases}$$
  其中 $p_{\text{solve}} \in [0, 1]$ 为当前 Breaker 对该题的破解率。Breaker 越容易识破，Mutator 得分越低；Breaker 全军覆没（$p_{\text{solve}}=0$）时，Mutator 拿满分 $+1.0$。

### 2. 角色 B：找茬者（Breaker / Counterexample Finder）
* **任务目标**：面对对抗题库中的函数对 $(f_{\text{ref}}, f_{\text{mut}})$，在 `<think>` 中比对控制流与分支边界差异，预测能触发输出不一致的整数输入 $x$。
* **差分判题**：沙箱并行执行 $f_{\text{ref}}(x)$ 与 $f_{\text{mut}}(x)$。
* **找茬奖励**：
  $$R_{\text{breaker}} = \begin{cases} 1.0, & \text{若 } f_{\text{ref}}(x) \neq f_{\text{mut}}(x) \\ 0.0, & \text{若 } f_{\text{ref}}(x) == f_{\text{mut}}(x) \text{ 或格式/越界错误} \end{cases}$$

---

## 📂 项目模块结构

```text
nano-SPIN/
├── core/
│   ├── grpo_engine.py      # 单模型共享权重 GRPO 引擎 (支持角色切换、零方差保护与 Schulman k3)
│   └── sandbox.py          # 安全子进程执行、三层突变校验探针与差分判题
├── roles/
│   ├── mutator.py          # Mutator 角色 Prompt、代码抽取正则与难度适中奖励计算
│   └── breaker.py          # Breaker 角色 Prompt、反例输入解析与客观差分奖励计算
├── data/
│   ├── seed_functions.py   # 40 个多样化干净种子函数库及常规基础测试集 X_sanity
│   └── heldout_eval.py     # 20 道严格隔离的外部基准测试集 (用于客观评估自博弈泛化演进)
├── tests/                  # 19 项单元回归测试 (涵盖沙箱、角色、数据与端到端闭环)
│   ├── test_sandbox.py
│   ├── test_roles.py
│   ├── test_data.py
│   ├── test_grpo_engine.py
│   └── test_spin_e2e.py
├── train_spin.py           # 对抗自博弈训练主循环 (Phase A 出题 -> Phase B 破题 -> Phase C 大考)
├── plot_evolution.py       # 绘制自博弈共演化曲线看板 (Held-out Pass@1、欺骗率与缓冲池增长)
└── requirements.txt        # 依赖规范清单
```

---

## ⚡ 快速开始与使用指南

### 1. 环境准备
```bash
cd /home/littlelei/github/nano-SPIN

# 推荐复用 nano-grpo 的 conda 环境
conda activate nano-grpo
pip install -r requirements.txt
```

### 2. 运行完整单元测试套件
```bash
# 验证沙箱防作弊、双角色解析、种子函数完备性与自博弈微闭环 (19 项测试秒级全绿)
python -m unittest discover tests/ -v
```

### 3. 启动对抗自博弈训练
```bash
# 模式一: 从 Qwen2.5-1.5B 基座开启自博弈冷启动 (5 轮左右互搏)
python train_spin.py --rounds 5 --seeds_per_round 8 --lr 2e-5

# 模式二: 加载 nano-GRPO 训练好的最佳 LoRA 适配器热启动自博弈 (突破静态题库上限)
python train_spin.py --rounds 5 --init_adapter ../nano-GRPO/checkpoints_counterexample/<run-id>/checkpoint-best
```

### 4. 绘制共演化曲线看板
```bash
# 从最新自博弈运行目录提取日志并绘制出版级 4 栏共演化图表
python plot_evolution.py
```

---

## 💻 硬件适配 (RTX 5070 8GB VRAM)

* **单模型共享**：通过提示词动态切换 Mutator 与 Breaker，显存开销等同于单模型训练（无需双模型常驻显存）。
* **轻量 LoRA + 梯度检查点**：bfloat16 全线性层 LoRA 挂载，静态显存仅占用 **~2980 MB**，探索与反向传播峰值显存维持在 **~3350 MB**（安全低于 8GB 显存红线）。
