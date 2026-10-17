"""
core/grpo_engine.py - nano-SPIN 共享权重双角色对抗 GRPO 训练引擎

核心设计:
  1. 单模型共享权重: 基于 Qwen2.5-1.5B-Base (bfloat16 + LoRA + 梯度检查点)，显存占用约 3.0 GB。
  2. 动态角色切换 (Role Switching): 通过提示词在 Mutator 与 Breaker 之间无缝切换，无需加载两套权重。
  3. 零方差过滤保护 (Zero-Variance Guard): 组内奖励方差低于 1e-6 时跳过策略梯度反传，防止无效噪声扰动。
  4. Schulman k3 无偏参考 KL 约束: 通过 peft disable_adapter 零显存获取基座参考概率分布，保持语言流利度。
  5. 严格 Token 级与题目级两级归一化，确保微批次划分下梯度数学等价性。
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel


def rollout_generation_config(
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 1.0,
    do_sample: bool = True
) -> GenerationConfig:
    """创建策略采样生成配置"""
    if top_p != 1.0:
        raise ValueError("GRPO requires top_p=1.0 for unbiased k3 reference KL estimation")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return GenerationConfig(
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=1.0,
        top_k=0 if do_sample else None,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def generate_with_policy(
    model: torch.nn.Module,
    generation_config: GenerationConfig,
    **inputs: Any
) -> torch.Tensor:
    """使用指定策略生成输出序列"""
    original = getattr(model, "generation_config", None)
    model.generation_config = generation_config
    try:
        return model.generate(**inputs, generation_config=generation_config)
    finally:
        model.generation_config = original


def compute_completion_mask(
    outputs: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    构造 completion 部分的有效 token 布尔掩码 (仅保留首个 EOS 及其之前的有效 token)。
    
    Returns:
        mask: (batch_size, gen_len) 布尔 Tensor, True 为有效生成 token
    """
    seq_len = outputs.shape[1]
    gen_len = seq_len - prompt_len
    if gen_len <= 0:
        return torch.zeros((outputs.shape[0], 0), dtype=torch.bool, device=device)

    gen_tokens = outputs[:, prompt_len:]
    is_eos = (gen_tokens == eos_token_id)
    has_eos = is_eos.any(dim=-1)
    first_eos_idx = torch.where(has_eos, is_eos.int().argmax(dim=-1), gen_len - 1)
    col_indices = torch.arange(gen_len, device=device).unsqueeze(0)
    resp_mask = (col_indices <= first_eos_idx.unsqueeze(1)).clone()
    return resp_mask


def compute_completion_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    仅对 completion 部分计算 token log 概率 (FP32 cross entropy)。
    
    Returns:
        logprobs: (batch_size, gen_len)
    """
    comp_logits = logits[:, prompt_len - 1 : -1, :].contiguous().float()
    comp_labels = input_ids[:, prompt_len :].contiguous()

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if temperature != 1.0:
        comp_logits = comp_logits / temperature

    logprobs = -F.cross_entropy(
        comp_logits.transpose(1, 2),
        comp_labels,
        reduction="none"
    )
    return logprobs


def get_reference_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    通过 peft disable_adapter 获取冻结基座模型的参考 completion log-probs (零额外权重显存)。
    """
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids).logits
                ref_logprobs = compute_completion_logprobs(ref_logits, input_ids, prompt_len, temperature=temperature)
                del ref_logits
        else:
            ref_logits = model(input_ids=input_ids).logits
            ref_logprobs = compute_completion_logprobs(ref_logits, input_ids, prompt_len, temperature=temperature)
            del ref_logits
    return ref_logprobs.detach()


def compute_grpo_loss(
    cur_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算单个组内的 PPO-Clipped GRPO 损失矩阵及关键诊断指标。
    """
    cur_logprobs = cur_logprobs.float()
    old_logprobs = old_logprobs.float()
    advantages = advantages.float()

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)  # (B, 1)

    # 1. PPO-Clip
    log_ratio = torch.where(mask, cur_logprobs - old_logprobs, 0.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)

    with torch.no_grad():
        active_tokens = mask.sum().float().clamp(min=1.0)
        clipped = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))) & mask
        clip_frac = (clipped.sum().float() / active_tokens).item()
        mean_ratio = ((ratio * mask).sum() / active_tokens).item()

    # 2. Schulman k3 KL Regularizer
    if kl_coef > 0.0 and ref_logprobs is not None:
        ref_logprobs = ref_logprobs.float()
        kl_log_ratio = torch.where(mask, ref_logprobs - cur_logprobs, 0.0)
        per_token_kl = torch.clamp(torch.expm1(kl_log_ratio) - kl_log_ratio, min=0.0)
        total_loss = policy_loss + kl_coef * per_token_kl
        with torch.no_grad():
            approx_kl = ((per_token_kl * mask).sum() / active_tokens).item()
    else:
        total_loss = policy_loss
        approx_kl = 0.0

    if not torch.isfinite(total_loss[mask]).all():
        raise FloatingPointError("Non-finite GRPO loss detected!")

    metrics = {
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "active_tokens": active_tokens.item()
    }
    return total_loss, metrics


def setup_lora_and_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "cuda",
    lora_r: int = 8,
    lora_alpha: int = 16,
    init_adapter: str = "",
    resume_checkpoint: str = ""
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    初始化基座模型并挂载轻量 LoRA 适配器 (适配 8GB 显存消费级硬件)。
    """
    print(f"[Model Setup] Loading tokenizer & model: {model_name} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if str(device).startswith("cuda") else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[Model Setup] Resuming LoRA from checkpoint: {resume_checkpoint}", flush=True)
        model = PeftModel.from_pretrained(base_model, resume_checkpoint, is_trainable=True)
    elif init_adapter and os.path.exists(init_adapter):
        print(f"[Model Setup] Initializing with pre-trained adapter: {init_adapter}", flush=True)
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        print(f"[Model Setup] Initializing fresh LoRA adapter (r={lora_r}, alpha={lora_alpha})...", flush=True)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)

    model.train()
    return tokenizer, model


class SPINGRPOTrainer:
    """
    nano-SPIN 统一双角色自博弈训练器。
    管理模型参数更新、优势估计与零方差保护。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm

    def backward_grpo_group(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        rewards: List[float],
        temperature: float = 0.7,
        num_groups_in_update: int = 1
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        对单个 Prompt 的 G 路候选执行 GRPO 前向损失计算与反向传播。
        包含零方差保护: 当组内奖励方差低于 1e-6 时跳过，返回 is_effective = False。

        Returns:
            (is_effective, group_loss, metrics)
        """
        G = len(rewards)
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std(unbiased=False)

        # 零方差保护
        if r_std.item() < 1e-6:
            return False, 0.0, {
                "r_mean": r_mean.item(),
                "r_std": r_std.item(),
                "approx_kl": 0.0,
                "clip_frac": 0.0
            }

        advantages = (r_tensor - r_mean) / (r_std + 1e-8)

        prompt_len = prompt_ids.shape[1]
        device = prompt_ids.device
        mask = compute_completion_mask(generated_ids, prompt_len, self.tokenizer.eos_token_id, device)

        # 1. 基座参考 logprobs (PEFT disable_adapter)
        ref_logprobs = get_reference_logprobs(self.model, generated_ids, prompt_len, temperature=temperature)

        # 2. 策略旧 logprobs (采样分布基准)
        with torch.no_grad():
            old_logits = self.model(input_ids=generated_ids).logits
            old_logprobs = compute_completion_logprobs(old_logits, generated_ids, prompt_len, temperature=temperature)
            del old_logits

        # 3. 策略当前 logprobs (带梯度)
        cur_logits = self.model(input_ids=generated_ids).logits
        cur_logprobs = compute_completion_logprobs(cur_logits, generated_ids, prompt_len, temperature=temperature)

        # 4. 计算损失
        per_token_loss, metrics = compute_grpo_loss(
            cur_logprobs=cur_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
            clip_eps=self.clip_eps,
            kl_coef=self.kl_coef
        )

        # 5. Token 级归一化 + 题目级平均
        active_tokens = mask.sum().float().clamp(min=1.0)
        sample_loss = (per_token_loss * mask).sum() / active_tokens
        scaled_loss = sample_loss / max(1, num_groups_in_update)

        # 反向传播累积梯度
        scaled_loss.backward()

        metrics["r_mean"] = r_mean.item()
        metrics["r_std"] = r_std.item()
        return True, sample_loss.item(), metrics

    def step_optimizer(self) -> float:
        """执行优化器单步更新与梯度裁剪，返回梯度范数"""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm
        ).item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_norm
