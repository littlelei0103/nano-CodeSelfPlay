"""
Evaluate the untrained base model (Qwen/Qwen2.5-1.5B) directly on both
the validation benchmark (20 problems) and held-out test benchmark (20 problems)
under the exact numeric-selfplay-v2 evaluation protocol.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.experiment import PROTOCOL_VERSION, source_fingerprints
from core.grpo_engine import generate_with_policy, rollout_generation_config
from data.heldout_eval import get_test_benchmark, get_validation_benchmark
from roles.breaker import compute_breaker_reward, extract_breaker_input, format_breaker_prompt


def evaluate_suite(model, tokenizer, problems, device, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    cfg_greedy = rollout_generation_config(tokenizer, max_new_tokens=24, do_sample=False)
    cfg_sample = rollout_generation_config(tokenizer, max_new_tokens=24, temperature=0.7, do_sample=True)

    greedy_hits = 0
    sample_hits = 0
    total_candidates = 0
    total_candidate_hits = 0
    per_problem_results = []

    for prob in problems:
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
        is_greedy_hit = (r_g > 0.0)
        if is_greedy_hit:
            greedy_hits += 1

        # 2. Sample Pass@4
        rep_inputs = {k: v.repeat(4, 1) for k, v in inputs.items()}
        with torch.no_grad():
            out_s = generate_with_policy(model, cfg_sample, **rep_inputs)
        solved_any = False
        candidates = []
        for i in range(4):
            total_candidates += 1
            raw_s = tokenizer.decode(out_s[i][p_len:], skip_special_tokens=True).strip()
            val_s, _, _ = extract_breaker_input("<input>" + raw_s, domain)
            r_s, _ = compute_breaker_reward(f_ref, f_mut, val_s, domain)
            hit = (r_s > 0.0)
            if hit:
                total_candidate_hits += 1
                solved_any = True
            candidates.append({"raw": raw_s, "parsed": val_s, "reward": r_s, "hit": hit})
        if solved_any:
            sample_hits += 1

        per_problem_results.append({
            "id": prob["id"],
            "spec": prob.get("spec", ""),
            "domain": list(domain),
            "greedy": {"raw": raw_g, "parsed": val_g, "reward": r_g, "hit": is_greedy_hit},
            "sample_pass4": solved_any,
            "candidates": candidates
        })

    total_probs = len(problems)
    return {
        "greedy_pass1": greedy_hits / max(1, total_probs),
        "sample_pass4": sample_hits / max(1, total_probs),
        "hit_density": total_candidate_hits / max(1, total_candidates),
        "total_problems": total_probs,
        "greedy_hits": greedy_hits,
        "sample_hits": sample_hits,
        "total_candidate_hits": total_candidate_hits,
        "total_candidates": total_candidates,
        "details": per_problem_results
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"reports/base_benchmark_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    base_model_name = args.model_name
    print(f"[Base Eval] Loading {base_model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    val_problems = get_validation_benchmark()
    test_problems = get_test_benchmark()

    # 1. Validation Benchmark
    print(f"\n[Base Eval] Auditing Validation Benchmark ({len(val_problems)} problems)...")
    val_res = evaluate_suite(model, tokenizer, val_problems, device, seed=42)
    print(f"   Validation Greedy Pass@1: {val_res['greedy_pass1']:.2%}")
    print(f"   Validation Sample Pass@4: {val_res['sample_pass4']:.2%}")
    print(f"   Validation Hit Density:   {val_res['hit_density']:.2%}")

    # 2. Test Benchmark across multiple sample seeds
    print(f"\n[Base Eval] Auditing Test Benchmark ({len(test_problems)} problems)...")
    seeds = [42, 43, 44]
    test_seed_results = {}
    for s in seeds:
        res = evaluate_suite(model, tokenizer, test_problems, device, seed=s)
        test_seed_results[s] = res
        print(f"   Seed {s}: Greedy={res['greedy_pass1']:.2%}, Pass@4={res['sample_pass4']:.2%}, HitDensity={res['hit_density']:.2%}")

    # Greedy is deterministic, so it's the same across seeds
    greedy_p1 = test_seed_results[42]["greedy_pass1"]
    sample_p4_list = [test_seed_results[s]["sample_pass4"] for s in seeds]
    hit_density_list = [test_seed_results[s]["hit_density"] for s in seeds]

    mean_p4 = sum(sample_p4_list) / len(sample_p4_list)
    mean_hit = sum(hit_density_list) / len(hit_density_list)

    final_report = {
        "protocol_version": PROTOCOL_VERSION,
        "base_model": base_model_name,
        "decoding": {"temperature": 0.7, "top_p": 1.0, "top_k": 0,
                     "max_new_tokens": 24, "sample_candidates": 4, "seeds": seeds},
        "evaluation_source_sha256": source_fingerprints([
            "scripts/eval_base_benchmark.py", "core/sandbox.py", "core/grpo_engine.py",
            "roles/breaker.py", "data/heldout_eval.py"]),
        "dataset_sha256": {"val": hashlib.sha256(json.dumps(val_problems, sort_keys=True).encode()).hexdigest(),
                           "test": hashlib.sha256(json.dumps(test_problems, sort_keys=True).encode()).hexdigest()},
        "device": device,
        "validation_benchmark": {
            "greedy_pass1": val_res["greedy_pass1"],
            "sample_pass4": val_res["sample_pass4"],
            "hit_density": val_res["hit_density"],
            "details": val_res["details"]
        },
        "test_benchmark": {
            "greedy_pass1": greedy_p1,
            "sample_pass4_by_seed": {s: test_seed_results[s]["sample_pass4"] for s in seeds},
            "hit_density_by_seed": {s: test_seed_results[s]["hit_density"] for s in seeds},
            "mean_sample_pass4": mean_p4,
            "mean_hit_density": mean_hit,
            "seed_42_details": test_seed_results[42]["details"]
        }
    }

    report_path = args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n[Base Eval] Audit complete! Report written to {report_path}")


if __name__ == "__main__":
    main()
