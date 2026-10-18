"""Versioned experiment state. Incompatible historical runs are weights-only inputs."""
import hashlib
import json
import os
from pathlib import Path
import random

import torch

PROTOCOL_VERSION = "numeric-selfplay-v2"
CONFIG_KEYS = (
    "model_name", "device", "seed", "mutator_warmup_steps", "eval_interval", "mut_rollouts",
    "breaker_rollouts", "lr", "lr_mutator", "lr_breaker", "lr_breaker_min",
    "cosine_decay_start", "curriculum_tighten_step", "anchor_ratio", "kl_coef", "temperature",
    "freeze_mutator_after_warmup", "breaker_only_static",
)


def source_fingerprints(names):
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def experiment_manifest(args, eval_interval):
    names = ["train_selfplay.py", "core/experiment.py", "core/run_files.py", "core/sandbox.py", "core/hard_buffer.py",
             "core/grpo_engine.py", "roles/mutator.py", "roles/breaker.py",
             "data/seed_functions.py", "data/anchor_tasks.py", "data/heldout_eval.py"]
    files = source_fingerprints(names)
    config = {key: getattr(args, key) for key in CONFIG_KEYS}
    config["eval_interval"] = eval_interval
    return {"protocol_version": PROTOCOL_VERSION, "config": config, "source_sha256": files,
            "initial_adapter": args.init_adapter or None,
            "torch_version": str(torch.__version__)}


def read_training_state(checkpoint):
    state = torch.load(os.path.join(checkpoint, "train_state.pt"), map_location="cpu", weights_only=True)
    required = {"step", "round", "best_val_pass1", "best_step", "manifest", "baseline",
                "schedule_total_steps", "hard_buffer_items", "optimizer_mutator", "optimizer_breaker",
                "torch_rng_state", "python_rng_state", "cuda_rng_states"}
    if required - state.keys() or state.get("manifest", {}).get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Checkpoint lacks v2 experiment state. Use --init_adapter for a new weights-only run.")
    return state


def validate_resume(state, manifest):
    prior = state["manifest"]
    for k in CONFIG_KEYS:
        val_prior = prior["config"].get(k, False if k in ("freeze_mutator_after_warmup", "breaker_only_static") else None)
        val_manifest = manifest["config"].get(k, False if k in ("freeze_mutator_after_warmup", "breaker_only_static") else None)
        if val_prior != val_manifest:
            raise ValueError(f"Resume configuration changed: {k} ({val_prior} != {val_manifest}). Start a new experiment instead.")
    if prior["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Training code or dataset changed; exact resume is not supported. Start a new run.")
    if prior["torch_version"] != manifest["torch_version"]:
        raise ValueError("PyTorch version changed; start a new run instead of exact resume.")


def restore_training_state(state, opt_mutator, opt_breaker):
    # Fail explicitly rather than silently continuing with fresh optimizer/RNG state.
    opt_mutator.load_state_dict(state["optimizer_mutator"])
    opt_breaker.load_state_dict(state["optimizer_breaker"])
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    if state["cuda_rng_states"] is not None:
        if not torch.cuda.is_available() or len(state["cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("CUDA device topology differs from the saved experiment")
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])


def select_validation_best(score, step, best_score, best_step):
    """Keep the earliest checkpoint on a tie; update state before saving any snapshot."""
    if score > best_score:
        return score, step, True
    return best_score, best_step, False


def discover_checkpoint(run_dir):
    root = Path(run_dir)
    candidates = list(root.glob("checkpoint-round-*")) + [root / "checkpoint-last"]
    complete = [p for p in candidates if (p / "train_state.pt").is_file()]
    if not complete:
        raise FileNotFoundError(f"No complete training checkpoint in {run_dir}")
    # Inspect every state: a corrupt checkpoint must not silently fall back to older weights.
    loaded = [(p, read_training_state(p)) for p in complete]
    path, state = max(loaded, key=lambda entry: entry[1]["step"])
    return str(path), state


def finite_json(value):
    """Encode unmeasured matrix entries as JSON null, never non-standard NaN."""
    import math
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
