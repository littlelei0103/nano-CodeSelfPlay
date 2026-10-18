"""Shared run filenames, including read support for historical training logs."""
from pathlib import Path


def training_metrics_path(run_dir):
    """Use the existing log, or the canonical name for a new run."""
    root = Path(run_dir)
    current = root / "metrics.jsonl"
    legacy = root / "metrics_spin.jsonl"
    if current.exists() and legacy.exists():
        raise ValueError(f"Ambiguous training logs in {root}: both {current.name} and {legacy.name} exist")
    return legacy if legacy.exists() else current
