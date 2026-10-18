# Figure provenance

Current release figures:

- `selfplay-overview.svg`: schematic illustration, not a training result.
- `v2_regime_comparison.{png,svg}`: three bar-chart panels comparing mean test scores across all four configurations and Base; error bars show sample SD. Individual seed results remain in `reports/v2_experiment_summary.md` and `reports/runs/`.
- `v2_training_evolution.{png,svg}`: validation and unique-pair pool trajectories from three runs per configuration.

Rebuild the latter two with `python scripts/aggregate_all_reports.py` then
`python scripts/plot_comparison.py`. Both use the published raw records in
`reports/runs/`, not model weights. Missing data is an error; no baseline score
is synthesized. SD uses ddof=1 and is not a confidence interval.

For a single local training run, use `python scripts/plot_training.py --run_dir
CHECKPOINT_RUN --output eval_outputs/training.png`. This plots runtime telemetry
from `metrics.jsonl` (or the historical `metrics_spin.jsonl`) without loading a
model; it does not replace the multi-run release figures above. New diagnostic
figures from `eval_selfplay.py` are named `crossplay_heatmap.*` and
`four_dimension_eval.*`.

The older `spin_*` figures and `eval_results.json` are retained historical
artifacts, not the release comparison. Their collection protocols and source
runs may differ. The v2 Seed 42 cross-play audit is separately published as
`reports/crossplay_seed42.json`; it uses training seed functions and is a
retrospective diagnostic, not an independent generalization benchmark.
