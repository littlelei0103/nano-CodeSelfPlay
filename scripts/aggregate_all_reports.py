"""Rebuild release tables from published raw reports; never invent missing values."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.run_files import training_metrics_path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'breaker_only_static': ('Static GRPO', 'checkpoints_v2_breaker_static'),
    'pure_selfplay': ('Self-play', 'checkpoints_v2_pure'),
    'mixed_replay': ('Mixed replay (configured ratio 0.1)', 'checkpoints_v2_mixed'),
    'frozen_mutator_ablation': ('Frozen after warmup', 'checkpoints_v2_frozen_mutator'),
}
METRICS = ('best_val_greedy_pass1', 'test_greedy_pass1', 'test_sample_pass4', 'test_hit_rate')
JUDGE_FILES = ('core/sandbox.py', 'core/grpo_engine.py', 'roles/breaker.py',
               'data/anchor_tasks.py', 'data/heldout_eval.py', 'data/seed_functions.py')


def calc_stats(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else None
    return mean, std


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_report(report, group):
    if report.get('protocol_version') != 'numeric-selfplay-v2' or report.get('selection_split') != 'validation':
        raise ValueError('Expected v2 evaluation and validation-based selection')
    cfg = report['manifest']['config']
    static = cfg.get('breaker_only_static', False)
    frozen = cfg.get('freeze_mutator_after_warmup', False)
    expected = {'breaker_only_static': (True, False, 0), 'pure_selfplay': (False, False, 0),
                'mixed_replay': (False, False, .1), 'frozen_mutator_ablation': (False, True, 0)}[group]
    if (static, frozen, cfg['anchor_ratio']) != expected:
        raise ValueError(f'Run configuration does not match group {group}')
    if report['test_size'] != 20 or report['manifest'].get('initial_adapter') is not None:
        raise ValueError('Release experiment expects 20 test problems and no warm-start adapter')
    for key in METRICS:
        val = report[key]
        if not isinstance(val, (float, int)) or not math.isfinite(val) or not 0 <= val <= 1:
            raise ValueError(f'Invalid metric {key}: {val}')
    return cfg


def collect_runs(paths, group, audit=None, expected_seeds=(42, 43, 44), base_root=ROOT):
    runs = {}
    for path in sorted(paths):
        raw = json.loads(path.read_text())
        cfg = validate_report(raw, group)
        seed = cfg['seed']
        if seed in runs:
            raise ValueError(f'Duplicate seed {seed} in {group}; choose runs explicitly')
        digest = file_sha256(path)
        evidence = (audit or {}).get('runs', {}).get(group, {}).get(str(seed), {})
        if evidence and evidence['final_report_sha256'] != digest:
            raise ValueError(f'Checkpoint audit does not match report {path}')
        telem = raw.get('training_telemetry', {})
        updates = evidence.get('breaker_updates_last', telem.get('actual_breaker_updates'))
        if evidence and telem.get('actual_breaker_updates', updates) != updates:
            raise ValueError('Report telemetry disagrees with checkpoint audit')
        metrics_path = path.with_name(path.stem + '.metrics.jsonl') if path.name.startswith('seed') else training_metrics_path(path.parent)
        if not metrics_path.is_file():
            raise FileNotFoundError(f'Missing validation trajectory: {metrics_path}')
        trajectory = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if not trajectory or trajectory[-1]['step'] != 75:
            raise ValueError('This release table requires complete 75-step experiments')
        best = max(row['val_greedy_pass1'] for row in trajectory)
        chosen = next(row['step'] for row in trajectory if row['val_greedy_pass1'] == best)
        if chosen != raw['selected_step'] or best != raw['best_val_greedy_pass1']:
            raise ValueError('Selection does not match earliest best validation checkpoint')
        runs[seed] = {
            'seed': seed, 'source_report': str(path.relative_to(base_root)) if path.is_relative_to(base_root) else str(path),
            'source_sha256': digest, 'manifest': raw['manifest'], 'test_decoding': raw['test_decoding'],
            'selected_step': raw['selected_step'], **{k: raw[k] for k in METRICS},
            'pool_size': evidence.get('last_pool_size'), 'trajectory': trajectory,
            'training_telemetry': telem,
            'cost': {'breaker_updates': updates, 'breaker_updates_best': evidence.get('breaker_updates_best'),
                     'mutator_updates': evidence.get('mutator_updates_last', telem.get('actual_mutator_updates')),
                     'breaker_candidates': telem.get('total_breaker_generations'),
                     'mutator_candidates': telem.get('total_mutator_generations'),
                     'walltime_seconds': telem.get('total_training_walltime_sec')},
        }
    if set(runs) != set(expected_seeds):
        raise ValueError(f'{group}: expected seeds {list(expected_seeds)}, found {sorted(runs)}')
    return runs


def group_summary(runs):
    rows = list(runs.values())
    result = {'seeds': sorted(runs), 'num_seeds': len(runs), 'runs': runs,
              'val_best': calc_stats([r['best_val_greedy_pass1'] for r in rows])}
    for key in ('test_greedy_pass1', 'test_sample_pass4', 'test_hit_rate', 'pool_size'):
        vals = [r[key] for r in rows]
        result[key] = calc_stats(vals) if all(v is not None for v in vals) else (None, None)
    result['cost'] = {}
    for key in rows[0]['cost']:
        vals = [r['cost'][key] for r in rows]
        result['cost'][key] = calc_stats(vals) if all(v is not None for v in vals) else (None, None)
    return result


def build_summary(root=ROOT, source='archive'):
    audit_path = root/'reports/checkpoint_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    result = {'schema_version': 2, 'protocol_version': 'numeric-selfplay-v2',
              'statistics': 'Mean and sample standard deviation (ddof=1); not confidence intervals.',
              'budget_note': '75 outer steps each; static has 75 Breaker opportunities, other groups have 70 plus optional rescue. Not matched compute.',
              'evaluation_note': 'Historical 20-problem test benchmark. Base variation is decoding-only; trained-run variation includes training and decoding.',
              'base_model': json.loads((root/'reports/base_benchmark_eval.json').read_text())}
    signatures = []
    for key, (_, folder) in GROUPS.items():
        if source == 'archive':
            paths = list((root/'reports/runs'/key).glob('seed*.json'))
        else:
            paths = list((root/folder).glob('*/final_test_step_75.json'))
        runs = collect_runs(paths, key, audit, base_root=root)
        for run in runs.values():
            manifest = run['manifest']
            cfg = manifest['config']
            signatures.append((cfg['model_name'], cfg['breaker_rollouts'], cfg['temperature'], cfg['kl_coef'],
                               cfg['lr_breaker'], cfg['lr_breaker_min'], cfg['cosine_decay_start'], cfg['eval_interval'],
                               tuple(manifest['source_sha256'][p] for p in JUDGE_FILES),
                               json.dumps(run['test_decoding'], sort_keys=True)))
        result[key] = group_summary(runs)
    if len(set(signatures)) != 1:
        raise ValueError('Model, judge, data, decoding, or shared Breaker settings differ across runs')
    return result


def format_stat(values, percent=False):
    mean, std = values
    if mean is None:
        return 'N/A (not recorded)'
    scale = 100 if percent else 1
    return f'{mean*scale:.2f}' + (f' ± {std*scale:.2f}' if std is not None else '') + ('%' if percent else '')


def render_markdown(summary):
    base = summary['base_model']
    test = base['test_benchmark']
    lines = ['# nano-CodeSelfPlay · v2 results', '',
             'Twelve completed runs: four configurations × training seeds 42, 43, 44. Each model is selected by validation Greedy Pass@1, keeping the earliest tie.', '',
             'Mean ± **sample SD (ddof=1)**. SD is not a confidence interval. The test benchmark contains 20 historically inspected problems; it is not a newly unseen blind set.', '',
             '| Method | Validation best | Test greedy | Test Pass@4 | Candidate hit rate |',
             '| :--- | ---: | ---: | ---: | ---: |',
             f"| Base | {base['validation_benchmark']['greedy_pass1']:.2%} | {test['greedy_pass1']:.2%} | {format_stat(calc_stats(list(test['sample_pass4_by_seed'].values())), True)} | {format_stat(calc_stats(list(test['hit_density_by_seed'].values())), True)} |"]
    for key, (label, _) in GROUPS.items():
        g = summary[key]
        lines.append('| ' + label + ' | ' + ' | '.join(format_stat(g[k], True) for k in ['val_best', 'test_greedy_pass1', 'test_sample_pass4', 'test_hit_rate']) + ' |')
    lines += ['', 'Base sampling seeds repeat decoding of one pretrained model; other rows vary training and final decoding. Mixed replay uses a configured selection probability of 0.1, not a measured 10% share.', '',
              '## Training cost', '', summary['budget_note'], '',
              '| Method | Breaker updates at last | Breaker candidates | Mutator candidates | Wall time (s) |',
              '| :--- | ---: | ---: | ---: | ---: |']
    for key, (label, _) in GROUPS.items():
        g = summary[key]
        lines.append('| ' + label + ' | ' + ' | '.join(format_stat(g['cost'][k]) for k in ['breaker_updates','breaker_candidates','mutator_candidates','walltime_seconds']) + ' |')
    lines += ['', 'Update counts are checked against AdamW step counters in [checkpoint_audit.json](checkpoint_audit.json). Missing historical counters/time are N/A, not estimates. Static wall time covers the training loop, periodic validation and checkpoint saves; it excludes loading, the initial baseline audit and final test.', '',
              'A frozen Mutator still generates new candidates throughout training. The configured 75 × 4 = 300 Mutator candidates must not be described as 20. Old runs did not record all Breaker probe/rescue generation counts.', '', '## Per-seed results', '']
    for key, (label, _) in GROUPS.items():
        lines += ['### ' + label, '', '| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |', '| ---: | ---: | ---: | ---: | ---: | :--- |']
        for seed, r in sorted(summary[key]['runs'].items()):
            c = r['cost']
            lines.append(f"| {seed} | {r['selected_step']} | {r['test_greedy_pass1']:.2%} | {r['test_sample_pass4']:.2%} | {r['test_hit_rate']:.2%} | {c['breaker_updates_best']} / {c['breaker_updates']} |")
        lines += ['']
    lines += ['## Provenance', '', 'Raw final reports and validation trajectories are bundled in [runs/](runs/), including training/data fingerprints and decoding settings. The original model weights remain untracked. Rebuild this file with `python scripts/aggregate_all_reports.py`.', '']
    return '\n'.join(lines)



def render_readme_results(summary):
    test = summary['base_model']['test_benchmark']
    rows = ['| 方案 | 测试 Greedy Pass@1 | 测试 Sample Pass@4 | 候选命中率 |',
            '| :--- | ---: | ---: | ---: |',
            f"| Base | {test['greedy_pass1']:.2%} | {format_stat(calc_stats(list(test['sample_pass4_by_seed'].values())), True)} | {format_stat(calc_stats(list(test['hit_density_by_seed'].values())), True)} |"]
    for key, (label, _) in GROUPS.items():
        rows.append('| ' + label + ' | ' + ' | '.join(format_stat(summary[key][k], True)
                    for k in ['test_greedy_pass1','test_sample_pass4','test_hit_rate']) + ' |')
    return '\n'.join(rows)


def render_readme_cost(summary):
    rows = ['| 方案 | Breaker 有效更新（完整运行） | Breaker 候选数 | Mutator 候选数 | 耗时（秒） |',
            '| :--- | ---: | ---: | ---: | ---: |']
    for key, (label, _) in GROUPS.items():
        rows.append('| ' + label + ' | ' + ' | '.join(format_stat(summary[key]['cost'][k])
                    for k in ['breaker_updates','breaker_candidates','mutator_candidates','walltime_seconds']) + ' |')
    return '\n'.join(rows)


def update_readme(path, summary):
    import re
    content = path.read_text()
    for name, table in [('results', render_readme_results(summary)), ('cost', render_readme_cost(summary))]:
        pattern = rf'<!-- {name}:start -->.*?<!-- {name}:end -->'
        content, n = re.subn(pattern, f'<!-- {name}:start -->\n{table}\n<!-- {name}:end -->', content, flags=re.S)
        if n != 1:
            raise ValueError(f'Expected exactly one README marker for {name}')
    path.write_text(content)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-readme', action='store_true', help='Refresh only the marked result/cost tables')
    parser.add_argument('--source', choices=['archive','runs'], default='archive', help='Default: published raw reports; runs: completed local checkpoints')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary = build_summary(args.root.resolve(), args.source)
    output = args.output_dir or args.root/'reports'
    output.mkdir(parents=True, exist_ok=True)
    (output/'v2_experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'v2_experiment_summary.md').write_text(render_markdown(summary))
    if args.update_readme:
        update_readme(args.root/'README.md', summary)
    print(f'Rebuilt validated 12-run summary in {output}')


if __name__ == '__main__':
    main()
