"""Release regressions: static/frozen routing, cumulative telemetry, and honest reports."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_review_regressions as fixtures
from core.hard_buffer import HardBuffer
import train_selfplay
from scripts.aggregate_all_reports import ROOT, build_summary, calc_stats, collect_runs, update_readme
from scripts.plot_comparison import prepare_plot_data


class ControlledTrainer:
    """Force effective gradients to test main-loop routing independently of reward noise."""
    def __init__(self, model, tokenizer, optimizer_mutator, optimizer_breaker, **kwargs):
        self.model = model
        self.opt_mutator = optimizer_mutator
        self.opt_breaker = optimizer_breaker

    def backward_grpo_group(self, **kwargs):
        getattr(self.model, self.model.active_adapter).square().backward()
        return True, 0.0, {}

    def step_optimizer(self, role):
        opt = self.opt_mutator if role == 'mutator' else self.opt_breaker
        opt.step()
        opt.zero_grad(set_to_none=True)
        return 1.0


class PopulatedBuffer(HardBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.add('def f_ref(x): return x', 'def f_mut(x): return x+1', candidate_hits=0)


class TestReleaseTraining(unittest.TestCase):
    def run_controlled(self, args, buffer=HardBuffer):
        with patch.object(train_selfplay, 'SelfPlayGRPOTrainer', ControlledTrainer):
            fixtures.TestTrainingLifecycle().run_main(args, [(.5,.5,.3)]*30, buffer_class=buffer)

    def test_static_skips_all_mutator_generation_and_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixtures.TestTrainingLifecycle().args(tmp, total_steps=3, breaker_only_static=True)
            with patch.object(train_selfplay, 'format_mutator_prompt', side_effect=AssertionError('Mutator executed')):
                self.run_controlled(args)
            run = next(Path(tmp).iterdir())
            state = torch.load(run/'checkpoint-last/train_state.pt', weights_only=True)
            self.assertEqual(state['optimizer_mutator']['state'], {})
            self.assertEqual(state['hard_buffer_items'], {})
            initial = torch.load(run/'checkpoint-round-0/mutator/weight.pt', weights_only=True)
            final = torch.load(run/'checkpoint-last/mutator/weight.pt', weights_only=True)
            self.assertTrue(torch.equal(initial, final))
            t = state['training_telemetry']
            self.assertEqual(t['actual_breaker_updates'], 3)
            self.assertEqual(t['total_breaker_generations'], 6)
            self.assertEqual(t['total_mutator_generations'], 0)

    def test_frozen_weights_stay_fixed_but_generation_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixtures.TestTrainingLifecycle().args(tmp, total_steps=7, freeze_mutator_after_warmup=True)
            self.run_controlled(args, PopulatedBuffer)
            run = next(Path(tmp).iterdir())
            weights = [torch.load(run/f'checkpoint-round-{s}/mutator/weight.pt', weights_only=True) for s in (5,6,7)]
            self.assertTrue(torch.equal(weights[0], weights[1]) and torch.equal(weights[1], weights[2]))
            state = torch.load(run/'checkpoint-last/train_state.pt', weights_only=True)
            t = state['training_telemetry']
            self.assertEqual(t['actual_mutator_updates'], 5)
            self.assertEqual(t['actual_breaker_updates'], 2)
            self.assertEqual(t['total_mutator_generations'], 14)

    def test_resume_and_audit_only_resume_preserve_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixtures.TestTrainingLifecycle().args(tmp, total_steps=2, breaker_only_static=True)
            self.run_controlled(args)
            run = next(Path(tmp).iterdir())
            # Historical log names must continue in place, without splitting the run.
            (run/'metrics.jsonl').rename(run/'metrics_spin.jsonl')
            args.resume_dir = str(run)
            args.total_steps = 4
            self.run_controlled(args)
            state = torch.load(run/'checkpoint-last/train_state.pt', weights_only=True)
            t = state['training_telemetry']
            self.assertEqual(t['actual_breaker_updates'], 4)
            self.assertEqual(t['total_breaker_generations'], 8)
            self.run_controlled(args)  # No new steps: evaluate the existing selection only.
            report = json.loads((run/'final_test_step_4.json').read_text())
            self.assertEqual(report['training_telemetry']['actual_breaker_updates'], 4)
            self.assertEqual(report['training_telemetry']['total_breaker_generations'], 8)
            self.assertFalse((run/'metrics.jsonl').exists())
            self.assertTrue((run/'metrics_spin.jsonl').is_file())

    def test_breaker_probes_are_included_in_generation_totals(self):
        class ValidCodeTokenizer(fixtures.TinyTokenizer):
            def decode(self, tokens, **kwargs):
                return 'def f_mut(x):\n    return x + (1 if x == 4 else 0)'
        details = {'sanity_passed': True, 'witnesses':[4], 'sparsity':1/7,
                   'clean_f_ref_src':'def f_ref(x): return x',
                   'clean_f_mut_src':'def f_mut(x): return x + (1 if x == 4 else 0)'}
        with tempfile.TemporaryDirectory() as tmp:
            args = fixtures.TestTrainingLifecycle().args(tmp, total_steps=1, mutator_warmup_steps=0)
            with patch.object(fixtures, 'TinyTokenizer', ValidCodeTokenizer), \
                 patch.object(train_selfplay, 'validate_mutation', return_value=(True,'valid_mutation',details)), \
                 patch.object(train_selfplay, 'compute_breaker_reward', return_value=(0.0,{})):
                self.run_controlled(args)
            run = next(Path(tmp).iterdir())
            state = torch.load(run/'checkpoint-last/train_state.pt', weights_only=True)
            t = state['training_telemetry']
            self.assertEqual(t['breaker_probe_candidates'], 8)  # Two mutations × four probes.
            self.assertEqual(t['breaker_training_candidates'], 2)
            self.assertEqual(t['total_breaker_generations'], 10)


class TestReleaseReports(unittest.TestCase):
    def test_unmeasured_statistics_are_not_zero(self):
        self.assertEqual(calc_stats([]), (None,None))
        self.assertEqual(calc_stats([1]), (1,None))
        self.assertEqual(calc_stats([0,1,2]), (1,1))

    def test_duplicate_and_missing_seeds_are_rejected(self):
        source = ROOT/'reports/runs/pure_selfplay/seed42.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('seed42.json','seed42copy.json'):
                shutil.copyfile(source, root/name)
            shutil.copyfile(source.with_suffix('.metrics.jsonl'), root/'seed42.metrics.jsonl')
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                collect_runs(list(root.glob('*.json')), 'pure_selfplay')
            with self.assertRaisesRegex(ValueError, 'expected seeds'):
                collect_runs([root/'seed42.json'], 'pure_selfplay')

    def test_archive_rebuild_needs_no_ignored_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT/'reports/runs', root/'reports/runs')
            for name in ('base_benchmark_eval.json','checkpoint_audit.json'):
                shutil.copyfile(ROOT/'reports'/name, root/'reports'/name)
            summary = build_summary(root)
            self.assertEqual(summary['pure_selfplay']['num_seeds'], 3)
            self.assertAlmostEqual(summary['breaker_only_static']['test_greedy_pass1'][0], 19/30)
            self.assertEqual(summary['pure_selfplay']['cost']['walltime_seconds'], (None,None))
            self.assertNotEqual(summary['frozen_mutator_ablation']['cost']['breaker_updates'][0], 48)

    def test_plots_use_raw_values_and_sample_standard_deviation(self):
        summary = build_summary()
        data = prepare_plot_data(summary)
        self.assertEqual(data['test']['breaker_only_static']['test_greedy_pass1'], [.55,.60,.75])
        self.assertEqual(data['trajectories']['frozen_mutator_ablation']['steps'], [0,15,30,45,60,75])
        bad = copy.deepcopy(summary)
        bad['pure_selfplay']['test_greedy_pass1'] = [1.0,0.0]
        with self.assertRaisesRegex(ValueError, 'aggregate and raw'):
            prepare_plot_data(bad)
        del bad['breaker_only_static']
        with self.assertRaises(KeyError):
            prepare_plot_data(bad)

    def test_readme_refresh_only_updates_marked_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'README.md'
            path.write_text('keep\n<!-- results:start -->old<!-- results:end -->\nend')
            update_readme(path, build_summary())
            self.assertTrue(path.read_text().startswith('keep\n'))
            self.assertTrue(path.read_text().endswith('\nend'))
            self.assertIn('63.33', path.read_text())
            self.assertNotIn('cost:start', path.read_text())
            self.assertNotIn('N/A (not recorded)', path.read_text())
