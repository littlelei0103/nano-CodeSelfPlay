"""Regressions for batch/point semantics, failure handling, selection, and resume."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

from core.experiment import (
    PROTOCOL_VERSION, experiment_manifest, finite_json, read_training_state,
    restore_training_state, validate_resume,
)
from core.grpo_engine import SelfPlayGRPOTrainer
from core.hard_buffer import HardBuffer
from core.sandbox import execute_batch_safely, evaluate_breaker_counterexample, validate_mutation
import train_selfplay
import eval_selfplay


class TestExecutionConsistency(unittest.TestCase):
    ref = "def f_ref(x):\n    return x\n"

    def test_mutable_defaults_cannot_create_batch_only_witnesses(self):
        mut = "def f_mut(x, state=[0]):\n    state[0] += 1\n    return x if state[0] <= 3 else x + 1\n"
        results = execute_batch_safely(mut, "f_mut", list(range(7)))
        self.assertEqual([r["result"] for r in results], list(range(7)))
        valid, _, details = validate_mutation(self.ref, mut, [0, 1, 2], (0, 6))
        self.assertFalse(valid)
        self.assertEqual(details["witnesses"], [])

    def test_each_reported_witness_reproduces_independently(self):
        mut = "def f_mut(x, state=[0]):\n    state[0] += 1\n    return x + (1 if x == 4 else 0)\n"
        valid, _, details = validate_mutation(self.ref, mut, [0, 1, 2], (0, 6))
        self.assertTrue(valid)
        self.assertEqual(details["witnesses"], [4])
        for x in details["witnesses"]:
            self.assertTrue(evaluate_breaker_counterexample(self.ref, mut, x, (0, 6))[0])

    def test_batch_timeout_is_not_a_witness(self):
        mut = "def f_mut(x):\n    return sum(range(1000000000)) if x == 3 else x\n"
        valid, reason, details = validate_mutation(self.ref, mut, [0, 1, 2], (0, 6), timeout_sec=0.3)
        self.assertFalse(valid, reason)
        self.assertEqual(details["witnesses"], [])
        self.assertIn("unavailable", reason)
        self.assertFalse(evaluate_breaker_counterexample(self.ref, mut, 3, (0, 6), timeout_sec=0.3)[0])

    def test_real_runtime_exception_still_counts(self):
        mut = "def f_mut(x):\n    return 1 // 0 if x == 3 else x\n"
        valid, _, details = validate_mutation(self.ref, mut, [0, 1, 2], (0, 6))
        self.assertTrue(valid)
        self.assertEqual(details["witnesses"], [3])
        self.assertTrue(evaluate_breaker_counterexample(self.ref, mut, 3, (0, 6))[0])

    def test_bad_reference_and_unsupported_result_never_rewarded(self):
        bad_reference = "def f_ref(x):\n    return 1 // 0\n"
        self.assertFalse(evaluate_breaker_counterexample(bad_reference, self.ref, 1, (0, 6))[0])
        unsupported = "def f_mut(x):\n    return [x] if x == 3 else x\n"
        self.assertFalse(validate_mutation(self.ref, unsupported, [0, 1, 2], (0, 6))[0])
        self.assertFalse(evaluate_breaker_counterexample(self.ref, unsupported, 3, (0, 6))[0])

    def test_malformed_worker_response_is_infrastructure_failure(self):
        import subprocess
        proc = subprocess.CompletedProcess([], 0, '[{"success":true,"result":1}]', '')
        with patch("core.sandbox.subprocess.run", return_value=proc):
            rows = execute_batch_safely(self.ref, "f_ref", [1, 2])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["error_kind"] == "infrastructure" for r in rows))

    def test_non_integer_inputs_are_not_truncated(self):
        for value in (3.9, True, float('nan'), float('inf')):
            self.assertFalse(evaluate_breaker_counterexample(self.ref, self.ref, value, (0, 6))[0])


class TestBufferAccounting(unittest.TestCase):
    def test_draw_evaluation_and_candidate_counts_are_distinct(self):
        buf = HardBuffer()
        buf.add("def f(x): return x", "def g(x): return x+1", candidate_hits=1, total_candidates=4)
        item = buf.sample()[0]
        buf.update_result(item["key"], candidate_hits=1, total_candidates=8)
        self.assertEqual(item["times_sampled"], 1)
        self.assertEqual(item["groups_evaluated"], 1)
        self.assertEqual(item["times_solved"], 1)
        self.assertAlmostEqual(item["solve_rate"], 2 / 12)
        self.assertEqual(buf.get_stats()["avg_group_solve_rate"], 1)
        with self.assertRaises(ValueError):
            buf.update_result(item["key"], solved=True)


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text('{}')

    def __call__(self, prompt, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[3, 4]])})

    def decode(self, tokens, **kwargs):
        return "invalid"


class TinyLifecycleModel(torch.nn.Module):
    """No downloads: main-loop orchestration uses tiny, serializable role weights."""
    def __init__(self):
        super().__init__()
        self.mutator = torch.nn.Parameter(torch.tensor(1.0))
        self.breaker = torch.nn.Parameter(torch.tensor(2.0))
        self.active_adapter = None

    def set_adapter(self, role):
        self.active_adapter = role
        self.mutator.requires_grad_(role == "mutator")
        self.breaker.requires_grad_(role == "breaker")

    def save_pretrained(self, path):
        for role in ("mutator", "breaker"):
            subdir = Path(path, role)
            subdir.mkdir()
            (subdir / "adapter_config.json").write_text('{}')
            torch.save(getattr(self, role).detach(), subdir / "weight.pt")

    def load_adapter(self, path, adapter_name):
        self.loaded_audit_adapter = (str(path), adapter_name)


def generated(model, config, input_ids, **kwargs):
    return torch.cat([input_ids, torch.full((len(input_ids), 1), 5)], dim=1)


class TestTrainingLifecycle(unittest.TestCase):
    def args(self, root, **changes):
        with patch('sys.argv', ['train_selfplay.py']):
            args = train_selfplay.parse_args()
        args.output_dir = root
        args.device = 'cpu'
        args.total_steps = 1
        args.eval_interval = 1
        args.mutator_warmup_steps = 5
        args.mut_rollouts = 2
        args.breaker_rollouts = 2
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def run_main(self, args, evaluations, trainer=None, buffer_class=HardBuffer):
        model = TinyLifecycleModel()
        seed = {"id": "seed", "func_src": "def f_ref(x): return x", "domain": (0, 6),
                "sanity_inputs": [0, 1, 2], "description": "identity", "family": "math"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(train_selfplay, 'parse_args', return_value=args))
            stack.enter_context(patch.object(train_selfplay, 'setup_lora_and_model', return_value=(TinyTokenizer(), model)))
            stack.enter_context(patch.object(train_selfplay, 'generate_with_policy', side_effect=generated))
            stack.enter_context(patch.object(train_selfplay, 'get_seed_functions', return_value=[seed]))
            stack.enter_context(patch.object(train_selfplay, 'get_validation_benchmark', return_value=[{"id": "val"}]))
            stack.enter_context(patch.object(train_selfplay, 'get_test_benchmark', return_value=[{"id": "test"}]))
            stack.enter_context(patch.object(train_selfplay, 'HardBuffer', buffer_class))
            evaluate = stack.enter_context(patch.object(train_selfplay, 'evaluate_on_heldout', side_effect=evaluations))
            if trainer is not None:
                stack.enter_context(patch.object(train_selfplay, 'SelfPlayGRPOTrainer', return_value=trainer))
            train_selfplay.main()
        return model, evaluate

    def test_improved_best_survives_resume_with_worse_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            self.run_main(args, [(.5, .5, .3), (.8, .9, .6), (.7, .8, .5)])
            run = next(Path(tmp).iterdir())
            for folder in ('checkpoint-last', 'checkpoint-round-1', 'checkpoint-best'):
                state = read_training_state(run / folder)
                self.assertEqual(state['best_val_pass1'], .8)
                self.assertEqual(state['best_step'], 1)
            # Run-level buffer and ahead-of-checkpoint logs are not resume truth.
            (run / 'hard_buffer.jsonl').write_text('not a checkpoint snapshot')
            with (run / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': 999, 'round': 999}) + '\n')
            args.resume_dir = str(run)
            args.total_steps = 2
            model, evaluate = self.run_main(args, [(.7, .8, .5), (.8, .9, .6)])
            state = read_training_state(run / 'checkpoint-last')
            self.assertEqual((state['step'], state['best_step'], state['best_val_pass1']), (2, 1, .8))
            self.assertEqual(read_training_state(run / 'checkpoint-best')['step'], 1)
            self.assertEqual(evaluate.call_args.kwargs['adapter_name'], 'breaker_final_audit')
            self.assertTrue(model.loaded_audit_adapter[0].endswith('checkpoint-best/breaker'))
            report = json.loads((run / 'final_test_step_2.json').read_text())
            self.assertEqual(report['selected_step'], 1)
            self.assertEqual(report['test_greedy_pass1'], .8)
            records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            self.assertNotIn(999, [record['step'] for record in records])

    def test_baseline_remains_best_when_training_regresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(self.args(tmp), [(.8, .9, .6), (.7, .8, .5), (.8, .9, .6)])
            run = next(Path(tmp).iterdir())
            self.assertEqual(read_training_state(run / 'checkpoint-best')['step'], 0)
            self.assertEqual(read_training_state(run / 'checkpoint-last')['best_val_pass1'], .8)

    def test_breaker_only_static_runs_without_mutator(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, breaker_only_static=True, total_steps=1)
            model, evaluate = self.run_main(args, [(.5, .5, .3), (.75, .8, .6), (.75, .8, .6)])
            run = next(Path(tmp).iterdir())
            report = json.loads((run / 'final_test_step_1.json').read_text())
            self.assertIn("training_telemetry", report)
            telemetry = report["training_telemetry"]
            self.assertEqual(telemetry["actual_mutator_updates"], 0)
            self.assertEqual(telemetry["total_mutator_generations"], 0)
            self.assertGreaterEqual(telemetry["total_breaker_generations"], 2)

    def test_command_line_help_renders(self):
        for module in (train_selfplay, eval_selfplay):
            with patch('sys.argv', ['program', '--help']), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    module.parse_args()
            self.assertEqual(raised.exception.code, 0)

    def test_changed_config_or_code_refuses_resume(self):
        args = self.args('/tmp/unused')
        manifest = experiment_manifest(args, 1)
        state = {'manifest': copy.deepcopy(manifest)}
        args.anchor_ratio = .5
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            validate_resume(state, experiment_manifest(args, 1))
        changed = copy.deepcopy(manifest)
        changed['source_sha256']['core/sandbox.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'code or dataset changed'):
            validate_resume(state, changed)

    def test_incomplete_state_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save({'step': 1}, Path(tmp, 'train_state.pt'))
            with self.assertRaisesRegex(ValueError, 'weights-only'):
                read_training_state(tmp)

    def test_zero_variance_rescue_reaches_optimizer(self):
        class PopulatedBuffer(HardBuffer):
            def __init__(self, capacity):
                super().__init__(capacity)
                self.add('def f_ref(x): return x', 'def f_mut(x): return x+1', candidate_hits=0)
        class Trainer:
            opt_breaker = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
            def __init__(self):
                self.effects = iter([False, False, True])
                self.steps = []
            def backward_grpo_group(self, **kwargs):
                return next(self.effects), 0, {}
            def step_optimizer(self, role):
                self.steps.append(role)
                return 1.0
        trainer = Trainer()
        with tempfile.TemporaryDirectory() as tmp, patch.object(train_selfplay.random, 'random', return_value=.99):
            args = self.args(tmp, anchor_ratio=.1, mutator_warmup_steps=0)
            self.run_main(args, [(.5,.5,.3),(.6,.7,.4),(.6,.7,.4)], trainer, PopulatedBuffer)
            self.assertIn('breaker', trainer.steps)
            run = next(Path(tmp).iterdir())
            record = json.loads((run / 'metrics.jsonl').read_text().splitlines()[-1])
            self.assertTrue(record['brk_effective_update'])
            self.assertEqual(record['brk_total_candidates'], 4)


class TestCrossPlayMissingData(unittest.TestCase):
    def test_empty_verified_suite_is_na_and_auditable(self):
        seed = {"id": "identity", "func_src": "def f_ref(x): return x", "domain": (0, 6),
                "sanity_inputs": [0, 1, 2], "description": "identity"}
        audit = {}
        with patch.object(eval_selfplay, 'generate_with_policy', side_effect=generated), contextlib.redirect_stdout(io.StringIO()):
            solve, deception, vmr, deception0 = eval_selfplay.run_cross_play_evaluation(
                TinyLifecycleModel(), TinyTokenizer(), [0, 1], [seed], 'cpu', audit)
        self.assertTrue(np.isnan(solve).all())
        self.assertTrue(np.isnan(deception).all())
        self.assertEqual(vmr, {0: 0, 1: 0})
        self.assertTrue(np.isnan(deception0[0]))
        self.assertEqual(audit['suite_sizes'], {0: 0, 1: 0})
        self.assertEqual(audit['total_candidates'], {0: 4, 1: 4})
        encoded = json.dumps(finite_json({'matrix': solve.tolist()}), allow_nan=False)
        self.assertEqual(json.loads(encoded)['matrix'], [[None, None], [None, None]])
        with tempfile.TemporaryDirectory() as tmp:
            svg = str(Path(tmp, 'matrix.svg'))
            with contextlib.redirect_stdout(io.StringIO()):
                eval_selfplay.plot_crossplay_heatmap(solve, [0, 1], str(Path(tmp, 'matrix.png')), svg)
            self.assertIn('N/A', Path(svg).read_text())

    def test_nonempty_suite_is_shared_and_reports_real_denominators(self):
        model = TinyLifecycleModel()
        class Tokenizer(TinyTokenizer):
            def decode(self, tokens, **kwargs):
                return 'def f_mut(x):\n    return x <= 3' if model.active_adapter.startswith('mutator') else '3'
        seed = {"id": "boundary", "func_src": "def f_ref(x): return x < 3", "domain": (0, 6),
                "sanity_inputs": [0, 6], "description": "threshold"}
        audit = {}
        with patch.object(eval_selfplay, 'generate_with_policy', side_effect=generated), contextlib.redirect_stdout(io.StringIO()):
            solve, deception, vmr, _ = eval_selfplay.run_cross_play_evaluation(
                model, Tokenizer(), [0, 1], [seed], 'cpu', audit)
        np.testing.assert_array_equal(solve, np.ones((2,2)))
        np.testing.assert_array_equal(deception, np.zeros((2,2)))
        self.assertEqual(vmr, {0: 1, 1: 1})
        self.assertEqual(audit['suite_sizes'], {0: 1, 1: 1})
        self.assertEqual(len(audit['mutation_trials']), 8)
        self.assertEqual(len(audit['breaker_trials']), 4)

    def test_eval_main_writes_strict_json_with_missing_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for r in (0, 1):
                (root / f'checkpoint-round-{r}').mkdir()
            args = argparse.Namespace(run_dir=tmp, output_dir=str(root/'output'), split='val', seed=42,
                                      device='cpu', base_model='tiny-test-model', num_eval_seeds=1)
            seed = {"id": "identity", "func_src": "def f_ref(x): return x", "domain": (0, 6),
                    "sanity_inputs": [0, 1, 2], "description": "identity"}
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(eval_selfplay, 'parse_args', return_value=args))
                stack.enter_context(patch.object(eval_selfplay, 'load_multi_round_model',
                                               return_value=(TinyTokenizer(), TinyLifecycleModel())))
                stack.enter_context(patch.object(eval_selfplay, 'get_validation_benchmark', return_value=[{'id':'val'}]))
                stack.enter_context(patch.object(eval_selfplay, 'evaluate_heldout_suite', return_value=(.5,.75,.4)))
                stack.enter_context(patch.object(eval_selfplay, 'get_seed_functions', return_value=[seed]))
                stack.enter_context(patch.object(eval_selfplay, 'generate_with_policy', side_effect=generated))
                eval_selfplay.main()
            raw = (root/'output/eval_results.json').read_text()
            self.assertNotIn('NaN', raw)
            report = json.loads(raw)
            self.assertEqual(report['solve_matrix'], [[None, None], [None, None]])
            self.assertEqual(report['pool_sizes'], [None, None])
            self.assertEqual(report['crossplay_audit']['suite_sizes'], {'0':0, '1':0})
            self.assertEqual(report['heldout_eval']['Validation Set']['size'], 1)
            self.assertEqual(report['training_manifest']['protocol_version'], 'legacy-unrecorded')
            self.assertIn('core/sandbox.py', report['evaluation_source_sha256'])



class TestRealDualLoraResume(unittest.TestCase):
    def test_two_role_update_isolation_and_exact_cpu_resume(self):
        torch.manual_seed(7)
        random.seed(7)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             max_position_embeddings=32, attention_dropout=0.0)
        base = Qwen2ForCausalLM(config)
        initial = copy.deepcopy(base.state_dict())
        def lora():
            return LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','v_proj'], lora_dropout=0.0, task_type='CAUSAL_LM')
        model = get_peft_model(base, lora(), adapter_name='breaker')
        model.add_adapter('mutator', lora())
        def optimizers(m):
            return [torch.optim.AdamW([p for n,p in m.named_parameters() if f'.{role}.' in n], lr=.01)
                    for role in ('mutator','breaker')]
        opts = optimizers(model)
        trainer = SelfPlayGRPOTrainer(model, TinyTokenizer(), *opts, device='cpu', kl_coef=.01)
        prompt = torch.tensor([[3,4]])
        samples = torch.tensor([[3,4,5,2], [3,4,6,2]])
        def update(m, t, role):
            m.set_adapter(role)
            m.train()
            before = {n:p.detach().clone() for n,p in m.named_parameters()}
            effective, _, _ = t.backward_grpo_group(prompt, samples, [1.0,0.0])
            self.assertTrue(effective)
            self.assertGreater(t.step_optimizer(role), 0)
            self.assertTrue(any(not torch.equal(p, before[n]) for n,p in m.named_parameters() if f'.{role}.' in n))
            other = 'mutator' if role == 'breaker' else 'breaker'
            self.assertTrue(all(torch.equal(p, before[n]) for n,p in m.named_parameters() if f'.{other}.' in n))
        update(model, trainer, 'mutator')
        update(model, trainer, 'breaker')
        with tempfile.TemporaryDirectory() as tmp:
            train_selfplay.save_checkpoint(model, TinyTokenizer(), tmp + '/checkpoint', *opts,
                                             step=1, best_val_pass1=.7)
            state = torch.load(tmp + '/checkpoint/train_state.pt', weights_only=True)
            expected_random = (random.random(), torch.rand(3))
            update(model, trainer, 'mutator')
            update(model, trainer, 'breaker')
            expected = copy.deepcopy(model.state_dict())
            base2 = Qwen2ForCausalLM(config)
            base2.load_state_dict(initial)
            restored = PeftModel.from_pretrained(base2, tmp+'/checkpoint/breaker', adapter_name='breaker', is_trainable=True)
            restored.load_adapter(tmp+'/checkpoint/mutator', adapter_name='mutator', is_trainable=True)
            opts2 = optimizers(restored)
            restore_training_state(state, *opts2)
            self.assertEqual(random.random(), expected_random[0])
            torch.testing.assert_close(torch.rand(3), expected_random[1], rtol=0, atol=0)
            trainer2 = SelfPlayGRPOTrainer(restored, TinyTokenizer(), *opts2, device='cpu', kl_coef=.01)
            update(restored, trainer2, 'mutator')
            update(restored, trainer2, 'breaker')
            for name, parameter in restored.state_dict().items():
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            for opt in opts2:
                self.assertTrue(opt.state)
                self.assertTrue(all(entry['step'] == 2 for entry in opt.state.values()))


if __name__ == '__main__':
    unittest.main()
