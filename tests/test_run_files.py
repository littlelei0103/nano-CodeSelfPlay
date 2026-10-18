"""Renamed entry points must still read historical experiment logs."""
from pathlib import Path
import tempfile
import unittest

from core.run_files import training_metrics_path


class TestRunFiles(unittest.TestCase):
    def test_new_runs_use_canonical_logs_and_old_runs_keep_their_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(training_metrics_path(root), root/'metrics.jsonl')
            legacy = root/'metrics_spin.jsonl'
            legacy.write_text('{"step": 1}\n')
            self.assertEqual(training_metrics_path(root), legacy)
            legacy.rename(root/'metrics.jsonl')
            self.assertEqual(training_metrics_path(root), root/'metrics.jsonl')

    def test_conflicting_logs_are_rejected_instead_of_silently_ignoring_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('metrics.jsonl', 'metrics_spin.jsonl'):
                (root/name).write_text('{"step": 1}\n')
            with self.assertRaisesRegex(ValueError, 'Ambiguous training logs'):
                training_metrics_path(root)
