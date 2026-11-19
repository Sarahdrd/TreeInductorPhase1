"""Runner failure/resume/persistence tests; fits are mocked, no benchmark fits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

from .checkpoints import IntegrityError, Store, atomic_write, read_verified, writer_lock
from .infrastructure import ROOT, digest, json_bytes
from .runner import Runner, parameters, plan


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runner = Runner('unit-test-only', store_base=self.temp.name)
        self.tasks = plan(datasets=['iris'], models=['dummy_most_frequent'], seeds=[42], repeats=[1], folds=[1, 2])

    def record(self, key, attempt=1, failed=False):
        _, _, _, expected = self.runner.context(key)
        now = datetime.now(timezone.utc).isoformat()
        return dict(expected, attempt=attempt, record_version='fold-attempt-v1',
                    execution_status='failed' if failed else 'success',
                    macro_f1=None if failed else .5, runtime_seconds=.1, warnings=[],
                    started_at=now, finished_at=now,
                    error=dict(type='test.Injected', message='test failure', traceback='test traceback') if failed else None)

    def test_inventory_and_dry_run_never_fit_or_generate(self):
        self.assertEqual(len(plan()), 20400)
        self.assertEqual(len(set(t[1] for t in plan())), 16)
        with patch('experiments.protocol.infrastructure.generate_splits', side_effect=AssertionError('No regeneration')):
            with patch.object(self.runner, 'evaluate', side_effect=AssertionError('No fitting')):
                status = self.runner.run(self.tasks)
        self.assertEqual(status['pending'], 2)
        self.assertFalse(self.runner.store.root.exists())
        for filters in ({'datasets':['abalone']}, {'seeds':[1]}, {'folds':[0]}, {'models':['ripper','ripper']}):
            with self.assertRaises(ValueError):
                plan(**filters)

    def test_safety_guards(self):
        for kwargs in ({}, {'max_new_folds':21}, {'max_new_folds':-1}, {'retry_failed':True, 'max_new_folds':1}):
            with self.assertRaises(ValueError):
                self.runner.run(self.tasks, execute=True, **kwargs)
        self.assertFalse(self.runner.store.root.exists())
        other = Runner('unit-test-only', official=True, store_base=self.temp.name)
        self.assertNotEqual(other.store.root, self.runner.store.root)

    def test_checkpoint_budget_restart_resume_and_duplicates(self):
        with patch.object(self.runner, 'evaluate', side_effect=self.record) as evaluate:
            first = self.runner.run(self.tasks, execute=True, max_new_folds=1)
            self.assertEqual(evaluate.call_count, 1)
        self.assertEqual((first['completed'], first['pending']), (1, 1))
        restarted = Runner('unit-test-only', store_base=self.temp.name)
        with patch.object(restarted, 'evaluate', side_effect=self.record) as evaluate:
            resumed = restarted.run(self.tasks, execute=True, resume=True, max_new_folds=1)
            self.assertEqual(evaluate.call_count, 1)
        self.assertEqual((resumed['completed'], resumed['skipped']), (2, 1))
        with patch.object(restarted, 'evaluate', side_effect=AssertionError('Must skip')):
            repeated = restarted.run(self.tasks, execute=True, resume=True, max_new_folds=2)
        self.assertEqual((repeated['executed'], repeated['skipped']), (0, 2))
        with self.assertRaises(ValueError):
            restarted.run(self.tasks, execute=True, max_new_folds=1)
        with self.assertRaises(IntegrityError):
            restarted.store.save(self.tasks[0], self.record(self.tasks[0], 2))

    def test_failed_requires_explicit_retry_and_preserves_attempt(self):
        key = self.tasks[0]
        with patch.object(self.runner, 'evaluate', side_effect=lambda k,a: self.record(k,a,True)):
            failed = self.runner.run([key], execute=True, max_new_folds=1)
        self.assertEqual(failed['failed'], 1)
        with patch.object(self.runner, 'evaluate', side_effect=AssertionError('Do not retry implicitly')):
            skipped = self.runner.run([key], execute=True, resume=True, max_new_folds=1)
        self.assertEqual(skipped['skipped'], 1)
        with patch.object(self.runner, 'evaluate', side_effect=self.record):
            final = self.runner.run([key], execute=True, resume=True, retry_failed=True, max_new_folds=1)
        self.assertEqual(final['completed'], 1)
        attempts = self.runner.store.attempts(key)
        self.assertEqual([r['execution_status'] for r in attempts], ['failed','success'])

    def test_corruption_provenance_score_and_partial_files(self):
        key = self.tasks[0]
        with patch.object(self.runner, 'evaluate', side_effect=self.record):
            self.runner.run([key], execute=True, max_new_folds=1)
        path = self.runner.store.folder(key) / 'attempt_000001.json'
        original = path.read_bytes()
        for mutation in ('truncated', 'checksum', 'provenance', 'score'):
            path.write_bytes(original)
            if mutation == 'truncated':
                path.write_text('{')
            else:
                envelope = json.loads(original)
                envelope['payload']['macro_f1'] = 2. if mutation == 'score' else .4
                if mutation == 'provenance':
                    envelope['payload']['environment_id'] = 'other'
                if mutation != 'checksum':
                    envelope['sha256'] = digest(json_bytes(envelope['payload']))
                path.write_text(json.dumps(envelope))
            summary, _ = self.runner.audit([key])
            self.assertEqual(summary['corrupt'], 1)
            with patch.object(self.runner, 'evaluate', side_effect=AssertionError('Never refit corrupt')):
                with self.assertRaises(IntegrityError):
                    self.runner.run([key], execute=True, resume=True, retry_failed=True, max_new_folds=1)
        path.write_bytes(original)
        folder = self.runner.store.folder(self.tasks[1]); folder.mkdir(parents=True)
        (folder / 'interrupted.tmp').write_text('partial')
        summary, _ = self.runner.audit(self.tasks)
        self.assertEqual((summary['pending'], summary['orphan_temporary_files']), (1, 1))

    def test_atomic_interruption_and_writer_exclusion(self):
        root = Path(self.temp.name) / 'atomic'
        with writer_lock(root):
            with self.assertRaises(RuntimeError):
                with writer_lock(root):
                    pass
            path = root / 'result.json'
            with patch('experiments.protocol.checkpoints.os.rename', side_effect=OSError('injected interruption')):
                with self.assertRaises(OSError):
                    atomic_write(path, {'value':1})
            self.assertFalse(path.exists())
            atomic_write(path, {'value':1})
            self.assertEqual(read_verified(path), {'value':1})
            with self.assertRaises(FileExistsError):
                atomic_write(path, {'value':2})
        with writer_lock(root):
            pass

    def test_keyboard_interrupt_keeps_committed_fold_and_unlocks(self):
        def interrupt(key, attempt):
            if key == self.tasks[1]:
                raise KeyboardInterrupt()
            return self.record(key, attempt)
        with patch.object(self.runner, 'evaluate', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.run(self.tasks, execute=True, max_new_folds=2)
        summary, _ = self.runner.audit(self.tasks)
        self.assertEqual((summary['completed'], summary['pending']), (1, 1))
        with writer_lock(self.runner.store.root):
            pass

    def test_fold_fit_uses_only_saved_indices_and_records_failure(self):
        key = self.tasks[0]
        obj, (train, valid), meta, expected = self.runner.context(key)
        with patch.object(DummyClassifier, 'fit', autospec=True, side_effect=RuntimeError('injected fit failure')):
            failure = self.runner.evaluate(key, 1)
        self.assertEqual(failure['execution_status'], 'failed')
        self.assertIsNone(failure['macro_f1'])
        self.assertIn('RuntimeError', failure['error']['traceback'])
        with patch.object(DummyClassifier, 'fit', autospec=True) as fit:
            with patch.object(DummyClassifier, 'predict', autospec=True, return_value=np.zeros(len(valid), dtype=int)) as predict:
                success = self.runner.evaluate(key, 1)
        np.testing.assert_array_equal(fit.call_args.args[1].index, obj['X'].iloc[train].index)
        np.testing.assert_array_equal(fit.call_args.args[2], obj['y'][train])
        np.testing.assert_array_equal(predict.call_args.args[1].index, obj['X'].iloc[valid].index)
        self.assertEqual(success['execution_status'], 'success')
        self.runner.validate_attempt(success, expected, meta)

    def test_changed_run_manifest_refused(self):
        with patch.object(self.runner, 'evaluate', side_effect=self.record):
            self.runner.run(self.tasks[:1], execute=True, max_new_folds=1)
        for key in ('environment', 'config_sha256', 'source_sha256'):
            manifest = dict(self.runner.manifest, **{key: 'changed'})
            with self.assertRaises(IntegrityError):
                Store(self.runner.store.root, manifest).check_manifest()

    def test_all_model_descriptors_stable_no_fit(self):
        for key in plan(datasets=['iris'], seeds=[42], repeats=[1], folds=[1]):
            _, _, _, a = self.runner.context(key)
            fresh = Runner('unit-test-only', store_base=self.temp.name)
            _, _, _, b = fresh.context(key)
            self.assertEqual(a['configuration_id'], b['configuration_id'])
            self.assertEqual(a['evaluation_id'], b['evaluation_id'])


if __name__ == '__main__':
    unittest.main()
