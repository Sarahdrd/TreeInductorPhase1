"""Production-runner invariants; mock external calls, never fit scientific models."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from . import block8_full as b
from .checkpoints import IntegrityError

class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runner=b.Production()
    def test_exact_plans(self):
        self.assertEqual(len(b.plan()),5100);self.assertEqual(len(set(b.plan())),5100);self.assertEqual(len(b.plan(True)),68)
        self.assertEqual({k[2:] for k in b.plan(True)},{(42,1,1)});self.assertTrue(set(b.plan(True)).issubset(b.plan()))
    def test_shared_splits_and_representation(self):
        for dataset in self.runner.manifest['datasets']:
            ids=[self.runner.identity((dataset,m,42,1,1)) for m in b.MODELS]
            self.assertEqual(len({r['split_id'] for r in ids}),1)
            self.assertEqual(ids[0]['representation'],ids[2]['representation']);self.assertEqual(ids[1]['representation'],ids[3]['representation'])
    def test_default_resume_preserves_failures(self):
        for status in ['external_timeout','execution_failure','no_returned_tree','certified_optimal']:
            r=[dict(termination_status=status,execution_status='success' if status=='certified_optimal' else 'failure')]
            self.assertFalse(b.should_run(r));self.assertEqual(b.should_run(r,True),status!='certified_optimal')
        self.assertTrue(b.should_run([dict(termination_status='interrupted',execution_status='failure')]))
    def test_full_gate(self):
        with self.assertRaises(PermissionError):self.runner.run(first=False)
    def test_budget_and_timeout_no_score(self):
        key=b.plan(True)[0];r=dict(**self.runner.identity(key),execution_status='failure',prediction_available=False,training_macro_f1=None,validation_macro_f1=None)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(b,'external',return_value=dict(external_status='external_timeout',returncode=124,wall_seconds=120.,peak_job_memory_bytes=123)) as mock:
                self.runner.execute(key,r,Path(tmp));args,kwargs=mock.call_args
                self.assertEqual(args[0][4],'120');self.assertEqual(kwargs['seconds'],120);self.assertEqual(kwargs['memory_limit'],4*1024**3)
        self.assertFalse(r['prediction_available']);self.assertIsNone(r['validation_macro_f1']);self.assertEqual(r['termination_status'],'external_timeout')
    def test_invalid_failure_score_rejected(self):
        key=b.plan(True)[0];r=dict(**self.runner.identity(key),execution_status='failure',prediction_available=False,validation_macro_f1=0.,training_macro_f1=None)
        with patch.object(self.runner.store,'attempts',return_value=[r]):
            with self.assertRaises(IntegrityError):self.runner.attempts(key)
    def test_configuration_parameters(self):
        c=self.runner.manifest;self.assertEqual(c['cart_parameters']['max_depth'],2);self.assertEqual(c['policy']['murtree_internal_seconds'],120)
        self.assertEqual(c['policy']['quant']['certification'],'feasible but uncertified')
        self.assertIn('block8_process.py',c['source_sha256'])

if __name__=='__main__':unittest.main()
