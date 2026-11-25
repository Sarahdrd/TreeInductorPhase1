"""Block 7 invariants; synthetic scores are test-only and never exported as results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import block7 as b7
from . import block7_analysis as analysis
from .checkpoints import Store,IntegrityError,writer_lock
from .infrastructure import generate_splits,digest,config
from .models import make_model


class Block7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner=b7.MagicRunner(pilot=True)

    def test_source_and_order(self):
        X,y,m=b7.inspect_data();self.assertEqual(X.shape,(19020,10));self.assertEqual(y.value_counts().to_dict(),{'g':12332,'h':6688});self.assertEqual(m['duplicate_full_rows'],115)
        self.assertEqual(X.iloc[0,0],28.7967);self.assertEqual(digest(b7.SOURCE.read_bytes()),b7.SHA)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.data';p.write_bytes(b7.SOURCE.read_bytes()+b'\n')
            with self.assertRaises(IntegrityError):b7.inspect_data(p)

    def test_nested_saved_indices_and_proportions(self):
        self.assertFalse(set(b7.SUBSEEDS)&set(b7.CVSEEDS));seen=[]
        for seed in b7.SUBSEEDS:
            regenerated=b7.nested_indices(self.runner.y,seed);previous=set()
            for n in b7.SIZES:
                meta,ids,X,y,splits=self.runner.context[f's{seed}_n{n}'];np.testing.assert_array_equal(ids,regenerated[n]);self.assertTrue(previous.issubset(set(ids)));previous=set(ids)
                self.assertEqual(len(ids),n)
                for label,count in meta['class_counts'].items():self.assertLessEqual(abs(count-n*self.runner.dataset['class_counts'][label]/19020),1)
            seen.append(tuple(regenerated[200]))
        self.assertEqual(len(set(seen)),5)

    def test_saved_cv_reproducibility(self):
        for name in ['s101_n200','s202_n500','s505_n10000']:
            meta,ids,X,y,splits=self.runner.context[name]
            for seed in b7.CVSEEDS:
                original=generate_splits(y,seed)
                for (a,b),(c,d) in zip(splits[seed][0],original):
                    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d);self.assertEqual(len(set(ids[a])&set(ids[b])),0)
        for name in self.runner.context:
            identities=[self.runner.identity((name,m,42,1,1)) for m in b7.MODELS]
            self.assertEqual(len({r['split_id'] for r in identities}),1)

    def test_model_configuration_and_leakage(self):
        X=pd.DataFrame({'x':[1.,2.,3.,1000.]});y=pd.Series(['g','h','g','h'])
        for model in b7.MODELS:
            pipe=make_model(model,X,y)
            actual=pipe.named_steps['model'].get_params(deep=False)
            for key,value in config()['models'][model].items():self.assertEqual(actual[key],value)
            self.assertEqual(b7.parameters(make_model(model,self.runner.X,self.runner.y)),self.runner.parameters[model])
            pre=pipe.named_steps['preprocess'];self.assertFalse(hasattr(pre,'transformers_'))
            pre.fit(X.iloc[:3]);num=pre.named_transformers_['num'];self.assertEqual(num.named_steps['imputer'].statistics_[0],2.)
            if model.startswith('logistic'):self.assertEqual(num.named_steps['scaler'].mean_[0],2.)
            pre.transform(X.iloc[3:])
            self.assertEqual(num.named_steps['imputer'].statistics_[0],2.)

    def test_fixed_reference_no_clipping(self):
        ref=b7.reference();self.assertEqual(ref,self.runner.cfg['normalization'])
        en,vn,d=analysis.project(ref['E_max_ref']+.1,ref['V_min_ref']-.1,ref)
        self.assertGreater(en,1);self.assertLess(vn,0);self.assertEqual(d,.7*en+.3*vn)

    def test_five_seed_aggregation_and_guard(self):
        rows=[]
        for key in self.runner.plan:
            r=self.runner.identity(key);r['macro_f1']=.6+.001*r['fold']+.002*r['repeat']+.00001*r['cv_seed'];r['execution_status']='success';rows.append(r)
        frame=pd.DataFrame(rows);t=analysis.aggregate(frame,b7.reference())
        seed=frame[(frame.sample_id=='s101_n200')&(frame.model==b7.MODELS[0])&(frame.cv_seed==42)].macro_f1
        actual=t['model_seed_statistics'].query('sample_id=="s101_n200" and model=="decision_tree_default" and cv_seed==42').iloc[0]
        self.assertAlmostEqual(actual.within_seed_sd,seed.std(ddof=1),15)
        comp=t['subsample_difficulty_components'].iloc[0];self.assertAlmostEqual(comp.variability,seed.std(ddof=1),15);self.assertAlmostEqual(comp.error,1-comp.mean_macro_f1,15)
        self.assertEqual(len(t['sample_size_summary']),7);self.assertEqual(len(t['adjacent_size_changes']),30)
        for bad in [frame.iloc[:-1],pd.concat([frame.iloc[:-1],frame.iloc[:1]])]:
            with self.assertRaises(ValueError):analysis.aggregate(bad,b7.reference())
        bad=frame.copy();bad.loc[0,'split_id']='wrong'
        with self.assertRaises(ValueError):analysis.aggregate(bad,b7.reference())

    def test_checkpoint_failure_retry_and_no_overwrite(self):
        runner=self.runner;old=runner.store
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runner.store=Store(Path(tmp),old.manifest)
                with patch.object(b7,'make_model',side_effect=RuntimeError('injected test-only failure')):
                    with self.assertRaises(RuntimeError):runner.run(limit=1)
                key=('s101_n200',b7.MODELS[0],42,1,1);attempts=runner.store.attempts(key);self.assertEqual(attempts[0]['execution_status'],'failure')
                record=dict(**runner.identity(key),attempt=2,execution_status='success',macro_f1=.5)
                with writer_lock(runner.store.root):runner.store.save(key,record)
                self.assertEqual(runner.completed(key)['attempt'],2)
                with self.assertRaises(IntegrityError):runner.store.save(key,record)
                bad=Store(Path(tmp),{**old.manifest,'purpose':'changed'})
                with self.assertRaises(IntegrityError):bad.check_manifest()
            finally:runner.store=old

    def test_full_run_gate_and_separation(self):
        runner=self.runner;old=runner.pilot
        try:
            runner.pilot=False
            with self.assertRaises(PermissionError):runner.run()
        finally:runner.pilot=old
        self.assertTrue(runner.store.root.is_relative_to(b7.BASE.resolve()) or str(b7.BASE) in str(runner.store.root))
        self.assertNotIn('block2b',str(runner.store.root))
        with self.assertRaises(ValueError):analysis.aggregate(pd.DataFrame(),b7.reference())


if __name__=='__main__':unittest.main()
