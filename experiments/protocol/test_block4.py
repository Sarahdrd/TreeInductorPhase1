"""Synthetic and saved-artifact tests. No training or split regeneration."""
import ast
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import block4_analysis as b4
from . import block3_analysis as b3
from .runner import plan
from .infrastructure import ROOT,config,digest,json_bytes


class Block4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[]
        for i,(d,m,s,r,f) in enumerate(plan(models=list(b4.DUMMIES))):
            value=.1+(.15 if m=='dummy_stratified' else 0)+.01*config()['cv']['seeds'].index(s)+.001*((r-1)*5+f-1)
            rows.append(dict(dataset=d,model=m,seed=s,repeat=r,fold=f,macro_f1=value,
                purpose='official',execution_status='success',protocol_version=config()['protocol_version'],evaluation_id=str(i)))
        cls.frame=pd.DataFrame(rows)

    def test_exact_grid_seeds_selection_and_pilots(self):
        self.assertEqual(len(self.frame),2550)
        b4.aggregate_dummies(self.frame)
        for bad in [self.frame.iloc[:-1],self.frame.assign(seed=999),self.frame.assign(purpose='pilot'),
                    self.frame.assign(model='decision_tree_default'),pd.concat([self.frame.iloc[:-1],self.frame.iloc[:1]])]:
            with self.assertRaises(ValueError): b4.aggregate_dummies(bad)

    def test_dummy_aggregation(self):
        wide,seed,long=b4.aggregate_dummies(self.frame)
        expected=.25+.02+.007
        np.testing.assert_allclose(wide.dummy_stratified_mean_f1,expected,rtol=0,atol=1e-16)
        np.testing.assert_allclose(wide.dummy_most_frequent_mean_f1,expected-.15,rtol=0,atol=1e-16)
        np.testing.assert_allclose(wide.dummy_stratified_mean_within_seed_sd,np.std(np.arange(15)*.001,ddof=1),atol=1e-16)
        values=np.concatenate([.25+i*.01+np.arange(15)*.001 for i in range(5)])
        np.testing.assert_allclose(wide.dummy_stratified_pooled_sd,values.std(ddof=1),atol=1e-16)
        self.assertTrue(seed['count'].eq(15).all());self.assertTrue(long.fold_count.eq(75).all())

    def test_formula_aggregate_not_mean_ratios(self):
        f=np.array([.4,.9]);dummy=np.array([.1,.6])
        improvement,error,den=b4.corrected_error(f.mean(),dummy.mean())
        self.assertAlmostEqual(float(error),(1-f.mean())/(1-dummy.mean()))
        self.assertAlmostEqual(float(improvement),1-float(error))
        self.assertNotAlmostEqual(float(error),float(np.mean((1-f)/(1-dummy))))

    def test_safeguards_never_clip(self):
        for f,dummy in [(.5,1),(.5,1-1e-10),(.2,.3),(.5,float('nan')),(1.1,.2),(.5,-.1)]:
            with self.assertRaises(ValueError): b4.corrected_error(f,dummy)
        self.assertEqual(float(b4.corrected_error(1,.3)[1]),0)
        self.assertEqual(float(b4.corrected_error(.3,.3)[1]),1)

    def test_variability_weights_identity_and_rank(self):
        primary=b3.rank_components(pd.DataFrame(dict(dataset=config()['primary_datasets'],mean_macro_f1=np.linspace(.95,.4,17),variability=np.linspace(.01,.1,17))),'fixture')
        before=primary.copy(deep=True)
        dummy,_,_=b4.aggregate_dummies(self.frame)
        result=b4.correct_ranking(primary,dummy).set_index('dataset').loc[primary.dataset]
        pd.testing.assert_frame_equal(primary,before)
        np.testing.assert_array_equal(result.original_variability,primary.variability)
        np.testing.assert_array_equal(result.normalized_variability,primary.normalized_variability)
        e=(1-primary.mean_macro_f1.to_numpy())/(1-dummy.set_index('dataset').loc[primary.dataset,'dummy_stratified_mean_f1'].to_numpy())
        np.testing.assert_array_equal(result.chance_corrected_error,e)
        np.testing.assert_allclose(result.normalized_chance_corrected_error,(e-e.min())/(e.max()-e.min()),atol=0,rtol=0)
        np.testing.assert_array_equal(result.chance_corrected_difficulty,.70*result.normalized_chance_corrected_error+.30*result.normalized_variability)
        self.assertEqual(result.corrected_stratum.value_counts().reindex(['easy','intermediate','hard']).tolist(),[6,6,5])
        self.assertEqual(result.corrected_rank.tolist(),list(range(1,18)))
        with self.assertRaises(ValueError): b4.correct_ranking(primary.iloc[:-1],dummy)

    def test_no_fit_or_cv_calls_and_preserved_canonical(self):
        for name in ['block4_analysis.py','block4_figures.py']:
            tree=ast.parse((ROOT/'experiments/protocol'/name).read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Call):
                    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                    self.assertNotIn(name,{'fit','fit_transform','predict','run','generate_splits','split','cross_validate','cross_val_score'})
        if b4.OUTPUT.exists():
            saved=b4.load_outputs();primary=b3.load_outputs()['five_seed_primary'].set_index('dataset')
            d=saved['original_vs_corrected'].set_index('dataset').loc[primary.index]
            for left,right in [('original_mean_macro_f1','mean_macro_f1'),('original_error','error'),('original_variability','variability'),('original_difficulty','difficulty_score'),('original_rank','rank')]:
                np.testing.assert_array_equal(d[left],primary[right])
        protected=json.loads((ROOT/'experiments/protocol/block4_preservation.json').read_text())
        for path,expected in protected['files'].items():
            if path.endswith('New_Characterization_Experiments.ipynb'): continue
            self.assertEqual(digest((ROOT/path).read_bytes()),expected,path)


if __name__=='__main__': unittest.main()
