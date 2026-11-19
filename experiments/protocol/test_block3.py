"""Synthetic aggregation and exact-source scoring checks; never fits models."""
import ast
import json
import unittest
import numpy as np
import pandas as pd

from .block3_analysis import CANONICAL, NS, aggregate, rank_components, validate_grid
from .canonical_scoring import assign_predictive_difficulty_groups, minmax_normalize_on_pool
from .infrastructure import ROOT, config
from .runner import plan


class Block3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[]
        for i,(d,m,s,r,f) in enumerate(plan(models=list(CANONICAL))):
            rows.append(dict(dataset=d,model=m,seed=s,repeat=r,fold=f,
                             macro_f1=.1+((r-1)*5+f-1)*.04+config()['cv']['seeds'].index(s)*.01,
                             purpose='official',execution_status='success',protocol_version=config()['protocol_version'],
                             evaluation_id='synthetic-'+str(i)))
        cls.frame=pd.DataFrame(rows)

    def test_canonical_inventory_and_complete_grid(self):
        self.assertEqual(set(CANONICAL),set(config()['canonical_models']))
        self.assertEqual(len(self.frame),5100)
        validate_grid(self.frame,CANONICAL)
        for bad in [self.frame.iloc[:-1],pd.concat([self.frame.iloc[:-1],self.frame.iloc[:1]]),
                    self.frame.assign(model='svm_rbf'),self.frame.assign(purpose='pilot'),
                    self.frame.assign(macro_f1=float('nan')),self.frame.assign(macro_f1=1.1)]:
            with self.assertRaises(ValueError): validate_grid(bad,CANONICAL)

    def test_seed_mean_sd_ddof_and_five_seed_aggregation(self):
        variants,seed,model,per_seed=aggregate(self.frame)
        values=.1+np.arange(15)*.04
        row=seed.loc[seed.seed.eq(42)].iloc[0]
        self.assertAlmostEqual(row.mean_f1,values.mean(),places=15)
        self.assertAlmostEqual(row.sd_f1,values.std(ddof=1),places=15)
        self.assertNotAlmostEqual(row.sd_f1,values.std(ddof=0),places=8)
        self.assertTrue(seed['count'].eq(15).all());self.assertTrue(model.fold_count.eq(75).all())
        self.assertAlmostEqual(model.iloc[0].mean_f1_75,values.mean()+.02,places=15)
        self.assertAlmostEqual(model.iloc[0].mean_within_seed_sd,values.std(ddof=1),places=15)
        self.assertAlmostEqual(model.iloc[0].pooled_sd,np.concatenate([values+s*.01 for s in range(5)]).std(ddof=1),places=15)
        self.assertEqual(len(per_seed),85)
        self.assertTrue(np.allclose(model.mean_f1_75,model.direct_mean_75,rtol=0,atol=5e-16))

    def test_minmax_and_constant_components(self):
        result=minmax_normalize_on_pool(pd.Series([2.,4.,6.,100.]),pd.Series([True,True,True,False]))
        np.testing.assert_array_equal(result.iloc[:3],[0.,.5,1.]);self.assertTrue(np.isnan(result.iloc[3]))
        np.testing.assert_array_equal(minmax_normalize_on_pool(pd.Series([3.,3.]),pd.Series([True,True])),[0.,0.])

    def test_exact_weights_strata_and_names(self):
        data=pd.DataFrame(dict(dataset=config()['primary_datasets'],mean_macro_f1=np.linspace(.95,.2,17),variability=np.linspace(.01,.2,17)))
        result=rank_components(data,'test')
        np.testing.assert_array_equal(result.difficulty_score,.70*result.normalized_error+.30*result.normalized_variability)
        self.assertEqual(result.stratum.value_counts().reindex(['easy','intermediate','hard']).tolist(),[6,6,5])
        self.assertEqual(set(result.dataset),set(config()['primary_datasets']))

    def test_exact_tie_breaks_and_no_rounding(self):
        d=pd.DataFrame(dict(dataset=['d','c','b','a'],predictive_difficulty_score=[.5]*4,
                            baseline_macro_f1_mean=[.7,.8,.8,.8],cv_variability=[.01,.03,.02,.02],ranking_eligible=True))
        _,rank=assign_predictive_difficulty_groups(d)
        self.assertEqual(d.loc[rank.sort_values().index,'dataset'].tolist(),['a','b','c','d'])
        d.loc[0,'predictive_difficulty_score']=.5-1e-15
        _,rank=assign_predictive_difficulty_groups(d)
        self.assertEqual(rank.iloc[0],1)

    def test_extracted_source_and_notebook_crosscheck(self):
        path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
        n=json.loads(path.read_text(encoding='utf8'));source=''.join(n['cells'][12]['source'])
        nodes=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)]
        ns={'np':np,'pd':pd};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        extracted=ast.parse((ROOT/'experiments/protocol/canonical_scoring.py').read_text())
        definitions={x.name:ast.dump(x) for x in extracted.body if isinstance(x,ast.FunctionDef)}
        for node in nodes:self.assertEqual(ast.dump(node),definitions[node.name])
        data=pd.DataFrame(dict(dataset=config()['primary_datasets'],mean_macro_f1=np.linspace(.9,.3,17),variability=np.linspace(.1,.01,17)))
        actual=rank_components(data,'test')
        d=data.rename(columns={'mean_macro_f1':'baseline_macro_f1_mean','variability':'cv_variability'}).assign(ranking_eligible=True)
        E=ns['minmax_normalize_on_pool'](1-d.baseline_macro_f1_mean,d.ranking_eligible)
        V=ns['minmax_normalize_on_pool'](d.cv_variability,d.ranking_eligible)
        d['predictive_difficulty_score']=.70*E+.30*V
        labels,ranks=ns['assign_predictive_difficulty_groups'](d)
        reference=d.assign(stratum=labels,rank=ranks).set_index('dataset').loc[actual.dataset]
        np.testing.assert_array_equal(actual.difficulty_score,reference.predictive_difficulty_score)
        np.testing.assert_array_equal(actual['rank'],reference['rank'])
        self.assertEqual(actual.stratum.tolist(),reference.stratum.tolist())


if __name__=='__main__':unittest.main()
