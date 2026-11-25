"""Synthetic complete-grid tests; no training or CV generation."""
import ast
import json
import unittest
import numpy as np
import pandas as pd
from . import block3_analysis as b3
from . import block6_analysis as b6
from .runner import plan
from .infrastructure import ROOT,config,digest


class Block6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models=b6.DT+b6.LR+tuple(m for v in b6.panels().values() for m in v)
        rows=[]
        for i,(d,m,s,r,f) in enumerate(plan(models=list(models))):
            base=.4+.001*((r-1)*5+f-1)+.002*config()['cv']['seeds'].index(s)
            delta=.02 if m not in b6.DT+b6.LR else 0
            if m.endswith('_balanced'):delta+=.03
            rows.append(dict(dataset=d,model=m,seed=s,repeat=r,fold=f,macro_f1=base+delta,
                dataset_fingerprint='data-'+d,split_id=f'{d}-{s}-{r}-{f}',configuration_id='config-'+m,
                evaluation_id=str(i),environment_id='fixture',purpose='official',execution_status='success',protocol_version=config()['protocol_version']))
        cls.frame=pd.DataFrame(rows);cls.primary=b3.load_outputs()['five_seed_primary']

    def test_selection_and_aggregation(self):
        self.assertEqual(set(b6.DT),{'decision_tree_default','decision_tree_balanced'})
        components,seed,model=b6.aggregate_models(self.frame,b6.DT)
        self.assertEqual(len(seed),170);self.assertEqual(len(model),34)
        np.testing.assert_allclose(components.mean_macro_f1,.4+.007+.004+.015,atol=1e-16)
        np.testing.assert_allclose(components.variability,np.std(np.arange(15)*.001,ddof=1),atol=1e-16)
        self.assertTrue(seed['count'].eq(15).all());self.assertTrue(model.seeds.eq(5).all())

    def test_exact_pairing_count_sign_and_provenance(self):
        paired=b6.paired_deltas(self.frame)
        self.assertEqual(len(paired),10200)
        np.testing.assert_allclose(paired.delta_macro_f1,.02,atol=1e-16)
        for _,g in paired.groupby('class_weight'):
            self.assertEqual(g.model_unpruned.nunique(),1)
            self.assertEqual(g.model_unpruned.iloc[0],'decision_tree_balanced' if g.class_weight.iloc[0]=='balanced' else 'decision_tree_default')
        self.assertTrue(paired.configuration_id_regularized.str.startswith('config-decision_tree_').all())
        self.assertTrue(paired.configuration_id_unpruned.isin(['config-'+m for m in b6.DT]).all())
        for column,value in [('split_id','wrong'),('dataset_fingerprint','wrong'),('environment_id','wrong')]:
            bad=self.frame.copy();idx=bad.index[bad.model.eq('decision_tree_leaf_2_default')][0];bad.loc[idx,column]=value
            with self.assertRaises(ValueError):b6.paired_deltas(bad)
        missing=self.frame.drop(self.frame.index[self.frame.model.eq('decision_tree_default')][0])
        with self.assertRaises(ValueError):b6.paired_deltas(missing)
        with self.assertRaises(ValueError):b6.paired_deltas(pd.concat([self.frame,self.frame.iloc[:1]]))

    def test_dataset_and_stratum_summaries_equal_weight(self):
        paired=b6.paired_deltas(self.frame)
        order=config()['primary_datasets']
        paired['delta_macro_f1']=paired.dataset.map({d:(i-8)*.01 for i,d in enumerate(order)})
        t=b6.effect_tables(paired,self.primary)
        self.assertEqual(len(t['dataset_regularization_effects']),68)
        self.assertEqual(len(t['class_weight_effects']),136)
        self.assertEqual(len(t['seed_regularization_effects']),1020)
        self.assertEqual(len(t['stratum_regularization_effects']),36)
        for _,r in t['stratum_regularization_effects'].iterrows():
            datasets=self.primary.loc[self.primary.stratum.eq(r.canonical_stratum),'dataset']
            expected=np.array([(order.index(d)-8)*.01 for d in datasets])
            self.assertAlmostEqual(r.mean_dataset_delta,float(expected.mean()),places=15)
            self.assertAlmostEqual(r.sd_across_datasets,float(expected.std(ddof=1)),places=15)
            self.assertEqual(r.datasets,len(datasets))
        for _,r in t['dataset_regularization_effects'].iterrows():
            self.assertEqual(r['count'],150)
            self.assertAlmostEqual(r.mean_delta,(order.index(r.dataset)-8)*.01,places=15)
            self.assertAlmostEqual(r.mean_delta,(r.mean_delta_unbalanced+r.mean_delta_balanced)/2,places=15)
            self.assertEqual(r.fraction_improved+r.fraction_worsened+r.fraction_unchanged,1)

    def test_panel_normalization_and_exact_reference(self):
        for panel,models in {'dt_only':b6.DT,**b6.panels()}.items():
            c,_,_=b6.aggregate_models(self.frame,models)
            c['mean_macro_f1']=np.linspace(.2,.9,17);c['variability']=np.linspace(.02,.12,17)**2
            r=b3.rank_components(c,panel)
            np.testing.assert_allclose(r.normalized_error,(r.error-r.error.min())/(r.error.max()-r.error.min()),rtol=0,atol=0)
            np.testing.assert_allclose(r.normalized_variability,(r.variability-r.variability.min())/(r.variability.max()-r.variability.min()),rtol=0,atol=0)
            self.assertEqual(r.stratum.value_counts().reindex(['easy','intermediate','hard']).tolist(),[6,6,5])
        saved=b3.load_outputs();variants,_,_,_=b3.aggregate(saved['canonical_fold_scores'])
        pd.testing.assert_frame_equal(variants['C'],self.primary,check_exact=True,check_dtype=False)

    def test_deterministic_ties_and_no_rounding(self):
        c=pd.DataFrame(dict(dataset=config()['primary_datasets'],mean_macro_f1=.8,variability=.02))
        first=b3.rank_components(c,'test');second=b3.rank_components(c.sample(frac=1,random_state=42),'test')
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(first.dataset.tolist(),sorted(c.dataset))
        d=sorted(c.dataset)[-1];c.loc[c.dataset.eq(d),'mean_macro_f1']+=1e-14
        self.assertEqual(b3.rank_components(c,'test').dataset.iloc[0],d)

    def test_no_fitting_significance_or_chance_labels(self):
        tree=ast.parse((ROOT/'experiments/protocol/block6_analysis.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                self.assertNotIn(name,{'fit','fit_transform','predict','generate_splits','split','cross_validate','ttest_ind','ttest_rel','wilcoxon','mannwhitneyu'})
        source=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='effect_tables'))
        self.assertNotIn('chance',source)
        snapshot=json.loads((ROOT/'experiments/protocol/block6_preservation.json').read_text())
        for path,sha in snapshot['files'].items():
            if '/analysis/' in path:self.assertEqual(digest((ROOT/path).read_bytes()),sha,path)


if __name__=='__main__':unittest.main()
