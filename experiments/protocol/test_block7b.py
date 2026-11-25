"""Stage B tests: saved-data arithmetic and notebook checks, with no fitting."""
import ast
import json
import unittest
import numpy as np
import pandas as pd
from .block7b_reporting import tables,derived,SUPPLEMENT
from .block3_analysis import load_outputs
from .infrastructure import ROOT,digest,json_bytes


class Block7BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t=tables()

    def test_raw_aggregation(self):
        raw=self.t['fold_level_scores'];self.assertEqual(len(raw),10500)
        self.assertTrue(raw.execution_status.eq('success').all())
        for row in self.t['subsample_difficulty_components'].itertuples():
            g=raw[raw.sample_id.eq(row.sample_id)]
            seed=g.groupby(['model','cv_seed']).macro_f1.agg(['mean','std'])
            self.assertAlmostEqual(row.mean_macro_f1,seed['mean'].mean(),14)
            self.assertAlmostEqual(row.variability,seed['std'].mean(),14)
            self.assertAlmostEqual(row.error,1-row.mean_macro_f1,14)
            self.assertEqual(len(g),300)

    def test_summary_dispersion_and_directions(self):
        c=self.t['subsample_difficulty_components'];s=self.t['sample_size_summary']
        for row in s.itertuples():
            g=c[c.sample_size.eq(row.sample_size)]
            self.assertEqual(len(g),5)
            self.assertAlmostEqual(row.variability_std,g.variability.std(ddof=1),14)
        a=self.t['adjacent_size_changes'];self.assertEqual(int((a.delta_variability<0).sum()),29);self.assertEqual(int((a.delta_difficulty_score<0).sum()),30)
        for metric in ['mean_macro_f1','error','variability','difficulty_score']:
            for seed,g in c.groupby('subsampling_seed'):
                ordered=g.sort_values('sample_size');saved=a[a.subsampling_seed.eq(seed)].sort_values('from_size')
                np.testing.assert_allclose(np.diff(ordered[metric]),saved['delta_'+metric],rtol=0,atol=1e-15)

    def test_supplement_exact_and_decomposition(self):
        saved=load_outputs(SUPPLEMENT)
        for name,t in derived(self.t).items():pd.testing.assert_frame_equal(t,saved[name],check_exact=True,check_dtype=False)
        d=saved['score_change_decomposition'];np.testing.assert_allclose(d.error_contribution+d.variability_contribution,d.difficulty_change,rtol=0,atol=1e-15)

    def test_notebook_preservation_and_analysis_only(self):
        old=json.loads((ROOT/'experiments/protocol/block7b_preservation.json').read_text());n=json.loads((ROOT/'experiments/New_Characterization_Experiments.ipynb').read_text(encoding='utf8'))
        self.assertEqual(n['metadata'],old['notebook_metadata']);self.assertEqual([digest(json_bytes(c)) for c in n['cells'][:56]],old['notebook_cells'])
        self.assertEqual(len(n['cells']),64)
        for c in n['cells'][56:]:
            if c['cell_type']=='code':
                self.assertIsNotNone(c['execution_count']);tree=ast.parse(''.join(c['source']))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call):self.assertNotIn(getattr(node.func,'attr',getattr(node.func,'id','')),['fit','fit_transform','generate_splits','nested_indices','run'])


if __name__=='__main__':unittest.main()
