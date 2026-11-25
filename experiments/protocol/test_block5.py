"""Pure calculation and artifact checks for Block 5. No evaluations."""
import ast
import json
import unittest
import numpy as np
import pandas as pd
from . import block3_analysis as b3
from . import block5_analysis as b5
from .canonical_scoring import minmax_normalize_on_pool
from .infrastructure import ROOT,config,digest


class Block5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=b3.load_outputs()
        cls.primary=cls.saved['five_seed_primary'];cls.seeds=cls.saved['per_seed_rankings']
        cls.tables=b5.compute_tables(cls.primary,cls.seeds)

    def test_reference_and_model_pool(self):
        canonical=self.saved['canonical_fold_scores']
        b3.validate_grid(canonical,b3.CANONICAL)
        self.assertEqual(set(canonical.model),set(config()['canonical_models']))
        variants,_,_,per_seed=b3.aggregate(canonical)
        b5.assert_reference(variants['C'],self.primary)
        b5.assert_reference(b5.score(self.primary,.70),self.primary)
        pd.testing.assert_frame_equal(per_seed,self.seeds,check_exact=True,check_dtype=False)

    def test_grid_normalization_weights_and_sizes(self):
        grid=self.tables['scenario_rankings']
        self.assertEqual(set(grid.alpha),{.51,.60,.70,.80,.90})
        self.assertEqual(len(grid),510);self.assertEqual(grid.scenario.nunique(),30)
        self.assertEqual(grid.loc[grid.family.eq('combined'),'scenario'].nunique(),25)
        for _,g in grid.groupby('scenario'):
            self.assertEqual(g.stratum.value_counts().reindex(b5.LABELS).tolist(),[6,6,5])
            self.assertEqual(set(g.dataset),set(self.primary.dataset))
            mask=pd.Series(True,index=g.index)
            np.testing.assert_array_equal(g.normalized_error,minmax_normalize_on_pool(g.error,mask))
            np.testing.assert_array_equal(g.normalized_variability,minmax_normalize_on_pool(g.variability,mask))
            a=g.alpha.iloc[0]
            np.testing.assert_array_equal(g.difficulty_score,a*g.normalized_error+(.30 if a==.70 else 1-a)*g.normalized_variability)
        with self.assertRaises(ValueError): b5.score(self.primary,0)
        with self.assertRaises(ValueError): b5.score(self.primary,1)

    def test_reference_denominators_and_counts(self):
        grid=self.tables['scenario_rankings'];primary=self.primary.set_index('dataset')
        for name,denominator in [('seed',5),('weight',5),('combined',25)]:
            frame=grid.loc[grid.family.eq('aggregate_weight' if name=='weight' else 'combined')]
            if name=='seed':frame=frame.loc[frame.alpha.eq(.70)]
            for _,r in self.tables[name+'_stability'].iterrows():
                g=frame.loc[frame.dataset.eq(r.dataset)]
                expected=int(g.stratum.eq(primary.loc[r.dataset,'stratum']).sum())
                self.assertEqual(r[name+'_matches'],expected)
                self.assertEqual(r[name+'_stability'],expected/denominator)
                self.assertEqual(r.scenario_count,denominator)
        updated=self.tables['canonical_ranking_with_stability']
        b5.assert_reference(updated,self.primary)

    def test_deterministic_ties_and_no_rounding(self):
        d=self.primary.copy();d['mean_macro_f1']=.8;d['variability']=.01
        ranked=b5.score(d,.70)
        self.assertEqual(ranked.dataset.tolist(),sorted(d.dataset))
        shuffled=b5.score(d.sample(frac=1,random_state=17),.70)
        pd.testing.assert_frame_equal(ranked,shuffled)
        # Tiny unrounded F difference must break the all-equal ordering.
        name=sorted(d.dataset)[-1];d.loc[d.dataset.eq(name),'mean_macro_f1']=.8+1e-14
        self.assertEqual(b5.score(d,.70).dataset.iloc[0],name)

    def test_transitions_and_boundary_exchanges(self):
        grid=self.tables['scenario_rankings'];transitions=self.tables['label_transition_details']
        self.assertEqual(len(transitions),int((~grid.label_matches).sum()))
        for _,r in self.tables['boundary_exchanges'].iterrows():
            self.assertEqual(len(r.moved_to_harder_side.split(';')),r.count_each_direction)
            self.assertEqual(len(r.moved_to_easier_side.split(';')),r.count_each_direction)
        gaps=self.tables['canonical_boundary_gaps']
        self.assertEqual(gaps.lower_rank.tolist(),[6,12])
        self.assertTrue((gaps.score_gap>0).all())

    def test_invalid_reference_and_seed_inputs_stop(self):
        bad=self.primary.copy();bad.loc[0,'difficulty_score']+=1e-12
        with self.assertRaises(AssertionError): b5.scenarios(bad,self.seeds)
        with self.assertRaises(ValueError): b5.scenarios(self.primary,self.seeds.iloc[:-1])
        with self.assertRaises(ValueError): b5.score(self.primary.iloc[:-1],.70)

    def test_no_fit_chance_correction_or_regeneration(self):
        source=(ROOT/'experiments/protocol/block5_analysis.py').read_text()
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                self.assertNotIn(name,{'fit','fit_transform','predict','generate_splits','split','cross_validate','cross_val_score'})
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='score')
        text=ast.unparse(function)
        self.assertNotIn('chance',text);self.assertNotIn('block4',text)
        saved=json.loads((ROOT/'experiments/protocol/block5_preservation.json').read_text())
        for path,expected in saved['files'].items():
            if '/analysis/block3-' in path or '/analysis/block4-' in path:
                self.assertEqual(digest((ROOT/path).read_bytes()),expected,path)


if __name__=='__main__':unittest.main()
