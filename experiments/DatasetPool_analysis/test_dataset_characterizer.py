"""Regression tests for canonical formulas, exclusions and fold-local processing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from core.DatasetCharacterizer import (
    DatasetCharacterizer, NON_DUMMY_BASELINE_MODELS, _compute_pool_profile,
    assign_predictive_difficulty_groups, build_preprocessor, minmax_normalize_on_pool,
)


class CharacterizerTests(unittest.TestCase):
    def test_pool_normalization(self):
        mask = pd.Series([True, True, False])
        result = minmax_normalize_on_pool(pd.Series([2., 4., 999.]), mask)
        self.assertEqual(result.iloc[:2].tolist(), [0., 1.])
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertEqual(minmax_normalize_on_pool(pd.Series([2., 2., 999.]), mask).iloc[:2].tolist(), [0., 0.])

    def test_exact_ties_and_uneven_thirds(self):
        frame = pd.DataFrame(dict(dataset=['b', 'a', 'c', 'd', 'e', 'f', 'g', 'excluded'],
            ranking_eligible=[True]*7+[False], predictive_difficulty_score=[.2, .2, .2+1e-14, .4, .5, .6, .7, 0],
            baseline_macro_f1_mean=[.7, .8, .99, .5, .4, .3, .2, 1], cv_variability=[.1]*8))
        labels, ranks = assign_predictive_difficulty_groups(frame)
        self.assertEqual(ranks.iloc[:3].tolist(), [2, 1, 3])
        self.assertEqual(labels.value_counts().to_dict(), {'easy': 3, 'intermediate': 2, 'hard': 2, 'review_separately': 1})
        frame['strategy_sensitivity'] = np.arange(8)*100
        frame['computational_burden_score'] = np.arange(8)*-100
        self.assertTrue(assign_predictive_difficulty_groups(frame)[0].equals(labels))

    def test_aggregate_protocol_and_sample_sd(self):
        rows = []
        for i, model in enumerate(NON_DUMMY_BASELINE_MODELS + ['dummy_stratified']):
            scores = np.linspace(.2 + i*.1, .3 + i*.1, 15)
            rows.append(dict(dataset='sample', model=model, scores=scores, n_scores=15,
                error='', f1_macro_mean=scores.mean(), f1_macro_std=scores.std(ddof=1),
                runtime_seconds=1., mean_tree_nodes=3., mean_tree_depth=1., mean_tree_leaves=2.))
        frame = pd.DataFrame(rows)
        audit = pd.DataFrame([dict(dataset='sample', comparable_cv=True, min_class_count=5, cv_issue='')])
        profile = _compute_pool_profile(frame, audit).iloc[0]
        self.assertAlmostEqual(profile.baseline_macro_f1_mean, .4)
        self.assertAlmostEqual(profile.cv_variability, np.std(rows[0]['scores'], ddof=1))
        self.assertEqual(profile.predictive_difficulty_score, 0)
        for failure in ['missing', 'error', 'nonfinite', 'duplicate']:
            broken = frame.copy(deep=True)
            if failure == 'missing': broken = broken.iloc[1:]
            if failure == 'error': broken.loc[0, 'error'] = 'failed'
            if failure == 'nonfinite': broken.at[0, 'scores'] = np.full(15, np.nan)
            if failure == 'duplicate': broken = pd.concat([broken, broken.iloc[[0]]])
            value = _compute_pool_profile(broken, audit).iloc[0]
            self.assertFalse(value.ranking_eligible)
            self.assertTrue(pd.isna(value.predictive_difficulty_score))

    def test_preprocessing_train_only(self):
        train = pd.DataFrame({'number': [1., 3., np.nan], 'category': ['a', 'a', np.nan]})
        transformer = build_preprocessor(train)
        transformer.fit(train)
        test = pd.DataFrame({'number': [np.nan, 999.], 'category': ['unseen', 'a']})
        transformed = transformer.transform(test)
        self.assertEqual(transformed[0, 0], 2.)
        self.assertEqual(transformed[0, 1], 0.)
        self.assertEqual(transformer.named_transformers_['num']['imputer'].statistics_[0], 2.)

    def test_failures_rare_classes_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'x': range(8), 'label': ['a']*4+['b']*3+[None]}).to_csv(root/'rare.csv', index=False)
            pd.DataFrame({'x': range(10), 'label': ['a']*10}).to_csv(root/'invalid.csv', index=False)
            c = DatasetCharacterizer(target_config={'rare': 'label', 'invalid': 'label'},
                                     id_columns={'rare': [], 'invalid': []})
            result = c.characterize_pool({'rare': root/'rare.csv', 'invalid': root/'invalid.csv', 'missing': root/'missing.csv'})
            self.assertFalse(result.ranking_eligible.any())
            self.assertEqual(result.loc[0, 'n_missing_targets_removed'], 1)
            self.assertEqual(result.status.tolist(), ['review_separately', 'invalid_dataset', 'error'])
            self.assertTrue(result.ranking_exclusion_reason.str.len().gt(0).all())
            self.assertEqual(c.get_difficulty_groups(), {'Easy': [], 'Intermediate': [], 'Hard': []})
            self.assertTrue(c.characterize_pool({}).empty)
            self.assertTrue(c.baseline_results_.empty)
            self.assertEqual(c.get_datasets_by_difficulty(['Easy', 'Hard']), {})
            with self.assertRaises(ValueError): c.get_datasets_by_difficulty('unknown')
            only_errors = c.characterize_pool({'missing': root/'absent.csv'})
            self.assertEqual(only_errors.status.tolist(), ['error'])

    def test_fits_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'x': range(30), 'label': ['a']*15+['b']*15}).to_csv(root/'good.csv', index=False)
            c = DatasetCharacterizer(target_config={'good': 'label'}, id_columns={'good': []})
            summary = c.characterize_pool({'good': root/'good.csv'})
            self.assertTrue(summary.ranking_eligible.all())
            self.assertTrue(c.baseline_results_.n_scores.eq(15).all())
            self.assertEqual(c.get_datasets_by_difficulty('Easy'), {'good': root/'good.csv'})
            self.assertEqual(c.get_datasets_by_difficulty(['Easy', 'Hard']), {'good': root/'good.csv'})


if __name__ == '__main__':
    unittest.main()
