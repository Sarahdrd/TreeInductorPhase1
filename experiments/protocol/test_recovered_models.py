"""Small compatibility fixtures only; no benchmark evaluation or CV loops."""
import ast
import json
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import get_tags

from . import recovered_cross_models as recovered
from .infrastructure import ROOT, config
from .models import make_model


class RecoveryTests(unittest.TestCase):
    compatibility = None

    def test_original_compatibility_check(self):
        # The authoritative check consumes only the FIRST partition for each
        # of its two tiny iris fixtures, never all 15 CV partitions.
        result = recovered.check_ripper_compatibility()
        type(self).compatibility = result
        self.assertEqual(result['status'], 'complete', result['error'])

    def test_source_fidelity(self):
        source = (ROOT / 'experiments/protocol/recovered_source/cross_model_cell_14.py.txt').read_text(encoding='utf-8')
        extracted = (ROOT / 'experiments/protocol/recovered_cross_models.py').read_text(encoding='utf-8')
        def definitions(text):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(text))
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        original, actual = definitions(source), definitions(extracted)
        self.assertEqual(original, actual)

    def test_parameters_clone_and_tags(self):
        model = recovered.SklearnRIPPER(random_state=42)
        expected = config()['models']['ripper']
        self.assertEqual(model.get_params(), expected)
        self.assertEqual(clone(model).get_params(), expected)
        self.assertTrue(is_classifier(model))
        self.assertFalse(get_tags(model).classifier_tags.multi_class)
        self.assertTrue(is_classifier(OneVsRestClassifier(clone(model), n_jobs=1)))
        with self.assertRaises(ValueError):
            make_model('ripper', pd.DataFrame({'x': [1, 2]}))

    def test_fold_local_preprocessing(self):
        train = pd.DataFrame({'number': [1., 3., np.nan], 'nominal': ['b', None, 'b'],
                              'empty_num': [np.nan]*3, 'empty_cat': pd.Series([None]*3, dtype=object)})
        test = pd.DataFrame({'number': [999., np.nan], 'nominal': ['unseen', None],
                             'empty_num': [np.nan]*2, 'empty_cat': [None]*2})
        pre = recovered.RipperPreprocessor().fit(train)
        before = dict(pre.fill_)
        transformed = pre.transform(test)
        self.assertEqual(pre.fill_, before)
        self.assertEqual(transformed.number.tolist(), [999., 2.])
        self.assertEqual(transformed.nominal.tolist(), ['unseen', 'b'])
        self.assertEqual(transformed.empty_num.tolist(), [0., 0.])
        self.assertEqual(transformed.empty_cat.tolist(), ['__MISSING__']*2)
        self.assertFalse(transformed.isna().any().any())
        self.assertEqual(transformed.number.dtype.kind, 'f')

    def test_binary_multiclass_feature_types(self):
        rng = np.random.default_rng(81)
        for classes in (2, 3):
            y = np.tile(np.arange(classes), 36)
            X = pd.DataFrame({'number': y*3 + rng.normal(0, .3, len(y)),
                              'nominal': pd.Series(['label_'+str(k) for k in y], dtype='str')})
            X.loc[0, 'number'] = np.nan
            X.loc[1, 'nominal'] = None
            train = np.arange(len(y))[np.arange(len(y)) % 4 != 0]
            valid = np.arange(len(y))[np.arange(len(y)) % 4 == 0]
            # Ensure both classes occur in validation for the binary fixture.
            if classes == 2:
                valid = np.arange(0, len(y), 5)
                train = np.setdiff1d(np.arange(len(y)), valid)
            for columns in (['number'], ['nominal'], ['number', 'nominal']):
                with self.subTest(classes=classes, columns=columns):
                    pipe = clone(make_model('ripper', X[columns], y))
                    self.assertTrue(is_classifier(pipe))
                    self.assertEqual(isinstance(pipe.named_steps['model'], OneVsRestClassifier), classes > 2)
                    pipe.fit(X[columns].iloc[train], y[train])
                    predictions = np.asarray(pipe.predict(X[columns].iloc[valid]))
                    self.assertEqual(predictions.shape, (len(valid),))
                    self.assertTrue(np.isin(predictions, np.unique(y)).all())
                    probabilities = np.asarray(pipe.predict_proba(X[columns].iloc[valid]))
                    self.assertEqual(probabilities.shape, (len(valid), classes))
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                    model = pipe.named_steps['model']
                    members = model.estimators_ if classes > 2 else [model]
                    self.assertTrue(all(is_classifier(clone(m)) for m in members))
                    if 'number' in columns:
                        self.assertTrue(all(m.n_discretize_bins == 10 for m in members))
                        self.assertTrue(all('number' in m.bin_transformer_.bins_ for m in members))

    def test_svm_pipeline_binary_multiclass_missing(self):
        rng = np.random.default_rng(3)
        for classes in (2, 3):
            y = np.tile(np.arange(classes), 15)
            X = pd.DataFrame({'number': y+rng.normal(0, .1, len(y)),
                              'nominal': ['x' if v else 'y' for v in y]})
            X.loc[0, 'number'] = np.nan
            X.loc[1, 'nominal'] = None
            pipe = clone(make_model('svm_rbf', X, y))
            params = pipe.named_steps['model'].get_params()
            self.assertEqual(params['probability'], 'deprecated')
            self.assertEqual({**params, 'probability': False}, config()['models']['svm_rbf'])
            pipe.fit(X.iloc[:-classes*2], y[:-classes*2])
            self.assertFalse(pipe.named_steps['model']._effective_probability)
            predicted = pipe.predict(X.iloc[-classes*2:])
            self.assertTrue(np.isin(predicted, np.unique(y)).all())
            self.assertTrue(is_classifier(pipe))


if __name__ == '__main__':
    unittest.main()
