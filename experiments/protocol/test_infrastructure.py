"""Lightweight tests; no estimator fitting, scores or rankings generated."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from .datasets import load_dataset
from .infrastructure import (ROOT, config, data_digest, generate_splits, load_splits,
                             save_splits, validate_dataset, validate_splits)
from .models import make_model
from .results import validate_record, save_records


class InfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'experiments/configs/dataset_manifest.json').read_text())
        cls.entry = next(e for e in cls.manifest['datasets'] if e['dataset'] == 'iris')
        cls.obj = load_dataset('iris', ROOT / cls.entry['local_path'])

    def test_entire_saved_benchmark(self):
        cfg = config()
        self.assertEqual(len(cfg['primary_datasets']), 17)
        self.assertEqual(self.manifest['dataset_order'], cfg['primary_datasets'])
        self.assertEqual(len(self.manifest['datasets']), 20)
        count = 0
        for entry in self.manifest['datasets']:
            obj = load_dataset(entry['dataset'], ROOT / entry['local_path'])
            validate_dataset(obj, entry, ROOT / entry['local_path'])
            self.assertEqual(entry['eligible'], entry['min_class_count'] >= 5)
            if entry['eligible']:
                for seed in cfg['cv']['seeds']:
                    path = ROOT / 'experiments/splits/saved_cv_indices' / cfg['protocol_version'] / entry['dataset'] / f'seed_{seed}.npz'
                    partitions, _ = load_splits(path, entry, obj['y'])
                    count += len(partitions)
        self.assertEqual(count, 1275)

    def test_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_splits(directory, self.entry, self.obj['y'], 7, 'test-environment')
            save_splits(directory, self.entry, self.obj['y'], 7, 'test-environment')
            splits, meta = load_splits(path, self.entry, self.obj['y'])
            self.assertEqual(len(splits), 15)
            with self.assertRaises(ValueError):
                save_splits(directory, self.entry, self.obj['y'], 7, 'other-environment')
            meta['partitions'][0]['split_id'] = 'tampered'
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_splits(path, self.entry, self.obj['y'])

    def test_invalid_splits_and_order(self):
        splits = generate_splits(self.obj['y'], 42)
        bad = copy.deepcopy(splits)
        bad[0][0][0] = bad[0][1][0]
        with self.assertRaises(ValueError):
            validate_splits(bad, self.obj['y'])
        with self.assertRaises(ValueError):
            generate_splits(np.array([0]*20 + [1]*4), 42)
        obj = dict(self.obj, X=self.obj['X'].iloc[::-1].reset_index(drop=True))
        with self.assertRaises(ValueError):
            validate_dataset(obj, self.entry, ROOT / self.entry['local_path'])

    def test_model_configs_without_fit(self):
        for name, parameters in config()['models'].items():
            model = make_model(name, self.obj['X'], self.obj['y'])
            estimator = model.named_steps['model'] if hasattr(model, 'named_steps') else model
            if name == 'ripper' and hasattr(estimator, 'estimator'):
                estimator = estimator.estimator
            for key, value in parameters.items():
                if name == 'svm_rbf' and key == 'probability':
                    self.assertEqual(estimator.get_params()[key], 'deprecated')
                    continue
                self.assertEqual(estimator.get_params()[key], value)
            self.assertFalse(hasattr(estimator, 'classes_'))

    def test_record_schema_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_splits(directory, self.entry, self.obj['y'], 42, 'test-env')
            _, meta = load_splits(path, self.entry, self.obj['y'])
            # Synthetic record exists only in a temporary test directory.
            record = dict(experiment_id='synthetic-test', protocol_version=config()['protocol_version'],
                          dataset='iris', dataset_fingerprint=self.entry['dataset_fingerprint'],
                          model='test_adapter', model_configuration={'test_only': True},
                          seed=42, repeat=1, fold=1, split_id=meta['partitions'][0]['split_id'],
                          macro_f1=0.5, environment_id='test-env')
            validate_record(record, meta)
            for change in ({'macro_f1': float('nan')}, {'fold': 0}, {'split_id': 'wrong'}, {'seed': True}):
                with self.assertRaises(ValueError):
                    validate_record(dict(record, **change), meta)
            destination = Path(directory) / 'test.jsonl'
            with self.assertRaises(ValueError):
                save_records(destination, [record, record], {('iris', 42): meta})
            self.assertFalse(destination.exists())
            save_records(destination, [record], {('iris', 42): meta})
            with self.assertRaises(FileExistsError):
                save_records(destination, [record], {('iris', 42): meta})

    def test_extracted_source_matches_notebooks(self):
        notebook = json.loads((ROOT / 'experiments/New_Characterization_Experiments.ipynb').read_text(encoding='utf-8'))
        for module, index in [('datasets.py', 6), ('preprocessing.py', 9)]:
            source = ast.parse(''.join(notebook['cells'][index]['source']))
            original = {n.name: ast.dump(n, include_attributes=False) for n in source.body if isinstance(n, ast.FunctionDef)}
            extracted = ast.parse((ROOT / 'experiments/protocol' / module).read_text())
            for node in extracted.body:
                if isinstance(node, ast.FunctionDef):
                    self.assertEqual(ast.dump(node, include_attributes=False), original[node.name])
        other = json.loads((ROOT / 'experiments/Dataset_Characterization.ipynb').read_text(encoding='utf-8'))
        def functions(cell):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(''.join(cell['source'])).body
                    if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions(notebook['cells'][6]), functions(other['cells'][6]))
        # Documented target fallback discrepancy: both resolve the actual CSV to 'class'.
        balance = load_dataset('balance_scale', ROOT / 'data/balance_scale.csv')
        self.assertEqual(balance['target_col'], 'class')


if __name__ == '__main__':
    unittest.main()
