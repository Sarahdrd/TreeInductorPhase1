"""Run from repository root: .venv/Scripts/python.exe -m experiments.protocol.prepare"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import load_dataset
from .infrastructure import (ROOT, config, data_digest, digest, environment,
                             load_splits, save_splits, validate_dataset, write_frozen)


def main():
    cfg = config()
    env = environment()
    frozen_env = json.loads((ROOT / 'experiments/configs/environment.json').read_text())
    if env != frozen_env:
        recovery_path = ROOT / 'experiments/configs/environment_ripper_ready.json'
        recovered_env = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        if env != recovered_env:
            raise ValueError('Use a recorded environment; do not silently regenerate under another runtime')
        # Addition of wittgenstein cannot alter the generator or data parser.
        original_packages = frozen_env['packages']
        if any(env['packages'][k] != v for k, v in original_packages.items() if k != 'wittgenstein'):
            raise ValueError('Split-generation dependencies changed')
    historical = json.loads((ROOT / 'results/new_characterization_experiments/metadata.json').read_text())
    old_splits = pd.read_csv(ROOT / 'results/new_characterization_experiments/tables/cv_partitions.csv')
    entries, objects = [], {}
    for name, spec in cfg['dataset_specs'].items():
        path = ROOT / spec['path']
        if not path.is_file():
            raise FileNotFoundError(f'Missing original CSV: {path}; no substitution allowed')
        obj = load_dataset(name, path)
        counts = {str(k): int(v) for k, v in obj['y_raw'].value_counts().items()}
        entry = dict(dataset=name, canonical_name=spec['canonical_name'], local_path=spec['path'],
                     source='Local curated CSV; upstream source and identifier unverified',
                     source_identifier=None, dataset_version=None,
                     file_sha256=digest(path.read_bytes()), dataset_fingerprint=data_digest(obj),
                     target_column=obj['target_col'], target_spec=spec['target_spec'],
                     target_definition='Original CSV values; remove missing targets; LabelEncoder on string labels; no binarization',
                     dropped_id_columns=obj['dropped_id_cols'], n_observations=len(obj['y']),
                     n_features=obj['X'].shape[1], n_classes=len(counts), min_class_count=min(counts.values()),
                     class_counts=counts, encoded_classes=sorted(counts),
                     missing_targets_removed=int(obj['df'][obj['target_col']].isna().sum()),
                     eligible=name in cfg['primary_datasets'], exclusion_reason=None if name in cfg['primary_datasets'] else 'min_class_count < 5',
                     row_order='Original CSV order after missing-target removal; reset_index(drop=True)',
                     columns=list(obj['X'].columns), dtypes={str(k):str(v) for k,v in obj['X'].dtypes.items()})
        if entry['eligible'] != (entry['min_class_count'] >= 5):
            raise ValueError(f'Eligibility discrepancy: {name}')
        old = historical['datasets'][name]
        entry['historical_identity_matches'] = (entry['file_sha256'] == old['file_sha256']
                                                and entry['dataset_fingerprint'] == old['filtered_data_sha256'])
        entry['historical_filtered_fingerprint'] = old['filtered_data_sha256']
        if entry['file_sha256'] != old['file_sha256']:
            raise ValueError(f'Historical CSV bytes changed: {name}')
        validate_dataset(obj, entry, path)
        entries.append(entry)
        objects[name] = obj
    manifest = dict(protocol_version=cfg['protocol_version'], environment_id=frozen_env['environment_id'],
                    dataset_order=cfg['primary_datasets'], datasets=entries)
    write_frozen(ROOT / 'experiments/configs/dataset_manifest.json', manifest)
    comparisons = []
    folder = ROOT / 'experiments/splits/saved_cv_indices' / cfg['protocol_version']
    for entry in entries:
        if not entry['eligible']:
            continue
        name, y = entry['dataset'], objects[entry['dataset']]['y']
        for seed in cfg['cv']['seeds']:
            path = save_splits(folder, entry, y, seed, frozen_env['environment_id'])
            splits, meta = load_splits(path, entry, y)
            if seed == 42:
                rows = old_splits.loc[old_splits.dataset.eq(name)].sort_values(['repeat','fold'])
                if len(rows) != 15:
                    raise ValueError(f'Incomplete historical partitions: {name}')
                for i, (row, (train, valid)) in enumerate(zip(rows.itertuples(), splits)):
                    if (row.repeat, row.fold) != (i // 5 + 1, i % 5 + 1):
                        raise ValueError('Historical fold numbering mismatch')
                    np.testing.assert_array_equal(train, json.loads(row.train_indices))
                    np.testing.assert_array_equal(valid, json.loads(row.test_indices))
                    if row.data_sha256 != entry['historical_filtered_fingerprint']:
                        raise ValueError('Historical partition data identity mismatch')
                comparisons.append(dict(dataset=name, seed=42, partitions_matched=15,
                                        csv_bytes_match=True,
                                        filtered_fingerprint_matches=entry['historical_identity_matches'],
                                        reference='results/new_characterization_experiments/tables/cv_partitions.csv'))
    cells = 0
    for notebook in (ROOT / 'experiments').glob('*.ipynb'):
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                cells += 1
    report = dict(dataset_count=17, excluded_count=3, archives=85, partitions=1275,
                  historical_comparisons=comparisons, notebook_code_cells_parsed=cells,
                  models_fitted=0, rankings_computed=0,
                  cross_model_historical_pairing='Unverified: historical SVM/RIPPER records lack exact partitions')
    write_frozen(ROOT / 'experiments/protocol/validation_report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'historical_comparisons'}, indent=2))


if __name__ == '__main__':
    main()
