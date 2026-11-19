"""Versioned data identities and reusable CV partitions. Never fits estimators."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiments/configs/experiment_config.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def write_frozen(path, value):
    """Idempotent writes; refuse replacement of a different frozen artifact."""
    path = Path(path)
    content = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Frozen artifact differs: {path}; use a new protocol version')
        return
    with path.open('xb') as stream:
        stream.write(content)


def config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def environment():
    packages = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib',
                 'joblib', 'threadpoolctl', 'wittgenstein', 'nbformat', 'ipython']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    value = dict(python=platform.python_version(), packages=packages,
                 platform=platform.platform(), executable=sys.executable)
    value['environment_id'] = digest(json_bytes(value))
    return value


def data_digest(obj):
    # Same filtered-data identity as the historical ExperimentStore.
    return digest(pd.util.hash_pandas_object(obj['X'], index=True).values.tobytes()
                  + pd.util.hash_pandas_object(pd.Series(obj['y']), index=True).values.tobytes()
                  + repr([(str(c), str(t)) for c, t in obj['X'].dtypes.items()]).encode())


def validate_dataset(obj, entry, path):
    if digest(Path(path).read_bytes()) != entry['file_sha256']:
        raise ValueError('CSV bytes changed')
    if data_digest(obj) != entry['dataset_fingerprint']:
        raise ValueError('Dataset values, dtypes or row ordering changed')
    if obj['target_col'] != entry['target_column'] or obj['dropped_id_cols'] != entry['dropped_id_columns']:
        raise ValueError('Target or identifiers changed')
    if obj['X'].shape != (entry['n_observations'], entry['n_features']):
        raise ValueError('Dataset dimensions changed')
    counts = {str(k): int(v) for k, v in obj['y_raw'].value_counts().items()}
    if counts != entry['class_counts'] or not obj['X'].columns.is_unique:
        raise ValueError('Class distribution or predictor columns invalid')


def generate_splits(y, seed):
    cv = config()['cv']
    if seed not in cv['seeds']:
        raise ValueError('Seed outside frozen protocol')
    y = np.asarray(y)
    if y.ndim != 1 or len(np.unique(y)) < 2 or np.unique(y, return_counts=True)[1].min() < cv['folds']:
        raise ValueError('Dataset is ineligible for stratified five-fold CV')
    splits = [(a.astype('<i8'), b.astype('<i8')) for a, b in RepeatedStratifiedKFold(
        n_splits=cv['folds'], n_repeats=cv['repeats'], random_state=seed).split(np.zeros(len(y)), y)]
    validate_splits(splits, y)
    return splits


def validate_splits(splits, y):
    cv = config()['cv']
    n = len(y)
    if len(splits) != cv['folds'] * cv['repeats']:
        raise ValueError('Incomplete split collection')
    labels = np.unique(y)
    for train, valid in splits:
        for indices in (train, valid):
            if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
                raise ValueError('Indices must be nonempty integer vectors')
            if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= n:
                raise ValueError('Duplicate or out-of-bounds indices')
            if not np.array_equal(np.unique(np.asarray(y)[indices]), labels):
                raise ValueError('Every class must occur in both fold partitions')
        if np.intersect1d(train, valid).size or len(train) + len(valid) != n:
            raise ValueError('Partitions overlap or omit observations')
    for repeat in range(cv['repeats']):
        tests = [b for _, b in splits[repeat * cv['folds']:(repeat + 1) * cv['folds']]]
        if not np.array_equal(np.sort(np.concatenate(tests)), np.arange(n)):
            raise ValueError('Validation folds do not partition a repetition')
        for label in labels:
            sizes = [int(np.sum(np.asarray(y)[b] == label)) for b in tests]
            if max(sizes) - min(sizes) > 1:
                raise ValueError('Fold class allocation is not stratified')


def split_id(entry, seed, index, train, valid):
    return digest(json_bytes([config()['protocol_version'], entry['dataset'],
                             entry['dataset_fingerprint'], seed, index])
                  + train.astype('<i8').tobytes() + valid.astype('<i8').tobytes())


def save_splits(folder, entry, y, seed, environment_id):
    if not entry['eligible'] or entry['dataset'] not in config()['primary_datasets']:
        raise ValueError('Dataset outside primary benchmark')
    splits = generate_splits(y, seed)
    folder = Path(folder) / entry['dataset']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'seed_{seed}.npz'
    rows = [dict(repeat=i // 5 + 1, fold=i % 5 + 1,
                 split_id=split_id(entry, seed, i, a, b)) for i, (a, b) in enumerate(splits)]
    meta = dict(protocol_version=config()['protocol_version'], dataset=entry['dataset'],
                dataset_fingerprint=entry['dataset_fingerprint'], file_sha256=entry['file_sha256'],
                seed=seed, environment_id=environment_id, index_base=0,
                row_order='CSV order after missing-target filtering and index reset', partitions=rows)
    if path.exists() or path.with_suffix('.json').exists():
        old, oldmeta = load_splits(path, entry, y)
        if oldmeta != meta or any(not np.array_equal(a, c) or not np.array_equal(b, d)
                                  for (a, b), (c, d) in zip(old, splits)):
            raise ValueError('Refusing to replace saved partitions')
        return path
    arrays = {f'{part}_{i:02d}': value for i, pair in enumerate(splits)
              for part, value in zip(('train', 'valid'), pair)}
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_frozen(path.with_suffix('.json'), meta)
    load_splits(path, entry, y)
    return path


def load_splits(path, entry, y, *, verify_regeneration=True):
    """Validate saved indices; evaluation sets verify_regeneration=False.

    Call validate_dataset first to verify X and CSV identity as well as y.
    Structural, class-coverage and split-ID checks always run. Regeneration is
    reserved for preparation tests and is never needed by the fold runner.
    """
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    for field in ('dataset', 'dataset_fingerprint', 'file_sha256'):
        if meta[field] != entry[field]:
            raise ValueError(f'Split identity mismatch: {field}')
    if meta['protocol_version'] != config()['protocol_version'] or meta['seed'] not in config()['cv']['seeds']:
        raise ValueError('Split protocol mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {f'{p}_{i:02d}' for p in ('train', 'valid') for i in range(15)}:
            raise ValueError('Unexpected split archive contents')
        splits = [(archive[f'train_{i:02d}'], archive[f'valid_{i:02d}']) for i in range(15)]
    validate_splits(splits, y)
    rows = [dict(repeat=i // 5 + 1, fold=i % 5 + 1, split_id=split_id(entry, meta['seed'], i, a, b))
            for i, (a, b) in enumerate(splits)]
    if meta['partitions'] != rows:
        raise ValueError('Partition metadata or checksum mismatch')
    if verify_regeneration:
        expected = generate_splits(y, meta['seed'])
        if any(not np.array_equal(a, c) or not np.array_equal(b, d)
               for (a, b), (c, d) in zip(splits, expected)):
            raise ValueError('Splits differ from deterministic frozen generator')
    return splits, meta
