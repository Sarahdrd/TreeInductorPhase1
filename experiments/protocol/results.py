"""Fold record validation and exclusive JSONL serialization, without evaluation."""
import json
import math
from pathlib import Path
from .infrastructure import config

REQUIRED = ('experiment_id', 'protocol_version', 'dataset', 'dataset_fingerprint',
            'model', 'model_configuration', 'seed', 'repeat', 'fold', 'split_id',
            'macro_f1', 'environment_id')


def validate_record(record, split_metadata=None):
    missing = set(REQUIRED) - record.keys()
    if missing:
        raise ValueError(f'Missing fields: {sorted(missing)}')
    for key in ('experiment_id', 'protocol_version', 'dataset', 'dataset_fingerprint',
                'model', 'split_id', 'environment_id'):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f'Invalid {key}')
    if record['protocol_version'] != config()['protocol_version']:
        raise ValueError('Unsupported protocol; introduce an explicit schema version for future studies')
    if record['dataset'] not in config()['primary_datasets']:
        raise ValueError('Dataset outside benchmark')
    for key, allowed in [('seed', config()['cv']['seeds']), ('repeat', range(1, 4)), ('fold', range(1, 6))]:
        if type(record[key]) is not int or record[key] not in allowed:
            raise ValueError(f'Invalid {key}')
    score = record['macro_f1']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('macro_f1 must be a finite number in [0,1]')
    if not isinstance(record['model_configuration'], dict) or not record['model_configuration']:
        raise ValueError('Record explicit resolved estimator and preprocessing parameters')
    json.dumps(record, allow_nan=False)
    if split_metadata is not None:
        for key in ('dataset', 'dataset_fingerprint', 'seed', 'protocol_version'):
            if record[key] != split_metadata[key]:
                raise ValueError(f'Fold provenance mismatch: {key}')
        partition = split_metadata['partitions'][(record['repeat'] - 1) * 5 + record['fold'] - 1]
        if record['split_id'] != partition['split_id']:
            raise ValueError('Wrong split identity')
    return record


def save_records(path, records, split_metadata):
    """Validate the whole batch before writing; never overwrite an existing run.

    split_metadata maps (dataset, seed) to metadata returned by load_splits.
    Additional fields may store warnings, runtime, tree complexity, adapter version,
    sampling indices/seed, and model preprocessing identity.
    """
    records = list(records)
    seen = set()
    for record in records:
        validate_record(record, split_metadata[(record['dataset'], record['seed'])])
        identity = tuple(record[k] for k in ('experiment_id', 'dataset', 'model', 'seed', 'repeat', 'fold'))
        if identity in seen:
            raise ValueError('Duplicate fold record')
        seen.add(identity)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
