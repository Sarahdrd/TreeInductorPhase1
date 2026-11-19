"""Saved-fold evaluation only. Default CLI invocation is a read-only dry run."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
import logging
import math
from pathlib import Path
import re
import time
import traceback
import warnings

import numpy as np
from sklearn.metrics import f1_score

from .checkpoints import IntegrityError, Store, writer_lock
from .datasets import load_dataset
from .infrastructure import (ROOT, config, digest, environment, json_bytes,
                             load_splits, validate_dataset)
from .models import make_model
from .results import validate_record

LOG = logging.getLogger(__name__)
RUNNER_VERSION = 'saved-fold-runner-v1'
SAFE_FOLD_LIMIT = 20


def qualified(value):
    return value.__module__ + '.' + value.__qualname__


def parameters(value):
    """Stable full recursive estimator/preprocessor parameters, never memory repr."""
    if isinstance(value, np.generic):
        return parameters(value.item())
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'float': str(value)}
    if isinstance(value, type):
        return {'type': qualified(value)}
    if hasattr(value, 'get_params'):
        return {'class': qualified(type(value)), 'parameters': parameters(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(k): parameters(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [parameters(v) for v in value]
    if callable(value):
        return {'callable': qualified(value)}
    raise TypeError(f'Unsupported configuration value: {type(value)}')


def select(requested, allowed, label):
    selected = list(allowed) if requested is None else list(requested)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
        raise ValueError(f'Invalid/duplicate {label}: {selected}')
    return [x for x in allowed if x in selected]


def plan(datasets=None, models=None, seeds=None, repeats=None, folds=None):
    cfg = config()
    expected = set(cfg['canonical_models']) | {'svm_rbf', 'ripper', 'dummy_most_frequent', 'dummy_stratified'}
    regularized = [m for panel in cfg['regularization_panels'].values() for m in panel]
    if (len(cfg['primary_datasets']) != 17 or len(cfg['canonical_models']) != 4
            or len(regularized) != 8 or len(set(regularized)) != 8
            or set(cfg['models']) != expected | set(regularized) or len(cfg['models']) != 16
            or cfg['cv']['seeds'] != [42, 7, 21, 84, 123]
            or (cfg['cv']['folds'], cfg['cv']['repeats']) != (5, 3)):
        raise ValueError('Frozen benchmark/configuration inventory changed')
    return list(product(select(datasets, cfg['primary_datasets'], 'datasets'),
                        select(models, cfg['models'], 'models'),
                        select(seeds, cfg['cv']['seeds'], 'seeds'),
                        select(repeats, range(1, 4), 'repeats'),
                        select(folds, range(1, 6), 'folds')))


class Runner:
    def __init__(self, experiment_id='block2a-pilot', *, official=False, store_base=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', experiment_id):
            raise ValueError('Use a simple experiment ID (letters, digits, _, -, .)')
        self.cfg = config()
        self.env = environment()
        expected_env = json.loads((ROOT / self.cfg['runtime_environment']).read_text())
        if self.env != expected_env:
            raise ValueError('Runtime differs from the validated .venv environment')
        manifest_path = ROOT / 'experiments/configs/dataset_manifest.json'
        self.data_manifest = json.loads(manifest_path.read_text())
        if (self.data_manifest['dataset_order'] != self.cfg['primary_datasets']
                or self.data_manifest['protocol_version'] != self.cfg['protocol_version']):
            raise ValueError('Dataset manifest protocol/order mismatch')
        self.entries = {r['dataset']: r for r in self.data_manifest['datasets'] if r['eligible']}
        if set(self.entries) != set(self.cfg['primary_datasets']):
            raise ValueError('Eligible pool mismatch')
        self.experiment_id, self.official = experiment_id, official
        source_files = ['runner.py', 'checkpoints.py', 'datasets.py', 'preprocessing.py',
                        'models.py', 'recovered_cross_models.py', 'infrastructure.py', 'results.py']
        source = {p: digest((Path(__file__).parent / p).read_bytes()) for p in source_files}
        self.implementation_id = digest(json_bytes(source))
        self.manifest = dict(experiment_id=experiment_id, protocol_version=self.cfg['protocol_version'],
                             purpose='official' if official else 'pilot', environment=self.env,
                             runner_version=RUNNER_VERSION, source_sha256=source,
                             config_sha256=digest(json_bytes(self.cfg)),
                             dataset_manifest_sha256=digest(manifest_path.read_bytes()),
                             success_schema_sha256=digest((ROOT / 'experiments/configs/fold_result.schema.json').read_bytes()),
                             scoring=self.cfg['scoring'], config=self.cfg)
        base = Path(store_base) if store_base is not None else ROOT / 'experiments/results'
        self.store = Store(base / ('official' if official else 'pilots') / self.cfg['protocol_version'] / experiment_id,
                           self.manifest)
        self._data, self._splits, self._configs = {}, {}, {}

    def context(self, key):
        dataset, model, seed, repeat, fold = key
        entry = self.entries[dataset]
        if dataset not in self._data:
            obj = load_dataset(dataset, ROOT / entry['local_path'])
            validate_dataset(obj, entry, ROOT / entry['local_path'])
            self._data[dataset] = obj
        obj = self._data[dataset]
        if (dataset, seed) not in self._splits:
            path = ROOT / 'experiments/splits/saved_cv_indices' / self.cfg['protocol_version'] / dataset / f'seed_{seed}.npz'
            splits, meta = load_splits(path, entry, obj['y'], verify_regeneration=False)
            if meta['seed'] != seed or meta['environment_id'] != self.data_manifest['environment_id']:
                raise IntegrityError('Saved split provenance does not match requested seed/environment')
            self._splits[dataset, seed] = (splits, meta, path, digest(path.read_bytes()), digest(path.with_suffix('.json').read_bytes()))
        splits, meta, *_ = self._splits[dataset, seed]
        if (dataset, model) not in self._configs:
            estimator = make_model(model, obj['X'], obj['y'])
            self._configs[dataset, model] = dict(frozen_parameters=self.cfg['models'][model],
                                                resolved_pipeline=parameters(estimator))
        configuration = self._configs[dataset, model]
        index = (repeat - 1) * 5 + fold - 1
        expected = dict(experiment_id=self.experiment_id, protocol_version=self.cfg['protocol_version'],
                        dataset=dataset, dataset_fingerprint=entry['dataset_fingerprint'],
                        model=model, model_configuration=configuration,
                        configuration_id=digest(json_bytes(configuration)), seed=seed, repeat=repeat,
                        fold=fold, split_id=meta['partitions'][index]['split_id'],
                        environment_id=self.env['environment_id'], implementation_id=self.implementation_id,
                        runner_version=RUNNER_VERSION, purpose='official' if self.official else 'pilot')
        expected['evaluation_id'] = digest(json_bytes(expected))
        return obj, splits[index], meta, expected

    def validate_attempt(self, record, expected, split_meta):
        if type(record.get('attempt')) is not int or record['attempt'] < 1:
            raise IntegrityError('Invalid attempt number')
        if any(record.get(k) != v for k, v in expected.items()):
            raise IntegrityError('Fold provenance/configuration mismatch')
        if record.get('record_version') != 'fold-attempt-v1':
            raise IntegrityError('Unsupported attempt version')
        runtime = record.get('runtime_seconds')
        if type(runtime) not in (int, float) or not math.isfinite(runtime) or runtime < 0:
            raise IntegrityError('Invalid runtime')
        if not isinstance(record.get('warnings'), list) or not all(
                isinstance(w, dict) and set(w) == {'category', 'message', 'count'}
                and isinstance(w['category'], str) and isinstance(w['message'], str)
                and type(w['count']) is int and w['count'] > 0 for w in record['warnings']):
            raise IntegrityError('Invalid warning records')
        try:
            for field in ('started_at', 'finished_at'):
                if datetime.fromisoformat(record[field]).tzinfo is None:
                    raise ValueError('Missing UTC offset')
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError('Invalid timestamps') from exc
        if record.get('execution_status') == 'success':
            validate_record(record, split_meta)
            if record.get('error') is not None:
                raise IntegrityError('Successful record has an error')
        elif record.get('execution_status') == 'failed':
            if record.get('macro_f1') is not None or not isinstance(record.get('error'), dict):
                raise IntegrityError('Failure contains a score or lacks error information')
            if not all(isinstance(record['error'].get(k), str) for k in ('type', 'message', 'traceback')):
                raise IntegrityError('Invalid failure diagnostics')
        else:
            raise IntegrityError('Invalid execution status')

    def audit(self, tasks):
        if config() != self.cfg or environment() != self.env:
            raise IntegrityError('Configuration/environment changed since runner initialization')
        if any(digest((Path(__file__).parent / p).read_bytes()) != h
               for p, h in self.manifest['source_sha256'].items()):
            raise IntegrityError('Implementation files changed; restart with a new experiment ID')
        if digest((ROOT / 'experiments/configs/dataset_manifest.json').read_bytes()) != self.manifest['dataset_manifest_sha256']:
            raise IntegrityError('Dataset manifest changed')
        # A new audit must not trust data or splits cached by a prior invocation.
        self._data.clear()
        self._splits.clear()
        self._configs.clear()
        self.store.check_manifest()
        states, issues = {}, []
        for key in tasks:
            try:
                _, _, meta, expected = self.context(key)
                attempts = self.store.attempts(key)
                for record in attempts:
                    self.validate_attempt(record, expected, meta)
                states[key] = attempts[-1]['execution_status'] if attempts else 'pending'
            except (ValueError, KeyError, TypeError, OSError) as exc:
                states[key] = 'corrupt'
                issues.append(dict(key=list(key), error=str(exc)))
        return dict(planned=len(tasks), completed=sum(v == 'success' for v in states.values()),
                    failed=sum(v == 'failed' for v in states.values()),
                    pending=sum(v == 'pending' for v in states.values()),
                    corrupt=sum(v == 'corrupt' for v in states.values()), issues=issues,
                    orphan_temporary_files=len(list(self.store.root.rglob('*.tmp'))) if self.store.root.exists() else 0), states

    def evaluate(self, key, attempt):
        obj, (train, valid), meta, expected = self.context(key)
        dataset, name, seed, _, _ = key
        # Recheck input integrity immediately before every actual fit.
        validate_dataset(obj, self.entries[dataset], ROOT / self.entries[dataset]['local_path'])
        _, _, path, data_hash, meta_hash = self._splits[dataset, seed]
        if digest(path.read_bytes()) != data_hash or digest(path.with_suffix('.json').read_bytes()) != meta_hash:
            raise IntegrityError('Saved partitions changed during execution')
        start = time.perf_counter()
        record = dict(expected, record_version='fold-attempt-v1', attempt=attempt,
                      started_at=datetime.now(timezone.utc).isoformat(), macro_f1=None,
                      execution_status='failed', error=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                estimator = make_model(name, obj['X'], obj['y'])
                if parameters(estimator) != expected['model_configuration']['resolved_pipeline']:
                    raise IntegrityError('Fresh estimator configuration changed')
                estimator.fit(obj['X'].iloc[train], obj['y'][train])
                predicted = np.asarray(estimator.predict(obj['X'].iloc[valid]))
                if predicted.shape != (len(valid),) or not np.isin(predicted, np.unique(obj['y'])).all():
                    raise ValueError('Invalid prediction shape or label space')
                record['macro_f1'] = float(f1_score(obj['y'][valid], predicted, average='macro', zero_division=0))
                record['execution_status'] = 'success'
                fitted = estimator.named_steps['model'] if hasattr(estimator, 'named_steps') else estimator
                if hasattr(fitted, 'tree_'):
                    record['tree_complexity'] = dict(nodes=int(fitted.tree_.node_count),
                                                     depth=int(fitted.tree_.max_depth), leaves=int(fitted.tree_.n_leaves))
            except IntegrityError:
                raise
            except Exception as exc:
                record['macro_f1'] = None
                record['execution_status'] = 'failed'
                record['error'] = dict(type=qualified(type(exc)), message=str(exc), traceback=traceback.format_exc())
        unique = {}
        for warning in caught:
            k = (warning.category.__name__, str(warning.message))
            unique[k] = unique.get(k, 0) + 1
        record['warnings'] = [dict(category=c, message=m, count=n) for (c, m), n in unique.items()]
        record['runtime_seconds'] = time.perf_counter() - start
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        self.validate_attempt(record, expected, meta)
        return record

    def run(self, tasks, *, execute=False, resume=False, retry_failed=False,
            max_new_folds=None, authorize_full_run=False):
        # Validate even API callers, not just the CLI selections.
        allowed = set(plan())
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= allowed:
            raise ValueError('Invalid task selection')
        if max_new_folds is not None and (type(max_new_folds) is not int or max_new_folds < 0):
            raise ValueError('max_new_folds must be a nonnegative integer')
        if execute and not authorize_full_run and (max_new_folds is None or max_new_folds > SAFE_FOLD_LIMIT):
            raise ValueError('Execution requires --max-new-folds 0..20 or --authorize-full-run')
        if retry_failed and not resume:
            raise ValueError('--retry-failed requires --resume')
        if not execute:
            summary, _ = self.audit(tasks)
            summary.update(mode='dry-run', max_new_folds=max_new_folds,
                           planned_new_attempts=min(summary['pending'] + (summary['failed'] if retry_failed else 0),
                                                    max_new_folds if max_new_folds is not None else len(tasks)))
            return summary
        started = time.perf_counter()
        with writer_lock(self.store.root):
            summary, states = self.audit(tasks)
            if summary['corrupt']:
                raise IntegrityError(f'Preflight rejected corrupt/incompatible records: {summary["issues"]}')
            if not resume and any(v != 'pending' for v in states.values()):
                raise ValueError('Existing attempts require --resume; nothing was retrained')
            self.store.initialize()
            executed, skipped = 0, 0
            for key in tasks:
                state = states[key]
                if state == 'success' or (state == 'failed' and not retry_failed):
                    skipped += 1
                    continue
                if max_new_folds is not None and executed >= max_new_folds:
                    continue
                attempts = self.store.attempts(key)
                LOG.info('Starting dataset=%s model=%s seed=%s repeat=%s fold=%s', *key)
                record = self.evaluate(key, len(attempts) + 1)
                self.store.save(key, record)
                states[key] = record['execution_status']
                executed += 1
                LOG.info('Saved %s; completed=%d failed=%d remaining=%d skipped=%d runtime=%.2fs',
                         record['execution_status'], sum(v == 'success' for v in states.values()),
                         sum(v == 'failed' for v in states.values()), sum(v == 'pending' for v in states.values()),
                         skipped, time.perf_counter() - started)
            final, _ = self.audit(tasks)
            final.update(mode='execute', executed=executed, skipped=skipped,
                         runtime_seconds=time.perf_counter() - started)
            return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Default; validates/plans without fitting or writing')
    mode.add_argument('--status', action='store_true', help='Read-only integrity/progress audit')
    parser.add_argument('--experiment-id', default='block2a-pilot')
    parser.add_argument('--official', action='store_true', help='Separate official store (default is pilot)')
    for name in ('datasets', 'models', 'seeds', 'repeats', 'folds'):
        parser.add_argument('--' + name, nargs='+', type=int if name in ('seeds', 'repeats', 'folds') else str)
    parser.add_argument('--max-new-folds', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--authorize-full-run', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        tasks = plan(**{k: getattr(args, k) for k in ('datasets', 'models', 'seeds', 'repeats', 'folds')})
        runner = Runner(args.experiment_id, official=args.official)
        if args.status:
            summary, _ = runner.audit(tasks)
            summary['mode'] = 'status'
        else:
            summary = runner.run(tasks, execute=args.execute, resume=args.resume,
                                 retry_failed=args.retry_failed, max_new_folds=args.max_new_folds,
                                 authorize_full_run=args.authorize_full_run)
        summary['selection'] = {k: list(dict.fromkeys(t[i] for t in tasks))
                                for i, k in enumerate(('datasets', 'models', 'seeds', 'repeats', 'folds'))}
        summary['store'] = str(runner.store.root)
        print(json.dumps(summary, indent=2))
        return 1 if summary.get('corrupt') or (args.execute and summary['failed']) else 0
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error('%s', exc)
        return 2
    except KeyboardInterrupt:
        LOG.warning('Interrupted; committed attempts remain valid. Resume to continue pending folds.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
