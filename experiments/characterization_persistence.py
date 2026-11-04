"""Auditable persistence for the new characterization notebook (no model fitting)."""
import hashlib
import json
import types
from importlib.metadata import version
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def data_digest(obj):
    return digest(pd.util.hash_pandas_object(obj['X'], index=True).values.tobytes()
                  + pd.util.hash_pandas_object(pd.Series(obj['y']), index=True).values.tobytes()
                  + repr([(str(c), str(t)) for c, t in obj['X'].dtypes.items()]).encode())


def code_digest(function):
    # Ignore notebook execution counters, filenames and line numbers in code identity.
    def semantic(code):
        return [code.co_code.hex(), code.co_names, code.co_varnames,
                [semantic(c) if isinstance(c, types.CodeType) else repr(c) for c in code.co_consts]]
    return digest(json.dumps(semantic(function.__code__), sort_keys=True).encode())


class ExperimentStore:
    def __init__(self, root, data, paths, functions, target_config, id_columns, missing_tokens):
        self.root = Path(root)
        self.cache = self.root / 'cache'
        self.tables = self.root / 'tables'
        for folder in (self.cache, self.tables, self.root / 'figures'):
            folder.mkdir(parents=True, exist_ok=True)
        self.splits, self.identities, self.events = {}, {}, []
        versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                        scipy=scipy.__version__, sklearn=sklearn.__version__,
                        matplotlib=version('matplotlib'), joblib=version('joblib'), threadpoolctl=version('threadpoolctl'))
        self.protocol = dict(versions=versions, cv=dict(folds=5, repeats=3, seed=42),
                             scoring='macro_f1; zero_division=0', std_ddof=1,
                             code={f.__name__: code_digest(f) for f in functions},
                             target_config=target_config, id_columns=id_columns, missing_tokens=missing_tokens)
        split_rows = []
        for name, obj in data.items():
            identity = dict(path=str(Path(paths[name]).resolve()), file_sha256=digest(Path(paths[name]).read_bytes()),
                            filtered_data_sha256=data_digest(obj), target=obj['target_col'],
                            resolved_ids=obj['dropped_id_cols'], n_samples=len(obj['y']))
            self.identities[name] = identity
            eligible = pd.Series(obj['y']).value_counts().min() >= 5
            splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(obj['X'], obj['y'])) if eligible else []
            self.splits[name] = splits
            for k, (train, test) in enumerate(splits):
                split_rows.append(dict(dataset=name, repeat=k // 5 + 1, fold=k % 5 + 1,
                    data_sha256=identity['filtered_data_sha256'], train_indices=json.dumps(train.tolist()),
                    test_indices=json.dumps(test.tolist()), split_sha256=digest(train.tobytes()+test.tobytes())))
        self.split_table = pd.DataFrame(split_rows)
        self.split_table.to_csv(self.tables / 'cv_partitions.csv', index=False)
        write_json(self.root / 'metadata.json', dict(protocol=self.protocol, datasets=self.identities))

    def signature(self, dataset, model, estimator, n_jobs):
        params = {k: repr(v) for k, v in estimator.get_params(deep=True).items()}
        spec = dict(protocol=self.protocol, data=self.identities[dataset], model=model,
                    resolved_pipeline_parameters=params, n_jobs=n_jobs,
                    splits=self.split_table.loc[self.split_table.dataset.eq(dataset), 'split_sha256'].tolist())
        return digest(json.dumps(spec, sort_keys=True).encode()), spec

    def read(self, dataset, model, signature):
        path = self.cache / f'{dataset}__{model}.json'
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload['signature'] != signature:
                raise ValueError('incompatible data/protocol/preprocessing/versions/parameters')
            row = payload['row']
            validate_evaluations(pd.DataFrame([row]), [dataset], [model])
            self.events.append(dict(dataset=dataset, model=model, status='cache_hit'))
            return row
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            message = f'Cache rejected for {dataset}/{model}: {exc}'
            print(message)
            self.events.append(dict(dataset=dataset, model=model, status=message))
            return None

    def save(self, row, signature, spec):
        # Failed evaluations remain in exported tables, never reusable cache entries.
        if row['error'] or row['n_scores'] != 15:
            return
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        write_json(self.cache / f"{row['dataset']}__{row['model']}.json",
                   dict(signature=signature, specification=spec, row=clean))
        self.events.append(dict(dataset=row['dataset'], model=row['model'], status='fitted'))

    def export(self, frame, stem):
        serialized = frame.copy()
        for col in ('scores', 'tree_nodes', 'tree_depths', 'tree_leaves'):
            serialized[col] = serialized[col].map(json.dumps)
        serialized.to_csv(self.tables / f'{stem}_results.csv', index=False)
        rows = []
        for r in frame.to_dict('records'):
            for k, score in enumerate(r['scores']):
                train, test = self.splits[r['dataset']][k]
                fold = dict(dataset=r['dataset'], model=r['model'], cv_seed=r['cv_seed'],
                            repeat=k // 5 + 1, fold=k % 5 + 1, macro_f1=score,
                            data_sha256=self.identities[r['dataset']]['filtered_data_sha256'],
                            split_sha256=digest(train.tobytes()+test.tobytes()))
                for key in ('tree_nodes', 'tree_depths', 'tree_leaves'):
                    fold[key] = r[key][k] if r[key] else np.nan
                rows.append(fold)
        pd.DataFrame(rows).to_csv(self.tables / f'{stem}_fold_scores.csv', index=False)
        pd.DataFrame(self.events).to_csv(self.tables / 'cache_events.csv', index=False)


def validate_evaluations(frame, pool, models):
    if len(models) != len(set(models)) or any(m.startswith('dummy') for m in models):
        # Dummy validation is supported individually for cache, but never in panels.
        if len(models) != 1 or not models[0].startswith('dummy'):
            raise ValueError('Duplicate or dummy models in required panel')
    selected = frame.loc[frame.dataset.isin(pool) & frame.model.isin(models)]
    expected = {(d, m) for d in pool for m in models}
    if selected.duplicated(['dataset', 'model']).any() or set(zip(selected.dataset, selected.model)) != expected:
        raise ValueError('Missing or duplicate dataset/model evaluations')
    for row in selected.to_dict('records'):
        scores = np.asarray(row['scores'], dtype=float)
        if row['cv_seed'] != 42 or row['cv_folds'] != 5 or row['cv_repeats'] != 3:
            raise ValueError('Mixed or invalid CV protocol')
        if row['error'] or row['n_scores'] != 15 or scores.shape != (15,) or not np.isfinite(scores).all():
            raise ValueError(f"Incomplete evaluation: {row['dataset']}/{row['model']}: {row['error']}")
        if not np.allclose([row['f1_macro_mean'], row['f1_macro_std']], [scores.mean(), scores.std(ddof=1)], rtol=1e-12, atol=1e-14):
            raise ValueError('Stored summary differs from fold scores')
        if row['model'].startswith('decision_tree'):
            for key, mean in [('tree_nodes', 'mean_tree_nodes'), ('tree_depths', 'mean_tree_depth'), ('tree_leaves', 'mean_tree_leaves')]:
                values = np.asarray(row[key], dtype=float)
                if values.shape != (15,) or not np.isfinite(values).all() or not np.isclose(values.mean(), row[mean]):
                    raise ValueError('Invalid tree complexity scores')
    return selected
