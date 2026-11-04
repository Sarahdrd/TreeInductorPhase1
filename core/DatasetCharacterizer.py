"""Standalone implementation of the canonical Dataset_Characterization notebook.

Methodology: cells 4, 8, 27, 29 and 30 (zero-based). No notebook is read
at runtime. Labels are relative to the entire successfully evaluated pool.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ColumnSpec = str | int | Sequence[str | int]


TARGET_CONFIG = {'abalone': 'target',
 'anneal': 'target',
 'arrhythmia': 'target',
 'bridges': 'target',
 'balance_scale': ['class', 'Class', 'balance', 'Balance', 0],
 'car': 'target',
 'cylinder_bands': 'target',
 'glass': 'target',
 'heart_disease': '13',
 'ionosphere': '34',
 'semeion': 'target',
 'sonar': '60',
 'tic_tac_toe': '9',
 'breast_cancer': 'target',
 'wine_red': 'target',
 'iris': 'target',
 'haberman': '3',
 'parkinsons': ['status', 'Status'],
 'banknote': '4',
 'credit_approval': '15'}

ID_COLUMNS = {'abalone': [],
 'anneal': [],
 'arrhythmia': [],
 'bridges': [],
 'balance_scale': [],
 'car': [],
 'cylinder_bands': ['feature_001'],
 'glass': [],
 'heart_disease': [],
 'ionosphere': [],
 'semeion': [],
 'sonar': [],
 'tic_tac_toe': [],
 'breast_cancer': [],
 'wine_red': [],
 'iris': [],
 'haberman': [],
 'parkinsons': [['name', 'Name']],
 'banknote': [],
 'credit_approval': []}

MISSING_TOKENS = ["?", "NA", "N/A", "na", "n/a", "", " ", "null", "None"]

def _resolve_column(df: pd.DataFrame, spec: ColumnSpec, dataset_name: str, purpose: str) -> str:
    """Resolve a column spec that can be a name, integer position, or fallback list."""
    specs = spec if isinstance(spec, (list, tuple)) else [spec]

    for candidate in specs:
        if isinstance(candidate, int):
            if -len(df.columns) <= candidate < len(df.columns):
                return df.columns[candidate]
        elif candidate in df.columns:
            return candidate

    available = list(df.columns)
    raise ValueError(
        f"Could not resolve {purpose} for {dataset_name}. "
        f"Tried {specs}; available columns: {available}"
    )

def _resolve_existing_columns(df: pd.DataFrame, specs: Sequence[ColumnSpec]) -> list[str]:
    """Return only the configured columns that exist in this concrete CSV."""
    resolved = []
    for spec in specs:
        try:
            col = _resolve_column(df, spec, "dataset", "ID column")
            if col not in resolved:
                resolved.append(col)
        except ValueError as exc:
            raise ValueError("Configured identifier is missing") from exc
    return resolved

def build_preprocessor(X: pd.DataFrame, for_linear: bool = False) -> ColumnTransformer:
    """Build fold-fitted numeric and categorical transformations."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if for_linear:
        numeric_steps.append(("scaler", StandardScaler()))

    # Use sparse_output when available; fall back for older scikit-learn.
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")

CV_N_SPLITS, CV_N_REPEATS, CV_RANDOM_STATE = 5, 3, 42

N_JOBS = 1

NON_DUMMY_BASELINE_MODELS = ['decision_tree_default', 'decision_tree_balanced',
                            'logistic_regression', 'logistic_regression_balanced']

TREE_BASELINE_MODELS = NON_DUMMY_BASELINE_MODELS[:2]

DUMMY_REFERENCE_MODELS = ['dummy_most_frequent', 'dummy_stratified']

TREE_CONFIG = dict(criterion='gini', splitter='best', max_depth=None,
                   min_samples_split=2, min_samples_leaf=1, max_features=None,
                   min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                   min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0,
                   random_state=42)

LR_CONFIG = dict(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000,
                 tol=1e-4, class_weight=None, random_state=42, fit_intercept=True)

MODEL_CONFIGS = {
    'decision_tree_default': TREE_CONFIG,
    'decision_tree_balanced': {**TREE_CONFIG, 'class_weight': 'balanced'},
    'logistic_regression': LR_CONFIG,
    'logistic_regression_balanced': {**LR_CONFIG, 'class_weight': 'balanced'},
    'dummy_most_frequent': dict(strategy='most_frequent', random_state=42),
    'dummy_stratified': dict(strategy='stratified', random_state=42),
}

def make_baseline(name: str, X: pd.DataFrame) -> Any:
    """Construct one of the six fixed notebook baselines."""
    config = MODEL_CONFIGS[name]
    if name in DUMMY_REFERENCE_MODELS:
        return DummyClassifier(**config)
    linear = name.startswith('logistic')
    estimator = LogisticRegression(**config) if linear else DecisionTreeClassifier(**config)
    return Pipeline([('preprocess', build_preprocessor(X, for_linear=linear)),
                     ('model', estimator)])

def evaluate_baselines(data: Mapping[str, dict[str, Any]], cv_seed: int = 42, include_dummy: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate common splits and preserve per-model scores, warnings and errors."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    rows, audits = [], []
    models = NON_DUMMY_BASELINE_MODELS + (DUMMY_REFERENCE_MODELS if include_dummy else [])
    for name, obj in data.items():
        X, y = obj['X'], obj['y']
        minimum = int(np.bincount(y).min())
        eligible = minimum >= CV_N_SPLITS
        issue = '' if eligible else f'min_class_count={minimum} < 5'
        audits.append(dict(dataset=name, min_class_count=minimum, comparable_cv=eligible,
                           cv_issue=issue, cv_n_splits=5, cv_n_repeats=3, cv_seed=cv_seed))
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                      random_state=cv_seed).split(X, y)) if eligible else []
        for model_name in models:
            row = dict(dataset=name, model=model_name, cv_seed=cv_seed, cv_folds=5,
                cv_repeats=3, n_scores=0, scores=[], error=issue, warnings='',
                f1_macro_mean=np.nan, f1_macro_std=np.nan, f1_macro_min=np.nan,
                f1_macro_max=np.nan, runtime_seconds=np.nan, mean_tree_nodes=np.nan,
                mean_tree_depth=np.nan, mean_tree_leaves=np.nan)
            caught = []
            if eligible:
                try:
                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        result = cross_validate(make_baseline(model_name, X), X, y,
                            cv=splits, scoring=make_scorer(f1_score, average='macro', zero_division=0),
                            n_jobs=N_JOBS, error_score='raise',
                            return_estimator=model_name in TREE_BASELINE_MODELS)
                    scores = result['test_score']
                    if len(scores) != 15 or not np.isfinite(scores).all():
                        raise ValueError('Invalid CV scores')
                    row.update(n_scores=len(scores), scores=scores.tolist(),
                        f1_macro_mean=scores.mean(), f1_macro_std=scores.std(ddof=1),
                        f1_macro_min=scores.min(), f1_macro_max=scores.max(),
                        runtime_seconds=time.perf_counter()-start)
                    if model_name in TREE_BASELINE_MODELS:
                        trees = [e.named_steps['model'].tree_ for e in result['estimator']]
                        row.update(mean_tree_nodes=np.mean([t.node_count for t in trees]),
                            mean_tree_depth=np.mean([t.max_depth for t in trees]),
                            mean_tree_leaves=np.mean([t.n_leaves for t in trees]))
                except Exception as exc:
                    row['error'] = repr(exc)
                    print(f'[FIT ERROR] {name}/{model_name}: {exc}')
                row['warnings'] = '\n'.join(sorted({f'{w.category.__name__}: {w.message}' for w in caught}))
                if row['warnings']:
                    print(f'[WARNINGS] {name}/{model_name}: {row["warnings"][:300]}')
            rows.append(row)
        print(f'CV seed {cv_seed}: {name} completed; comparable={eligible}')
    return pd.DataFrame(rows), pd.DataFrame(audits)

def successful_rows(frame: pd.DataFrame) -> pd.Series:
    """Identify complete, finite fifteen-score model evaluations."""
    return (frame['error'].fillna('').eq('') & frame['n_scores'].eq(15)
            & frame['scores'].map(lambda s: len(s) == 15 and np.isfinite(s).all())
            & frame[['f1_macro_mean', 'f1_macro_std']].notna().all(axis=1))

def aggregate_baselines(frame: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only complete four-model protocols; retain dummy references."""
    good = frame.loc[successful_rows(frame) & frame.model.isin(NON_DUMMY_BASELINE_MODELS)]
    agg = good.groupby('dataset').agg(n_successful_non_dummy_models=('model','nunique'),
        baseline_macro_f1_mean=('f1_macro_mean','mean'),
        baseline_macro_f1_best=('f1_macro_mean','max'),
        baseline_macro_f1_worst=('f1_macro_mean','min'),
        mean_model_cv_f1_std=('f1_macro_std','mean')).reset_index()
    agg = audit.merge(agg, on='dataset', how='left', validate='one_to_one')
    complete = pd.Series({name: len(g) == 4
        and set(g.model) == set(NON_DUMMY_BASELINE_MODELS)
        and successful_rows(g).all()
        for name, g in frame[frame.model.isin(NON_DUMMY_BASELINE_MODELS)].groupby('dataset')}, dtype=bool)
    agg['complete_non_dummy_protocol'] = agg.dataset.map(complete).fillna(False).astype(bool)
    agg['ranking_eligible'] = (agg.comparable_cv & agg.min_class_count.ge(5)
        & agg.complete_non_dummy_protocol
        & agg[['baseline_macro_f1_mean','mean_model_cv_f1_std']].notna().all(axis=1))
    agg['ranking_exclusion_reason'] = agg.apply(lambda r: '' if r.ranking_eligible else
        (r.cv_issue or 'Incomplete four-model protocol: inspect baseline_results.csv errors and score counts'), axis=1)
    # Partial models remain in baseline_df for inspection, never canonical aggregates.
    agg.loc[~agg.ranking_eligible, ['baseline_macro_f1_mean','baseline_macro_f1_best',
        'baseline_macro_f1_worst','mean_model_cv_f1_std']] = np.nan
    agg['baseline_macro_f1_spread'] = agg.baseline_macro_f1_best - agg.baseline_macro_f1_worst
    agg['cv_variability'] = agg.mean_model_cv_f1_std
    dummy = frame.loc[successful_rows(frame) & frame.model.isin(DUMMY_REFERENCE_MODELS)]
    agg['best_dummy_macro_f1'] = agg.dataset.map(dummy.groupby('dataset').f1_macro_mean.max())
    agg['baseline_gain_over_dummy'] = agg.baseline_macro_f1_mean - agg.best_dummy_macro_f1
    return agg

def minmax_normalize_on_pool(series: pd.Series, pool_mask: pd.Series) -> pd.Series:
    """Min-max normalize using only the eligible benchmark pool."""
    values = pd.to_numeric(series, errors="coerce")
    pool_mask = pd.Series(pool_mask, index=values.index).fillna(False).astype(bool)

    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pool_mask & values.notna()

    if valid.sum() == 0:
        return result

    min_val = values.loc[valid].min()
    max_val = values.loc[valid].max()

    if max_val == min_val:
        result.loc[valid] = 0.0
        return result

    result.loc[valid] = (
        values.loc[valid] - min_val
    ) / (
        max_val - min_val
    )

    return result

def assign_predictive_difficulty_groups(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sort exact ties deterministically and split eligible ranks into thirds."""
    labels = pd.Series("review_separately", index=df.index, dtype="object")
    ranks = pd.Series(np.nan, index=df.index, dtype=float)

    valid = (
        df["ranking_eligible"]
        & df["predictive_difficulty_score"].notna()
        & df["baseline_macro_f1_mean"].notna()
        & df["cv_variability"].notna()
    )

    if valid.sum() == 0:
        return labels, ranks

    ranked = df.loc[
        valid,
        [
            "dataset",
            "predictive_difficulty_score",
            "baseline_macro_f1_mean",
            "cv_variability",
        ],
    ].copy()

    # Exact equality implements SCORE_TIE_ATOL = 0 consistently in all analyses.
    ranked["_score_tie_key"] = (
        ranked["predictive_difficulty_score"]
    )

    ranked = ranked.sort_values(
        [
            "_score_tie_key",
            "baseline_macro_f1_mean",
            "cv_variability",
            "dataset",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    ranks.loc[ranked.index] = np.arange(1, len(ranked) + 1)

    thirds = np.array_split(ranked.index.to_numpy(), 3)

    for label, indices in zip(
        ["easy", "intermediate", "hard"],
        thirds,
    ):
        labels.loc[indices] = label

    return labels, ranks


def _compute_pool_profile(
    baselines: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    """Derive and normalize all dimensions after evaluation of the full pool."""
    profile = aggregate_baselines(baselines, audit)
    good = baselines.loc[successful_rows(baselines)
                         & baselines.model.isin(NON_DUMMY_BASELINE_MODELS)]
    runtime = good.groupby('dataset').runtime_seconds.mean()
    trees = good.loc[good.model.isin(TREE_BASELINE_MODELS)].groupby('dataset')
    profile['baseline_runtime_seconds_mean'] = profile.dataset.map(runtime)
    for column in ['mean_tree_nodes', 'mean_tree_depth', 'mean_tree_leaves']:
        profile[column] = profile.dataset.map(trees[column].mean())
    profile['predictive_error'] = 1 - profile.baseline_macro_f1_mean
    profile['strategy_sensitivity'] = profile.baseline_macro_f1_spread
    for source, target in [
        ('predictive_error', 'predictive_error_score'),
        ('cv_variability', 'cv_variability_score'),
        ('strategy_sensitivity', 'strategy_sensitivity_score'),
        ('baseline_runtime_seconds_mean', 'runtime_burden_score'),
        ('mean_tree_nodes', 'tree_size_burden_score'),
    ]:
        profile[target] = minmax_normalize_on_pool(profile[source], profile.ranking_eligible)
    profile['predictive_difficulty_score'] = (
        .70 * profile.predictive_error_score + .30 * profile.cv_variability_score)
    profile['computational_burden_score'] = (
        profile.runtime_burden_score + profile.tree_size_burden_score) / 2
    profile['predictive_difficulty_label'], profile['predictive_difficulty_rank'] = (
        assign_predictive_difficulty_groups(profile))
    return profile


class DatasetCharacterizer:
    """Characterize CSV datasets using the notebook's fixed seed-42 protocol.

    Supply a mapping of dataset names to CSV paths. Known notebook datasets have
    explicit target/ID defaults. For new datasets provide both ``target_config``
    and ``id_columns`` (an empty ID list is an explicit declaration).

    ``summary_`` retains all inputs, including failures and review-only datasets.
    ``baseline_results_`` retains the 15 scores, sample SD, errors and warnings
    per model. ``cv_audit_`` documents CV eligibility. Calling characterize_pool
    again replaces previous results. Runtime measurements are not deterministic.
    """

    def __init__(
        self,
        *,
        target_config: Mapping[str, ColumnSpec] | None = None,
        id_columns: Mapping[str, Sequence[ColumnSpec]] | None = None,
    ) -> None:
        self.target_config = deepcopy(TARGET_CONFIG)
        self.target_config.update(deepcopy(dict(target_config or {})))
        self.id_columns = deepcopy(ID_COLUMNS)
        self.id_columns.update(deepcopy(dict(id_columns or {})))
        self.summary_: pd.DataFrame | None = None
        self.baseline_results_ = pd.DataFrame()
        self.cv_audit_ = pd.DataFrame()
        self._datasets: dict[str, Path] = {}

    @staticmethod
    def default_pool(data_dir: str | Path | None = None) -> dict[str, Path]:
        """Return the notebook's 20-file pool; do not include other data CSVs."""
        root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / 'data'
        filenames = {'bridges': 'bridges_version1', 'wine_red': 'winequality_red',
                     'banknote': 'banknote_authentication'}
        return {name: root / (filenames.get(name, name) + '.csv') for name in TARGET_CONFIG}

    def _load_dataset(self, name: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply explicit targets, identifier removal and missing-target filtering."""
        df = pd.read_csv(path, na_values=MISSING_TOKENS)
        df.columns = [str(c).strip() for c in df.columns]
        if not df.columns.is_unique:
            raise ValueError('Duplicate column names after stripping whitespace')
        if name not in self.target_config or name not in self.id_columns:
            raise ValueError(f'Missing explicit target/ID configuration for {name}')
        target = _resolve_column(df, self.target_config[name], name, 'target column')
        ids = [c for c in _resolve_existing_columns(df, self.id_columns[name]) if c != target]
        valid = df[target].notna()
        X = df.loc[valid].drop(columns=[target] + ids).reset_index(drop=True)
        y_raw = df.loc[valid, target].reset_index(drop=True)
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        counts = np.bincount(y)
        probabilities = counts / counts.sum() if len(counts) else np.array([])
        numeric = len(X.select_dtypes(include=[np.number]).columns)
        record = dict(dataset=name, target_col=target, dropped_id_cols=', '.join(ids),
            n_samples=len(X), n_features=X.shape[1], n_classes=len(counts),
            n_numeric=numeric, n_categorical=X.shape[1] - numeric,
            n_missing_targets_removed=int((~valid).sum()),
            min_class_count=int(counts.min()) if len(counts) else 0,
            missing_pct=X.isna().mean().mean() * 100,
            class_entropy=float(-(probabilities * np.log2(probabilities)).sum()),
            imbalance_ratio=float(counts.max() / counts.min()) if len(counts) else np.nan,
            samples_per_feature=len(X) / max(X.shape[1], 1))
        return {'X': X, 'y': y}, record

    def characterize_pool(self, datasets: Mapping[str, str | Path]) -> pd.DataFrame:
        """Evaluate all datasets, then assign relative categories across eligible ones.

        A dataset load/validation failure is retained with a reason. Rare classes
        (<5 observations) and incomplete four-model fits remain review_separately.
        Preprocessing is fitted within each training fold, never across the pool.
        """
        self.summary_ = None
        self.baseline_results_ = pd.DataFrame()
        self.cv_audit_ = pd.DataFrame()
        self._datasets = {}
        records, frames, audits = [], [], []
        for name, source in datasets.items():
            record: dict[str, Any] = {'dataset': name}
            try:
                path = Path(source)
                self._datasets[name] = path
                obj, record = self._load_dataset(name, path)
                if record['n_samples'] == 0 or record['n_classes'] < 2 or record['n_features'] == 0:
                    record.update(status='invalid_dataset', ranking_exclusion_reason=
                                  'Requires nonempty data, at least two classes and at least one predictor')
                else:
                    frame, audit = evaluate_baselines({name: obj})
                    frames.append(frame)
                    audits.append(audit)
            except Exception as exc:
                record.update(status='error', ranking_exclusion_reason=repr(exc))
            records.append(record)
        if frames:
            self.baseline_results_ = pd.concat(frames, ignore_index=True)
            self.cv_audit_ = pd.concat(audits, ignore_index=True)
            profile = _compute_pool_profile(self.baseline_results_, self.cv_audit_)
        else:
            # Keep a useful schema for empty pools and pools containing only failures.
            profile = pd.DataFrame(columns=['dataset', 'comparable_cv', 'complete_non_dummy_protocol',
                'ranking_eligible', 'ranking_exclusion_reason', 'baseline_macro_f1_mean',
                'mean_model_cv_f1_std', 'predictive_error', 'cv_variability',
                'predictive_error_score', 'cv_variability_score', 'predictive_difficulty_score',
                'predictive_difficulty_rank', 'predictive_difficulty_label', 'strategy_sensitivity',
                'strategy_sensitivity_score', 'baseline_runtime_seconds_mean', 'mean_tree_nodes',
                'mean_tree_depth', 'mean_tree_leaves', 'runtime_burden_score',
                'tree_size_burden_score', 'computational_burden_score'])
        metadata = pd.DataFrame(records, columns=None if records else ['dataset'])
        reasons = metadata.pop('ranking_exclusion_reason') if 'ranking_exclusion_reason' in metadata else pd.Series('', index=metadata.index)
        # The metadata audit also covers datasets too invalid to enter the CV audit.
        profile = profile.drop(columns='min_class_count', errors='ignore')
        summary = metadata.merge(profile, on='dataset', how='left', validate='one_to_one')
        for column in ['ranking_eligible', 'comparable_cv', 'complete_non_dummy_protocol']:
            summary[column] = summary[column].eq(True)
        summary['ranking_exclusion_reason'] = summary.ranking_exclusion_reason.fillna(reasons).fillna('')
        summary['predictive_difficulty_label'] = summary.predictive_difficulty_label.fillna('review_separately')
        if 'status' not in summary:
            summary['status'] = None
        summary['status'] = summary.status.fillna(pd.Series(np.where(
            summary.ranking_eligible, 'ok', 'review_separately'), index=summary.index))
        summary['difficulty_category'] = summary.predictive_difficulty_label.map(
            {'easy': 'Easy', 'intermediate': 'Intermediate', 'hard': 'Hard',
             'review_separately': 'review_separately'})
        self.summary_ = summary
        return summary.copy(deep=True)

    def get_datasets_by_difficulty(self, categories: str | Sequence[str]) -> dict[str, Path]:
        """Return a name-to-path subpool for one or more case-insensitive categories."""
        if self.summary_ is None:
            raise RuntimeError('Call characterize_pool first')
        requested = [categories] if isinstance(categories, str) else list(categories)
        labels = {category.lower() for category in requested}
        if not labels <= {'easy', 'intermediate', 'hard'}:
            raise ValueError('Categories must be Easy, Intermediate or Hard')
        selected = self.summary_.loc[self.summary_.predictive_difficulty_label.isin(labels)]
        selected = selected.sort_values('predictive_difficulty_rank')
        return {name: self._datasets[name] for name in selected.dataset}

    def get_difficulty_groups(self) -> dict[str, list[str]]:
        """Return dataset names grouped by Easy, Intermediate and Hard."""
        return {label: list(self.get_datasets_by_difficulty(label))
                for label in ['Easy', 'Intermediate', 'Hard']}
