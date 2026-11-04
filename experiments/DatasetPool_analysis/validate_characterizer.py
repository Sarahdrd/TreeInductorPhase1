"""Recompute the pool and compare against both historical and canonical exports.

Run from the repository root: python -m experiments.DatasetPool_analysis.validate_characterizer
Writes only to results/dataset_characterizer_validation/.
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from core.DatasetCharacterizer import DatasetCharacterizer


def compare(actual: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Report every shared numeric/text field, including missing-value equality."""
    joined = actual.merge(reference, on='dataset', suffixes=('_new', '_reference'))
    rows = []
    for column in reference.columns.intersection(actual.columns).difference(['dataset']):
        for _, row in joined.iterrows():
            a, b = row[column + '_new'], row[column + '_reference']
            missing = pd.isna(a) and pd.isna(b)
            if pd.api.types.is_numeric_dtype(reference[column]):
                match = bool(np.isclose(a, b, atol=1e-10, rtol=0, equal_nan=True))
                delta = a - b
            else:
                # CSV parsing represents empty audit reasons as NaN.
                match = missing or (pd.isna(a) and b == '') or (pd.isna(b) and a == '') or a == b
                delta = np.nan
            rows.append(dict(dataset=row.dataset, metric=column, calculated=a,
                             reference=b, delta=delta, match=match))
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / 'results' / 'dataset_characterizer_validation'
    out.mkdir(parents=True, exist_ok=True)
    characterizer = DatasetCharacterizer()
    summary = characterizer.characterize_pool(characterizer.default_pool())
    summary.to_csv(out / 'summary.csv', index=False)
    characterizer.baseline_results_.to_csv(out / 'baseline_results.csv', index=False)
    characterizer.cv_audit_.to_csv(out / 'cv_audit.csv', index=False)
    for name, path in [
        ('historical', root / 'experiments/DatasetPool_analysis/DatasetLevelSummary.csv'),
        ('canonical', root / 'results/dataset_statistics/dataset_experimental_profile.csv'),
    ]:
        result = compare(summary, pd.read_csv(path))
        result.to_csv(out / f'{name}_comparison.csv', index=False)
        mismatches = result.loc[~result.match]
        print(name, 'mismatch counts:', mismatches.groupby('metric').size().to_dict(), flush=True)
    (out / 'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__), indent=2))
    print(characterizer.get_difficulty_groups(), flush=True)
    print(summary[['dataset', 'ranking_eligible', 'predictive_difficulty_rank',
                   'predictive_difficulty_label']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
