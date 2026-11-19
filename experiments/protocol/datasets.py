"""Extracted verbatim from New_Characterization_Experiments; no notebook execution."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def _resolve_column(df, spec, dataset_name, purpose):
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

def _resolve_existing_columns(df, specs):
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

def load_dataset(name, path):
    # Load a CSV and split into X and y.
    df = pd.read_csv(path, na_values=MISSING_TOKENS)
    df.columns = [str(c).strip() for c in df.columns]

    if name not in TARGET_CONFIG or name not in ID_COLUMNS:
        raise ValueError(f"Missing explicit target/ID configuration for {name}")
    target_spec = TARGET_CONFIG[name]
    target_col = _resolve_column(df, target_spec, name, "target column")

    id_cols = _resolve_existing_columns(df, ID_COLUMNS[name])
    id_cols = [c for c in id_cols if c != target_col]

    X = df.drop(columns=[target_col] + id_cols).copy()
    y = df[target_col].copy()

    valid_target = ~y.isna()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    y_encoded = LabelEncoder().fit_transform(y.astype(str))

    return {
        "df": df,
        "X": X,
        "y": y_encoded,
        "y_raw": y,
        "target_col": target_col,
        "dropped_id_cols": id_cols,
    }
