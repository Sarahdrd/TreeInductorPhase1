"""Original canonical normalization, assignment and scoring definitions.

Extracted without execution from the cleaned notebooks. No models or CV.
See block3_scoring_provenance.json; function bodies are preserved verbatim.
"""
import numpy as np
import pandas as pd

def minmax_normalize_on_pool(series, pool_mask):
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

def assign_predictive_difficulty_groups(df):
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

def score_profile(aggregated, ns, alpha=.70):
    result = aggregated.copy(deep=True)
    result['predictive_error'] = 1 - result.baseline_macro_f1_mean
    norm = ns['minmax_normalize_on_pool']
    result['predictive_error_score'] = norm(result.predictive_error, result.ranking_eligible)
    result['cv_variability_score'] = norm(result.cv_variability, result.ranking_eligible)
    result['predictive_difficulty_score'] = alpha * result.predictive_error_score + (.30 if alpha == .70 else 1-alpha) * result.cv_variability_score
    labels, ranks = ns['assign_predictive_difficulty_groups'](result)
    result['predictive_difficulty_label'], result['predictive_difficulty_rank'] = labels, ranks
    return result
