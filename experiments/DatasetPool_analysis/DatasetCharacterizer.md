# Standalone dataset characterization

`core/DatasetCharacterizer.py` implements the final canonical methodology in
`Dataset_Characterization.ipynb`, without reading or executing a notebook at runtime.
No GA modules, original CSV references, datasets, or notebook cells were modified.

## Usage

Run from the repository root:

```python
from core.DatasetCharacterizer import DatasetCharacterizer

characterizer = DatasetCharacterizer()
datasets = characterizer.default_pool()  # the notebook's explicit 20-file pool
summary = characterizer.characterize_pool(datasets)

easy = characterizer.get_datasets_by_difficulty("Easy")
train_pool = characterizer.get_datasets_by_difficulty(["Easy", "Intermediate"])
groups = characterizer.get_difficulty_groups()

# Name -> pathlib.Path mappings, ready for a future dataset-pool consumer.
print(easy)
# Name lists under Easy / Intermediate / Hard.
print(groups)
# Includes failed/ineligible datasets and their reasons.
print(summary[["dataset", "status", "ranking_exclusion_reason",
               "predictive_difficulty_score", "difficulty_category"]])
```

`default_pool(data_dir)` optionally uses a different data directory. To characterize
new CSVs, explicitly declare both targets and identifiers:

```python
characterizer = DatasetCharacterizer(
    target_config={"my_dataset": "label"},
    id_columns={"my_dataset": ["record_id"]},  # [] if there are no IDs
)
summary = characterizer.characterize_pool({"my_dataset": "data/my_dataset.csv"})
```

Targets can be column names, integer positions, or ordered fallback lists. Each ID
entry accepts the same forms. Unknown datasets never silently use the last column.
CSV reading retains pandas' normal header, delimiter and default NA behavior plus
the notebook's additional missing tokens. Returned mappings contain source CSVs;
a future consumer must honor the same target and ID configuration when loading them.

`summary_`, `baseline_results_`, and `cv_audit_` expose the last run's dataset,
model, and CV records. The model table retains fifteen fold scores, sample standard
deviations, errors, warnings, elapsed runtime and tree complexity. A second call
replaces all previous state. Invalid category names raise an error. Failed or
review-only datasets never appear in the three group accessors.

## Architecture and methodological provenance

Small module-level helpers implement preprocessing, baseline evaluation,
aggregation, eligible-pool normalization and rank partitioning. The class handles
loading, per-dataset error isolation, summary assembly and subset access.

The authoritative notebook cells are **zero-based**:

- Cell 2: the explicit 20-dataset pool.
- Cell 4: explicit target/ID configuration, missing tokens, exclusion of missing
  target rows, string conversion and LabelEncoder for targets.
- Cells 6 and 8: validity checks and supporting dataset descriptors. The reusable
  module reports failures per dataset instead of aborting the entire pool. The
  notebook's Parkinsons-specific fixture assertions are not general eligibility
  rules; explicit `status` target and `name` removal are retained.
- Cell 27: shared `RepeatedStratifiedKFold(5, 3, random_state=42)` splits, serial
  execution and six baseline models. The four non-dummy strategies are default
  and class-balanced decision trees and logistic regressions. Two dummy models
  are reference descriptors only. All model hyperparameters match this cell.
- Numerical predictors use median imputation. Logistic regressions additionally
  standardize them. Categoricals use most-frequent imputation and dense one-hot
  encoding with unknown categories ignored. All preprocessing fits within the
  training fold. All-missing feature handling remains sklearn's notebook behavior;
  warnings are retained rather than changing it.
- Eligibility requires at least five observations in every class and complete,
  finite fifteen-score evaluations from all four non-dummy strategies.
- `F` is the mean of their mean macro-F1 values; `E = 1 - F`; `V` is the mean of
  their within-model **sample** standard deviations (`ddof=1`).
- Cells 29–30: min-max normalization uses only the full eligible pool. Constant
  components normalize to zero. Ineligible normalized quantities remain NaN.
  `D = 0.70 * z(E) + 0.30 * z(V)`; `z` means min-max, not standard-score scaling.
- Ranking sorts by ascending D, then descending F, ascending V, and dataset name.
  Only exactly equal floating-point scores are ties; no rounding is applied.
  `numpy.array_split(..., 3)` partitions ranks, putting extra members in earlier
  groups. Even exact ties can cross group boundaries. Small pools follow this
  same rule; their groups may be empty.
- Strategy sensitivity is the range of non-dummy model means. Computational
  burden averages normalized elapsed runtime and normalized mean tree node count,
  requiring both components. Depth and leaf counts are also retained. These
  descriptors do not influence difficulty labels.
- Cell 30's group mean monotonicity is a validation observation, not a rule that
  changes labels. Cells 36 and 38–45 are sensitivity analyses and do not redefine
  the canonical protocol. They were not run by this module.
- Cells 50, 52 and 56 consume/export canonical results; their display formatting
  and heuristic issue flags do not redefine labels. Per-feature exploratory
  analyses and plotting are outside this reusable dataset-level module.

Canonical columns use the notebook's lowercase labels. `difficulty_category`
additionally offers Easy / Intermediate / Hard capitalization. Excluded records
retain `review_separately`, a status and an explicit reason. Labels are relative:
changing the pool can change normalization, ranks and subsets. Characterize the
complete intended pool before requesting training subsets.

## Validation performed

Environment matches the saved notebook environment: Python 3.12.14, NumPy 2.5.3,
pandas 3.0.5 and scikit-learn 1.9.0.

```text
python -m unittest experiments.DatasetPool_analysis.test_dataset_characterizer -v
python -m experiments.DatasetPool_analysis.validate_characterizer
```

Six regression tests passed: eligible-pool and constant normalization, exact ties
and uneven thirds, complete four-model aggregation/sample SD/dummy exclusion,
fold-local preprocessing with unseen categories, missing/invalid/rare-class inputs
and reset behavior, and actual CV evaluation with subset retrieval.

The full pool was recalculated from CSVs. Against the existing
`results/dataset_statistics/dataset_experimental_profile.csv`:

- All 17 eligible datasets reproduce F, V, normalized predictive components,
  difficulty scores, ranks and labels. Maximum difference in the compared
  predictive values after CSV serialization was **0.0** (tolerance 1e-10).
- Strategy sensitivity and tree node/depth/leaf statistics also match.
- All 20 eligibility decisions and review labels match. Abalone (minimum class
  count 1), Arrhythmia (2) and Bridges (1) remain review_separately.
- Only wall-clock runtimes and their derived burden scores differ, as expected.
  They cannot change difficulty labels.
- Mean F is Easy 0.921886, Intermediate 0.778932, Hard 0.513034, satisfying the
  notebook's monotonicity audit without altering the partition.

| Subset | Datasets in rank order |
|---|---|
| Easy | banknote, tic_tac_toe, breast_cancer, iris, semeion, car |
| Intermediate | ionosphere, credit_approval, balance_scale, cylinder_bands, anneal, parkinsons |
| Hard | sonar, haberman, glass, wine_red, heart_disease |

## Inconsistency in the requested CSV reference

`DatasetLevelSummary.csv` has only ten descriptive metrics; it contains **no
baseline performance, difficulty scores, ranks or labels**. The current notebook
exports `dataset_summary.csv` and `dataset_experimental_profile.csv`, not this
filename, so no current cell can be established as its exact generating cell.
The notebook explicitly says corrected results supersede stale manuscript tables.

Of its 200 dataset/metric comparisons, 183 match to absolute tolerance 1e-10.
All ten metrics match for 17 datasets. The remaining 17 metric differences occur
in three datasets:

| Dataset | Historical CSV | Current notebook and module |
|---|---|---|
| balance_scale | 5 classes, 3 numeric + 1 categorical predictors | Explicit first-column class target: 3 classes, 4 numeric predictors; entropy and imbalance also change |
| parkinsons | 196 rows, 196 classes, 23 categorical predictors | `status` target, remove `name`: 195 rows, 2 classes, 22 numeric predictors; related descriptors change |
| cylinder_bands | 39 predictors, 15 categorical, missingness 4.74359% | Remove serial identifier `feature_001`: 38 predictors, 14 categorical, missingness 4.86842%; sample/feature ratio also changes |

Balance Scale's old values are consistent with the last-column target mistake
explicitly documented in notebook cell 3. Parkinsons' exact historical CSV parsing
cannot be reconstructed from the current notebook; its corrected shape is explicitly
asserted in cell 6. Cylinder Bands reflects the identifier removal declared in cell
4. The module follows these current instructions without forcing historical matches.

Outputs are saved separately in `results/dataset_characterizer_validation/`:
`summary.csv`, `baseline_results.csv`, `cv_audit.csv`, `historical_comparison.csv`,
`canonical_comparison.csv`, and `environment.json`. Existing references were preserved.
The comparison files include every common metric, its reference and calculated
value, numeric difference where applicable, and match flag. The validation script
reports discrepancies rather than substituting reference values.