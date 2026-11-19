# Block 1: frozen protocol and reproducibility infrastructure

**Block 2A update:** the resumable saved-fold runner is implemented and validated
with a seven-fold diagnostic pilot. See [BLOCK2A_RUNNER.md](BLOCK2A_RUNNER.md)
for checkpoint/schema details, safe commands and the unexecuted Block 2B command.
Official results use a separate store; historical Block 1 findings below remain
as provenance context.

**Recovery update:** the authoritative SVM/RIPPER source has now been recovered,
integrated and smoke-tested in the existing `.venv`; wittgenstein 0.3.5 is installed.
See [RIPPER_RECOVERY.md](RIPPER_RECOVERY.md) for current readiness, exact source,
manifest differences and tests. The Block 1 inspection/environment findings below
are historical; its missing-adapter/install blockers are resolved by this update.

Prepared on branch `clean`. The three notebooks already had uncommitted edits;
this block does not modify them. No model fitting, scientific scores, difficulty
rankings, figures, package installation, commits or branch changes were performed.

## Files and entry points

All new files are under `experiments/`:

- `configs/experiment_config.json`: authoritative machine-readable protocol (JSON
  avoids introducing a YAML dependency into the project environment).
- `configs/dataset_manifest.json`: all 20 descriptive datasets, original targets,
  class counts, dimensions, eligibility, CSV checksums, ordered-data fingerprints,
  column dtypes and explicit unknown upstream source/version fields.
- `configs/environment.json`, `dependency_check.json`,
  `resolved_model_parameters.json`, `historical_cross_model_manifest.json`:
  installed provenance, dependency checks, resolved estimator defaults, and the
  preserved historical SVM/RIPPER specification.
- `configs/fold_result.schema.json`: common fold record schema.
- `protocol/datasets.py`, `preprocessing.py`: verbatim extracted definitions;
  `extraction_provenance.json` identifies their notebook source.
- `protocol/infrastructure.py`: data validation, deterministic split generation,
  compressed saving, exact loading, checksums and environment identity.
- `protocol/models.py`: unfitted DT/LR/SVM/dummy/RIPPER constructors; recovered
  SVM/RIPPER definitions are in `protocol/recovered_cross_models.py`. Pass encoded
  `y` when constructing RIPPER to preserve original binary/OvR selection.
- `protocol/results.py`: validated, exclusive JSONL result serialization.
- `protocol/prepare.py`, `test_infrastructure.py`: preparation and lightweight tests.
- `protocol/validation_report.json`: split counts and historical index comparisons.
- `protocol/historical_checksums.json`: pre-change hashes of 5,065 existing files.
- `protocol/preservation_report.json`: all 5,065 files verified unchanged after implementation.
- `splits/saved_cv_indices/revision2-five-seed-v1/<dataset>/seed_<seed>.npz`
  and adjacent JSON metadata: 85 archives containing 1,275 partitions.
- `requirements.txt`: pinned unified environment, now including verified wittgenstein 0.3.5.

From the repository root, use the explicit interpreter (bare `python` selects a
different Anaconda environment):

```powershell
.venv/Scripts/python.exe -m experiments.protocol.prepare
.venv/Scripts/python.exe -m unittest experiments.protocol.test_infrastructure -v
```

Preparation is idempotent and refuses to replace different frozen files. It reads
the local historical metadata/partitions as provenance dependencies. Those files
and the CSVs are locally present but excluded from Git by `.git/info/exclude`;
another checkout must obtain the exact originals and verify their checksums.
No remote dataset is downloaded or substituted.

## Repository inspection and reuse

`New_Characterization_Experiments.ipynb` defines the loader, preprocessing,
four canonical DT/LR strategies, dummy references, regularization panels and
the `ExperimentStore` helper in its tagged persistence cell.
`Default_Tree_Canonical_Strata.ipynb` loads those definitions, checks saved
canonical provenance, and evaluates default-tree structure by historical strata.
`Dataset_Characterization.ipynb` contains descriptive analysis and tagged alignment
helpers; its alignment workflow depends on saved canonical and default-tree
outputs and includes seed/alpha sensitivity. Setup cells load tagged helper cells
across notebooks, so notebook order and saved inputs matter.

The extracted loader/preprocessor retain their original function bodies; tests
compare their ASTs with the notebooks. The new fingerprint uses the historical
`ExperimentStore.data_digest` algorithm. The new split infrastructure extends
the historical seed-42 partition approach to five seeds without invoking the
historical evaluator (which explicitly rejects seeds other than 42).

The cleaned notebooks do not contain SVM/RIPPER construction code. Parameters are
recoverable from `results/dataset_statistics/cross_model_validation/manifest.json`,
including historical implementation hashes. They are preserved as historical
evidence, not represented as source recovered from notebook code.

## Frozen scientific choices

The primary pool is exactly the requested 17 datasets, with explicit dataset
order in the config/manifest. Abalone, Arrhythmia and Bridges remain descriptive
only because their minimum class counts are 1, 2 and 1. Other local CSVs are not
added. The loader retains original target values, removes missing-target rows,
resets row indices in original CSV order and uses `LabelEncoder` on string labels.
Heart Disease remains **5 classes** and Wine Quality Red **6 classes**. Manifest
counts come from the supplied local files, not external dataset descriptions.
Upstream source identifiers and curation versions are unknown; checksums identify
the exact local versions. Breast Cancer's local file has 569 rows and 30 features;
no upstream identifier is inferred from its generic name.

CV is `RepeatedStratifiedKFold(5, 3)` with seeds `[42, 7, 21, 84, 123]`: 15
partitions per seed, 75 per model configuration and dataset. Estimator random
states remain independent of CV seeds and retain their original configured values.
Macro-F1 uses `zero_division=0`; historical variability uses `ddof=1`.

The historical score remains `0.70 * minmax(E) + 0.30 * minmax(V)`, where E is
one minus the mean of four DT/LR strategy means, and V is the mean of their
within-model standard deviations. Constant components normalize to zero. Exact
score ties use descending mean F1, ascending V, then dataset name, with stable
sorting. `numpy.array_split` produces strata of 6/6/5. Dummy, SVM and RIPPER
results do not silently enter the canonical four-model panel.

Model parameters are fully listed in the config, with unspecified constructor
defaults separately recorded for the installed runtime:

| Model | Explicit configuration highlights |
| --- | --- |
| DT default/balanced | gini, best splitter, unlimited depth, split=2, leaf=1, alpha=0, seed=42; class weight None/balanced |
| LR default/balanced | L2, C=1, lbfgs, max_iter=5000, tol=1e-4, intercept enabled, seed=42; class weight None/balanced |
| SVM-RBF | C=1, gamma=scale, tol=.001, shrinking enabled, probability disabled, max_iter=-1, random_state=None; remaining saved parameters retained |
| RIPPER | k=2, prune_size=.33, dl_allowance=64, n_discretize_bins=10, alpha=1, seed=42; rule/condition limits None |
| Dummy | most_frequent and stratified, each random_state=42 |
| DT regularization | leaf=2, leaf=5, ccp_alpha=.001, ccp_alpha=.01; each with None/balanced class weights; other DT settings unchanged |

Numeric median imputation and categorical mode imputation plus dense one-hot
encoding (`handle_unknown='ignore'`) remain unchanged. LR and SVM also scale
numeric features. All transformations must be fit on training folds only, inside
the estimator pipeline. Dummy classifiers use their original direct construction.
RIPPER's historical description is median/mode imputation plus nominal strings,
with `n_discretize_bins=10`. Its exact numeric dtype handling and multiclass
adapter are unavailable: whether the nominal conversion bypassed numeric
discretization cannot be established from the manifest alone. Do not substitute
the DT/SVM preprocessing or a guessed one-vs-rest implementation.

## Reusing partitions and serializing future results

Load the dataset with `load_dataset`, select its manifest entry, then call
`validate_dataset(obj, entry, csv_path)` before `load_splits(npz_path, entry, obj['y'])`.
Indices are zero-based positions in the filtered dataset; use `X.iloc[train]`,
`X.iloc[valid]`, `y[train]`, `y[valid]`. Repetitions/folds in metadata are one-based.
Reuse these exact arrays for every model/panel. Every partition has a split ID
binding protocol, dataset fingerprint, seed, fold and exact train/validation bytes.

Validation checks every class in train and validation, class allocation balance,
disjointness, full coverage, unique integer indices, deterministic regeneration,
per-repeat coverage, archive contents and metadata hashes. Dataset validation
rejects changed bytes, dtypes, row order, target, identifiers and class counts.

Future records go under `experiments/results/revision2-five-seed-v1/folds/` using
unique run filenames. `save_records` validates every record against loaded split
metadata, rejects duplicate fold identities and refuses overwrite. Required fields
include experiment/protocol IDs, dataset/fingerprint, model/configuration, seed,
repeat, fold, split ID, macro-F1 and environment ID. Store full resolved pipeline
parameters in `model_configuration`; add runtime, warnings, tree structure,
adapter version and failure details as appropriate. Failed fits must not be
represented as successful fabricated scores. No scientific record was created;
the only synthetic record in tests is confined to a temporary directory.

Future adapters must expose fit/predict/get_params, preserve the encoded target
space and fit any preprocessing on training indices only. New study protocols
need explicit schema extensions before execution. MurTree/Quant-BnB are listed
only as future work: feature binarization, target/multiclass support, missing-value
handling, capacity and time budgets require investigation before comparison.

## Environment findings and unresolved choices

The candidate unified runtime is the existing `.venv`: Python 3.12.14, NumPy
2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0, matplotlib 3.11.1,
joblib 1.6.0 and threadpoolctl 3.6.0. No packages were changed. Requirements
describe verified installed versions, not verified availability on a package
index. The environment lacks pip and wittgenstein. A direct Requires-Dist check
of 32 active installed requirements found no conflicts; this is not a full
cross-model execution/compatibility test. RIPPER 0.3.5 is historically recorded
but remains unverified in this runtime and is commented out in requirements.

The task describes DT/LR results using scikit-learn 1.9.0/NumPy 2.5.3/pandas
3.0.5. However, the local `results/new_characterization_experiments/metadata.json`
records Python 3.11.5, scikit-learn 1.3.0, NumPy 1.23.5, pandas 2.3.3 and
SciPy 1.15.3, matching bare Anaconda `python`. The older cross-model manifest
records Python 3.12.12, scikit-learn 1.8.0, NumPy 2.4.1, pandas 2.3.3,
SciPy 1.17.0 and wittgenstein 0.3.5. These histories are not interchangeable.

All local CSV hashes match the canonical metadata. Pandas 3 infers string dtype
where pandas 2 used object; seven descriptive datasets therefore have different
filtered-data fingerprints (five in the primary pool). The current manifest
records both identities. This block does not cast columns or change preprocessing
to hide that discrepancy. Missing-string/imputer behavior, all-empty columns,
LR parameter deprecations and RIPPER compatibility must be checked before full
Block 2 evaluation. At completion of Block 1, a usable environment for **all** models was pending.
The recovery update resolves SVM/RIPPER installation and smoke compatibility;
full DT/LR and benchmark evaluations remain Block 2 work.

All 255 new seed-42 partitions match historical DT/LR train/validation indices
exactly, with unchanged CSV bytes. This establishes index identity for those
files, not equality of transformed matrices, predictions or results. SVM/RIPPER
records lack exact partition indices; paired comparisons with them remain
unverified. Existing results are never imported as new five-seed evaluations.

Balance Scale has a documented notebook discrepancy: New Characterization uses
target fallbacks `['class','Class','balance','Balance',0]`, whereas Dataset
Characterization uses `['class']`. Both select `class` on the supplied file.
The extracted loader preserves the New Characterization definition exactly.

## Validation and Block 2 handoff

Six lightweight tests pass: full benchmark validation, source extraction identity,
invalid splits/row-order rejection, unfitted model parameters, record serialization,
and round-trip/tamper/overwrite rejection. All 58 notebook code cells parse.
Preparation was rerun to verify idempotence. No model CV was executed.

Original Block 1 handoff (items 1–2 are updated by RIPPER_RECOVERY.md):

1. Recover the original RIPPER adapter/source using its saved implementation
   provenance; verify numeric representation, discretization and multiclass rules.
2. Resolve the historical environment discrepancy and verify the candidate
   runtime's categorical/missing-value and estimator behavior. Validate RIPPER's
   dependency compatibility before adding an active requirement. If a different
   environment is chosen, freeze a new version rather than overwriting this one.
3. Validate every CSV against the manifest, load the saved splits, and implement
   the fold runner using the preserved pipelines and complete fold records.
4. Run the authorized Block 2 baseline/panel evaluations under that single
   verified environment, recording errors and all resolved parameters; preserve
   historical results. Do not reuse unverified historical SVM/RIPPER pairing.
5. Defer aggregated ranking to Block 3. Finalize the variability choice, chance
   correction formula and label-stability definition before those analyses.

Future configurations retain pooled 75-score SD and mean five within-seed SD as
unselected alternatives. Alpha sensitivity is `[.51,.60,.70,.80,.90]`, with
error-only/variability-only ablations at 1/0. Subsampling has requested sizes,
explicit future independent seeds/replicate slots and no selected dataset.
Separate identifiers distinguish historical canonical, five-seed, chance-corrected,
subsampling and alternative-inducer studies. None of those future analyses ran.

| Reviewer comment | Infrastructure contribution | Remaining work |
| --- | --- | --- |
| R1.5 | Environment/dependency snapshots, dataset hashes, shared partitions | Verify full unified environment and RIPPER |
| R1.2 | Five-seed partitions and reconstructible fold schema | Block 3 aggregation and variability decision |
| R1.1 | Both frozen dummy configs and chance-comparison metadata | Fit dummy baselines; finalize correction |
| R1.3–R1.4 | Default/balanced DT and all four paired regularization panels | Evaluate on common indices |
| R2.2 | Subsample sizes, independent seed/replicate configuration slots | Select dataset and freeze replicate design |
| R2.1/R3.1 | Extensible model metadata and adapter contract | Verify and integrate MurTree/Quant-BnB later |
