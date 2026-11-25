# Block 7 — MAGIC controlled subsampling (Stage A)

## Status and authorization boundary

Infrastructure and pilot completed on branch `clean`. Full execution is **not authorized and has not started**: 0/10,500 main-run records. The separate pilot contains 16 successful records, not scientific evidence for sample-size trends. Section 17 of the supplied Block 7 specification requires stopping here for explicit authorization. The optional 19,020-row reference is excluded from this frozen version; the main experiment has exactly seven requested sizes.

## Dataset verification

Source: local `data/magic04.data`; [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), DOI 10.24432/C52C8B. The local dimensions and counts match UCI documentation. No remote byte-identity claim is made.

- 19,020 observations; ten finite numerical predictors; target in column 11.
- Feature order: fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist.
- Original labels: `g` = 12,332; `h` = 6,688. Labels are preserved in fitting and scoring.
- Missing values: 0. Malformed rows: 0. Duplicate complete rows beyond the first occurrence: 115; retained.
- Exact SHA-256: `e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a`.
- The `.data` file already contains comma-separated values without a header. It is parsed directly with explicit UCI column names and round-trip float parsing. No CSV copy or change to the source is needed.
- Original row ID is the zero-based physical line number; no filtering or deduplication. Sample-local row order is ascending original row ID, and saved CV positions map back through that list.

The experiment-specific `dataset_manifest.json` also records the parsed fingerprint, target runs in source order, duplicate-feature count, parser, and source attribution. The existing benchmark manifest is untouched.

## Frozen design

Artifacts: `experiments/block7-magic-subsampling-v1/`.

Sizes: `[200, 500, 1000, 1500, 2000, 5000, 10000]`.
Subsampling seeds: `[101, 202, 303, 404, 505]`, distinct from CV seeds `[42, 7, 21, 84, 123]`.

Each subsampling seed initializes NumPy default_rng (PCG64). In sorted label order, one independent permutation is created for each class. At each size, integer class allocations use the floor of the exact proportional allocation and distribute remaining observations to the largest fractional remainder; ties use label order. Prefixes of each class permutation are selected and the resulting original IDs are sorted. Explicit set-inclusion checks verify nesting. Each class allocation is within one observation of its ideal proportional count. Different seeds give distinct randomized sample sequences, although observations may overlap across replicates.

Saved before any model evaluation: 35 sample-index manifests, 175 sample-specific CV collections, and their checksums. Metadata includes experiment and sample IDs, original dataset fingerprint/checksum, subsampling seed and replicate, requested/actual size, original row IDs, class counts, sample checksum, and algorithm. `preparation_tables/` provides dataset/sample manifests, actual class proportions, and the normalization reference as CSV.

Each sample has five RepeatedStratifiedKFold collections, each 5 folds × 3 repetitions. This yields 2,625 shared folds and 10,500 model evaluations. All four models use identical saved folds. Every partition was checked for class coverage, disjoint training/validation indices, valid original-row mapping, and exactly one validation appearance per row per repetition. Evaluation only loads these partitions; it does not regenerate them.

## Models and checkpointing

Only `decision_tree_default`, `decision_tree_balanced`, `logistic_regression`, and `logistic_regression_balanced` are selected. The existing frozen `make_model` constructor supplies median imputation, LR numeric scaling, original estimator parameters, class weights and random states. A fresh pipeline is fitted on each training partition. Macro-F1 uses `zero_division=0`. The model random-state policy is unchanged across CV seeds.

The runner reuses the existing immutable checksummed Store and OS writer lock. Results include both seeds, original dataset identity, sample checksum, exact fold ID, complete resolved configuration identity, environment identity, timing and warnings. Failures are recorded and stop execution; retry creates a new attempt. Successful folds cannot be overwritten. Resume validates provenance and skips completed successes. Code, frozen config, source data, prepared indices, model parameters, normalization reference and environment are checked before evaluation. One numerical thread is used for execution; this does not change estimator parameters.

Pilot and main result stores are separate under `results/pilot` and `results/full`; neither uses the official Block 2B store. The pilot is capped at 20 new evaluations per invocation and its fixed plan contains 16 evaluations. Full execution requires the `run --authorize-full` action after user authorization.

## Pilot and validation

The pilot covered sizes 200 and 500, all four models, CV seed 42, repetition 1, folds 1 and 2. It ran eight folds, resumed for the other eight, then reran with zero additions. The first batch remained byte-identical. All 16 evaluations succeeded. LR emitted the existing scikit-learn `penalty` deprecation warning; no convergence warnings or failures occurred. Frozen hyperparameters were retained.

All **33 Block 3–7 tests passed**. Coverage includes parsing and checksum rejection, unchanged row order/labels, nested saved sampling, proportional class allocations, distinct seed sequences, reproducible saved CV, shared model folds, training-only preprocessing, frozen parameters and resolved pipelines, five-seed aggregation, sample SD with ddof=1, incomplete-result rejection, fixed normalization without clipping, failure/retry/no-overwrite behavior, full-run authorization gate and store separation.

The preservation audit checked **20,735 existing files unchanged**, including all existing result records, prior analyses, configuration, splits, protocol modules and notebooks in its inventory. The original MAGIC file, canonical endpoints and unified environment were also verified. No notebook training cells were executed, no dependencies changed, and no Git commit/push/merge/branch change was performed. Details: `block7-magic-subsampling-v1/stage_a_validation.json` and `pilot_validation.json`.

## Fixed-reference analysis prepared for Stage B

Normalization is frozen from the primary Block 3 five-seed ranking:

| Endpoint | Value |
|---|---:|
| E_min_ref | 0.017872678841215017 |
| E_max_ref | 0.703213787821136 |
| V_min_ref | 0.007066193748940045 |
| V_max_ref | 0.07944060073210513 |

For each model/sample/CV seed, compute the mean and sample SD of its 15 fold scores. Average the five seed means and the five within-seed SDs separately, then average across the four canonical models. E = 1 − F; V is the mean within-seed SD. The pooled 75-fold SD and between-CV-seed SD of means are retained separately as diagnostics.

E_norm = (E − E_min_ref)/(E_max_ref − E_min_ref); V_norm uses the corresponding fixed V endpoints. D_pred = 0.70 E_norm + 0.30 V_norm. No clipping, MAGIC-fitted endpoints, rank or Easy/Intermediate/Hard assignment is applied. Out-of-reference components are flagged.

`protocol/block7_analysis.py` requires all 10,500 planned successes with matching shared fold identities before creating scientific outputs. It implements replicate-level components, size-level means/SD/median/range, model-specific summaries, within-replicate adjacent-size changes, direction consistency and descriptive Spearman associations. Constant trajectories have undefined correlations rather than fabricated values. No independent-fold significance tests are computed.

Prepared figures: predictive error, CV variability, fixed-reference difficulty, joint components and model-specific trends, using log sample-size axes, individual replicate trajectories and between-replicate SD bars; output formats are 300-dpi PNG and vector PDF. They will be generated and visually verified only from complete results. Tables and figures will be written to a new `experiments/analysis/block7-magic-subsampling-v1/` directory. No scientific results or plots were created from pilot/synthetic test scores.

## Commands

Run from the repository root with the existing `.venv`:

```powershell
# Read-only workload/completeness inspection
.\.venv\Scripts\python.exe -m experiments.protocol.block7 dry-run

# Pilot: resume or verify already completed pilot folds
.\.venv\Scripts\python.exe -m experiments.protocol.block7 pilot --limit 16

# Full experiment: ONLY after explicit user authorization; same command resumes
.\.venv\Scripts\python.exe -m experiments.protocol.block7 run --authorize-full

# Validate main results, then analyze only when all 10,500 succeeded
.\.venv\Scripts\python.exe -m experiments.protocol.block7 validate
.\.venv\Scripts\python.exe -m experiments.protocol.block7_analysis

# Tests
.\.venv\Scripts\python.exe -m unittest experiments.protocol.test_block3 experiments.protocol.test_block4 experiments.protocol.test_block5 experiments.protocol.test_block6 experiments.protocol.test_block7
```

Preparation can be verified idempotently with `python -m experiments.protocol.block7 prepare` using the same explicit virtual-environment executable. It refuses incompatible frozen artifacts. Full-run provenance changes require a deliberately reviewed new version; never edit frozen manifests to bypass validation.

## Pending Stage B and interpretation limits

Pending authorization: execute and validate the 10,500-fold main run, inspect warnings/failures, generate full scientific tables and figures, assess monotonicity/diminishing changes and inconsistent E/V directions, append an analysis-only Block 7 notebook section after Block 6, and complete this report's empirical findings. The notebook remains unchanged during Stage A.

Within-CV variability, between-subsample variability and between-CV-seed variation are distinct. Five independently randomized subsampling sequences are the summary units; sizes are nested within sequence, folds overlap, and replicate samples can share observations. The duplicate rows retained in the standard dataset can also span CV partitions, a limitation of the unchanged row-level evaluation design.

Changing sample size changes training information, validation-fold size, sampling composition and estimation stability simultaneously. Results on this one underlying dataset and four fixed configurations cannot establish a universal size–difficulty relationship or prove that V exclusively measures sample size. No full-run findings are available yet.

Reviewer traceability: R2.2 is addressed by the controlled nested size experiment and planned V trajectories; previous R2.5 by separating sample-size sensitivity from a universal descriptor–variability claim. This study does not remove all sample-size confounding in the original 17-dataset benchmark or replace its canonical characterization.
