> **Final status — repository review:** Block 2B subsequently completed all 20,400 official evaluations. This report preserves the original seven-fold pilot and handoff; its pending counts and unexecuted-command statements are historical. See BLOCK3_FIVE_SEED_RANKING.md and EXPERIMENT_INDEX.md.

# Block 2A: saved-fold runner and Block 2B handoff

The runner is ready for Block 2B execution under the pinned `.venv`. Block 2A ran
**seven diagnostic folds only**, with zero failures. The official benchmark was
dry-run validated, not executed. Its workload is **17 × 16 × 5 × 3 × 5 = 20,400
fold evaluations**, or 75 per dataset/configuration. No difficulty rankings or
dataset-level scientific summaries were calculated.

## Architecture and files

- `protocol/runner.py`: selection, preflight, fresh pipeline construction, saved
  index evaluation, provenance validation, progress/status and execution guards.
- `protocol/checkpoints.py`: checksummed immutable JSON attempts, flushed atomic
  writes, and one OS-locked writer per run directory. Extended Windows paths
  support long OneDrive paths.
- `protocol/pilot.py`: a fixed seven-fold diagnostic selection with an absolute
  seven-new-fold invocation cap. Defaults to a dry run.
- `protocol/test_runner.py`: ten infrastructure tests with mocked fits.
- `configs/fold_attempt.schema.json`: versioned checkpoint envelope; the original
  `fold_result.schema.json` remains unchanged for successful records.
- `protocol/block2a_validation_report.json`: pilot coverage, warnings, resumption
  counts and verification evidence. It does not aggregate scientific scores.
- `protocol/block2a_split_checksums.json`: hashes of all 170 pre-pilot split files.

Existing `datasets.py`, `models.py`, `preprocessing.py`, recovered cross-model
definitions, config, manifest and result validator are reused. The only existing
runtime helper changed is `load_splits`: it now accepts
`verify_regeneration=False`. The runner always uses that option and never calls
the split generator. Existing preparation tests retain their original optional
regeneration check; Block 2A did not run partition preparation or rewrite splits.

The three cleaned notebooks, source datasets and historical results are unchanged.
No dependencies, Git branches or scientific settings were changed.

## Supported configurations

The runner checks the inventory against the frozen configuration on selection:

| Family | Stable identifiers |
| --- | --- |
| Canonical DT | `decision_tree_default`, `decision_tree_balanced` |
| Canonical LR | `logistic_regression`, `logistic_regression_balanced` |
| Cross-model | `svm_rbf`, `ripper` |
| Dummy | `dummy_most_frequent`, `dummy_stratified` |
| Leaf 2 | `decision_tree_leaf_2_default`, `decision_tree_leaf_2_balanced` |
| Leaf 5 | `decision_tree_leaf_5_default`, `decision_tree_leaf_5_balanced` |
| Alpha .001 | `decision_tree_ccp_001_default`, `decision_tree_ccp_001_balanced` |
| Alpha .01 | `decision_tree_ccp_01_default`, `decision_tree_ccp_01_balanced` |

No tuning occurs. CV seeds remain `[42,7,21,84,123]`; estimator random states are
the original configuration values. RIPPER's binary/OvR choice and its original
preprocessing are unchanged. SVM's documented probability sentinel is retained.
Model construction is centralized, so a future adapter can reuse the store,
indices and success schema. Adding future algorithms requires an explicit new
inventory/protocol decision; MurTree and Quant-BnB are not implemented.

## Frozen inputs and fold evaluation

Every invocation checks the recorded `environment_ripper_ready.json`, exact
configuration, manifest and implementation hashes. Selected datasets are loaded
and verified against their original byte checksum, filtered-data fingerprint,
target/identifier settings and class counts. All selected model pipelines are
constructed without fitting during preflight, so resolved defaults are auditable.

The runner reads:

```
experiments/splits/saved_cv_indices/revision2-five-seed-v1/<dataset>/seed_<seed>.npz
```

It verifies the adjacent metadata, requested seed, split-generation environment,
all index bounds, disjointness, coverage, class allocation and split IDs. It does
not regenerate random partitions. All models share the same arrays. Repetition
and fold numbers are one-based; indices are zero-based positions in filtered CSV
order. Dataset fingerprints and split-file hashes are checked again immediately
before every actual fit.

For each fold a fresh unfitted model is constructed; the complete original
pipeline fits only `X.iloc[train]` and `y[train]`. Predictions use `X.iloc[valid]`.
Labels/shapes are checked and macro-F1 is computed with `zero_division=0`.
Tree node/depth/leaf counts are retained when available. Runtime covers model
construction, fitting, prediction and scoring, excluding checkpoint disk I/O.

Resolved configuration serialization recursively records estimator classes and
all preprocessing/estimator parameters (including column selectors and nested
OvR configuration). Non-finite parameter sentinels such as imputer NaN are typed
JSON representations; scientific scores must remain finite numbers.

## Result layout and compatibility

Pilot and official records are physically separated:

```
experiments/results/pilots/revision2-five-seed-v1/<experiment-id>/
experiments/results/official/revision2-five-seed-v1/<experiment-id>/
  run.json
  .writer.lock
  records/<dataset>/<model>/seed_<seed>/r<repeat>_f<fold>/attempt_000001.json
```

`run.json` pins the environment, code/configuration/manifest hashes, scoring and
run purpose. Each attempt file contains `{ "payload": {...}, "sha256": "..." }`.
Successful payloads satisfy the original fold-result schema and add status,
runtime, warning messages/counts, UTC timestamps, attempt number, purpose,
configuration ID, implementation ID and evaluation ID. The evaluation ID binds
all fold provenance, including the exact split and resolved pipeline.

Failed payloads use the separately versioned `fold-attempt-v1` wrapper with
`execution_status="failed"`, `macro_f1=null` and exception type/message/traceback.
They are **not** successful fold-result records. This extension preserves the
original success schema and does not admit missing scores into it. Pending
evaluations have no committed attempt. No synthetic score is assigned to failure.

Future analysis should read verified attempts and take the successful payload
for each evaluation identity. Never combine pilot and official stores or treat
failed attempts as scores. No aggregation/export analysis is implemented here.
Existing local Git exclusion rules ignore `results/`; checkpoint retention and
backup are therefore independent of Git commits.

## Checkpointing, interruption, corruption and concurrency

Each completed fold is validated and written before the next begins. A unique
temporary file is created in the final directory, flushed and `fsync`ed, then
renamed. POSIX directory metadata is also fsynced. Existing final paths are never
replaced. This protects ordinary process interruption; storage hardware/cloud
sync and catastrophic power-loss guarantees remain filesystem-dependent.

Ctrl+C leaves committed attempts intact; an interrupted, uncommitted fold remains
pending. Restart with `--resume`. Resumption validates run provenance, checksums,
record schema, expected full configuration, split and environment before skipping
any success. The fold budget counts **new attempts**, including failures, not
skipped records. Budget exhaustion stops at a fold boundary; it does not cancel
an active fit. There is no per-fit hard timeout in this serial implementation.

Failed folds are skipped unless both `--resume --retry-failed` are supplied.
A retry appends `attempt_000002.json`; prior errors remain intact. A successful
fold is never retried. A changed environment/config/code requires a new experiment
ID rather than relabeling or overwriting existing attempts. Do not edit implementation
files during a long run.

Corrupt committed records, missing manifests, changed provenance and inconsistent
attempt sequences fail preflight before any selected fold is fitted. They are not
silently deleted or recomputed. `--status` reports fold-level corruption; invalid
run manifests produce a nonzero command error. Restore a known-good backup or
investigate and start a separately identified run. Orphan `.tmp` files are counted
and ignored as uncommitted attempts; they never count as successful evaluations.

One writer per run is enforced with a nonblocking OS file lock. Process exit
releases the lock automatically; do not delete `.writer.lock`. Different hosts
writing to the same OneDrive-synced run are **prohibited**: local OS locks are not
distributed locks. Read-only status uses atomic files and can be inspected during
a run, but its counts are a moving snapshot while a writer is active.

## Commands (from the repository root)

Default invocation is a dry run and cannot fit models:

```powershell
.venv/Scripts/python.exe -m experiments.protocol.runner
```

Filters accept space-separated values. A bounded generic pilot example:

```powershell
.venv/Scripts/python.exe -m experiments.protocol.runner --execute --experiment-id my-small-pilot --datasets iris --models decision_tree_default logistic_regression --seeds 42 --repeats 1 --folds 1 --max-new-folds 2
```

Execution requires `--execute` plus an explicit budget of 0–20 new folds, or the
explicit `--authorize-full-run` flag. `--official` selects a different store but
does not itself authorize unbounded execution. `--retry-failed` requires resume.
Dry runs validate inputs and report exact selected workload without creating the
run store, fitting models or generating splits.

The actual Block 2A seven-fold pilot was stopped/resumed as follows:

```powershell
.venv/Scripts/python.exe -m experiments.protocol.pilot --execute --max-new-folds 3
.venv/Scripts/python.exe -m experiments.protocol.pilot --execute --max-new-folds 4
.venv/Scripts/python.exe -m experiments.protocol.pilot --execute --max-new-folds 7
```

The first command executed 3 folds; the second skipped 3 and executed 4; the
third skipped all 7 and executed 0. Commands run in separate Python processes.
The pilot helper cannot exceed seven selected folds and never writes official
results. Without `--execute`, it only validates and reports status.

Read-only run status (add filters to inspect a subset):

```powershell
.venv/Scripts/python.exe -m experiments.protocol.runner --status --official --experiment-id block2b-five-seed
```

The exact future **Block 2B** command is below. **It has not been executed.**

```powershell
.venv/Scripts/python.exe -m experiments.protocol.runner --execute --official --experiment-id block2b-five-seed --authorize-full-run --resume
```

Use that same command after interruption. To explicitly retry failed folds, add
`--retry-failed` (successes remain skipped). For a capped continuation, also add
`--max-new-folds N`. Logs show dataset/model/seed/repeat/fold at start and saved
status, completed/failed/pending/skipped counts and runtime after each commit.
Exit codes: 0 successful command, 1 selected failed/corrupt evaluations,
2 preflight/argument/store error, 130 keyboard interruption.

## Validation and remaining concerns

Ten runner tests pass with mocked fits, covering full inventory, stable IDs,
exact saved-index fitting, no-generator/dry-run behavior, budgets, interruption,
resume, retries, duplicates, atomic persistence, lock exclusion, corruption and
provenance rejection. Command:

```powershell
.venv/Scripts/python.exe -m unittest experiments.protocol.test_runner -v
```

The seven real pilot folds (all seed 42, repeat 1, fold 1) are:

| Dataset | Configurations | Purpose |
| --- | --- | --- |
| Banknote Authentication | DT default, LR default, most-frequent dummy, stratified dummy | Numeric binary execution |
| Car Evaluation | RIPPER, SVM-RBF | Categorical multiclass execution, original OvR |
| Iris | Balanced DT with min_samples_leaf=2 | Numeric multiclass regularization |

All seven successful envelopes validate against the versioned JSON schema. The
official full dry run validated all 17 datasets, 16 model constructions per dataset,
and all 85 saved split archives; it reported 20,400 pending, zero corrupt records,
and performed zero fits. No environment/package changes were needed.

The pilot captures the known LR `penalty` deprecation warning and RIPPER's pandas
string-dtype deprecation warning. Neither caused failure; original parameters
were preserved. Other datasets/folds may still expose model-specific failures,
which the runner records without changing methodology. Historical SVM/RIPPER fold
pairing and numerical reproduction across old environments remain unverified;
these do not block new evaluations on the shared frozen indices.

There are no observed execution blockers for starting Block 2B. It still requires
the user's separate authorization. Rankings, chance correction, stability,
subsampling and new induction algorithms remain outside this runner.
