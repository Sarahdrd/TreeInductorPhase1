# Random vs. stratified SDD experiment

Open `Ex4_Random_vs_Stratified_SDD.ipynb`. All scientific parameters and selections
are written to `results/random_vs_stratified_sdd/` before search. The notebook
defaults to **pilot**, uses a spawn-safe script, and finds an installed compatible
Python (or uses `SDD_PYTHON`). The existing `EvoML` environment supports the GA;
the base environment lacks `pymoo`.

From the repository root, in the compatible environment:

```powershell
python -m unittest experiments.test_random_vs_stratified_sdd -v
python experiments/run_random_vs_stratified_sdd.py --prepare-only
python experiments/run_random_vs_stratified_sdd.py --mode pilot
```

After inspecting the pilot, continue all ten searches, then explicitly evaluate
the frozen configurations:

```powershell
python experiments/run_random_vs_stratified_sdd.py --mode full_search
python experiments/run_random_vs_stratified_sdd.py --mode final_evaluation
```

No full searches or final evaluation are triggered by the default mode.
`full_search` resumes the first pair from generation 2 and starts the remaining
four pairs. A completed search is reused, not repeated. All ten searches must
finish successfully before the final mode can evaluate anything. Repeating the
final mode retains recorded failures as well as successes; it does not selectively
retry failed held-out evaluations.

Confirm the pilot using the notebook's checkpoint status table: both first-pair
conditions must reach generation 2 with a viable candidate. Generation 0 is only
the initial population. An absent `pilot_summary.csv` and a missing stratified
checkpoint mean the pilot has not completed, even if its process has stopped.
`pilot` resumes an interrupted first pair; `full_search` can also continue the
same checkpoint directly, but provides no completed pilot cost report beforehand.

The 2026-11-03 repair fixes a serialization bug where CLI default `900` and an
explicit `900.0` timeout compared equal but produced different JSON hashes.
`prepare()` now returns the exact persisted representation after validating the
requested protocol. The one-time recovery script
`experiments/repair_sdd_numeric_identity.py` proves that the only executable code
change is that return statement before migrating dependent identities. It keeps
the original cache entries and backups and does not change results, RNG state,
fitness or scientific configuration. The applied migration report is under
`results/random_vs_stratified_sdd/recovery/2026-11-03_numeric_identity/`.

`--n-workers` defaults to 4 and `--gen-dataset-timeout-s` to 900. Keep these and the
code/environment unchanged to continue existing checkpoints. Changed protocols
require a new `--output` directory; the old pilot cannot become part of that
revised main plan. The timeout covers one gene on one dataset, all five folds
together, including process startup. It is never reduced based on performance.
No additional whole-run parallelism is used.

The canonical characterization source must already exist at
`results/new_characterization_experiments/` (override with `--source`). Required
files are `metadata.json`, `canonical_pool.json`, and the four individual
non-dummy model JSON cache entries for every dataset with at least five
observations per class. Missing or incompatible inputs produce an explicit
error; the experiment never executes a notebook or retrains the baselines.

The experiment-specific module imports the existing GA's gene domains, repair,
population generator, tournament and crossover, and reuses its mutation rule
and mu+lambda structure. It uses the existing decoder/tree unchanged. Its
isolated execution loop adds persistence, deterministic ties, strict fitness
failure handling, and an evaluator that cannot access held-out datasets during
search. The original clamped initial generation and crossover modulo correction
are deliberately retained. The numerical ordinal preprocessing is fit per fold;
unknown categories become missing values, which the inducer handles itself.

Main artifacts:

| Artifact | Meaning |
|---|---|
| `reserve_manifest.json`, `.csv` | Sorted eligible names, fixed reserve and available pool |
| `available_pool_ranking.csv` | Normalization and groups reconstructed on the available pool |
| `selection_manifest.json`, `selections.csv` | All ten selections fixed before training |
| `initial_populations.json`, `protocol.json` | Paired initialization, seeds, splits, data/code/environment identities |
| `target_id_audit.csv` | Resolved target/ID columns and removed-target row counts |
| `characterization_cost.json` | Previously paid baseline computation cost, separate from search |
| `runs/<run>/checkpoint.json` | Atomic state after initial population and each complete generation |
| `runs/<run>/history.csv`, `evaluation_events.csv` | Fitness, macro-F1, budgets, times, cache hits and failures |
| `cache/` | Only complete reusable gene–dataset evaluations |
| `attempts/` | Actual attempted evaluations, including incomplete fold records; never reused as cache |
| `pilot_summary.csv`, `pilot_dataset_cost.csv` | Measured pilot timing, unique genes, actual fit attempts and failures |
| `pilot_cost_estimate.json` | Explicitly approximate extrapolations, separate from measurements |
| `search_histories.csv`, `figures/search_fitness_diagnostic.*` | Search diagnostics, not cross-condition utility comparisons |
| `frozen_winners.json`, `winners.csv` | Final configurations selected by search SDD only |
| `final_fold_results.csv`, `final_dataset_results.csv` | Five-fold results including all failure slots |
| `final_run_summary.csv`, `paired_differences.csv` | Complete-only aggregates and paired descriptive comparisons |

`cache_hits` counts reused unique gene–dataset evaluations, not population slots.
`candidates_requested` includes duplicates; `batch_unique_genes` counts distinct
genes within a generation; `cumulative_unique_genes` counts distinct genes over
the run. `fits_started` includes interrupted fits; `fits_completed` counts folds
whose fit and scoring completed successfully. Dataset task times overlap across
workers and cannot be summed as wall time. Measured process overhead covers timed
dataset children that return results; it does not isolate all process-pool startup
cost. Generation wall times include dispatch, data loading and parent processing.

Low SDD alone is not evidence of useful transfer. Always report macro-F1 with
SDD. Final utility is the unweighted mean of the five held-out dataset means;
an incomplete run has unavailable aggregate macro-F1 and SDD, with coverage
reported explicitly. Five pairs on one fixed reserve support descriptive,
exploratory conclusions only. Folds are not independent replications.

The repository ignores `results/`; copy that directory separately when preserving
or sharing a run. Do not delete protocol or checkpoint files to bypass a mismatch.
