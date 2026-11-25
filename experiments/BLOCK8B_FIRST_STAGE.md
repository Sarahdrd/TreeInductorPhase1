> **Final status — repository review:** The remaining evaluations subsequently completed: all 5,100 official fits succeeded. This report preserves the historical 68-fit first-stage observations. Final scientific findings, primary F and historical H sensitivity results are in BLOCK8_OPTIMAL_TREES.md; the full-run analysis is block8_scientific.py.

# Block 8B — Official first-stage execution

## Scope and immutable protocol

Completed exactly 68 official evaluations: all 17 canonical datasets, both solvers and both representation-matched depth-2 CART configurations, using only seed 42, repetition 1, fold 1. The remaining 5,032 evaluations were not launched. Phase 8A pilots remain separate and are not copied into this new configuration. These official stage-one records are retained for full-run resume.

Frozen configuration: `block8-optimal-trees-v1/official-depth2-v1/configuration.json`; configuration ID `61be9b209866b127a38981d819b856dd2b72df7f85f442ed97b1cc87742deff7`. Settings: depth 2; MurTree at most 3 decision nodes, zero sparsity penalty, at most 16 numeric cuts per original feature and internal time limit 120 seconds; unchanged Quant-BnB native depth-2 settings; 120-second external wall and 4-GiB committed-memory job limits; serial execution and one numerical thread. The same external envelope is applied to separate CART worker processes. Preprocessing is training-only and occurs outside the solver/worker fit envelope, with its duration recorded separately. Total attempt time includes serialization, fitting and prediction checks.

The original authors’ implementations and binaries, Julia project, canonical runtime, preprocessing/runner/worker/controller source hashes, dataset manifest, all 85 saved split collections, model parameters and canonical label reference are pinned. Resume refuses a changed configuration. No canonical constructor, dataset, split, score or previous artifact was modified.

## Execution outcomes

| model | evaluated | successful | timed_out | failed | wall_mean | wall_max | job_peak_max_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cart_depth2_murtree | 17 | 17 | 0 | 0 | 3.260 | 3.766 | 121217024 |
| cart_depth2_quant_bnb | 17 | 17 | 0 | 0 | 3.127 | 3.369 | 120836096 |
| murtree | 17 | 17 | 0 | 0 | 0.105 | 0.308 | 10084352 |
| quant_bnb | 17 | 17 | 0 | 0 | 10.098 | 23.175 | 489086976 |

Successful predictions: 68/68. Timeouts without usable predictions: 0. Other failures: 0.

Macro-F1 is absent for unsuccessful fits. Quant-BnB predictions are feasible but uncertified. MurTree certification comes from the original solver flag; an external kill or missing returned tree never supplies a certificate. CART is recorded as a completed heuristic, not a globally optimal tree. All successful outputs pass label/length/depth/node validation, and solver training objectives are checked against returned training predictions.

## Per-dataset coverage

| dataset | cart_depth2_murtree | cart_depth2_quant_bnb | murtree | quant_bnb |
| --- | --- | --- | --- | --- |
| anneal | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| balance_scale | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| banknote | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| breast_cancer | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| car | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| credit_approval | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| cylinder_bands | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| glass | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| haberman | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| heart_disease | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| ionosphere | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| iris | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| parkinsons | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| semeion | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| sonar | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| tic_tac_toe | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |
| wine_red | heuristic_complete | heuristic_complete | certified_optimal | feasible_uncertified |

## Runtime and memory interpretation

wall_seconds measures external process startup through termination, including Julia startup/JIT and worker import overhead. solver_seconds, where supplied, measures the inner operation and is not directly interchangeable across languages. peak_job_memory_bytes is the Windows job peak committed memory, not Phase 8A’s polled working set. Memory allocation failures can yield nonzero exits; the controller does not infer a memory-limit cause from an arbitrary exit code. The 4-GiB limit is enforced for the attached job and descendants. Wall time has a small polling/cleanup overhead. Windows job kill-on-close handles interruptions and descendant cleanup. Logs and return codes are preserved per attempt.

## Execution difficulties and coverage risk

No first-stage execution difficulties were observed.

One saved fold per dataset does not establish feasibility of the remaining 74. Different partitions may change preprocessing dimensions, search behavior or runtime; successes here are not a guarantee of complete 75-fold predictive coverage. Any committed failure or timeout remains an explicit missing predictive outcome. Subsequent analysis must distinguish attempted coverage from successful coverage and cannot aggregate incomplete datasets as if they had 75 valid scores. This stage makes no claim about stratum usefulness or comparative predictive superiority.

## Resume and retry semantics

A normal resume skips all committed successful, timed-out and failed outcomes. Interrupted attempts are automatically retried. Explicit --retry-failed appends a new attempt for unsuccessful outcomes; it never replaces history or reruns a successful fit. Changed budgets, source, environment or preprocessing require a new configuration version.

After explicit authorization for the remaining folds, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.protocol.block8_full resume --authorize-remaining
```

Read-only coverage validation:

```powershell
.\.venv\Scripts\python.exe -m experiments.protocol.block8_full validate
```

Reissuing `first-stage` checks/skips its existing committed outcomes. It cannot expand to other folds. `resume` without `--authorize-remaining` is rejected. The full plan is 5,100 fits; the two solver configurations account for 2,550. Their worst-case external budget alone is 85 serial hours before overhead; CART/serialization adds time. This stage is complete and execution has stopped.

## Artifacts and preservation

Coverage, per-evaluation diagnostics and difficulties are in `analysis/block8-official-first-stage-v1/`, with checksums and record provenance. Full precision is retained. Work directories contain inputs, preprocessing, logs, native outputs and successful predictions, all linked by record hashes. No publication figures or notebook/manuscript edits were made. Validation results are saved separately in `block8-optimal-trees-v1/official-depth2-v1/first_stage_validation.json`.

## Completed validation

All 14 production/controller tests passed. The final audit verified all 68 exact first-stage keys, recomputed every successful training and validation macro-F1 exactly from saved predictions, and confirmed identical fold and representation identities for both matched CART comparisons. All 31,821 protected prior files remained unchanged, including prior results and notebook sections. Runtime/environment and immutable configuration checks passed. A first-stage resume with fitting explicitly blocked produced zero new fits and left all 68 result-record bytes unchanged. Git whitespace checks passed on branch clean.
