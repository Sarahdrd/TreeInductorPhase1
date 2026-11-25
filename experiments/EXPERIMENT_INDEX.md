# Completed experiment index

All computational experiments are complete; the saved evidence remains immutable.
F is the primary five-seed analysis reference, H the manuscript's historical
seed-42 reference. Block 8 publishes both explicitly. Use the analysis-only
sections of `New_Characterization_Experiments.ipynb` independently of its earlier
training cells.

| Block | Report | Principal code in `protocol/` | Final outputs in `analysis/` |
|---|---|---|---|
| 1 | REPRODUCIBILITY.md; RIPPER_RECOVERY.md | infrastructure, datasets, preprocessing, models, recovered_cross_models | Frozen configs and saved splits |
| 2 | BLOCK2A_RUNNER.md | runner, checkpoints, results | 20,400 local official records; subsequent analyses consume them |
| 3 | BLOCK3_FIVE_SEED_RANKING.md | block3_analysis, block3_figures, canonical_scoring | block3-five-seed-v1 |
| 4 | BLOCK4_CHANCE_CORRECTION.md | block4_analysis, block4_figures | block4-chance-correction-v1 |
| 5 | BLOCK5_LABEL_STABILITY.md | block5_analysis, block5_figures | block5-label-stability-v1 |
| 6 | BLOCK6_TREE_REGULARIZATION.md | block6_analysis, block6_figures | block6-tree-regularization-v1 |
| 7 | BLOCK7_MAGIC_SUBSAMPLING.md | block7, block7_analysis, block7b_reporting | block7-magic-subsampling-v1; block7-magic-report-v1 |
| 8 | BLOCK8_OPTIMAL_TREES.md | block8_full, block8_process, block8_scientific, block8_scientific_figures | block8-scientific-v1 |

`published_analysis.py` is the portable, checksum-verifying display loader.
Publication manifests list exactly which small tables and figures are distributed;
the original full manifests are preserved for archive regeneration. Historical
stage-specific statements in reports remain evidence of the work at that time,
not current pending tasks. Completed Block 7 has 10,500 successful fits; completed
Block 8 has 5,100, including the original 68 first-stage records.

## Retained historical modules (not deleted)

- `block8.py`: frozen original feasibility adapter/controller. **Do not use its
  old controller for production.** Use `block8_full.py` and `block8_process.py`.
- `block8_full_report.py`, `block8_full_audit.py`: explicitly first-stage-only,
  with 68-record assertions. They are not full-run scientific analysis commands.
- `block8_scientific.py`: current full-run analysis and artifact validation;
  refuses missing/duplicate coverage and output-directory overwrite.
- Old report/notebook generation scripts recreate their stage-era reports/cells;
  do not rerun them over finalized documentation. Original source hashes remain
  intact. The final report renderer is `block8_scientific_report.py`.
- Some historical preservation tests assert complete old notebook byte hashes
  or the former 64-cell notebook. They test stage-era snapshots; authorized
  output cleanup and Block 8 appending are validated by the final review audit.
  Scientific arithmetic tests remain applicable.

## Reproduction scope

For analysis regeneration, each `block*_analysis` entry point uses existing
saved evaluations and a new output directory. Read its report for dependencies;
several require the full private archive. None should silently recreate data or
partitions to make a missing archive pass. Full training commands are preserved
in the original reports but were not invoked during repository finalization.
See [README](../README.md), [dataset availability](../data/README.md),
[external setup](EXTERNAL_SOLVERS.md), and [final validation](repository_review/VALIDATION.md).
