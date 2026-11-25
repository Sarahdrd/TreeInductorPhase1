# Final repository validation

Validation date: 2026-11-24. Branch: `clean`. No models were fitted, solvers invoked, splits generated, dependencies installed, or Git staging/commit/push operations performed.

## Scientific coverage and preservation

The new Block 8 analysis verified 5,100 successful official evaluations: 17 datasets, 75 saved folds per dataset and four configurations. Both solver/CART comparisons use identical fold keys and representation metadata. All 38,250 referenced artifact hashes were checked. MurTree has 1,275 certified constrained-optimum outcomes; Quant-BnB has 1,275 feasible, uncertified outcomes. No official failures or timeouts were found. The unrestricted unified CART comparison is explicitly separate from representation-matched depth-2 CART.

Primary F and historical H summaries are exported separately, including their differing Sonar/Parkinsons assignments. Previously verified six-decimal stratum differences are reproduced from saved records. Dataset means, rather than repeated folds treated as independent datasets, are the stratum-level units.

`preservation_validation.json` records 41,951 protected files unchanged, no unexpected changes, unchanged installed environment, and 170 unchanged saved split files. The snapshot includes 36,043 stored result records across the existing experiment directories, including preliminary records; this is not the official Block 8 count. Canonical scores, rankings, data and prior results were preserved. The manuscript was absent at the initial snapshot and was not edited.

## Tests and notebook checks

- 31 applicable lightweight tests passed; see `test_results.json`.
- The initial 32-test run had one failure: `test_block4.Block4Tests.test_no_fit_or_cv_calls_and_preserved_canonical` checks an archived whole-notebook byte hash. Authorized cached-output cleanup changes that hash. The test and historical checksum remain unchanged; the failure is not a scientific-result discrepancy.
- All three notebooks pass nbformat validation. Original scientific source and notebook metadata are checked against `notebook_source_baseline.json`. Only five documented display-loader cells changed in earlier sections, alongside cached-output cleanup and the appended Block 8 section.
- `Dataset_Characterization.ipynb` received cached-output cleanup only. `Default_Tree_Canonical_Strata.ipynb` is unchanged. Oversized cached figures were removed from the new-experiments notebook; saved figures remain available.
- 23 analysis-only cells from Blocks 3–8 executed with 71 display calls and zero fitting/solver calls. This was a programmatic notebook/display check, not an interactive Jupyter UI inspection or execution of training cells.
- A temporary review-only checkout, containing the proposed publication files but no raw records, datasets or solver sidecars, passed the same analysis-cell checks. See `review_only_validation.json`.
- All nine curated `published_checksums.json` manifests were verified. Original full-output manifests remain intact and may reference locally retained, excluded intermediates.
- Four new Block 8 figures were visually reviewed; PNG and vector PDF versions and generation code are included.
- `git diff --check` passed.

The lightweight display validation can be repeated from the repository root with:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.protocol.validate_review
```

## Publication limits

The curated checkout supports inspection of prepared analyses. Exact regeneration requires the original dataset bytes, archived records when auditing prior fits, and external solver dependencies described in `../EXTERNAL_SOLVERS.md` and `../../data/README.md`. A fresh end-to-end experimental bootstrap was not tested and is not claimed.

The inclusion inventory found no credential-pattern matches or files above 10 MiB. Pattern scanning is not a security guarantee. Eleven historical provenance locations retain recorded absolute source/executable paths; their locations are listed in `inventory_summary.json`. These immutable records were not rewritten to conceal their origins. No absolute machine paths remain in the primary notebook content detected by that scan.

Repository and external-source redistribution licenses require owner review, especially the Quant-BnB source license. External sources, runtimes and raw data are excluded from the proposed publication set. No scientific evidence was deleted or moved.
