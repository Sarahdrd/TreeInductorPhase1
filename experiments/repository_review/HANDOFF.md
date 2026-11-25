# Clean-branch publication handoff

The work remains on `clean`, uncommitted and unpushed. No staging, history changes, branch switches, scientific-result deletion or manuscript editing was performed.

## Completed work

Block 8 now has a full scientific report, seven final CSV tables, configuration/provenance manifests, four PNG/PDF figure pairs, and an analysis-only notebook section. All 5,100 official evaluations are successful. F is the primary grouping and H is explicitly reported as sensitivity analysis. Quant-BnB remains feasible but uncertified. Both matched CART and separate unrestricted unified CART comparisons are included.

The README, experiment index, data availability instructions, external-solver setup notes and shared ignore rules now describe the repository. Historical report bodies are preserved beneath clearly marked final-status updates. Original scientific notebook source is preserved except for five documented changes to load curated display outputs.

## Exact inclusion inventory

`prospective_files.tsv` lists each proposed file's path, current tracked/untracked state, size and SHA-256, excluding recursive hashes of the inventory's own three outputs. `stage-paths.txt` is the complete explicit path list. `inventory_summary.json` lists size totals, largest files, credential-pattern findings and historical machine-path locations. Review these files before staging; a subsequent local change requires a fresh review.

Currently tracked and modified files:

- `experiments/BLOCK2A_RUNNER.md`
- `experiments/BLOCK3_FIVE_SEED_RANKING.md`
- `experiments/BLOCK4_CHANCE_CORRECTION.md`
- `experiments/Dataset_Characterization.ipynb`
- `experiments/New_Characterization_Experiments.ipynb`
- `experiments/REPRODUCIBILITY.md`
- `experiments/RIPPER_RECOVERY.md`

Recommended untracked additions are enumerated individually in the inventory: root README and `.gitignore`; data documentation; later-block reports and experiment index; required protocol modules/tests; lightweight frozen configurations and external bridge/runtime metadata; curated analysis CSVs, figures and manifests; and this review evidence. A green/untracked file is not yet versioned. The explicit staging list includes already tracked unchanged files too, which is harmless.

## Files retained locally and excluded

Raw datasets, historical result stores, full fold checkpoints, subsamples, solver work directories, native source/build trees, portable runtimes, environments, caches, logs, compiled binaries and local notebook/snapshot backups remain on disk. Large intermediate fold tables are excluded in favor of small final outputs. Credentials and temporary artifacts should not be uploaded.

Ignore rules do not remove already tracked files. In particular, the existing Block 3/4 record-checksum manifests (approximately 3 MB each) remain tracked as provenance; no untracking was performed. No prospective file exceeds 10 MiB. See the inventory for exact sizes.

Eleven historical provenance locations contain absolute paths and are intentionally retained without altering frozen evidence. No credential-pattern matches were found; owner review is still appropriate. Raw-data distribution/access and repository/external licensing remain publication considerations. Exact fresh-clone numerical regeneration is not claimed without the prerequisites in the README. The manuscript was absent when this task began and was not modified.

## Validation

See [VALIDATION.md](VALIDATION.md): 31 applicable tests passed; all notebooks validate; 23 analysis cells execute in a review-only checkout; 41,951 protected files are unchanged. One archived whole-notebook checksum test fails following authorized output cleanup, while current source/metadata preservation checks pass. No historical test or checksum was changed to hide this distinction.

## Commands for the owner to review and execute

Run from the repository root. These staging, commit and push commands have **not** been executed. Confirm the configured `origin` destination before pushing.

```powershell
git branch --show-current
git status --short
git diff --check
git remote -v
Get-Content experiments/repository_review/stage-paths.txt
git --literal-pathspecs add --dry-run --pathspec-from-file=experiments/repository_review/stage-paths.txt

# Only after reviewing the exact proposed inclusion list:
git --literal-pathspecs add --pathspec-from-file=experiments/repository_review/stage-paths.txt
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Finalize experimental analyses and reproducibility materials"
git push origin clean
```

The branch is prepared for review, not a Git-clean working tree: the intentional pending changes remain visible until the owner approves and commits them.
