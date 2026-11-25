# Dataset difficulty characterization for decision-tree experiments

This review repository contains the frozen protocol, analysis code, three main
notebooks, and compact results from the completed experimental revision. The
score combines normalized predictive error (0.70) and within-CV variability
(0.30). It is relative to the evaluated models and dataset pool, not an intrinsic
or universal dataset property.

## Start with the results

- [Experiment index](experiments/EXPERIMENT_INDEX.md): Blocks 1–8, code, reports and outputs.
- [New characterization experiments](experiments/New_Characterization_Experiments.ipynb):
  historical methodology plus analysis-only sections for completed Blocks 3–8.
- [Dataset characterization](experiments/Dataset_Characterization.ipynb): descriptive analyses.
- [Default-tree canonical strata](experiments/Default_Tree_Canonical_Strata.ipynb): historical tree profiles.
- [Final additional-inductor results](experiments/BLOCK8_OPTIMAL_TREES.md).
- [Validation and handoff](experiments/repository_review/VALIDATION.md).

**Do not use Run All to inspect results:** early notebook cells fit models.
Execute the appended analysis-only sections independently. They read curated,
checksummed CSV tables and figures; they do not invoke training or solvers.
Large cached notebook image outputs were cleared, while useful small tables
were retained. Figures are loaded from `experiments/analysis/` when displayed.

## Three references that must not be mixed

- **H:** historical seed-42 manuscript characterization. Parkinsons is Intermediate;
  Sonar is Hard. This remains the manuscript's canonical reference.
- **U:** unified-environment seed-42 evaluation.
- **F:** primary Block 3 five-seed aggregate. Sonar is Intermediate; Parkinsons is Hard.

Block 8 publishes primary F summaries and separate H sensitivity summaries.
Neither alternative inductor changes the canonical score or assignments.
The historical archive used inside the original Block 3 comparison is separately
identified in that report and is not silently substituted for manuscript H.

## Layout and what is distributed

| Location | Contents |
|---|---|
| `experiments/protocol/` | Reusable loaders, frozen runners/adapters, scoring, analysis, plotting and tests |
| `experiments/configs/` | Frozen model, dataset, environment and schema metadata |
| `experiments/splits/` | 85 saved benchmark CV collections: 17 datasets × five seeds |
| `experiments/analysis/` | Curated small tables, PNG/PDF figures, provenance and publication manifests |
| `experiments/*REPORT*.md`, `BLOCK*.md` | Methodology, scientific findings and historical implementation evidence |
| `data/README.md` | Required input identities and acquisition limitations |
| `experiments/repository_review/` | Exact staging recommendation, checks and caveats |

Raw datasets, historical result archives, large fold-level intermediates,
execution checkpoints, solver work inputs, logs, source checkouts and portable
runtimes remain local and are ignored. Nothing was deleted or moved to prepare
this review. A fresh clone can inspect published summaries after installing
analysis dependencies; it **cannot reproduce the exact experiments from scratch
without the separately archived inputs**. There is no verified public archive
URL for all those inputs. See [data instructions](data/README.md).

`published_checksums.json` lists the portable subset in each analysis directory.
The display-only `published_analysis.load_outputs` verifies that subset. Original
`output_checksums.json` manifests remain unchanged and may reference excluded
local intermediate files. Original full-analysis loaders intentionally require
the full archive; no checksum requirement was silently weakened.

## Python environment

The unified execution used Python 3.12.14 and the exact versions in
`experiments/requirements.txt`, including scikit-learn 1.9.0 and wittgenstein 0.3.5.
Use the existing `.venv` for this checkout; it has not been changed. For a new
review environment, install the recorded requirements in a separate virtual
environment using your Python interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/requirements.txt
```

These are setup instructions, not commands executed during cleanup. Package
availability and a new installation were not tested. The frozen execution
identity includes platform and executable path; moving the archive or changing
versions can require a separately versioned run configuration. Do not edit an
old frozen manifest to bypass an identity failure. Historical H used Python
3.11.5, NumPy 1.23.5, pandas 2.3.3, SciPy 1.15.3 and scikit-learn 1.3.0;
reviewing its saved results does not require recreating that environment.

## Reproduce analysis, not training

Display published Block 8 tables:

```python
from pathlib import Path
from experiments.protocol.published_analysis import load_outputs
tables = load_outputs(Path("experiments/analysis/block8-scientific-v1"))
print(tables["stratum_paired_deltas"])
```

With the complete local record/prediction/source archive restored, regenerate
Block 8 analysis in a new directory (never overwrite prior outputs):

```powershell
.\.venv\Scripts\python.exe -B -m experiments.protocol.block8_scientific --output experiments/analysis/block8-scientific-reproduction
```

Block 7 is already complete (10,500 successful fits); Block 8 has 5,100 successful
fits. No new execution is needed to review either result. Original full fitting
commands and authorization gates are retained in the corresponding reports.
Some old scripts and tests are stage-specific snapshots, not current full-run
entry points; consult the index before invoking them.

## External tree solvers

MurTree: [authors' source](https://github.com/MurTree/murtree), commit
`129e0466acf43f5b4759fba8222800ce40aa32a3`, C++14, portable Zig 0.13.0.
Quant-BnB: [authors' source](https://github.com/mengxianglgal/Quant-BnB), commit
`0a381a5bc6d689f66e5fca34a767da2c21918da0`, Julia 1.10.10, JSON 0.21.4.
See [external setup](experiments/EXTERNAL_SOLVERS.md). Native runtimes and
downloaded source trees are not redistributed. The original Quant-BnB checkout
had no explicit license file, so obtain it from its authors rather than treating
this repository as permission to redistribute it. No license for this project's
code is inferred; the owner must choose publication licensing before release.

## Git handoff

Work remains on `clean`, uncommitted and unpushed. Review the explicit inclusion
list and commands in `experiments/repository_review/HANDOFF.md`. Ignore rules do
not untrack existing files: two already tracked record-checksum inventories and
historical preservation metadata remain tracked. No automatic Git removal,
staging, commit, push, merge or branch change was performed.
