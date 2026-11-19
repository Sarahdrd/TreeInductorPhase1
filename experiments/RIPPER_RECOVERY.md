# Recovered SVM/RIPPER implementation and Block 2 readiness

**Block 2A update:** the common fold runner is now implemented; its seven-fold
pilot passed, including original RIPPER/OvR on Car Evaluation. See
[BLOCK2A_RUNNER.md](BLOCK2A_RUNNER.md) for current runner readiness and commands.
The recovery validation below remains a record of the preceding step.

The current `.venv` is ready for the recovered SVM/RIPPER portion of Block 2:
installation, dependency checks and lightweight runtime tests pass. No remaining
RIPPER compatibility blocker was found within the tested scope. Full benchmark
behavior has not been tested, and the fold runner remains Block 2 implementation
work. Historical cross-model pairing is still unverified.

## Authoritative source and fidelity

The user referred to `Methodology_Only_Tree(3).ipynb`; the supplied accessible file
is `C:/Users/sarah/Downloads/Methodology_Only_Tree.ipynb`. The actual filename and
whole-notebook SHA-256 are recorded in
`configs/recovered_source_provenance.json`. Cross-Model Validation is cell 14
(zero-based); cell 16 provides the manifest/signature logic. Cell 9 provides the
shared SVM preprocessing definition.

`protocol/recovered_cross_models.py` extracts the original `RipperPreprocessor`,
`SklearnRIPPER`, `make_svm_model`, `make_cross_model`,
`check_ripper_compatibility`, `check_cross_predictions`, `SVM_CONFIG` and original
RIPPER import/configuration guard. Definition ASTs match the recovered notebook
exactly. Only module imports/initialization were supplied; the notebook's
top-level execution, result-writing and full evaluation loops are not invoked.
Read-only reference cell text is retained in `protocol/recovered_source/`.
The source file and the original three cleaned notebooks are not modified.

The shared `build_preprocessor` AST matches the existing extracted preprocessor,
so it is reused without duplication. `models.make_model(name, X, y=None)` now
delegates SVM/RIPPER construction to the recovered factory. RIPPER requires the
dataset's original encoded `y` to select the historical binary/multiclass path.
For future fold fitting, `X` and `y` must remain in the manifest's original order.

## Preserved behavior

- Numeric columns: training-fold median imputation, fallback `0.0` for all-missing
  training columns, preserved as floats. Native RIPPER discretizes numeric values
  using `n_discretize_bins=10`; tests verify fitted numeric bin definitions exist.
- Categorical columns: training-fold string mode, fallback `__MISSING__` for
  all-missing training columns, nominal strings. No RIPPER one-hot encoding or
  scaling. Missing validation values use training fills; unseen categories do
  not alter them.
- Binary: `SklearnRIPPER(random_state=42)` inheriting original RIPPER fit/predict.
  The infrastructure loader supplies encoded binary labels 0/1. Arbitrary raw
  string binary targets are outside this loader contract.
- Multiclass: the original `OneVsRestClassifier(estimator, n_jobs=1)` only when
  `len(np.unique(y)) > 2`. No target conversion or new multiclass strategy added.
- sklearn metadata: original `ClassifierMixin, RIPPER, BaseEstimator` inheritance
  and `classifier_tags.multi_class=False` retained.
- SVM: original scaled-numeric, median/mode-imputed, dense one-hot pipeline;
  `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=None)`.

## Installation and environment

Before changing the environment, the exact install command, dry-run result and
release dependency findings were reported to the user. The
[PyPI release metadata](https://pypi.org/pypi/wittgenstein/0.3.5/json) declares
Python >=3.8 and dependencies `pandas` and `numpy` without version bounds. The
release metadata and artifact digests are saved in
`configs/ripper_package_metadata.json`. Metadata compatibility alone was not
treated as runtime validation.

Executed command, using the existing `.venv`:

```powershell
uv --no-cache pip install --python .venv/Scripts/python.exe --no-deps wittgenstein==0.3.5
```

Only wittgenstein 0.3.5 was added. Python 3.12.14, scikit-learn 1.9.0, NumPy
2.5.3, pandas 3.0.5 and the other recorded packages retain their versions. No
historical environment was recreated. The installed package was built from its
published source distribution; build tooling was isolated from the target venv.
`uv --no-cache pip check --python .venv/Scripts/python.exe` checked 55 installed
packages and reported no dependency conflicts. Requirements now include the
active pin `wittgenstein==0.3.5`.

`configs/environment_ripper_ready.json` is the runtime environment snapshot for
future evaluations. `configs/environment.json` remains the original split-generation
environment. `prepare.py` accepts the explicitly recorded updated environment
only while its split/data dependencies match the original; existing manifests,
split IDs and split-generation provenance remain untouched. Future fold results
must record the **new runtime** environment ID, not the split-generation ID.
The original config is archived as `configs/experiment_config.block1.json`;
the active config updates recovery/runtime metadata without changing scientific
parameters, targets, weights or CV settings.

## Compatibility and manifest comparison

Run from repository root:

```powershell
.venv/Scripts/python.exe -m experiments.protocol.check_recovery
.venv/Scripts/python.exe -m experiments.protocol.prepare
```

The recovery audit passes **12 tests**: six recovery tests plus six Block 1
regression tests. Its fit scope is ten small fixture pipeline fits: two original
compatibility checks (each consumes only the first CV partition), six RIPPER
binary/multiclass × numeric/categorical/mixed cases, and two SVM cases. Those
smoke fits are not benchmark evaluations. The original helper checks finite F1
internally, but no scientific scores or ranking outputs are saved.

Coverage includes numeric and categorical missing values, all-missing imputation
fallbacks, train-only fill values, classifier tags before/after fitting, cloning,
Pipeline/OvR compatibility, prediction labels, probability shape/range/sums,
native numeric discretization and exact source AST fidelity. All 1,275 saved
partitions also revalidate; preparation succeeds without altering them.

| Manifest comparison | Finding |
| --- | --- |
| RIPPER hyperparameters | Exact match, including all defaults and n_discretize_bins=10 |
| RIPPER adapter and multiclass choice | Match recovered source; original OvR and metadata inheritance retained |
| SVM parameters | All match except sklearn 1.9 default `probability='deprecated'` versus historical `False` |
| SVM effective behavior | Installed BaseLibSVM source and fitted smoke tests confirm sentinel resolves to `_effective_probability=False`; original factory left unchanged |
| Preprocessing description | Historical description is abbreviated; recovered code confirms numeric preservation and nominal categorical strings |
| Implementation hashes | Both current bytecode signatures differ from historical signatures; exact values recorded in report |
| Library environment | Intentionally newer unified environment; historical versions are not restored |
| Warnings | pandas `Pandas4Warning` from wittgenstein's object/string dtype selection; non-fatal on pinned pandas 3.0.5 |

Historical signatures depend on Python bytecode and import/compilation context.
The Python patch versions and module context differ; the exact cause of hash
differences is not asserted. Exact extracted-source AST agreement is established,
but historical bytecode/cache identity is **not** claimed. Do not reuse historical
caches as new evaluations or claim matching predictions from these smoke tests.

`protocol/recovery_validation_report.json` records test output, warnings, both
sets of parameters/hashes, installation command, environment, and package changes.
The pandas warning is retained rather than suppressed or patched in the original
algorithm. A future pandas upgrade requires renewed compatibility tests.

## Remaining Block 2 work

1. Implement the common fold runner around `make_model(name, X, y)` and validated
   saved indices; collect complete fold records with actual resolved pipeline
   parameters, warnings and the new environment ID. Nothing here launches it.
2. Run authorized baseline/regularization evaluations in this unified environment.
   Smoke tests do not guarantee every dataset or fold will succeed. DT/LR full
   runtime behavior is outside this SVM/RIPPER recovery validation.
3. Preserve the previously documented historical environment and dtype differences.
   They do not prevent new evaluations but prevent assuming historical numerical
   reproduction. Exact historical SVM/RIPPER fold identity remains unavailable.
4. Keep five-seed aggregation, chance-correction/variability decisions and ranking
   calculations deferred to their designated later blocks.

No full cross-validation experiment, new difficulty ranking, canonical-score
change, historical-result overwrite, commit or push was performed.
