"""Write the Block 3 narrative from verified exported tables, never models."""
import json
from .block3_analysis import OUTPUT, load_outputs
from .infrastructure import ROOT


def markdown(frame):
    def fmt(value):
        if isinstance(value, float): return f'{value:.4f}'
        return str(value).replace('|','\\|')
    lines=['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def write_report():
    t=load_outputs()
    provenance=json.loads((OUTPUT/'provenance.json').read_text())
    manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names={r['dataset']:r['canonical_name'] for r in manifest['datasets']}
    def display(frame):
        f=frame.copy()
        if 'dataset' in f: f['dataset']=f.dataset.map(names)
        return f
    summary=t['comparison_summary'].set_index('comparison')
    primary=summary.loc['B_vs_C']; history=summary.loc['A_vs_B']; sensitivity=summary.loc['C_vs_D']
    wide=t['ranking_comparison'].set_index('dataset')
    boundary=t['boundary_focus'].set_index('dataset')
    text=f'''# Block 3 — Five-seed aggregated canonical difficulty ranking

## Findings

The primary unified seed-42 versus five-seed comparison changes **{int(primary.changed_ranks)} of 17 ranks**
and **{int(primary.changed_strata)} of 17 stratum assignments ({primary.changed_strata_percent:.4f}%)**.
Spearman rank correlation is **{primary.spearman_rank:.4f}** and Kendall tau-b is
**{primary.kendall_tau_b:.4f}**. Mean absolute rank displacement is
**{primary.mean_absolute_rank_displacement:.4f}**, with maximum
**{int(primary.maximum_absolute_rank_displacement)}** positions.

Sonar's actual primary aggregated assignment is **{wide.loc['sonar','five_seed_stratum'].title()}**,
at rank **{int(wide.loc['sonar','five_seed_rank'])}**. Parkinsons is
**{wide.loc['parkinsons','five_seed_stratum'].title()}**, at rank
**{int(wide.loc['parkinsons','five_seed_rank'])}**. These are computed findings,
not an assumed outcome of the reviewer's suggestion.

## Provenance and completeness

All **20,400 official successful evaluations** passed the existing checkpoint
checksum, runner provenance and result-schema validators. The exact Cartesian
grid includes 17 datasets, 16 configurations, 5 seeds, 3 repetitions and 5 folds.
All evaluation IDs are unique, all scores finite and in [0,1], and the dataset
fingerprints, resolved model configurations and saved split identities match.
No pilot, failed, missing or corrupt score was included. Exactly **5,100** records
belong to the four canonical models, with 15 scores per seed/model/dataset and
75 per model/dataset. No model was fit or evaluated and no split regenerated.

Canonical selection is only DT default, DT balanced, LR default and LR balanced.
SVM, RIPPER, dummy baselines and all regularized trees were validated as part of
the full store but excluded from scoring. Official input:
`experiments/results/official/revision2-five-seed-v1/block2b-five-seed/`.
Its run manifest hash is `{provenance['validation']['run_manifest_sha256']}`.
The complete record checksum inventory is exported separately.

## Definition and reference identities

A = manuscript-era stored seed-42 profile at
`{provenance['historical']['path']}` with its original `reproducibility.json`.
B = reconstructed seed 42 from the new official unified-environment results.
C = five-seed primary, using mean within-seed SD.
D = `five_seed_pooled_sd_sensitivity`.

The later profile at
`results/new_characterization_experiments/tables/dataset_experimental_profile.csv`
is a cleaned-notebook reproduction, not A. Its stored history note identifies A
as the older reference. Both older profiles have the same ranks/labels but differ
slightly numerically. This distinction prevents silently choosing the newer
reproduction as the manuscript-era comparison. The actual publication's exact
bytes were not independently verified.

For each model/seed, mean and sample SD use its 15 scores (`ddof=1`). For C,
average the five means and the five SDs separately; then average across the four
canonical models to obtain F and V, with E = 1 − F. Equal fold counts make the
five-mean average equivalent to the direct 75-score mean; this was numerically
checked for all 68 dataset/model pairs. D instead uses each model's sample SD
over 75 scores, then averages those four SDs.

Original notebook normalization, tie-breaking and assignment function bodies are
reused in `canonical_scoring.py`. E and V are min-max normalized separately on
the fixed 17 datasets; constant components map to zero. The score remains exactly
**0.70 × normalized E + 0.30 × normalized V**. Ordering uses exact score equality,
then higher F, lower V and dataset name; stable sorting and `numpy.array_split`
produce Easy/Intermediate/Hard groups of **6/6/5**. No pre-ranking rounding occurs.
The canonical literal 0.30 is retained, matching the original main score and
alignment helper; the generic panel helper's `1-alpha` differs only in possible
last-bit floating-point arithmetic and is not substituted for the canonical formula.

The user's Block 3 instruction explicitly resolves the formerly pending
variability choice in favor of C. Evaluation configs were not modified: this
analysis choice is recorded in Block 3 provenance, preserving runner identities.

## Full primary five-seed ranking

Numbers below are display-rounded to four decimals only. CSV exports retain
17 significant digits.

{markdown(display(t['five_seed_primary'][['dataset','mean_macro_f1','error','variability','normalized_error','normalized_variability','difficulty_score','rank','stratum']]))}

## Historical versus unified seed 42 (A versus B)

A→B changes **{int(history.changed_ranks)} ranks** and **{int(history.changed_strata)} strata**.
Spearman = {history.spearman_rank:.4f}; Kendall tau-b = {history.kendall_tau_b:.4f}.
This is a reproducibility/environment-change check, not evidence that one
environment is superior or that all differences can be causally attributed to
library versions. Historical CSV identities/string dtypes and execution histories
were documented in Blocks 1–2. Exact historical SVM/RIPPER pairing is irrelevant
to this four-model calculation and remains unverified.

The complete component differences (B minus A) are:

'''
    hist=t['comparison_details'].query("comparison == 'A_vs_B'")
    text+=markdown(display(hist[['dataset','mean_macro_f1_difference','error_difference','variability_difference',
                                 'difficulty_score_difference','rank_difference','stratum_changed']]))
    text+='\n\n## Ranking comparisons\n\nAll correlations below compare deterministic ordinal ranks, not p-values.\n\n'
    text+=markdown(t['comparison_summary'][['comparison','spearman_rank','kendall_tau_b','changed_ranks','changed_strata',
                                           'changed_strata_percent','mean_absolute_rank_displacement','maximum_absolute_rank_displacement']])
    text+='\n\nPrimary B→C changes by dataset:\n\n'
    text+=markdown(display(t['ranking_comparison'][['dataset','historical_rank','historical_stratum','new_seed42_rank',
                           'new_seed42_stratum','five_seed_rank','five_seed_stratum','score_difference','rank_displacement','stratum_changed']]))
    text+='\n\n## Sonar, Parkinsons, Haberman and boundary changes\n\n'
    for dataset in ['sonar','parkinsons','haberman']:
        r=wide.loc[dataset];b=boundary.loc[dataset]
        text+=(f"- **{names[dataset]}:** historical D={r.historical_difficulty_score:.4f}, rank {int(r.historical_rank)} "
               f"({r.historical_stratum}); unified seed-42 D={r.new_seed42_difficulty_score:.4f}, rank {int(r.new_seed42_rank)} "
               f"({r.new_seed42_stratum}); five-seed D={r.five_seed_difficulty_score:.4f}, rank {int(r.five_seed_rank)} "
               f"({r.five_seed_stratum}). B→C displacement: {int(r.rank_displacement):+d}. "
               f"Immediate easier/harder neighbors: {names.get(b.adjacent_easier_dataset,'none')} / "
               f"{names.get(b.adjacent_harder_dataset,'none')}. Signed distance to the rank-12/13 midpoint: "
               f"{b.rank_distance_to_intermediate_hard_midpoint:+.4f} rank positions and "
               f"{b.score_distance_to_intermediate_hard_midpoint:+.4f} score units; "
               f"{int(b.ranks_to_cross_intermediate_hard)} rank move(s) required to cross that boundary.\n")
    text+='\nThe midpoint at rank 12.5 and the average score of ranks 12 and 13 are descriptive distance conventions only. '
    text+='Strata are assigned by ordered group membership, not by a newly introduced score threshold.\n\n'
    crossers=t['comparison_details'].query("comparison == 'B_vs_C' and stratum_changed")
    text+='Primary boundary crossers: '+(', '.join(names[x] for x in crossers.dataset) if len(crossers) else 'none')+'.\n\n'
    focus=t['per_seed_rankings'].loc[t['per_seed_rankings'].dataset.isin(['sonar','parkinsons','haberman']),
                                    ['dataset','seed','difficulty_score','rank','stratum']].sort_values(['dataset','seed'])
    text+='Observed per-seed results in the new unified environment (not assumed from reviewer recollection):\n\n'+markdown(display(focus))
    text+=f'''

## Pooled-SD sensitivity and interpretation

C→D yields Spearman **{sensitivity.spearman_rank:.4f}**, Kendall tau-b
**{sensitivity.kendall_tau_b:.4f}**, **{int(sensitivity.changed_ranks)} rank changes**,
and **{int(sensitivity.changed_strata)} stratum changes**. The full D ranking is
exported separately. Pooled SD incorporates both within-seed variation and
between-seed mean differences; it is not numerically interchangeable with the
mean of within-seed SDs. The finite-sample denominator also differs, so pooled SD
need not be greater than mean within-seed SD.

The primary formulation preserves the original within-CV variability interpretation
while reducing dependence on the choice of a single seed. More folds alone do
not establish that the ranking is better. Repeated-CV folds overlap and are not
75 independent experimental replicates. No significance tests, independence-based
confidence intervals or causal environment claims are made.

## Tables, figures and reproducibility

Output directory: `experiments/analysis/block3-five-seed-v1/`.

- Four separate full rankings: historical A, unified seed42 B, primary C, pooled-SD D.
- `canonical_fold_scores.csv`: the validated 5,100 canonical scores.
- `seed_model_statistics.csv`: 340 dataset/model/seed means and sample SDs.
- `five_seed_model_statistics.csv`: 68 aggregated model rows, including both SD definitions.
- `per_seed_rankings.csv`: 85 dataset/seed rankings and labels for later Block 5.
- `comparison_details.csv`, `comparison_summary.csv`, `ranking_comparison.csv`:
  all six pairwise comparisons, exact component differences and boundary crossers.
- `boundary_focus.csv`: Sonar, Parkinsons, Haberman and all datasets changing
  assignments in any comparison, with primary-ranking neighbors and distances.
- `provenance.json`, `official_record_checksums.json`, `output_checksums.json`:
  input identity, method decisions and export integrity.

Publication plots are saved as 300-dpi PNG and vector PDF, with full dataset names:

![Seed-42 versus five-seed ranks](analysis/block3-five-seed-v1/rank_comparison.png)

![Rank displacement](analysis/block3-five-seed-v1/rank_displacement.png)

![Intermediate–Hard boundary](analysis/block3-five-seed-v1/intermediate_hard_boundary.png)

![Variability sensitivity](analysis/block3-five-seed-v1/variability_sensitivity.png)

Only a clearly marked analysis-only section is appended to
`New_Characterization_Experiments.ipynb`. Existing cells/results are preserved.
Execute the appended cells independently; the older notebook sections still
contain historical fitting workflows and must not be run for this analysis.
The new section loads verified exports through the shared module; it does not
duplicate scoring, runner or preprocessing implementations. Reproduction requires
the local official store, original CSVs, saved partitions and historical references,
which are not all tracked by Git. CSV/PDF/PNG files may also be locally ignored.

```powershell
.venv/Scripts/python.exe -m unittest experiments.protocol.test_block3 -v
.venv/Scripts/python.exe -m experiments.protocol.block3_analysis --output experiments/analysis/block3-five-seed-reproduction
```

The analysis command refuses to overwrite an existing output directory. All
20,400 records must pass validation before ranking computation. Parallelism is
limited to read-only checkpoint validation; no estimators are fitted. Extraction
provenance is in `protocol/block3_scoring_provenance.json`. Six lightweight tests
passed: canonical selection/completeness, 15/75 counts, means/SDs and ddof,
normalization/constants, weights/strata/names, exact ties/unrounded ordering and
original notebook-function cross-checks. Final preservation checks are recorded
in `protocol/block3_validation_report.json`.

Created: `block3_analysis.py`, `canonical_scoring.py`, `block3_figures.py`,
`block3_report.py`, `test_block3.py`, provenance/validation records, this report
and the Block 3 output directory. Modified: only the target notebook, by appending
the Block 3 section. Existing runner/config/data/model/preprocessing files and
unrelated notebooks are unchanged. No environment changes, commits, pushes, merges or branch changes.

## Reviewer traceability and next blocks

| Comment | Evidence provided | Deferred work |
| --- | --- | --- |
| R1.2a | B→C rank/stratum changes and explicit Sonar/Parkinsons boundary results | Manuscript adoption remains an editorial decision |
| R1.2b | Complete per-seed components, scores, ranks and strata | Combined seed/alpha label-stability column in Block 5 |
| R1.1 | Original score retained as the uncorrected reference | Chance correction in Block 4 |

Block 4 should compare the preserved primary score to a separately defined
chance-corrected alternative. Block 5 should finalize and calculate the combined
seed/alpha stability definition. Neither analysis was performed here, and no
historical or manuscript ranking was automatically replaced.
'''
    target=ROOT/'experiments/BLOCK3_FIVE_SEED_RANKING.md'
    if target.exists(): raise FileExistsError(target)
    target.write_text(text,encoding='utf-8')
    return target


if __name__=='__main__':print(write_report())
