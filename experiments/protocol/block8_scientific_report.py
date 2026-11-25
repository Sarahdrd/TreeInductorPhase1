"""Render the completed scientific report from checksummed, saved summaries."""
from .published_analysis import load_outputs
from .block8_scientific import OUTPUT
from .infrastructure import ROOT
import json


def markdown(table):
    def fmt(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
    return ('| '+' | '.join(table.columns)+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'
            +'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in table.itertuples(index=False,name=None)))


def render():
    t=load_outputs(OUTPUT);p=json.loads((OUTPUT/'provenance.json').read_text())
    performance=t['dataset_performance'].pivot(index='dataset',columns='model',values='mean_macro_f1').reset_index()
    deltas=t['dataset_paired_deltas'];strata=t['stratum_paired_deltas']
    parts=['# Block 8 — Completed additional-inductor evaluation',
        '**Final status:** the complete official depth-two experiment and scientific analysis are complete. The Phase 8A report retained below is historical, not current execution status. The first-stage report remains a separate historical diagnostic. No fitting was performed during this final analysis.',
        '## Motivation, scope and reviewer correspondence',
        'R2.1 and R3.1 request induction strategies absent from score construction. MurTree and Quant-BnB satisfy that design requirement across all 17 datasets. They offer complementary shallow-tree optimization objectives, but reuse the benchmark and saved partitions, so this is not fully independent statistical validation. The evidence supports organizing heterogeneous comparisons, not improved algorithm selection or generalization to unseen datasets.',
        '## Frozen implementation and evaluation protocol',
        'MurTree author source commit `129e0466acf43f5b4759fba8222800ce40aa32a3`; Quant-BnB commit `0a381a5bc6d689f66e5fca34a767da2c21918da0`. Original references: [MurTree, JMLR 2022](https://jmlr.org/papers/v23/20-520.html), [Quant-BnB, ICML 2022](https://proceedings.mlr.press/v162/mazumder22a.html). C++14 was built with portable Zig 0.13.0; Quant-BnB ran with Julia 1.10.10 and JSON 0.21.4. Python 3.12.14 and the existing unified package versions were unchanged. See configuration_summary.json for exact settings and packages; source files and binaries are bound by the immutable local configuration.',
        'Training-only numerical median and categorical mode imputation, constant removal, and categorical one-hot encoding were used. MurTree receives at most 16 evenly ranked training-value midpoint cuts per original numerical feature. Quant-BnB retains continuous numerical values. Native multiclass paths are used. MurTree: depth ≤2, ≤3 decision nodes, sparsity penalty zero. Quant-BnB: native depth 2, s=3, mid_method=2, mid_ratio=0.2, classification C, original greedy upper bound ×(1+1e−6); its iteration cap can terminate without certification. Each separate matched CART uses the exact solver representation, Gini splitting, no class weighting, max_depth=2, and random_state=42.',
        'All models use the same five saved CV seeds (42, 7, 21, 84, 123), three repetitions and five folds. External limits are 120 seconds and 4 GiB aggregate job committed memory; MurTree also has a 120-second internal limit. Execution is serial with one numerical thread using the tested v2 Windows job controller. Preprocessing is excluded from process wall time; startup, imports and Julia JIT are included. Budgets are not pure optimization-speed measures.',
        '## Final coverage and certification',
        f"All {p['records']}/5,100 fits succeeded; no missing evaluations, timeouts, failures or unsupported datasets. All 2,550 matched solver/CART pairs have identical folds and representations. MurTree reports 1,275 certified optimal solutions within the supplied constrained binary space; Quant-BnB reports 1,275 feasible, uncertified solutions. Both CART configurations complete 1,275 heuristic fits. All returned trees have depth two; Quant-BnB uses three decision nodes and the other configurations use two or three. Training-error optimality is not validation macro-F1 optimality.",
        markdown(t['solver_status']),
        '## Dataset-level validation performance',
        'Each value averages five seed means of 15 fold scores each (equivalently 75 scores in this balanced grid). Macro-F1 is a unitless fraction. No intermediate rounding is used.',markdown(performance),
        '## Representation-matched CART differences',
        'Solver minus matched depth-two CART is calculated within each exact fold, then averaged over 75 pairs per dataset. This comparison matches representation and depth, but the solver training-misclassification objective differs from greedy CART Gini. Positive, exactly zero and negative signs are classified before rounding.',
        markdown(deltas[deltas.reference.eq('representation_matched_depth2_CART')][['dataset','model','pairs','mean_delta','stratum_F','stratum_H']]),
        '## Primary F and historical H sensitivity',
        'F is the frozen primary Block 3 five-seed reference (Sonar Intermediate, Parkinsons Hard). H is the manuscript historical reference (Parkinsons Intermediate, Sonar Hard). All other memberships agree. Each stratum contains 6/6/5 equally weighted dataset observations. Fold scores are not independent datasets. These are fixed groupings; neither solver changes canonical scores or labels.',
        markdown(strata[strata.reference.eq('representation_matched_depth2_CART')].drop(columns='reference')),
        'F paired means (Easy, Intermediate, Hard) are MurTree +0.007742/+0.027024/+0.016812 and Quant-BnB +0.005729/+0.020069/+0.015130. H sensitivity means are +0.007742/+0.021977/+0.022870 and +0.005729/+0.018537/+0.016968. Counts are 4/6, 4/6, 4/5 positive for MurTree and 4/6, 5/6, 4/5 for Quant-BnB under either grouping; exact zeros are absent. Identical counts do not imply identical membership or means.',
        markdown(t['stratum_performance']),
        '## Unrestricted canonical CART — separate comparison',
        'This reference is the default unweighted CART from the completed unified five-seed baseline run, not the historical H execution and not an additional fit. Its saved dataset/split/environment identities match. Representation and depth are not matched, so these deltas cannot isolate the optimization algorithm. H here is only a grouping, never a substitution of historical prediction scores.',
        markdown(strata[strata.reference.eq('unrestricted_unified_CART')].drop(columns='reference')),
        'Both solvers have negative stratum-average differences against unrestricted unified CART under F and H. The matched shallow-tree gains must not be described as superiority to unrestricted trees. The complete dataset-level canonical comparisons are in dataset_paired_deltas.csv.',
        '## Runtime, complexity and scientific limitations',markdown(t['runtime_summary']),
        'Runtime summaries describe observed process durations (including startup/JIT); p95 is a descriptive quantile, not an inferential bound. The plot uses dataset mean runtimes. Quant-BnB has its largest dataset mean on Easy Semeion (about 32.969 s), while Hard group mean runtime is lower than Easy under H. Predictive difficulty does not order optimization cost. A shallow four-leaf ceiling is restrictive for ten-class Semeion; its macro-F1 is about 0.2149 under both solvers despite its Easy canonical assignment. Hard means are lower overall, but Easy and Intermediate are not consistently ordered. Individual datasets overlap; depth, representation, objective and training information affect the result. Repeated folds and seeds reuse observations. No significance tests treating folds as independent datasets, causal interpretations, or claims of unseen-dataset generalization are made.',
        '## Figures and reproducibility',
        'All figures use dataset-level observations except the explicitly labeled execution-status counts. F/H performance panels are separate; dataset deltas themselves do not depend on grouping. Runtime axes are logarithmic. PNG exports use 300 dpi; PDF exports preserve vector graphics.',
        '\n\n'.join(f'![{name}](analysis/block8-scientific-v1/{name}.png)\n\n[Vector PDF](analysis/block8-scientific-v1/{name}.pdf)' for name in ['performance_by_stratum','dataset_matched_differences','runtime_by_method','solver_status']),
        'The appended Block 8 notebook section loads published_checksums.json-verified exports and performs no fitting. To regenerate all scientific summaries from the private full record/artifact store, run `python -m experiments.protocol.block8_scientific --output experiments/analysis/block8-scientific-reproduction`. Existing directories are refused. Rendering the report is separate: `python -m experiments.protocol.block8_scientific_report` (refuses to prepend a second completed report). Review-only loading requires no runtime sidecars or raw fold stores. See ../README.md and EXPERIMENT_INDEX.md for archive dependencies and setup limitations.',
        '## Validation and preservation',
        f"The new audit verified {p['source_and_split_files_verified']} frozen source/runtime/split files and {p['fit_artifacts_verified']} fit artifacts, 5,100 checksummed official records, and 1,275 unrestricted CART records. The existing full_execution_validation.json also records exact score recomputation for all 5,100 saved prediction outputs. New paired checks verify 75 unique matching folds for every dataset and comparison. All previously verified F/H six-decimal means were reproduced from records. See repository_review/VALIDATION.md for final tests, notebook preservation and repository checks. Full source-record checksums are in provenance.json. No model was retrained and no saved record was overwritten.",
        '## Archived Phase 8A feasibility findings (original text follows)',
        'The status and future-tense statements below describe the historical pilot stage only. They are retained as implementation evidence and are superseded by the completed full-run status above.']
    path=ROOT/'experiments/BLOCK8_OPTIMAL_TREES.md';old=path.read_text(encoding='utf8')
    if old.startswith('# Block 8 — Completed'):raise FileExistsError('Completed report already exists')
    path.write_text('\n\n'.join(parts)+'\n\n'+old,encoding='utf8')


if __name__=='__main__':render()
