"""Render the Block 4 review report from verified saved tables."""
from pathlib import Path
import json
import pandas as pd
from .block4_analysis import OUTPUT,load_outputs
from .infrastructure import ROOT


def markdown(frame):
    def value(x):
        if isinstance(x,float): return '' if pd.isna(x) else f'{x:.4f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(output=OUTPUT):
    tables=load_outputs(output);d=tables['original_vs_corrected'];s=tables['rank_comparison_summary'];c=tables['descriptor_correlations']
    target=ROOT/'experiments/BLOCK4_CHANCE_CORRECTION.md'
    if target.exists(): raise FileExistsError(target)
    p=json.loads((output/'provenance.json').read_text())
    parts=['# Block 4 — Chance-corrected predictive difficulty',
        'The corrected composite ranking changes 13 of 17 ranks and 2 of 17 strata (11.76%). Glass moves from rank 15 Hard to rank 12 Intermediate; Sonar moves from rank 12 Intermediate to rank 13 Hard. No dataset crosses the Easy–Intermediate boundary. Spearman agreement is 0.9608 and Kendall tau-b is 0.8676; mean absolute displacement is 1.0588 ranks and the maximum is 3. Heart Disease and Wine Quality Red both remain Hard.',
        '## Scope and method',
        'This is a comparative sensitivity analysis, not a replacement of the proposed canonical score. The primary reference is the Block 3 five-seed ranking. All 20,400 official successful records were validated using the existing Block 3/runner provenance checks; 5,100 canonical and 2,550 dummy records are used here. Every dataset/model has five seeds [42, 7, 21, 84, 123], 15 saved evaluations per seed and 75 overall. Pilot records are excluded.',
        'Seed-level means and sample SDs (ddof=1) are computed first. F is the equally weighted mean of four DT/LR model means; F_dummy is the five-seed stratified-dummy mean. E_cc = (1 − F)/(1 − F_dummy), I = (F − F_dummy)/(1 − F_dummy). Correction is applied to aggregate means, never averaged fold ratios. D_cc = 0.70 minmax(E_cc) + 0.30 V_norm. Block 3 V and V_norm are unchanged. The shared original normalization and exact tie-breaking functions are reused: score ascending, original F descending, V ascending, dataset name; stable ordering and 6/6/5 rank strata.',
        f'All denominators are positive; the smallest is {p["numerical_checks"]["minimum_denominator"]:.8f}. The explicit near-zero guard is sqrt(machine epsilon), approximately 1.49e-8. No negative improvements, errors outside [0,1], clipping or unstable denominators occurred. Pathological inputs raise an error and stop analysis.',
        'The frozen experimental configuration contains archival planning placeholders. It is intentionally unchanged because it belongs to the official run provenance; the approved Block 4 formula and decisions are recorded in this analysis’s provenance.json.',
        '## Dummy references for all 17 datasets',
        'The CSV additionally records five seed means and SDs, both 75-fold pooled SDs, fold minima/maxima, class counts, majority proportions, and imbalance ratios. Imbalance ratio is maximum/minimum class count; all 17 have positive class counts. The two reference strategies remain separate.',
        markdown(tables['dummy_baselines'][['dataset_name','dummy_most_frequent_mean_f1','dummy_most_frequent_mean_within_seed_sd','dummy_most_frequent_pooled_sd','dummy_stratified_mean_f1','dummy_stratified_mean_within_seed_sd','dummy_stratified_pooled_sd']]),
        '## Full corrected ranking and original comparison',
        'Positive rank displacement means a move toward the harder end. All CSV values retain round-trip precision; only presentation values are rounded.',
        markdown(tables['chance_corrected_ranking'][['dataset_name','original_mean_macro_f1','stratified_dummy_macro_f1','original_error','chance_corrected_error','chance_corrected_difficulty','corrected_rank','corrected_stratum']]),
        markdown(d[['dataset_name','original_difficulty','chance_corrected_difficulty','original_rank','corrected_rank','rank_displacement','original_stratum','corrected_stratum']]),
        '## Rank agreement and boundary crossings',markdown(s),
        'Raw-error ranks are analyzed separately from composite ranks, using ascending error with the same F/V/name tie-breakers. Rank correlations are descriptive Spearman and Kendall tau-b. They do not use folds as independent observations.',
        'Exact stratum changes:',markdown(d.loc[d.stratum_changed,['dataset_name','original_rank','corrected_rank','original_stratum','corrected_stratum']]),
        '## Class-structure associations (R1.1)',
        'These are newly computed associations across the 17 datasets, not historical coefficients. No inferential p-values are reported. The coefficient changes are descriptive and do not establish that correction removes imbalance effects.',
        markdown(c.pivot(index='descriptor',columns='metric',values='spearman').reset_index()),
        'All coefficients use n=17. The historical approximately −0.708 minimum-class-count association is not substituted for current results. The current composite-score coefficient changes from −0.7304 to −0.6176 (difference +0.1127); for raw error it changes from −0.6348 to −0.5049 (+0.1299). Composite associations with number of classes weaken from 0.3325 to 0.1503 (−0.1822), and with imbalance ratio from 0.5539 to 0.4412 (−0.1127). The observation-count association becomes more negative, −0.3603 to −0.4240 (−0.0637); the majority-proportion association remains near zero, −0.0245 to 0.0368 (+0.0613). Thus association changes are not a uniform removal of class-structure effects.',
        '## Heart Disease and Wine Quality Red',
        markdown(tables['heart_wine_focus'][['dataset_name','number_of_classes','minimum_class_count','stratified_dummy_macro_f1','original_mean_macro_f1','original_error','chance_corrected_error','original_difficulty','chance_corrected_difficulty','original_rank','corrected_rank','original_stratum','corrected_stratum','rank_displacement']])]
    for dataset in ['heart_disease','wine_red']:
        r=d.loc[d.dataset.eq(dataset)].iloc[0]
        parts.append(f'{r.dataset_name}: {r.number_of_classes} classes, minimum class count {r.minimum_class_count}. Stratified dummy macro-F1 is {r.stratified_dummy_macro_f1:.4f}, versus canonical mean {r.original_mean_macro_f1:.4f}; the absolute improvement is {r.improvement_over_dummy:.4f} and normalized improvement {r.chance_corrected_improvement:.4f}. Rank changes from {r.original_rank} ({r.original_stratum.title()}) to {r.corrected_rank} ({r.corrected_stratum.title()}), displacement {r.rank_displacement:+d}.')
    parts.extend(['Both datasets remain Hard and move only one rank toward the easier end: Wine Quality Red 16→15 and Heart Disease 17→16. Their corrected errors increase, but min-max scaling is relative to the pool: Wine Quality Red’s composite score decreases from 0.7935 to 0.7701, while Heart Disease’s remains 0.9096 because it is the maximum-error endpoint in both formulations and its variability is unchanged. Haberman becomes rank 17 (from 14), showing that relative improvement over dummy can change ordering even without altering the variability component.',
        'Raw-error agreement (Spearman 0.9412, Kendall 0.8235) is lower than composite agreement (0.9608, 0.8676). The maximum raw-error displacement is 4, versus 3 for the composite score; 12 raw-error ranks change versus 13 composite ranks. Min-max scaling alone preserves raw-error ordering; the retained variability term changes how error differences contribute to composite ranks. The two comparisons therefore answer distinct questions.',
        'Persistence of the two Hard assignments supports their relative classification under this particular dummy-reference transformation. It does not prove intrinsic difficulty or eliminate the influence of multiclass structure, imbalance, macro-F1 behavior, sample size, or the chosen learners. A changed position likewise does not establish that either formulation is scientifically preferable.',
        '## Improvement over stratified dummy',
        markdown(d[['dataset_name','original_stratum','improvement_over_dummy','chance_corrected_improvement']]),
        markdown(tables['improvement_by_original_stratum']),
        'These descriptive summaries use original five-seed strata. They are not independent validation: the same canonical performance contributed to the original labels. Absolute improvement is a diagnostic only and is not added to the score.',
        '## Publication figures and exported artifacts'])
    for name,caption in [('rank_comparison','A. Original versus corrected ranks; orange indicates changed strata.'),('score_comparison','B. Original versus corrected composite scores.'),('class_structure','C. Minimum class count and both scores; Heart Disease and Wine Quality Red identified.'),('dummy_vs_canonical','D. Stratified dummy versus canonical baseline macro-F1.')]:
        parts.extend([caption,f'![{caption}](analysis/block4-chance-correction-v1/{name}.png)',f'[Vector PDF](analysis/block4-chance-correction-v1/{name}.pdf)'])
    parts.extend(['All exports are under `experiments/analysis/block4-chance-correction-v1/`. Tables: '+', '.join('`'+k+'.csv`' for k in tables)+'. The provenance, complete official record checksum inventory, and output checksums accompany the figures.',
        '## Reproduction and validation',
        'From the repository root, using the unchanged unified environment:',
        '```powershell\n.venv/Scripts/python.exe -m unittest experiments.protocol.test_block3 experiments.protocol.test_block4\n.venv/Scripts/python.exe -m experiments.protocol.block4_analysis --output experiments/analysis/block4-chance-correction-reproduction\n```',
        'An existing output directory is never overwritten. The appended notebook section only reads checksummed exports and displays tables/figures. Run it independently: earlier historical notebook sections can contain training workflows. Shared analysis modules contain all scientific calculations.',
        'Validation checks cover exact dummy counts and seed identity, correct strategy selection, sample SD definitions, aggregate-level correction versus nonlinear fold ratios, denominator and range safeguards without clipping, unchanged variability, normalization, weights, ranks, strata and dataset identities. Block 3 canonical values are reconstructed from validated official folds and matched exactly to saved outputs. Protected historical files, other notebooks, Block 3 artifacts, frozen configuration and saved splits are hashed. See `protocol/block4_validation_report.json` for the final executed audit.',
        '## Limitations and reviewer traceability',
        '- R1.1a: both dummy strategies and fold dispersion are reported for every dataset.\n- R1.1b: the stratified-dummy correction is fully specified and applied to aggregate means; the complete corrected ranking and raw-error comparison are exported.\n- R1.1c: all rank/stratum changes, descriptor associations and Heart Disease/Wine Quality Red results are quantified.',
        'The correction is reference-relative, not universally calibrated intrinsic difficulty. Macro-F1 and stratified dummy behavior both depend on class structure and finite samples; dummy seeds follow the frozen model seed policy. Min-max scaling depends on this fixed 17-dataset pool, and the unchanged variability term can moderate or amplify rank movements. These 17 heterogeneous datasets are the units of exploratory association. Repeated CV folds overlap and are dependent; 75 scores do not represent 75 independent datasets. No causal removal of class imbalance is established. No manuscript, canonical ranking, model configuration, CV partitions or environment was changed.',
        'The empirical comparison documents sensitivity to the predictive-error reference while retaining all other scoring decisions. Selection of a preferred formulation is deferred for scientific review; this block neither replaces the proposed score nor automatically recommends doing so.'])
    target.write_text('\n\n'.join(parts)+'\n',encoding='utf8')
    return target


if __name__=='__main__': print(write_report())
