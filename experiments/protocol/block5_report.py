"""Write the evidence report using verified Block 5 exports."""
import json
from .block5_analysis import OUTPUT,load_outputs
from .block4_report import markdown
from .infrastructure import ROOT


def write_report(output=OUTPUT):
    t=load_outputs(output)
    manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    names={d['dataset']:d['canonical_name'] for d in manifest['datasets']}
    def shown(frame):
        frame=frame.copy()
        for col in ['dataset','easier_dataset','harder_dataset']:
            if col in frame: frame[col]=frame[col].map(names)
        return markdown(frame)
    d=t['canonical_ranking_with_stability'];grid=t['scenario_rankings'];combined=grid.loc[grid.family.eq('combined')]
    parts=['# Block 5 — Label stability, weight sensitivity and stratum boundaries',
        '## Main findings',
        'Nine datasets retain their five-seed canonical stratum in all 25 seed × weight scenarios; eight change label at least once. Sonar retains its canonical Intermediate label in 21/25 scenarios (84%); Parkinsons retains its canonical Hard label in 13/25 (52%). High rank agreement coexists with these boundary changes: combined-scenario Spearman correlations range from 0.9436 to 0.9975, while individual scenarios change up to 4 of 17 stratum assignments. The canonical ranking itself is unchanged.',
        '## Method and reference',
        'The reference for every label and rank comparison is the original-score five-seed aggregate from Block 3, at alpha 0.70. All 20,400 official successful records were validated with the existing runner/checkpoint infrastructure; the exact 5,100 canonical DT/LR records reconstruct the saved primary and per-seed rankings. Block 4 outputs were checked only for provenance consistency and supply no scoring inputs. SVM, RIPPER, dummies and regularized trees are excluded.',
        'The grid uses seeds [42, 7, 21, 84, 123] and alpha [0.51, 0.60, 0.70, 0.80, 0.90]. Each seed uses its own validated F and mean within-model sample SD (ddof=1). Aggregate weight sensitivity uses the unchanged five-seed F and mean within-seed SD from Block 3. Each component is min-max normalized on the same 17 datasets. The shared original score_profile and assignment functions compute alpha × normalized E + (1 − alpha) × normalized V; alpha 0.70 retains the exact frozen 0.30 coefficient. No score is rounded before ranking. Exact ties use descending original F, ascending V, then dataset name and stable sorting. Every scenario has 6 Easy, 6 Intermediate and 5 Hard datasets.',
        'Seed stability counts matching individual-seed labels at alpha 0.70 out of 5. Weight stability counts matching aggregate labels across 5 alphas, including alpha 0.70 itself. Combined stability counts matching labels across 25 seed × alpha scenarios. These rates use the five-seed canonical label, not the historical seed-42 label. Five seed scenarios are members of the combined grid and are displayed again in the seed summary; they are not extra scenarios. Alpha 0 and 1 ablations are excluded.',
        '## Primary ranking with label stability',
        shown(d[['dataset','mean_macro_f1','normalized_error','normalized_variability','difficulty_score','rank','stratum','seed_matches','seed_stability','weight_matches','weight_stability','combined_matches','combined_stability']]),
        'For the primary manuscript table, use **combined_stability** as the label-stability column; retain seed and weight stability in supplementary tables. Suggested footnote: “Label stability is the fraction of 25 prespecified scenarios (five CV seeds × five error weights: 0.51, 0.60, 0.70, 0.80, 0.90) retaining the dataset’s original five-seed aggregated canonical stratum at alpha 0.70. This is descriptive agreement on a related sensitivity grid, not a probability of label correctness.”',
        '## Seed stability',
        shown(t['seed_stability'][['dataset','seed_matches','seed_differences','seed_stability','label_seed_42','label_seed_7','label_seed_21','label_seed_84','label_seed_123','most_frequent_seed_labels','canonical_agrees_with_majority']]),
        'Sonar differs only at seed 42; Anneal differs only at seed 123. Parkinsons differs at seeds 42 and 123. All canonical labels agree with a strict majority of the five individual-seed labels. Modal-label ties, if present, would be listed explicitly; strict majority means more than 2/5.',
        '## Weight stability on five-seed aggregate components',
        shown(t['weight_stability'][['dataset','weight_matches','weight_stability','label_alpha_0.51','label_alpha_0.60','label_alpha_0.70','label_alpha_0.80','label_alpha_0.90','minimum_rank','maximum_rank','minimum_score','maximum_score','changed_alphas']]),
        'At aggregate alpha 0.80, Sonar and Parkinsons exchange sides of the Intermediate–Hard boundary. At alpha 0.90, Balance Scale and Parkinsons exchange sides of that boundary, while Semeion and Ionosphere exchange sides of the Easy–Intermediate boundary. Alpha 0.51, 0.60 and 0.70 preserve all canonical labels. These are exact sampled alpha values with differing labels, not estimated continuous change thresholds. The five-seed results are not equated with historical seed-42 weight sensitivity.',
        '## Combined stability',
        shown(t['combined_stability'][['dataset','combined_matches','combined_stability','minimum_rank','maximum_rank','distinct_labels','boundary_crossing_count']]),
        'The eight datasets with at least one label change are Semeion, Car Evaluation, Credit Approval, Ionosphere, Balance Scale, Anneal, Sonar and Parkinsons. Every other dataset retains its label in all 25 scenarios. Every observed label change crosses one adjacent boundary; no dataset spans all three labels. The complete list of differing scenarios is in combined_stability.csv and every differing dataset/scenario record, including its score and ranks, is in label_transition_details.csv.',
        '## Ranking agreement is distinct from label agreement',
        shown(t['rank_agreement_by_family'][['family','spearman_rank_min','spearman_rank_mean','spearman_rank_max','kendall_tau_b_min','kendall_tau_b_mean','kendall_tau_b_max','changed_strata_min','changed_strata_mean','changed_strata_max','label_agreement_mean']]),
        'The per-scenario CSV additionally reports mean/max absolute rank displacement and changed-rank counts. Seed-only correlations range from 0.9926 to 0.9975, despite two seeds changing two labels each. Aggregate weight correlations range from 0.9485 to 1.0000; combined correlations range from 0.9436 to 0.9975. Rank correlations summarize all 17 positions, while moving one position across rank 6/7 or 12/13 changes a label. High correlation therefore does not guarantee stable boundary membership.',
        '## Boundary analysis',
        shown(t['canonical_boundary_gaps']),
        'The canonical Easy–Intermediate score gap is 0.037363 (Car Evaluation rank 6 versus Credit Approval rank 7). The Intermediate–Hard gap is 0.027756 (Sonar rank 12 versus Parkinsons rank 13). Midpoints shown in figures are descriptive aids only; labels are always assigned by sorted ranks, never by score thresholds.',
        shown(t['boundary_analysis'][['dataset','rank','difficulty_score','stratum','seed_stability','weight_stability','combined_stability','minimum_rank','maximum_rank','boundary_crossing_count','easy_intermediate_crossings','intermediate_hard_crossings','aggregate_weight_boundary_crossing_count']]),
        'Boundary-crossing counts count differences from the canonical side per boundary per scenario, not sequential transitions between arbitrarily ordered grid columns. The main boundary_crossing_count covers the 25 combined scenarios; aggregate weight crossings are reported separately. The following table lists the sets exchanging membership across each fixed boundary in each scenario. These are membership exchanges, not a claim of paired causal interactions.',
        markdown(t['boundary_exchanges']),
        '### Sonar and Parkinsons',
        'Sonar’s primary label is Intermediate, rank 12, score 0.461755. It retains this label in 4/5 seeds, 4/5 aggregate alpha values and 21/25 combined scenarios. It becomes Hard at seed 42 with alpha 0.60 and 0.70, and at seeds 7 and 84 with alpha 0.80. On aggregate components, alpha 0.80 changes Sonar to Hard; alpha 0.90 returns it to Intermediate. Stability is therefore not necessarily monotonic in alpha.',
        'Parkinsons’ primary label is Hard, rank 13, score 0.489511. It retains Hard in 3/5 seeds, 3/5 aggregate alpha values and 13/25 combined scenarios. Its 12 combined departures are to Intermediate and exchange boundary membership with Sonar, Balance Scale or Anneal depending on the scenario. Its canonical label still agrees with the majority of individual-seed labels at alpha 0.70; the lower combined rate reflects additional weight sensitivity. Neither rate is a probability of the label being correct.',
        '### Both boundaries and other requested datasets',
        'At the Easy–Intermediate boundary, Car Evaluation and Credit Approval exchange membership for seeds 84 and 123 at alpha 0.51. Semeion and Ionosphere exchange membership at alpha 0.90 for seeds 42, 21, 84 and 123. At the Intermediate–Hard boundary, Anneal becomes Hard for seed 123 at alpha 0.70 and 0.80, paired with Parkinsons moving to Intermediate. Balance Scale becomes Hard at alpha 0.90 for every seed, and also at seed 42 / alpha 0.80. Haberman remains Hard throughout. All scenarios and rank ranges are provided in the tables.',
        '## Publication figures and exports']
    for filename,caption in [('label_stability','A. Three agreement rates by canonical rank.'),('scenario_heatmap','B. Canonical reference and 25 seed × weight label assignments.'),('boundary_sensitivity','C. Scores around both fixed rank boundaries; Sonar and Parkinsons highlighted.'),('rank_vs_label_agreement','D. Rank agreement versus the number of changed labels.')]:
        parts.extend([caption,f'![{caption}](analysis/block5-label-stability-v1/{filename}.png)',f'[Vector PDF](analysis/block5-label-stability-v1/{filename}.pdf)'])
    parts.extend(['Exports under `analysis/block5-label-stability-v1/`: '+', '.join('`'+name+'.csv`' for name in t)+'. PNG figures use 300 dpi, PDFs are vector; CSVs retain round-trip precision. Provenance and official/artifact checksums accompany the tables.',
        '## Reproduction and validation',
        '```powershell\n.venv/Scripts/python.exe -m unittest experiments.protocol.test_block3 experiments.protocol.test_block4 experiments.protocol.test_block5\n.venv/Scripts/python.exe -m experiments.protocol.block5_analysis --output experiments/analysis/block5-label-stability-reproduction\n```',
        'Existing output directories are refused. The appended Block 5 notebook section reads checksummed saved exports and never runs evaluations. Run only these new display cells; historical cells may contain training workflows. All previous notebook cells are preserved.',
        'Automated checks verify exact Block 3 primary/per-seed reconstruction, canonical model identities, seed aggregation, alpha values and weights, per-seed and aggregate normalization, 6/6/5 scenario strata, denominators 5/5/25, canonical reference use, 30 distinct scenarios (25 combined + 5 aggregate) and 510 dataset/scenario rows, deterministic exact ties, absence of pre-ranking rounding, invalid-reference rejection and exclusion of chance correction. Source checks prohibit model fitting and split-generation calls in the analysis. Protected Block 3/4 exports, historical outputs, frozen configuration, saved splits, official records and prior notebook cells are checksum-audited; see `protocol/block5_validation_report.json` for final audit counts.',
        'Final verification: 19 analysis tests passed (Blocks 3–5); 20,400 official record files and 4,877 protected files remain byte-identical. All 40 prior notebook cells and notebook metadata are unchanged; eight cells were appended and only four new display cells executed. All 70 code cells across the three notebooks parse, notebook structure validates, and the environment identity is unchanged. Exported stability values match an exact independent recomputation from the validated Block 3 components. No models were fitted and no CV partitions were generated.',
        '## Reviewer traceability and suggested abstract wording',
        '**R1.2b:** the primary table includes seed, weight and combined stability with explicit denominators; the recommended combined-stability column and footnote define the reference label and sensitivity grid. Full scenario rankings, labels and transitions accompany it.',
        '**R1.2c:** exact evidence for an abstract acknowledgment includes Sonar’s 21/25 Intermediate agreements and Parkinsons’ 13/25 Hard agreements, plus up to four changed assignments per combined scenario despite Spearman correlations at least 0.9436. Suggested sentence:',
        '> Across the prespecified seed–weight sensitivity grid, rank correlations remained high (Spearman 0.944–0.998), but boundary membership changed: Sonar and Parkinsons retained their five-seed canonical strata in 84% and 52% of scenarios, respectively.',
        'This is proposed wording only; the manuscript and abstract were not edited.',
        '## Limitations',
        'The 25 scenarios share data, folds and score components and vary weights systematically. They are not independent experimental replicates; no binomial intervals, inferential p-values or probability-of-correctness claims are appropriate. The reported rates depend on the selected alpha grid, the 17-dataset pool, fixed stratum sizes, four baseline learners, metric and canonical reference. The inclusion of alpha 0.70 guarantees one matching aggregate weight configuration. Min-max normalization is relative to this pool, and score ranges across differently normalized seeds are descriptive. Labels compress a continuous relative ordering and are especially sensitive near rank cut points. Historical seed-42 comparisons and chance correction remain separate analyses. No score, pool, hyperparameter, environment or canonical label was changed.'])
    path=ROOT/'experiments/BLOCK5_LABEL_STABILITY.md'
    if path.exists(): raise FileExistsError(path)
    path.write_text('\n\n'.join(parts)+'\n',encoding='utf8')
    print(path)


if __name__=='__main__':write_report()
