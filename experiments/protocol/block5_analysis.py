"""Original-score stability against the immutable five-seed Block 3 reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import block3_analysis as b3
from .canonical_scoring import score_profile
from .checkpoints import IntegrityError
from .infrastructure import ROOT,config,digest,json_bytes

OUTPUT=ROOT/'experiments/analysis/block5-label-stability-v1'
ALPHAS=(.51,.60,.70,.80,.90)
LABELS=('easy','intermediate','hard')


def score(components,alpha):
    if alpha not in ALPHAS: raise ValueError('Alpha outside the prespecified stability grid')
    if components.dataset.duplicated().any() or set(components.dataset)!=set(config()['primary_datasets']):
        raise IntegrityError('Not the exact frozen 17-dataset pool')
    source=components[['dataset','mean_macro_f1','variability']].rename(columns={
        'mean_macro_f1':'baseline_macro_f1_mean','variability':'cv_variability'}).assign(ranking_eligible=True)
    if not np.isfinite(source[['baseline_macro_f1_mean','cv_variability']].to_numpy()).all(): raise ValueError('Non-finite components')
    ranked=score_profile(source,b3.NS,alpha=alpha).rename(columns=b3.RENAME)
    ranked['rank']=ranked['rank'].astype(int)
    if ranked.stratum.value_counts().reindex(LABELS).tolist()!=[6,6,5]: raise IntegrityError('Invalid strata sizes')
    return ranked[['dataset']+b3.VALUE_COLUMNS].sort_values('rank').reset_index(drop=True)


def assert_reference(actual,expected):
    columns=['dataset']+b3.VALUE_COLUMNS
    pd.testing.assert_frame_equal(actual[columns].sort_values('rank').reset_index(drop=True),
        expected[columns].sort_values('rank').reset_index(drop=True),check_exact=True,check_dtype=False)


def scenarios(primary,per_seed):
    expected={(d,s) for d in config()['primary_datasets'] for s in config()['cv']['seeds']}
    keys=list(per_seed[['dataset','seed']].itertuples(index=False,name=None))
    if len(keys)!=85 or set(keys)!=expected: raise IntegrityError('Incomplete/duplicate per-seed components')
    rows=[]
    for seed in config()['cv']['seeds']:
        components=per_seed.loc[per_seed.seed.eq(seed)]
        for alpha in ALPHAS:
            ranked=score(components,alpha)
            if alpha==.70: assert_reference(ranked,components)
            rows.append(ranked.assign(scenario=f'seed_{seed}_alpha_{alpha:.2f}',family='combined',seed=seed,alpha=alpha))
    for alpha in ALPHAS:
        ranked=score(primary,alpha)
        if alpha==.70: assert_reference(ranked,primary)
        rows.append(ranked.assign(scenario=f'aggregate_alpha_{alpha:.2f}',family='aggregate_weight',seed=pd.NA,alpha=alpha))
    result=pd.concat(rows,ignore_index=True)
    if len(result)!=510 or result.scenario.nunique()!=30: raise IntegrityError('Unexpected scenario count')
    reference=primary[['dataset','rank','difficulty_score','stratum']].rename(columns={
        'rank':'canonical_rank','difficulty_score':'canonical_difficulty_score','stratum':'canonical_stratum'})
    result=result.merge(reference,on='dataset',validate='many_to_one')
    result['label_matches']=result.stratum.eq(result.canonical_stratum)
    result['rank_displacement']=result['rank']-result.canonical_rank
    for edge,name in [(6,'easy_intermediate'),(12,'intermediate_hard')]:
        result[name+'_crossed']=(result['rank']<=edge)!=(result.canonical_rank<=edge)
    result['boundaries_crossed']=result.easy_intermediate_crossed.astype(int)+result.intermediate_hard_crossed.astype(int)
    return result


def stability_tables(primary,grid):
    combined=grid.loc[grid.family.eq('combined')]
    weight=grid.loc[grid.family.eq('aggregate_weight')]
    seed=combined.loc[combined.alpha.eq(.70)]
    tables={}
    for name,frame,denominator in [('seed',seed,5),('weight',weight,5),('combined',combined,25)]:
        rows=[]
        for d,g in frame.groupby('dataset',sort=False):
            if len(g)!=denominator: raise IntegrityError('Wrong stability denominator')
            matches=int(g.label_matches.sum())
            row=dict(dataset=d,canonical_rank=int(g.canonical_rank.iloc[0]),canonical_stratum=g.canonical_stratum.iloc[0],
                **{name+'_matches':matches,name+'_differences':denominator-matches,name+'_stability':matches/denominator},
                scenario_count=denominator,minimum_rank=int(g['rank'].min()),maximum_rank=int(g['rank'].max()),
                minimum_score=float(g.difficulty_score.min()),maximum_score=float(g.difficulty_score.max()),
                distinct_labels=int(g.stratum.nunique()),boundary_crossing_count=int(g.boundaries_crossed.sum()),
                easy_intermediate_crossings=int(g.easy_intermediate_crossed.sum()),
                intermediate_hard_crossings=int(g.intermediate_hard_crossed.sum()),
                differing_scenarios=';'.join(g.loc[~g.label_matches,'scenario']))
            if name=='seed':
                counts=g.stratum.value_counts();top=int(counts.max())
                modes=[label for label in LABELS if counts.get(label,0)==top]
                row.update(most_frequent_seed_labels=';'.join(modes),most_frequent_seed_count=top,
                    canonical_agrees_with_majority=matches>denominator/2,
                    canonical_is_modal_label=g.canonical_stratum.iloc[0] in modes)
                for _,r in g.iterrows():row[f'label_seed_{int(r.seed)}']=r.stratum
            if name=='weight':
                row['changed_alphas']=';'.join(f'{x:.2f}' for x in g.loc[~g.label_matches,'alpha'])
                for _,r in g.iterrows():row[f'label_alpha_{r.alpha:.2f}']=r.stratum
            rows.append(row)
        tables[name+'_stability']=pd.DataFrame(rows).sort_values('canonical_rank').reset_index(drop=True)
    updated=primary.copy(deep=True)
    for name in ['seed','weight','combined']:
        updated=updated.merge(tables[name+'_stability'][['dataset',name+'_stability',name+'_matches']],on='dataset',validate='one_to_one')
    tables['canonical_ranking_with_stability']=updated
    return tables


def rank_agreements(primary,grid):
    rows=[]
    for scenario,g in grid.groupby('scenario',sort=False):
        _,metrics=b3.compare(primary,g,'canonical_vs_'+scenario)
        metrics.update(scenario=scenario,family=g.family.iloc[0],seed=g.seed.iloc[0],alpha=float(g.alpha.iloc[0]),
            label_agreement=float(g.label_matches.mean()))
        rows.append(metrics)
        if g.family.iloc[0]=='combined' and g.alpha.iloc[0]==.70:
            rows.append({**metrics,'family':'seed'})
    result=pd.DataFrame(rows)
    numeric=['spearman_rank','kendall_tau_b','mean_absolute_rank_displacement','maximum_absolute_rank_displacement','changed_ranks','changed_strata','label_agreement']
    summary=result.groupby('family')[numeric].agg(['min','mean','max'])
    summary.columns=['_'.join(x) for x in summary.columns]
    return result,summary.reset_index()


def boundary_tables(primary,grid,stability):
    rows=[];exchanges=[]
    for edge,label in [(6,'Easy–Intermediate'),(12,'Intermediate–Hard')]:
        left=primary.loc[primary['rank'].eq(edge)].iloc[0];right=primary.loc[primary['rank'].eq(edge+1)].iloc[0]
        rows.append(dict(boundary=label,lower_rank=edge,upper_rank=edge+1,easier_dataset=left.dataset,
            harder_dataset=right.dataset,easier_score=left.difficulty_score,harder_score=right.difficulty_score,
            score_gap=right.difficulty_score-left.difficulty_score,descriptive_midpoint=(left.difficulty_score+right.difficulty_score)/2))
        for scenario,g in grid.groupby('scenario',sort=False):
            harder=g.loc[(g.canonical_rank<=edge)&(g['rank']>edge),'dataset'].tolist()
            easier=g.loc[(g.canonical_rank>edge)&(g['rank']<=edge),'dataset'].tolist()
            if len(harder)!=len(easier): raise IntegrityError('Boundary memberships do not balance')
            if harder:
                exchanges.append(dict(boundary=label,scenario=scenario,family=g.family.iloc[0],seed=g.seed.iloc[0],alpha=g.alpha.iloc[0],
                    moved_to_harder_side=';'.join(harder),moved_to_easier_side=';'.join(easier),count_each_direction=len(harder)))
    changed=set(grid.loc[~grid.label_matches,'dataset'])
    focus={'sonar','parkinsons','anneal','haberman','car','credit_approval'}|changed
    table=stability['canonical_ranking_with_stability'].loc[lambda x:x.dataset.isin(focus)].copy()
    stats=stability['combined_stability'][['dataset','minimum_rank','maximum_rank','boundary_crossing_count','easy_intermediate_crossings','intermediate_hard_crossings']]
    table=table.merge(stats,on='dataset',validate='one_to_one')
    weight=stability['weight_stability'][['dataset','boundary_crossing_count']].rename(columns={'boundary_crossing_count':'aggregate_weight_boundary_crossing_count'})
    table=table.merge(weight,on='dataset',validate='one_to_one')
    return table,pd.DataFrame(rows),pd.DataFrame(exchanges)


def compute_tables(primary,per_seed):
    grid=scenarios(primary,per_seed)
    tables=stability_tables(primary,grid)
    details,summary=rank_agreements(primary,grid)
    boundary,gaps,exchanges=boundary_tables(primary,grid,tables)
    tables.update(scenario_rankings=grid,rank_agreement_summary=details,rank_agreement_by_family=summary,
        boundary_analysis=boundary,canonical_boundary_gaps=gaps,boundary_exchanges=exchanges,
        label_transition_details=grid.loc[~grid.label_matches].copy())
    return tables


def run_analysis(output=OUTPUT):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/analysis').resolve()): raise ValueError('Output must be under experiments/analysis')
    if output.exists(): raise FileExistsError('Existing output is immutable; choose a new version directory')
    previous=b3.load_outputs()
    # Block 4 is inspected solely for artifact provenance, never scoring inputs.
    block4=ROOT/'experiments/analysis/block4-chance-correction-v1'
    b3.load_outputs(block4)
    canonical,validation,records=b3.load_official()
    variants,seed,models,per_seed=b3.aggregate(canonical)
    assert_reference(variants['C'],previous['five_seed_primary'])
    pd.testing.assert_frame_equal(per_seed,previous['per_seed_rankings'],check_exact=True,check_dtype=False)
    primary=previous['five_seed_primary']
    # Expected labels are a post-load audit only, never calculation inputs.
    expected=dict(easy={'banknote','tic_tac_toe','breast_cancer','iris','semeion','car'},
        intermediate={'credit_approval','ionosphere','balance_scale','cylinder_bands','anneal','sonar'},
        hard={'parkinsons','haberman','glass','wine_red','heart_disease'})
    for label,identities in expected.items():
        if set(primary.loc[primary.stratum.eq(label),'dataset'])!=identities: raise IntegrityError('Unexpected canonical reference labels')
    if list(ALPHAS)!=config()['analyses']['label_stability']['alpha_values']: raise IntegrityError('Weight grid differs from frozen plan')
    tables=compute_tables(primary,previous['per_seed_rankings'])
    output.mkdir(parents=True)
    for name,table in tables.items(): b3.export_csv(table,output/(name+'.csv'))
    (output/'official_record_checksums.json').write_bytes(json_bytes(records))
    provenance=dict(validation=validation,reference='Block 3 five-seed aggregate at alpha=0.70',
        block3_checksums_sha256=digest((b3.OUTPUT/'output_checksums.json').read_bytes()),
        block4_checksums_sha256=digest((block4/'output_checksums.json').read_bytes()),block4_used_for_scoring=False,
        alphas=list(ALPHAS),seeds=config()['cv']['seeds'],canonical_models=list(b3.CANONICAL),
        formula='alpha * normalized_error + (1-alpha) * normalized_variability; at alpha=0.70 use exact frozen 0.30',
        variability='within-seed sample SD ddof=1; five-seed aggregate uses mean of seed SDs, then mean of four models',
        denominators=dict(seed=5,weight=5,combined=25),weight_includes_canonical_alpha=True,
        unique_combined_scenarios=25,unique_aggregate_weight_scenarios=5,scenario_rows=510,
        rank_agreement_rows=35,seed_rows_note='Five alpha=0.70 combined scenarios are also summarized as seed variation; not extra independent scenarios',
        boundary_crossing_definition='relative to canonical rank side, counted per boundary per scenario; not sequential changes between grid columns',
        modal_ties='report all equally most frequent labels; strict majority requires more than 2 of 5 seed labels',
        scientific_scope='descriptive agreement grid; not independent replicates or probabilities of label correctness',
        models_fitted=0,splits_generated=0,canonical_modified=False,chance_correction_used=False,
        source_sha256={name:digest((Path(__file__).parent/name).read_bytes()) for name in ['block5_analysis.py','block5_figures.py','block3_analysis.py','canonical_scoring.py']})
    (output/'provenance.json').write_bytes(json_bytes(provenance))
    from .block5_figures import make_figures
    make_figures(tables,output)
    (output/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    print(tables['canonical_ranking_with_stability'][['dataset','rank','stratum','seed_matches','weight_matches','combined_matches']].to_string(index=False))
    print(tables['rank_agreement_by_family'].to_string(index=False))
    return tables


def load_outputs(output=OUTPUT): return b3.load_outputs(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run_analysis(parser.parse_args().output)
