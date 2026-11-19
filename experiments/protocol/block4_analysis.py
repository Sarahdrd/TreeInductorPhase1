"""Reference-relative sensitivity analysis; saved official evaluations only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from . import block3_analysis as b3
from .canonical_scoring import minmax_normalize_on_pool, assign_predictive_difficulty_groups
from .checkpoints import IntegrityError
from .infrastructure import ROOT, config, digest, json_bytes
from .runner import Runner, plan

OUTPUT = ROOT/'experiments/analysis/block4-chance-correction-v1'
DUMMIES = ('dummy_most_frequent', 'dummy_stratified')
DESCRIPTORS = ('number_of_classes','minimum_class_count','class_imbalance_ratio','number_of_observations','majority_class_proportion')


def descriptors():
    manifest = json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    rows=[]
    for d in manifest['datasets']:
        if d['dataset'] not in config()['primary_datasets']: continue
        counts=np.array(list(d['class_counts'].values()), dtype=int)
        if (counts<=0).any(): raise IntegrityError('Non-positive class count')
        rows.append(dict(dataset=d['dataset'], dataset_name=d['canonical_name'],
            number_of_classes=len(counts), minimum_class_count=int(counts.min()),
            number_of_observations=int(counts.sum()), majority_class_proportion=float(counts.max()/counts.sum()),
            class_imbalance_ratio=float(counts.max()/counts.min())))
    return pd.DataFrame(rows)


def aggregate_dummies(frame):
    b3.validate_grid(frame,DUMMIES)
    seed=frame.groupby(['dataset','model','seed']).macro_f1.agg(
        count='size', mean_f1='mean', sd_f1=lambda x:x.std(ddof=1)).reset_index()
    result=seed.groupby(['dataset','model']).agg(mean_f1=('mean_f1','mean'),
        mean_within_seed_sd=('sd_f1','mean'), seeds=('seed','nunique')).reset_index()
    pooled=frame.groupby(['dataset','model']).macro_f1.agg(
        fold_count='size', pooled_sd=lambda x:x.std(ddof=1), minimum='min', maximum='max').reset_index()
    result=result.merge(pooled,on=['dataset','model'],validate='one_to_one')
    if len(frame)!=2550 or not seed['count'].eq(15).all() or not result.fold_count.eq(75).all() or not result.seeds.eq(5).all():
        raise IntegrityError('Incomplete dummy grid')
    wide=descriptors()
    for model in DUMMIES:
        part=result.loc[result.model.eq(model)].drop(columns='model')
        wide=wide.merge(part.rename(columns={c:model+'_'+c for c in part if c!='dataset'}),on='dataset',validate='one_to_one')
    return wide,seed,result


def corrected_error(f, dummy):
    f,dummy=np.broadcast_arrays(np.asarray(f,dtype=float),np.asarray(dummy,dtype=float))
    if not np.isfinite(f).all() or not np.isfinite(dummy).all() or ((f<0)|(f>1)|(dummy<0)|(dummy>1)).any():
        raise ValueError('Invalid macro-F1 inputs; no clipping permitted')
    denominator=1-dummy
    # Conservative, explicit numerical guard: amplification beyond ~67 million.
    if (denominator<=np.sqrt(np.finfo(float).eps)).any():
        raise ValueError('Non-positive or numerically unstable chance-correction denominator')
    improvement=(f-dummy)/denominator
    error=(1-f)/denominator
    if (improvement<0).any() or ((error<0)|(error>1)).any():
        raise ValueError('Pathological below-dummy improvement / error outside [0,1]; inspect before proceeding, never clip')
    return improvement,error,denominator


def correct_ranking(primary,dummies):
    if primary.dataset.duplicated().any() or set(primary.dataset)!=set(config()['primary_datasets']) or set(dummies.dataset)!=set(primary.dataset) or dummies.dataset.duplicated().any():
        raise IntegrityError('Dataset identities differ')
    d=primary.merge(dummies,on='dataset',validate='one_to_one')
    d['stratified_dummy_macro_f1']=d.dummy_stratified_mean_f1
    d['improvement_over_dummy']=d.mean_macro_f1-d.stratified_dummy_macro_f1
    d['chance_corrected_improvement'],d['chance_corrected_error'],d['correction_denominator']=corrected_error(d.mean_macro_f1,d.stratified_dummy_macro_f1)
    d['normalized_chance_corrected_error']=minmax_normalize_on_pool(d.chance_corrected_error,pd.Series(True,index=d.index))
    np.testing.assert_array_equal(d.normalized_variability,minmax_normalize_on_pool(d.variability,pd.Series(True,index=d.index)))
    d['chance_corrected_difficulty']=.70*d.normalized_chance_corrected_error+.30*d.normalized_variability
    # Reuse original tie logic, retaining original F and V as tie-breakers.
    args=d.rename(columns={'mean_macro_f1':'baseline_macro_f1_mean','variability':'cv_variability'}).assign(
        ranking_eligible=True,predictive_difficulty_score=d.chance_corrected_difficulty)
    d['corrected_stratum'],ranks=assign_predictive_difficulty_groups(args)
    d['corrected_rank']=ranks.astype(int)
    d=d.rename(columns={'mean_macro_f1':'original_mean_macro_f1','error':'original_error',
        'normalized_error':'original_normalized_error','variability':'original_variability',
        'difficulty_score':'original_difficulty','rank':'original_rank','stratum':'original_stratum'})
    d['rank_displacement']=d.corrected_rank-d.original_rank
    d['stratum_changed']=d.corrected_stratum!=d.original_stratum
    for column,prefix in [('original_error','original'),('chance_corrected_error','corrected')]:
        # Raw-error associations retain ties; ordinal ranks use canonical F,V,name tie-breaks.
        order=d.sort_values([column,'original_mean_macro_f1','original_variability','dataset'],ascending=[True,False,True,True],kind='mergesort').index
        ranks=pd.Series(np.arange(1,18),index=order)
        d[prefix+'_raw_error_rank']=ranks
    d['raw_error_rank_displacement']=d.corrected_raw_error_rank-d.original_raw_error_rank
    return d.sort_values('corrected_rank').reset_index(drop=True)


def comparisons(d):
    rows=[]
    for label,left,right in [('composite_difficulty','original_rank','corrected_rank'),('raw_error','original_raw_error_rank','corrected_raw_error_rank')]:
        delta=d[right]-d[left]
        rows.append(dict(comparison=label,n=17,spearman_rank=float(spearmanr(d[left],d[right]).statistic),
            kendall_tau_b=float(kendalltau(d[left],d[right],variant='b').statistic),changed_ranks=int(delta.ne(0).sum()),
            mean_absolute_rank_displacement=float(delta.abs().mean()),maximum_absolute_rank_displacement=int(delta.abs().max()),
            changed_strata=int(d.stratum_changed.sum()) if label=='composite_difficulty' else None,
            changed_strata_percent=100*d.stratum_changed.mean() if label=='composite_difficulty' else None))
    for boundary,name in [(6,'easy_intermediate_crossers'),(12,'intermediate_hard_crossers')]:
        rows[0][name]=';'.join(d.loc[(d.original_rank<=boundary)!=(d.corrected_rank<=boundary),'dataset'])
    corr=[]
    for descriptor in DESCRIPTORS:
        for metric in ['original_error','chance_corrected_error','original_difficulty','chance_corrected_difficulty']:
            pair=d[[descriptor,metric]].dropna()
            corr.append(dict(descriptor=descriptor,metric=metric,n=len(pair),spearman=float(spearmanr(pair[descriptor],pair[metric]).statistic)))
    return pd.DataFrame(rows),pd.DataFrame(corr)


def run_analysis(output=OUTPUT):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/analysis').resolve()): raise ValueError('Output must be under experiments/analysis')
    if output.exists(): raise FileExistsError('Existing analyses are never overwritten; use a new version directory')
    previous=b3.load_outputs()
    canonical,validation,hashes=b3.load_official()
    reconstructed,_,_,_=b3.aggregate(canonical)
    # CSV round-trip infers pandas string dtype for labels; values remain exact.
    pd.testing.assert_frame_equal(reconstructed['C'],previous['five_seed_primary'],check_exact=True,check_dtype=False)
    # All attempts have just passed Block 3 provenance validation. Revalidate each
    # selected dummy against the frozen context while extracting its payload.
    runner=Runner('block2b-five-seed',official=True)
    rows=[]
    for key in plan(models=list(DUMMIES)):
        _,_,meta,expected=runner.context(key)
        attempts=runner.store.attempts(key)
        for record in attempts: runner.validate_attempt(record,expected,meta)
        for path in runner.store.folder(key).glob('attempt_*.json'):
            if digest(path.read_bytes())!=hashes[path.relative_to(runner.store.root).as_posix()]: raise IntegrityError('Official record changed during analysis')
        rows.append(attempts[-1])
    frame=pd.DataFrame(rows)
    dummy,seed,long=aggregate_dummies(frame)
    # Exact saved partition identity is shared by every canonical/dummy model.
    pairing=pd.concat([canonical,frame],ignore_index=True).groupby(['dataset','seed','repeat','fold'])
    for field in ['split_id','dataset_fingerprint','environment_id']:
        if not pairing[field].nunique().eq(1).all(): raise IntegrityError('Canonical/dummy partition or provenance mismatch')
    ranked=correct_ranking(previous['five_seed_primary'],dummy)
    summary,correlations=comparisons(ranked)
    tables=dict(dummy_baselines=dummy,dummy_seed_statistics=seed,dummy_model_statistics=long,
        dummy_fold_scores=frame[canonical.columns],chance_corrected_ranking=ranked,
        original_vs_corrected=ranked.sort_values('original_rank'),rank_comparison_summary=summary,
        descriptor_correlations=correlations,heart_wine_focus=ranked.loc[ranked.dataset.isin(['heart_disease','wine_red'])],
        improvement_by_original_stratum=ranked.groupby('original_stratum').improvement_over_dummy.agg(count='size',mean='mean',minimum='min',maximum='max').reset_index())
    output.mkdir(parents=True)
    for name,table in tables.items(): b3.export_csv(table,output/(name+'.csv'))
    (output/'official_record_checksums.json').write_bytes(json_bytes(hashes))
    provenance=dict(validation={**validation,'dummy_records':len(frame),'exact_seed_and_partition_matching':True},
        canonical_reference='block3-five-seed-v1/five_seed_primary.csv',
        block3_output_checksums_sha256=digest((b3.OUTPUT/'output_checksums.json').read_bytes()),
        source='official/revision2-five-seed-v1/block2b-five-seed',reference_model='dummy_stratified',
        formula='E_cc=(1-F)/(1-F_dummy); D_cc=0.70*minmax(E_cc)+0.30*Block3_V_norm',
        correction_level='aggregate dataset means, equally weighted four canonical models; not mean fold ratios',
        variability='unchanged Block 3 mean within-seed sample SD; ddof=1',weights=[.70,.30],
        seeds=config()['cv']['seeds'],pool=config()['primary_datasets'],canonical_models=list(b3.CANONICAL),
        original_score_replaced=False,models_fitted=0,splits_generated=0,
        numerical_checks=dict(minimum_denominator=float(ranked.correction_denominator.min()),
            instability_threshold=float(np.sqrt(np.finfo(float).eps)),negative_improvements=0,out_of_range_errors=0,clipping=False),
        frozen_config_note='Archival planning fields remain unchanged; the user-approved Block 4 formula is recorded here.',
        source_sha256={name:digest((Path(__file__).parent/name).read_bytes()) for name in ['block4_analysis.py','block4_figures.py','block3_analysis.py','canonical_scoring.py']})
    (output/'provenance.json').write_bytes(json_bytes(provenance))
    from .block4_figures import make_figures
    make_figures(tables,output)
    (output/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    print(summary.to_string(index=False))
    print(ranked[['dataset','original_rank','corrected_rank','original_stratum','corrected_stratum','chance_corrected_difficulty']].to_string(index=False))
    return tables


def load_outputs(output=OUTPUT):
    return b3.load_outputs(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run_analysis(parser.parse_args().output)
