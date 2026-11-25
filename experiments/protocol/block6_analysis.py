"""Saved-fold DT characterization and paired regularization effects; no fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import block3_analysis as b3
from .checkpoints import IntegrityError,read_verified,filesystem_path
from .infrastructure import ROOT,config,digest,json_bytes

OUTPUT=ROOT/'experiments/analysis/block6-tree-regularization-v1'
DT=('decision_tree_default','decision_tree_balanced')
LR=('logistic_regression','logistic_regression_balanced')
PANEL_ORDER=('leaf_2','leaf_5','ccp_001','ccp_01')
ZERO_TOL=1e-12  # Numerical zero for dataset means, not a practical-effect threshold.
KEYS=['dataset','dataset_fingerprint','seed','repeat','fold','split_id','environment_id','protocol_version','purpose']


def panels():
    result={p:tuple(config()['regularization_panels'][p]) for p in PANEL_ORDER}
    for p,models in result.items():
        if len(models)!=2 or {config()['models'][m]['class_weight'] for m in models}!={None,'balanced'}:
            raise IntegrityError('Panel does not contain exactly both class-weight settings')
        for m in models:
            if not m.startswith('decision_tree_'+p+'_'): raise IntegrityError('Unexpected panel configuration')
    return result


def aggregate_models(frame,models):
    selected=frame.loc[frame.model.isin(models)].copy()
    b3.validate_grid(selected,models)
    seed=selected.groupby(['dataset','model','seed']).macro_f1.agg(count='size',mean_f1='mean',sd_f1=lambda x:x.std(ddof=1)).reset_index()
    model=seed.groupby(['dataset','model']).agg(mean_f1=('mean_f1','mean'),mean_within_seed_sd=('sd_f1','mean'),seeds=('seed','nunique')).reset_index()
    if not seed['count'].eq(15).all() or not model.seeds.eq(5).all():raise IntegrityError('Incomplete model aggregation')
    components=model.groupby('dataset').agg(mean_macro_f1=('mean_f1','mean'),variability=('mean_within_seed_sd','mean')).reset_index()
    return components,seed,model


def paired_deltas(frame):
    expected=DT+tuple(m for values in panels().values() for m in values)
    b3.validate_grid(frame.loc[frame.model.isin(expected)],expected)
    result=[]
    for panel,models in panels().items():
        for model in models:
            weight=config()['models'][model]['class_weight']
            baseline='decision_tree_balanced' if weight=='balanced' else 'decision_tree_default'
            if config()['models'][baseline]['class_weight']!=weight: raise IntegrityError('Class-weight mismatch')
            left=frame.loc[frame.model.eq(model)].copy();right=frame.loc[frame.model.eq(baseline)].copy()
            if len(left)!=1275 or len(right)!=1275: raise IntegrityError('Incomplete pairing input')
            keep=KEYS+['model','macro_f1','configuration_id','evaluation_id']
            joined=left[keep].merge(right[keep],on=KEYS,how='outer',suffixes=('_regularized','_unpruned'),validate='one_to_one',indicator=True)
            if len(joined)!=1275 or not joined['_merge'].eq('both').all(): raise IntegrityError('Missing or mismatched split/fingerprint/environment pair')
            joined=joined.drop(columns='_merge')
            joined['panel']=panel;joined['class_weight']='balanced' if weight=='balanced' else 'unbalanced'
            joined['delta_macro_f1']=joined.macro_f1_regularized-joined.macro_f1_unpruned
            result.append(joined)
    result=pd.concat(result,ignore_index=True)
    if len(result)!=10200 or result.duplicated(['dataset','model_regularized','seed','repeat','fold']).any():raise IntegrityError('Wrong paired grid')
    return result


def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean_delta=float(a.mean()),median_delta=float(np.median(a)),
        pooled_sd=float(a.std(ddof=1)),minimum_delta=float(a.min()),maximum_delta=float(a.max()),
        fraction_improved=float((a>0).mean()),fraction_worsened=float((a<0).mean()),fraction_unchanged=float((a==0).mean()))


def effect_tables(paired,primary):
    seed_rows=[];model_rows=[];dataset_rows=[]
    for (dataset,panel),group in paired.groupby(['dataset','panel'],sort=False):
        for seed,g in group.groupby('seed'):
            for weight in ['unbalanced','balanced','both']:
                values=g if weight=='both' else g.loc[g.class_weight.eq(weight)]
                if len(values)!=(30 if weight=='both' else 15):raise IntegrityError('Incomplete seed effect group')
                seed_rows.append(dict(dataset=dataset,panel=panel,seed=int(seed),class_weight=weight,**describe(values.delta_macro_f1)))
        local=[]
        for weight,g in group.groupby('class_weight'):
            if len(g)!=75:raise IntegrityError('Incomplete class-weight effect group')
            means=g.groupby('seed').delta_macro_f1.mean()
            stats=describe(g.delta_macro_f1)
            stats['mean_delta']=float(means.mean())
            row=dict(dataset=dataset,panel=panel,class_weight=weight,**stats,
                mean_within_seed_sd=float(g.groupby('seed').delta_macro_f1.std(ddof=1).mean()),
                seed_mean_sd=float(means.std(ddof=1)),minimum_seed_mean=float(means.min()),maximum_seed_mean=float(means.max()))
            model_rows.append(row);local.append(row)
        stats=describe(group.delta_macro_f1)
        stats['mean_delta']=float(np.mean([r['mean_delta'] for r in local]))
        seed_means=group.groupby('seed').delta_macro_f1.mean()
        row=dict(dataset=dataset,panel=panel,class_weight='both',**stats,
            mean_within_seed_sd=float(group.groupby('seed').delta_macro_f1.std(ddof=1).mean()),
            seed_mean_sd=float(seed_means.std(ddof=1)),minimum_seed_mean=float(seed_means.min()),maximum_seed_mean=float(seed_means.max()),
            mean_delta_unbalanced=next(r['mean_delta'] for r in local if r['class_weight']=='unbalanced'),
            mean_delta_balanced=next(r['mean_delta'] for r in local if r['class_weight']=='balanced'))
        dataset_rows.append(row)
    labels=primary[['dataset','stratum','rank','difficulty_score']].rename(columns={'stratum':'canonical_stratum','rank':'canonical_rank','difficulty_score':'canonical_difficulty'})
    datasets=pd.DataFrame(dataset_rows).merge(labels,on='dataset',validate='many_to_one')
    weights=pd.DataFrame(model_rows).merge(labels,on='dataset',validate='many_to_one')
    seed_table=pd.DataFrame(seed_rows).merge(labels,on='dataset',validate='many_to_one')
    all_effects=pd.concat([datasets,weights],ignore_index=True)
    stratum_rows=[];influence=[];outliers=[]
    for (panel,weight,label),g in all_effects.groupby(['panel','class_weight','canonical_stratum'],sort=False):
        values=g.mean_delta.to_numpy();mean=float(values.mean());n=len(values)
        if n!=({'easy':6,'intermediate':6,'hard':5}[label]):raise IntegrityError('Wrong dataset count in stratum')
        stratum_rows.append(dict(panel=panel,class_weight=weight,canonical_stratum=label,datasets=n,
            mean_dataset_delta=mean,median_dataset_delta=float(np.median(values)),sd_across_datasets=float(values.std(ddof=1)),
            minimum_dataset_delta=float(values.min()),maximum_dataset_delta=float(values.max()),
            positive_datasets=int((values>ZERO_TOL).sum()),negative_datasets=int((values < -ZERO_TOL).sum()),
            approximately_unchanged_datasets=int((np.abs(values)<=ZERO_TOL).sum())))
        for removed in [1,2]:
            cases=[]
            for indices in combinations(range(n),removed):
                keep=np.ones(n,dtype=bool);keep[list(indices)]=False
                cases.append((float(values[keep].mean()),';'.join(g.iloc[list(indices)].dataset)))
            low=min(cases);high=max(cases)
            influence.append(dict(panel=panel,class_weight=weight,canonical_stratum=label,removed_datasets=removed,
                full_mean=mean,minimum_remaining_mean=low[0],maximum_remaining_mean=high[0],
                removed_for_minimum=low[1],removed_for_maximum=high[1],
                sign_can_change=bool(low[0]<-ZERO_TOL and high[0]>ZERO_TOL)))
        q1,q3=np.quantile(values,[.25,.75]);iqr=q3-q1
        for _,r in g.iterrows():
            remainder=float((values.sum()-r.mean_delta)/(n-1))
            outliers.append(dict(dataset=r.dataset,panel=panel,class_weight=weight,canonical_stratum=label,
                mean_delta=r.mean_delta,deviation_from_stratum_median=float(r.mean_delta-np.median(values)),
                outside_1_5_iqr=bool(r.mean_delta<q1-1.5*iqr or r.mean_delta>q3+1.5*iqr),
                stratum_mean_without_dataset=remainder,change_in_stratum_mean=remainder-mean,
                sign_opposes_stratum_mean=bool(r.mean_delta*mean < -ZERO_TOL**2)))
    comparison=datasets[['dataset','panel','canonical_stratum','canonical_rank','mean_delta_unbalanced','mean_delta_balanced']].copy()
    comparison['balanced_minus_unbalanced_delta']=comparison.mean_delta_balanced-comparison.mean_delta_unbalanced
    associations=[]
    for panel,g in datasets.groupby('panel'):
        associations.append(dict(panel=panel,datasets=len(g),spearman_difficulty_vs_delta=float(spearmanr(g.canonical_difficulty,g.mean_delta).statistic)))
    return dict(dataset_regularization_effects=datasets,class_weight_effects=weights,seed_regularization_effects=seed_table,
        stratum_regularization_effects=pd.DataFrame(stratum_rows),class_weight_comparison=comparison,
        regularization_outliers=pd.DataFrame(outliers),stratum_influence=pd.DataFrame(influence),difficulty_delta_associations=pd.DataFrame(associations))


def compute_tables(frame,primary):
    ranking,seed_stats,model_stats={},[],[]
    for panel,models in {'dt_only':DT,**panels()}.items():
        components,seed,model=aggregate_models(frame,models)
        ranking[panel]=b3.rank_components(components,panel)
        seed_stats.append(seed.assign(panel=panel));model_stats.append(model.assign(panel=panel))
    detail,summary=b3.compare(primary,ranking['dt_only'],'mixed_vs_dt_only')
    details=[];summaries=[]
    for reference_name,reference in [('mixed',primary),('dt_only',ranking['dt_only'])]:
        for panel in PANEL_ORDER:
            dd,ss=b3.compare(reference,ranking[panel],reference_name+'_vs_'+panel)
            for edge,name in [(6,'easy_intermediate_crossers'),(12,'intermediate_hard_crossers')]:
                ss[name]=';'.join(dd.loc[(dd.rank_reference<=edge)!=(dd.rank_alternative<=edge),'dataset'])
            details.append(dd.assign(panel=panel,reference=reference_name));summaries.append({**ss,'panel':panel,'reference':reference_name})
    for edge,name in [(6,'easy_intermediate_crossers'),(12,'intermediate_hard_crossers')]:
        summary[name]=';'.join(detail.loc[(detail.rank_reference<=edge)!=(detail.rank_alternative<=edge),'dataset'])
    lr,_,_=aggregate_models(frame,LR)
    performance=ranking['dt_only'][['dataset','mean_macro_f1','variability']].merge(lr,on='dataset',suffixes=('_dt','_lr'),validate='one_to_one')
    performance['dt_minus_lr_macro_f1']=performance.mean_macro_f1_dt-performance.mean_macro_f1_lr
    performance=performance.merge(detail[['dataset','rank_reference','rank_alternative','stratum_reference','stratum_alternative','stratum_changed']],on='dataset',validate='one_to_one')
    adjacent=primary.merge(ranking['dt_only'][['dataset','rank','stratum','difficulty_score']].rename(columns={'rank':'dt_only_rank','stratum':'dt_only_stratum','difficulty_score':'dt_only_difficulty'}),on='dataset',validate='one_to_one')
    paired=paired_deltas(frame)
    tables=dict(dt_only_ranking=ranking['dt_only'],mixed_vs_dt_only=detail,mixed_vs_dt_only_summary=pd.DataFrame([summary]),
        canonical_with_dt_only_assignments=adjacent,dt_lr_performance_diagnostic=performance,
        regularization_panel_rankings=pd.concat([ranking[p].assign(panel=p) for p in PANEL_ORDER],ignore_index=True),
        regularization_rank_agreement=pd.DataFrame(summaries),regularization_rank_transitions=pd.concat(details,ignore_index=True),
        model_seed_statistics=pd.concat(seed_stats,ignore_index=True),model_five_seed_statistics=pd.concat(model_stats,ignore_index=True),
        paired_fold_deltas=paired)
    tables.update(effect_tables(paired,primary))
    return tables


def load_selected(canonical,hashes):
    """Read selected payloads, checking bytes against the just-validated inventory."""
    root=filesystem_path(ROOT/'experiments/results/official/revision2-five-seed-v1/block2b-five-seed')
    models=set(DT)|{m for values in panels().values() for m in values}
    paths=[(name,sha) for name,sha in hashes.items() if Path(name).parts[2] in models]
    def read(item):
        name,expected=item;path=root/name
        if digest(path.read_bytes())!=expected:raise IntegrityError('Official record changed during extraction')
        return read_verified(path)
    with ThreadPoolExecutor(max_workers=8) as workers:raw=list(workers.map(read,paths))
    # Only a single successful attempt per selected fold is expected in this run.
    # If retries ever exist, stop for explicit inspection rather than drop duplicates.
    full=pd.DataFrame(raw)
    b3.validate_grid(full,tuple(models))
    if len(full)!=12750:raise IntegrityError('Wrong selected tree count')
    frame=pd.concat([full,canonical.loc[canonical.model.isin(LR)]],ignore_index=True)
    fields=sorted({k for record in raw for k in record})
    training_fields=[k for k in fields if ('train' in k.lower() and ('f1' in k.lower() or 'score' in k.lower())) or 'gap' in k.lower()]
    return frame,dict(tree_records=len(full),fields_inspected=fields,training_metric_fields=training_fields,
        training_scores_available=bool(training_fields),new_training_evaluations=0)


def run_analysis(output=OUTPUT):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/analysis').resolve()):raise ValueError('Output must be under experiments/analysis')
    if output.exists():raise FileExistsError('Choose a new output directory; never overwrite prior analyses')
    previous=b3.load_outputs()
    for block in ['block4-chance-correction-v1','block5-label-stability-v1']:
        b3.load_outputs(ROOT/'experiments/analysis'/block)
    canonical,validation,hashes=b3.load_official()
    variants,_,_,_=b3.aggregate(canonical)
    pd.testing.assert_frame_equal(variants['C'],previous['five_seed_primary'],check_exact=True,check_dtype=False)
    frame,training=load_selected(canonical,hashes)
    tables=compute_tables(frame,previous['five_seed_primary'])
    output.mkdir(parents=True)
    for name,table in tables.items():b3.export_csv(table,output/(name+'.csv'))
    (output/'official_record_checksums.json').write_bytes(json_bytes(hashes))
    (output/'training_metric_audit.json').write_bytes(json_bytes(training))
    provenance=dict(validation={**validation,'selected_tree_records':12750,'paired_records':10200},
        reference='Block 3 five-seed mixed canonical labels',dt_models=list(DT),regularization_panels=panels(),
        seeds=config()['cv']['seeds'],std_ddof=1,variability='mean of within-seed sample SDs, then equal mean of panel models',
        pairing_keys=KEYS,class_weight_pairing='regularized balanced vs unpruned balanced; regularized default vs unpruned default',
        configuration_identity='both resolved configuration IDs and evaluation IDs retained in every pair',
        delta='regularized macro-F1 minus matching unpruned macro-F1, constructed before aggregation',
        dataset_effect='equal mean of five seed mean deltas and two class-weight configurations',
        stratum_weighting='equal dataset weighting, fixed original mixed canonical labels',
        pooled_sd_note='descriptive spread of 75 per-weight or 150 combined paired scores; never an independent-replicate standard error',
        approximate_zero_tolerance=ZERO_TOL,zero_tolerance_note='numerical zero only; not a substantive equivalence margin; fold signs use exact zero',
        outliers='descriptive 1.5 IQR flags within panel/weight/stratum; all datasets retained; leave-one/two-out means check dominance',
        chance_corrected_labels_used=False,models_fitted=0,splits_generated=0,canonical_modified=False,
        previous_output_checksums={block:digest((ROOT/'experiments/analysis'/block/'output_checksums.json').read_bytes()) for block in ['block3-five-seed-v1','block4-chance-correction-v1','block5-label-stability-v1']},
        source_sha256={name:digest((Path(__file__).parent/name).read_bytes()) for name in ['block6_analysis.py','block6_figures.py','block3_analysis.py','canonical_scoring.py']})
    (output/'provenance.json').write_bytes(json_bytes(provenance))
    from .block6_figures import make_figures
    make_figures(tables,output)
    (output/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    print(tables['mixed_vs_dt_only_summary'].to_string(index=False))
    print(tables['stratum_regularization_effects'].loc[lambda x:x.class_weight.eq('both')].to_string(index=False))
    return tables


def load_outputs(output=OUTPUT):return b3.load_outputs(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    run_analysis(parser.parse_args().output)
