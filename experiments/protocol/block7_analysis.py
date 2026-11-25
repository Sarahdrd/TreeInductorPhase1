"""Analysis of complete MAGIC runs only; never fits models or generates partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import block7 as b7
from .infrastructure import ROOT,digest,json_bytes

OUTPUT=ROOT/'experiments/analysis/block7-magic-subsampling-v1'


def project(E,V,ref):
    en=(E-ref['E_min_ref'])/(ref['E_max_ref']-ref['E_min_ref'])
    vn=(V-ref['V_min_ref'])/(ref['V_max_ref']-ref['V_min_ref'])
    return en,vn,.70*en+.30*vn


def aggregate(frame,ref):
    keys=['sample_id','model','cv_seed','repeat','fold']
    expected={(f's{s}_n{n}',m,c,r,f) for s in b7.SUBSEEDS for n in b7.SIZES for m in b7.MODELS for c in b7.CVSEEDS for r in range(1,4) for f in range(1,6)}
    if len(frame)!=10500 or frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False,name=None))!=expected:raise ValueError('Scientific analysis requires all 10,500 unique planned successes')
    if not frame.execution_status.eq('success').all() or not np.isfinite(frame.macro_f1).all() or not frame.macro_f1.between(0,1).all():raise ValueError('Invalid/incomplete scores')
    for (sample,seed,rep,fold),g in frame.groupby(['sample_id','cv_seed','repeat','fold']):
        if g.split_id.nunique()!=1:raise ValueError('Models were not evaluated on shared folds')
    for s in b7.SUBSEEDS:
        for n in b7.SIZES:
            g=frame[frame.sample_id.eq(f's{s}_n{n}')]
            if not g.subsampling_seed.eq(s).all() or not g.sample_size.eq(n).all() or not g.replicate.eq(b7.SUBSEEDS.index(s)+1).all():raise ValueError('Sample metadata mismatch')
    group=['sample_id','subsampling_seed','replicate','sample_size','model']
    seeds=frame.groupby(group+['cv_seed'],sort=True).macro_f1.agg(mean_macro_f1='mean',within_seed_sd='std',folds='size').reset_index()
    models=seeds.groupby(group).agg(mean_macro_f1=('mean_macro_f1','mean'),variability=('within_seed_sd','mean'),between_cv_seed_sd=('mean_macro_f1','std')).reset_index()
    pooled=frame.groupby(group).macro_f1.std().rename('pooled_75_fold_sd').reset_index();models=models.merge(pooled,on=group,validate='one_to_one')
    models['error']=1-models.mean_macro_f1
    components=models.groupby(group[:-1]).agg(mean_macro_f1=('mean_macro_f1','mean'),variability=('variability','mean'),mean_pooled_75_fold_sd=('pooled_75_fold_sd','mean'),mean_between_cv_seed_sd=('between_cv_seed_sd','mean')).reset_index()
    components['error']=1-components.mean_macro_f1
    components['error_normalized'],components['variability_normalized'],components['difficulty_score']=project(components.error,components.variability,ref)
    for col in ['error_normalized','variability_normalized']:components[col+'_out_of_reference']=~components[col].between(0,1)
    metrics=['mean_macro_f1','error','variability','difficulty_score']
    summary=components.groupby('sample_size')[metrics].agg(['mean','std','median','min','max','count']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    changes=[];trends=[]
    for seed,g in components.groupby('subsampling_seed'):
        g=g.sort_values('sample_size')
        for a,b in zip(g.to_dict('records'),g.to_dict('records')[1:]):
            changes.append(dict(subsampling_seed=seed,from_size=a['sample_size'],to_size=b['sample_size'],**{'delta_'+m:b[m]-a[m] for m in metrics}))
        for m in metrics:
            delta=np.diff(g[m]);trends.append(dict(subsampling_seed=seed,metric=m,spearman_size=float(spearmanr(g.sample_size,g[m]).statistic) if g[m].nunique()>1 else np.nan,increasing_steps=int((delta>0).sum()),decreasing_steps=int((delta<0).sum()),unchanged_steps=int((delta==0).sum())))
    model_trends=[]
    for (model,seed),g in models.groupby(['model','subsampling_seed']):
        g=g.sort_values('sample_size')
        for metric in ['mean_macro_f1','error','variability']:
            delta=np.diff(g[metric]);model_trends.append(dict(model=model,subsampling_seed=seed,metric=metric,spearman_size=float(spearmanr(g.sample_size,g[metric]).statistic) if g[metric].nunique()>1 else np.nan,increasing_steps=int((delta>0).sum()),decreasing_steps=int((delta<0).sum())))
    adjacent=pd.DataFrame(changes);consistency=[]
    for (a,b),g in adjacent.groupby(['from_size','to_size']):
        for m in metrics:
            values=g['delta_'+m];consistency.append(dict(from_size=a,to_size=b,metric=m,replicates=len(g),mean_change=values.mean(),sd_change=values.std(ddof=1),minimum_change=values.min(),maximum_change=values.max(),increases=int((values>0).sum()),decreases=int((values<0).sum()),unchanged=int((values==0).sum())))
    return dict(fold_level_scores=frame,model_seed_statistics=seeds,model_five_seed_statistics=models,subsample_difficulty_components=components,sample_size_summary=summary,
        adjacent_size_changes=adjacent,adjacent_size_consistency=pd.DataFrame(consistency),replicate_trends=pd.DataFrame(trends),model_specific_trends=pd.DataFrame(model_trends))


def metadata_tables(runner):
    manifest={k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in runner.dataset.items()}
    samples=[];classes=[]
    for name,(meta,ids,X,y,splits) in runner.context.items():
        samples.append({k:v for k,v in meta.items() if k not in ['original_row_ids','class_counts']})
        for label,count in meta['class_counts'].items():classes.append(dict(sample_id=name,subsampling_seed=meta['subsampling_seed'],sample_size=len(ids),label=label,count=count,proportion=count/len(ids)))
    return dict(magic_dataset_manifest=pd.DataFrame([manifest]),subsample_manifest=pd.DataFrame(samples),subsample_class_distributions=pd.DataFrame(classes),normalization_reference=pd.DataFrame([runner.cfg['normalization']]))


def figures(tables,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    labels={'error':'Predictive error (1 − mean macro-F1)','variability':'Mean within-seed CV SD','difficulty_score':'Fixed-reference difficulty score'}
    comp=tables['subsample_difficulty_components']
    def draw(ax,data,metric,label=None):
        for seed,g in data.groupby('subsampling_seed'):
            g=g.sort_values('sample_size');ax.plot(g.sample_size,g[metric],color='0.65',alpha=.6,lw=.8)
        summary=data.groupby('sample_size')[metric].agg(['mean','std']);ax.errorbar(summary.index,summary['mean'],yerr=summary['std'],fmt='o-',capsize=3,label=label,color='#0072B2')
        ax.set(xscale='log',xlabel='Sample size (log scale)',ylabel=labels.get(metric,metric),xticks=b7.SIZES)
        ax.set_xticklabels([str(n) for n in b7.SIZES],rotation=35,ha='right');ax.grid(alpha=.2)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(output/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    for metric in labels:
        fig,ax=plt.subplots(figsize=(8,5));draw(ax,comp,metric);ax.set_title('Five nested subsampling replicates; bars = between-replicate SD');save(fig,'sample_size_'+metric)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,m in zip(axes,labels):draw(ax,comp,m)
    save(fig,'joint_components')
    models=tables['model_five_seed_statistics'];fig,axes=plt.subplots(2,4,figsize=(19,9))
    for j,model in enumerate(b7.MODELS):
        for i,metric in enumerate(['error','variability']):draw(axes[i,j],models[models.model.eq(model)],metric);axes[i,j].set_title(model.replace('_',' ').title())
    save(fig,'model_specific_trends')


def run(output=OUTPUT):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/analysis').resolve()) or output.exists():raise ValueError('Use a new analysis directory')
    runner=b7.MagicRunner();records=runner.validate_inventory()
    if len(records)!=10500:raise ValueError(f'Stage B incomplete: {len(records)}/10500. No scientific outputs generated.')
    tables=aggregate(pd.DataFrame(records),runner.cfg['normalization']);tables.update(metadata_tables(runner));output.mkdir(parents=True)
    for name,table in tables.items():table.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    figures(tables,output)
    provenance=dict(run_manifest=read_run(runner),record_checksums={str(p.relative_to(runner.store.root)):digest(p.read_bytes()) for p in runner.store.root.glob('records/**/attempt_*.json')},
        source_hashes={n:digest((Path(__file__).parent/n).read_bytes()) for n in ['block7_analysis.py','block7.py']},analysis_units='five subsampling sequences, sizes nested; folds not independent replicates',normalization='Block 3 reference, no clipping or stratum assignment')
    (output/'provenance.json').write_bytes(json_bytes(provenance));(output/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    return tables


def read_run(runner):
    from .checkpoints import read_verified
    return read_verified(runner.store.root/'run.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);run(parser.parse_args().output)
