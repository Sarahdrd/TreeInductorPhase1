"""Block 7B reporting derivatives. Reads saved analysis; no evaluation actions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .block3_analysis import load_outputs
from .block7_analysis import OUTPUT
from .infrastructure import ROOT,digest,json_bytes

SUPPLEMENT=ROOT/'experiments/analysis/block7-magic-report-v1'
METRICS=['mean_macro_f1','error','variability','difficulty_score']


def tables():
    return load_outputs(OUTPUT)


def derived(t):
    model=t['model_five_seed_statistics'].groupby(['model','sample_size'])[['mean_macro_f1','error','variability','between_cv_seed_sd','pooled_75_fold_sd']].agg(['mean','std','min','max'])
    model.columns=['_'.join(c) for c in model.columns];model=model.reset_index()
    dispersion=t['subsample_difficulty_components'].groupby('sample_size')[['mean_between_cv_seed_sd','mean_pooled_75_fold_sd']].mean().reset_index()
    summary=t['sample_size_summary'];adj=t['adjacent_size_consistency'].copy()
    adj['mean_change_per_1000_observations']=adj.mean_change*1000/(adj.to_size-adj.from_size)
    endpoints=[]
    for metric in METRICS:
        a=summary.iloc[0][metric+'_mean'];b=summary.iloc[-1][metric+'_mean'];endpoints.append(dict(metric=metric,n200=a,n10000=b,absolute_change=b-a,percent_change=100*(b-a)/a))
    comp=t['subsample_difficulty_components'];ref=t['normalization_reference'].iloc[0]
    contribution=[]
    for seed,g in comp.groupby('subsampling_seed'):
        g=g.sort_values('sample_size');a,b=g.iloc[0],g.iloc[-1]
        contribution.append(dict(subsampling_seed=seed,error_contribution=.7*(b.error-a.error)/(ref.E_max_ref-ref.E_min_ref),variability_contribution=.3*(b.variability-a.variability)/(ref.V_max_ref-ref.V_min_ref),difficulty_change=b.difficulty_score-a.difficulty_score))
    return dict(model_size_summary=model,variability_diagnostics=dispersion,adjacent_size_rates=adj,endpoint_changes=pd.DataFrame(endpoints),score_change_decomposition=pd.DataFrame(contribution))


def export():
    t=tables();extra=derived(t)
    if SUPPLEMENT.exists():raise FileExistsError(SUPPLEMENT)
    SUPPLEMENT.mkdir()
    for name,frame in extra.items():frame.to_csv(SUPPLEMENT/(name+'.csv'),index=False,float_format='%.17g')
    (SUPPLEMENT/'provenance.json').write_bytes(json_bytes(dict(source_analysis_checksums=digest((OUTPUT/'output_checksums.json').read_bytes()),source_code=digest(Path(__file__).read_bytes()),note='Descriptive reporting derivatives only; no fitting, new normalization or inference.')))
    (SUPPLEMENT/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in SUPPLEMENT.iterdir()}))
    return extra


if __name__=='__main__':
    x=export()
    print(x['endpoint_changes'].to_string(index=False));print(x['score_change_decomposition'].mean(numeric_only=True))
