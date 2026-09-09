"""Focused regression checks against the helper definitions in the notebook.

Run with the same Python environment used to execute the notebook.
Only small synthetic CV fits are used; canonical outputs are never changed.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

nb=json.loads(Path(__file__).with_name('sensitivity_ranking_test.ipynb').read_text(encoding='utf-8'))
names={'successful_rows','aggregate_baselines','minmax_normalize_on_pool',
       'assign_predictive_difficulty_groups','class_frequency_profile','compare_ranking'}
scope={'np':np,'pd':pd}
scope['NON_DUMMY_BASELINE_MODELS']=['decision_tree_default','decision_tree_balanced',
    'logistic_regression','logistic_regression_balanced']
scope['DUMMY_REFERENCE_MODELS']=['dummy_most_frequent','dummy_stratified']
for cell in nb['cells']:
    if cell['cell_type']!='code': continue
    for node in ast.parse(''.join(cell['source'])).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook helper','exec'),scope)

normalize=scope['minmax_normalize_on_pool']
result=normalize(pd.Series([2.,2.,999.]),pd.Series([True,True,False]))
assert result.iloc[:2].eq(0).all() and pd.isna(result.iloc[2])
result=normalize(pd.Series([1.,2.,999.]),pd.Series([True,True,False]))
assert result.iloc[:2].tolist()==[0.,1.] and pd.isna(result.iloc[2])

rows=[]
for i,model in enumerate(scope['NON_DUMMY_BASELINE_MODELS']):
    scores=np.linspace(.2+i*.1,.3+i*.1,15)
    rows.append(dict(dataset='fixture',model=model,error='',n_scores=15,scores=scores.tolist(),
        f1_macro_mean=scores.mean(),f1_macro_std=scores.std(ddof=1)))
rows.append(dict(dataset='fixture',model='dummy_stratified',error='',n_scores=15,
    scores=[1.]*15,f1_macro_mean=1.,f1_macro_std=0.))
frame=pd.DataFrame(rows)
audit=pd.DataFrame([dict(dataset='fixture',min_class_count=5,comparable_cv=True,cv_issue='')])
aggregate=scope['aggregate_baselines']
full=aggregate(frame,audit).iloc[0]
assert full.ranking_eligible
assert np.isclose(full.baseline_macro_f1_mean,.4)
assert np.isclose(full.mean_model_cv_f1_std,np.std(rows[0]['scores'],ddof=1))
assert np.isclose(full.baseline_macro_f1_spread,.3)
for kind in ['missing_model','error','short_scores','nonfinite','duplicate_model']:
    broken=frame.copy(deep=True)
    if kind=='missing_model': broken=broken.iloc[1:]
    if kind=='error': broken.loc[0,'error']='fit failure'
    if kind=='short_scores': broken.at[0,'scores']=[.5]*14
    if kind=='nonfinite': broken.at[0,'scores']=[np.nan]*15
    if kind=='duplicate_model': broken=pd.concat([broken,broken.iloc[[0]]],ignore_index=True)
    value=aggregate(broken,audit).iloc[0]
    assert not value.ranking_eligible and pd.isna(value.baseline_macro_f1_mean),kind
rare=audit.copy(); rare['min_class_count']=4; rare['comparable_cv']=False
assert not aggregate(frame,rare).iloc[0].ranking_eligible

fixture=pd.DataFrame(dict(dataset=['b','a','c','d','e','f','g'],ranking_eligible=[True]*6+[False],
    predictive_difficulty_score=[.2,.2,.3,.4,.5,.6,np.nan],
    baseline_macro_f1_mean=[.7,.8,.6,.5,.4,.3,np.nan],cv_variability=[.1]*7))
labels,ranks=scope['assign_predictive_difficulty_groups'](fixture)
assert ranks[1]==1 and ranks[0]==2
assert labels.value_counts().to_dict()=={'easy':2,'intermediate':2,'hard':2,'review_separately':1}
assert ranks.dropna().tolist()==[2.,1.,3.,4.,5.,6.]
print('PASS: eligible-pool normalization, four-model completeness, dummy exclusion, sample SD, exact ties and relative strata')

# Exercise the actual optional cell on small synthetic datasets, without touching
# paper outputs or enabling the expensive branch in the saved notebook.
import tempfile
with tempfile.TemporaryDirectory() as directory:
    smoke={}
    import_cell=next(''.join(c['source']) for c in nb['cells'] if '0. Imports and setup' in ''.join(c['source']))
    exec(import_cell,smoke)
    smoke['output_dir']=Path(directory)
    smoke['display']=lambda *args: None
    smoke['print']=lambda *args: None
    baseline_cell=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('def build_preprocessor'))
    exec(baseline_cell.split('baseline_df, cv_audit_df = evaluate_baselines(loaded)')[0],smoke)
    for name in ['minmax_normalize_on_pool','assign_predictive_difficulty_groups','compare_ranking']:
        for c in nb['cells']:
            if c['cell_type']!='code': continue
            for node in ast.parse(''.join(c['source'])).body:
                if isinstance(node,ast.FunctionDef) and node.name==name:
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook helper','exec'),smoke)
    rng=np.random.default_rng(42)
    smoke['loaded']={f'synthetic_{i}':{'X':pd.DataFrame(rng.normal(size=(30,3))),
        'y':np.tile([0,1],15)} for i in range(3)}
    baseline,audit=smoke['evaluate_baselines'](smoke['loaded'])
    profile=smoke['aggregate_baselines'](baseline,audit)
    profile['predictive_error_score']=normalize(1-profile.baseline_macro_f1_mean,profile.ranking_eligible)
    profile['cv_variability_score']=normalize(profile.cv_variability,profile.ranking_eligible)
    profile['predictive_difficulty_score']=.70*profile.predictive_error_score+.30*profile.cv_variability_score
    profile['predictive_difficulty_label'],profile['predictive_difficulty_rank']=smoke['assign_predictive_difficulty_groups'](profile)
    smoke.update(baseline_df=baseline, cv_audit_df=audit, canonical_snapshot_df=profile.copy(deep=True),
                 dataset_profile_df=profile.copy(deep=True),
                 baseline_characteristics_df=smoke['aggregate_baselines'](baseline,audit),
                 computational_characteristics_df=pd.DataFrame({'dataset':profile.dataset}))
    seed_sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'
        and (''.join(c['source']).startswith('RUN_SEED_ROBUSTNESS')
             or ''.join(c['source']).startswith('# SEED ROBUSTNESS:'))]
    # Helpers, five-seed execution, and canonical-protection assertions.
    for source in seed_sources[:3]:
        exec(source,smoke)
    results=smoke['seed_robustness_summary_df']
    assert results.seed.tolist()==[7,21,42,84,123]
    assert results.loc[results.seed.eq(42),'n_label_changes_vs_42'].eq(0).all()
    assert smoke['canonical_snapshot_df'].equals(profile)
    assert smoke['seed_robustness_assertions_df'].passed.all()

    # A failed or omitted model excludes only that dataset-seed pair. Recompute
    # min/max on complete pairs, and never count a failure as label agreement.
    models=smoke['seed_robustness_model_results_df'].query('seed == 7').copy(deep=True)
    cv_audit=smoke['seed_robustness_cv_audit_df'].query('cv_seed == 7').copy(deep=True)
    failed_dataset=models.iloc[0].dataset
    for mode in ['fit_failure','missing_model','short_scores','all_failed']:
        broken=models.copy(deep=True)
        if mode=='fit_failure': broken.loc[broken.index[0],'error']='injected failure'
        if mode=='missing_model': broken=broken.iloc[1:].copy()
        if mode=='short_scores': broken.at[broken.index[0],'scores']=[.5]*14
        if mode=='all_failed': broken['error']='injected failure'
        scores, failures=smoke['build_seed_dataset_profile'](broken,cv_audit,7)
        scores=smoke['assign_seed_difficulty_labels'](scores)
        assert not scores.set_index('dataset').loc[failed_dataset,'complete_seed_dataset'],mode
        assert failures.dataset.eq(failed_dataset).any(),mode
        valid=scores.complete_seed_dataset
        for raw, normalized in [('predictive_error','predictive_error_score'),('cv_variability','cv_variability_score')]:
            assert np.allclose(scores[normalized],normalize(scores[raw],valid),equal_nan=True)
        summary, comparison=smoke['compare_seed_profiles'](scores,profile)
        assert comparison.loc[~comparison.comparison_eligible,['label_changed','robustness_D_pred','robustness_rank']].isna().all().all()
        assert summary.iloc[0].n_datasets_compared==(0 if mode=='all_failed' else 2)
        assert summary.iloc[0].n_incomplete_datasets==(3 if mode=='all_failed' else 1)
        dataset_summary=smoke['summarize_seed_dataset_stability'](comparison).set_index('dataset')
        assert pd.isna(dataset_summary.loc[failed_dataset,'proportion_same_label_as_canonical'])
    # Disabling the section does not execute fits or recreate stale results.
    disabled={**smoke,'RUN_SEED_ROBUSTNESS':False}
    disabled['evaluate_baselines_for_cv_seed']=lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected fit'))
    for source in seed_sources[1:]:
        exec(source,disabled)
print('PASS: five-seed reruns, canonical reproduction, failure exclusion, within-seed normalization, comparison denominators and disabled branch')
