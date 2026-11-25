"""Audit legacy training metrics solely for the secondary editorial assessment."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .infrastructure import ROOT,config,digest,json_bytes


def audit():
    root=ROOT/'results/default_tree_strata';source=ROOT/'results/new_characterization_experiments'
    metadata=json.loads((root/'metadata.json').read_text())
    checks={name:digest((source/name).read_bytes())==sha for name,sha in metadata['source_sha256'].items()}
    if not all(checks.values()):raise ValueError('Historical source hash mismatch; do not use training metrics')
    folds=pd.read_csv(root/'tables/fold_metrics.csv',float_precision='round_trip')
    original=pd.read_csv(source/'tables/baseline_fold_scores.csv',float_precision='round_trip')
    original=original.loc[original.model.eq('decision_tree_default') & original.cv_seed.eq(42)]
    keys=['dataset','repeat','fold','data_sha256','split_sha256']
    joined=folds.merge(original,on=keys,validate='one_to_one')
    if len(joined)!=255 or len(folds)!=255 or set(folds.dataset)!=set(config()['primary_datasets']):raise ValueError('Historical fold grid mismatch')
    np.testing.assert_allclose(joined.f1_validation,joined.macro_f1,rtol=0,atol=1e-14)
    np.testing.assert_allclose(folds.generalization_gap,folds.f1_train-folds.f1_validation,rtol=0,atol=1e-14)
    if metadata['tree_parameters']!=config()['models']['decision_tree_default']:raise ValueError('Historical tree parameters differ')
    current_manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text())
    current={d['dataset']:d for d in current_manifest['datasets']}
    data_bytes={d:metadata['dataset_identities'][d]['file_sha256']==current[d]['file_sha256'] for d in folds.dataset.unique()}
    if not all(data_bytes.values()):raise ValueError('Historical dataset CSV bytes differ')
    d=folds.groupby('dataset').agg(training_mean=('f1_train','mean'),validation_mean=('f1_validation','mean'),gap_mean=('generalization_gap','mean')).reset_index()
    result=dict(historical_only=True,not_merged_with_unified_results=True,folds=255,datasets=17,
        source_checksums_verified=checks,dataset_csv_bytes_match=data_bytes,validation_matches_historical_baseline=True,
        gap_arithmetic_verified=True,tree_parameters_match=True,historical_versions=metadata['versions'],
        current_comparability='Same CSV bytes, scoring and tree parameters; historical Python/sklearn environment and filtered-data hashes differ. One seed only. Not a replacement for absent unified five-seed training metrics.',
        perfect_dataset_training_means=int(np.isclose(d.training_mean,1,rtol=0,atol=1e-12).sum()),
        dataset_training_means_at_least_099=int((d.training_mean>=.99).sum()),
        minimum_dataset_training_mean=float(d.training_mean.min()),
        spearman_gap_vs_validation=float(spearmanr(d.gap_mean,d.validation_mean).statistic),
        dataset_summaries=d.to_dict('records'),
        source_files_sha256={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [root/'metadata.json',root/'validation_report.json',root/'tables/fold_metrics.csv',root/'tables/dataset_metrics.csv']},
        editorial_assessment='Shorten the gap subsection: legacy train macro-F1 is effectively saturated for most datasets, so the gap largely restates validation error. Retain provenance-qualified exceptions and complementary structure/leaf-occupancy evidence; do not claim a new five-seed training result.')
    target=ROOT/'experiments/protocol/block6_historical_gap_audit.json'
    if target.exists():raise FileExistsError(target)
    target.write_bytes(json_bytes(result));print(json.dumps({k:v for k,v in result.items() if k not in ['dataset_summaries','source_files_sha256','source_checksums_verified','dataset_csv_bytes_match']},indent=2))


if __name__=='__main__':audit()
