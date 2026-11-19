"""Analysis-only five-seed ranking. No fit/predict or split generation calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .canonical_scoring import (assign_predictive_difficulty_groups,
                                minmax_normalize_on_pool, score_profile)
from .checkpoints import IntegrityError
from .infrastructure import ROOT, config, digest, json_bytes
from .runner import Runner, plan

OUTPUT = ROOT / 'experiments/analysis/block3-five-seed-v1'
CANONICAL = ('decision_tree_default', 'decision_tree_balanced',
             'logistic_regression', 'logistic_regression_balanced')
NS = dict(minmax_normalize_on_pool=minmax_normalize_on_pool,
          assign_predictive_difficulty_groups=assign_predictive_difficulty_groups)
RENAME = dict(baseline_macro_f1_mean='mean_macro_f1', predictive_error='error',
              cv_variability='variability', predictive_error_score='normalized_error',
              cv_variability_score='normalized_variability', predictive_difficulty_score='difficulty_score',
              predictive_difficulty_rank='rank', predictive_difficulty_label='stratum')
VALUE_COLUMNS = list(RENAME.values())


def validate_grid(frame, models):
    """Exact Cartesian grid, not just group sizes; rejects pilots and duplicates."""
    columns = ['dataset', 'model', 'seed', 'repeat', 'fold']
    expected = set(plan(models=list(models)))
    keys = list(frame[columns].itertuples(index=False, name=None))
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise IntegrityError('Missing/extra/duplicate dataset-model-seed-repeat-fold records')
    scores = frame['macro_f1'].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise IntegrityError('Non-finite or out-of-range macro-F1')
    if not frame.purpose.eq('official').all() or not frame.execution_status.eq('success').all():
        raise IntegrityError('Non-official or failed records in analysis')
    if not frame.protocol_version.eq(config()['protocol_version']).all():
        raise IntegrityError('Protocol mismatch')
    if frame.evaluation_id.duplicated().any():
        raise IntegrityError('Duplicate evaluation identity')


def load_official():
    """Read every attempt using the original checkpoint and runner validators."""
    runner = Runner('block2b-five-seed', official=True)
    runner.store.check_manifest()
    if not (runner.store.root / 'run.json').is_file():
        raise IntegrityError('Official completed run is absent')
    tasks = plan()
    records, files, identity = [], set(), {}
    # Preload read-only dataset/split/config contexts on one thread. Only file
    # reads and independent validation run in workers; never estimator fitting.
    contexts = []
    for key in tasks:
        _, _, split_meta, expected = runner.context(key)
        contexts.append((key, split_meta, expected))
    print('Validated dataset identities, saved split archives and unfitted configurations', flush=True)
    def read_one(context):
        key, split_meta, expected = context
        attempts = runner.store.attempts(key)
        if not attempts:
            raise IntegrityError(f'Missing official fold: {key}')
        for attempt in attempts:
            runner.validate_attempt(attempt, expected, split_meta)
        if attempts[-1]['execution_status'] != 'success':
            raise IntegrityError(f'Unresolved failed official fold: {key}')
        record = attempts[-1]
        row = {k: record[k] for k in ('dataset','model','seed','repeat','fold','macro_f1',
                       'evaluation_id','split_id','dataset_fingerprint','environment_id',
                       'protocol_version','purpose','execution_status','configuration_id')}
        hashes = {}
        for path in runner.store.folder(key).glob('attempt_*.json'):
            relative = path.relative_to(runner.store.root).as_posix()
            hashes[relative] = digest(path.read_bytes())
        return row, hashes
    with ThreadPoolExecutor(max_workers=8) as workers:
        for number, (row, hashes) in enumerate(workers.map(read_one, contexts), 1):
            records.append(row)
            files.update(hashes)
            identity.update(hashes)
            if number % 500 == 0:
                print(f'Validated {number}/{len(tasks)} official successful evaluations', flush=True)
    actual = {p.relative_to(runner.store.root).as_posix() for p in (runner.store.root / 'records').rglob('*.json')}
    if actual != files:
        raise IntegrityError('Unexpected/unselected JSON records in official store')
    if list(runner.store.root.rglob('*.tmp')):
        raise IntegrityError('Uncommitted temporary records require inspection before analysis')
    frame = pd.DataFrame(records)
    validate_grid(frame, tuple(config()['models']))
    canonical = frame.loc[frame.model.isin(CANONICAL)].copy()
    validate_grid(canonical, CANONICAL)
    if len(frame) != 20400 or len(canonical) != 5100:
        raise IntegrityError('Wrong official/canonical counts')
    validation = dict(successful_records=len(frame), canonical_records=len(canonical),
                      datasets=17, models=16, scores_per_dataset_model=75,
                      scores_per_dataset_model_seed=15, unique_evaluation_ids=frame.evaluation_id.nunique(),
                      selected_canonical_models=list(CANONICAL), failed_or_pilot_included=0,
                      run_manifest_sha256=digest((runner.store.root / 'run.json').read_bytes()),
                      record_inventory_sha256=digest(json_bytes(identity)), record_files=len(files),
                      environment_id=runner.env['environment_id'])
    return canonical, validation, identity


def rank_components(components, variant):
    if components.dataset.duplicated().any() or set(components.dataset) != set(config()['primary_datasets']):
        raise ValueError('Canonical scoring requires exactly the frozen 17-dataset pool')
    if not np.isfinite(components[['mean_macro_f1','variability']].to_numpy()).all():
        raise ValueError('Non-finite components')
    source = components.rename(columns={'mean_macro_f1':'baseline_macro_f1_mean',
                                        'variability':'cv_variability'}).assign(ranking_eligible=True)
    scored = score_profile(source, NS, alpha=.70).rename(columns=RENAME)
    scored['rank'] = scored['rank'].astype(int)
    scored['variant'] = variant
    if scored.stratum.value_counts().reindex(['easy','intermediate','hard']).tolist() != [6,6,5]:
        raise ValueError('Stratum sizes changed')
    return scored[['dataset','variant'] + VALUE_COLUMNS].sort_values('rank').reset_index(drop=True)


def aggregate(canonical):
    validate_grid(canonical, CANONICAL)
    seed = canonical.groupby(['dataset','model','seed'], sort=True).macro_f1.agg(
        count='size', mean_f1='mean', sd_f1=lambda s: s.std(ddof=1)).reset_index()
    if not seed['count'].eq(15).all():
        raise ValueError('Not 15 scores per seed')
    models = seed.groupby(['dataset','model'], sort=True).agg(
        mean_f1_75=('mean_f1','mean'), mean_within_seed_sd=('sd_f1','mean'), seeds=('seed','nunique')).reset_index()
    pooled = canonical.groupby(['dataset','model'], sort=True).macro_f1.agg(
        fold_count='size', direct_mean_75='mean', pooled_sd=lambda s: s.std(ddof=1)).reset_index()
    models = models.merge(pooled, on=['dataset','model'], validate='one_to_one')
    if not models.seeds.eq(5).all() or not models.fold_count.eq(75).all():
        raise ValueError('Incomplete five-seed group')
    np.testing.assert_allclose(models.mean_f1_75, models.direct_mean_75, rtol=0, atol=5e-16)
    variants = {}
    per_seed = []
    for value in config()['cv']['seeds']:
        components = seed.loc[seed.seed.eq(value)].groupby('dataset').agg(
            mean_macro_f1=('mean_f1','mean'), variability=('sd_f1','mean')).reset_index()
        ranking = rank_components(components, 'unified_seed_' + str(value))
        per_seed.append(ranking.assign(seed=value))
        if value == 42:
            variants['B'] = ranking
    for key, column, variant in [('C','mean_within_seed_sd','five_seed_mean_within_seed_sd_primary'),
                                  ('D','pooled_sd','five_seed_pooled_sd_sensitivity')]:
        components = models.groupby('dataset').agg(mean_macro_f1=('mean_f1_75','mean'),
                                                   variability=(column,'mean')).reset_index()
        variants[key] = rank_components(components, variant)
    return variants, seed, models, pd.concat(per_seed, ignore_index=True)


def historical():
    # The cleaned notebook explicitly calls this older profile its historical reference.
    path = ROOT / 'results/dataset_statistics/dataset_experimental_profile.csv'
    metadata = ROOT / 'results/dataset_statistics/reproducibility.json'
    old = pd.read_csv(path, float_precision='round_trip')
    old = old.loc[old.dataset.isin(config()['primary_datasets'])].copy()
    if len(old) != 17 or old.dataset.duplicated().any() or not old.ranking_eligible.all():
        raise IntegrityError('Historical profile does not contain the eligible fixed pool')
    result = old.rename(columns=RENAME)
    result['rank'] = result['rank'].astype(int)
    result['variant'] = 'historical_manuscript_seed_42'
    result = result[['dataset','variant'] + VALUE_COLUMNS].sort_values('rank').reset_index(drop=True)
    # Audit its internal definition without replacing its stored values.
    reconstructed = rank_components(result[['dataset','mean_macro_f1','variability']], 'historical_check')
    a, b = result.set_index('dataset').sort_index(), reconstructed.set_index('dataset').sort_index()
    for column in VALUE_COLUMNS[:-1]:
        np.testing.assert_allclose(a[column], b[column], rtol=1e-12, atol=1e-14)
    if a.stratum.tolist() != b.stratum.tolist():
        raise IntegrityError('Stored historical labels do not obey the frozen scoring definition')
    info = dict(path=str(path.relative_to(ROOT)), sha256=digest(path.read_bytes()),
                metadata_path=str(metadata.relative_to(ROOT)), metadata_sha256=digest(metadata.read_bytes()),
                metadata=json.loads(metadata.read_text()), historical_internal_reconstruction=True,
                alternative_reference='results/new_characterization_experiments/tables/dataset_experimental_profile.csv',
                reference_note='The later cleaned-notebook reproduction is distinct; historical means the manuscript-era stored profile, not a claim of independently verified publication bytes.')
    return result, info


def compare(reference, alternative, label):
    joined = reference.merge(alternative, on='dataset', suffixes=('_reference','_alternative'), validate='one_to_one')
    for column in VALUE_COLUMNS[:-1]:
        joined[column + '_difference'] = joined[column + '_alternative'] - joined[column + '_reference']
    joined['stratum_changed'] = joined.stratum_reference != joined.stratum_alternative
    changed = joined.loc[joined.stratum_changed, 'dataset'].tolist()
    summary = dict(comparison=label, spearman_rank=float(spearmanr(joined.rank_reference, joined.rank_alternative).statistic),
                   kendall_tau_b=float(kendalltau(joined.rank_reference, joined.rank_alternative, variant='b').statistic),
                   changed_ranks=int(joined.rank_difference.ne(0).sum()),
                   changed_strata=len(changed), changed_strata_percent=100*len(changed)/17,
                   mean_absolute_rank_displacement=float(joined.rank_difference.abs().mean()),
                   maximum_absolute_rank_displacement=int(joined.rank_difference.abs().max()),
                   boundary_crossers=';'.join(changed))
    joined['comparison'] = label
    return joined, summary


def comparisons(variants):
    details, summaries = [], []
    for left, right in [('A','B'),('B','C'),('C','D'),('A','C'),('B','D'),('A','D')]:
        detail, summary = compare(variants[left], variants[right], left+'_vs_'+right)
        details.append(detail); summaries.append(summary)
    wide = variants['A'][['dataset','rank','stratum','difficulty_score']].rename(
        columns={c:'historical_'+c for c in ['rank','stratum','difficulty_score']})
    for key, prefix in [('B','new_seed42'),('C','five_seed'),('D','pooled_sensitivity')]:
        wide = wide.merge(variants[key][['dataset','rank','stratum','difficulty_score']].rename(
            columns={c:prefix+'_'+c for c in ['rank','stratum','difficulty_score']}), on='dataset', validate='one_to_one')
    wide['score_difference'] = wide.five_seed_difficulty_score - wide.new_seed42_difficulty_score
    wide['rank_displacement'] = wide.five_seed_rank - wide.new_seed42_rank
    wide['stratum_changed'] = wide.five_seed_stratum != wide.new_seed42_stratum
    return pd.concat(details, ignore_index=True), pd.DataFrame(summaries), wide


def boundary_table(variants, detail, wide):
    pool = {'sonar','parkinsons','haberman'} | set(detail.loc[detail.stratum_changed,'dataset'])
    primary = variants['C'].sort_values('rank').reset_index(drop=True)
    edge = (primary.loc[primary['rank'].eq(12),'difficulty_score'].iloc[0]
            + primary.loc[primary['rank'].eq(13),'difficulty_score'].iloc[0]) / 2
    rows = []
    for index, row in primary.iterrows():
        if row.dataset not in pool:
            continue
        rows.append(dict(dataset=row.dataset, adjacent_easier_dataset=primary.iloc[index-1].dataset if index else None,
                         adjacent_harder_dataset=primary.iloc[index+1].dataset if index+1 < len(primary) else None,
                         rank_distance_to_intermediate_hard_midpoint=float(row['rank']-12.5),
                         score_distance_to_intermediate_hard_midpoint=float(row.difficulty_score-edge),
                         ranks_to_cross_intermediate_hard=int(13-row['rank'] if row['rank']<=12 else row['rank']-12),
                         boundary_score_midpoint=float(edge)))
    return wide.merge(pd.DataFrame(rows), on='dataset', how='inner').sort_values('five_seed_rank')


def export_csv(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


def run_analysis(output=OUTPUT):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/analysis').resolve()):
        raise ValueError('Block 3 writes only to experiments/analysis/')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing analysis is never overwritten')
    canonical, validation, record_checksums = load_official()
    # No ranking calculation occurs until ALL official records pass validation.
    variants, seed, model, per_seed = aggregate(canonical)
    variants['A'], history_info = historical()
    details, summaries, wide = comparisons(variants)
    boundary = boundary_table(variants, details, wide)
    output.mkdir(parents=True)
    tables = dict(canonical_fold_scores=canonical, seed_model_statistics=seed,
                  five_seed_model_statistics=model, per_seed_rankings=per_seed,
                  comparison_details=details, comparison_summary=summaries,
                  ranking_comparison=wide, boundary_focus=boundary)
    tables.update({name:variants[key] for key,name in [('A','historical_seed42'),('B','unified_seed42'),
                    ('C','five_seed_primary'),('D','five_seed_pooled_sd_sensitivity')]})
    for name, frame in tables.items():
        export_csv(frame, output / (name+'.csv'))
    (output/'official_record_checksums.json').write_bytes(json_bytes(record_checksums))
    provenance = dict(validation=validation, historical=history_info,
                      primary_variability='mean of five within-seed sample SDs per model, then mean of four models',
                      sensitivity_variability='pooled sample SD of 75 scores per model, then mean of four models',
                      std_ddof=1, weights=[.70,.30], canonical_models=list(CANONICAL),
                      boundary_definition='rank midpoint 12.5; descriptive score midpoint of ranks 12 and 13, not an additional classification threshold',
                      repeated_cv_note='Folds are dependent; no inferential p-values or independent-replicate tests',
                      analysis_source_sha256={p:digest((Path(__file__).parent/p).read_bytes())
                                             for p in ['block3_analysis.py','canonical_scoring.py']})
    (output/'provenance.json').write_bytes(json_bytes(provenance))
    from .block3_figures import make_figures
    make_figures(tables, output)
    (output/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    print(summaries.to_string(index=False))
    print(tables['five_seed_primary'].to_string(index=False))
    print(boundary.to_string(index=False))
    return tables, provenance


def load_outputs(output=OUTPUT):
    """Notebook display only. Checks published artifact integrity; never fits."""
    output = Path(output)
    checksums = json.loads((output/'output_checksums.json').read_text())
    for name, expected in checksums.items():
        if digest((output/name).read_bytes()) != expected:
            raise IntegrityError('Analysis artifact changed: '+name)
    return {p.stem:pd.read_csv(p, float_precision='round_trip') for p in output.glob('*.csv')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run_analysis(parser.parse_args().output)
