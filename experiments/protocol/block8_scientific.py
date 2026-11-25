"""Analysis-only full-run reporting. Never fits, predicts or generates partitions.

Regeneration requires the local archived inputs. Display uses the small curated
exports and does not require the raw records, datasets or external runtimes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .block3_analysis import load_outputs, RENAME
from .checkpoints import read_verified, filesystem_path, IntegrityError
from .infrastructure import ROOT, digest, json_bytes

BASE = ROOT / 'experiments/block8-optimal-trees-v1/official-depth2-v1'
OUTPUT = ROOT / 'experiments/analysis/block8-scientific-v1'
MODELS = ['murtree', 'quant_bnb', 'cart_depth2_murtree', 'cart_depth2_quant_bnb']
KEY = ['dataset', 'seed', 'repeat', 'fold']
STRATA = ['easy', 'intermediate', 'hard']


def verify_files(pairs):
    def check(item):
        path, expected = item
        if digest(filesystem_path(path).read_bytes()) != expected:
            raise IntegrityError('Checksum mismatch: ' + str(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, pairs))


def load_records():
    configuration = json.loads((BASE / 'configuration.json').read_text())
    configuration_id = digest(json_bytes(configuration))
    if read_verified(BASE / 'results/run.json') != configuration:
        raise IntegrityError('Frozen run/configuration mismatch')
    checks = [(ROOT / 'experiments/protocol' / p, sha)
              for p, sha in configuration['source_sha256'].items()]
    checks += [(ROOT / p, sha) for p, sha in configuration['saved_split_hashes'].items()]
    author = configuration['author_implementations']
    checks += [(ROOT / p, sha) for p, sha in author['artifacts'].items()]
    for algorithm, dirname in [('murtree', 'murtree'), ('quant_bnb', 'Quant-BnB')]:
        source = BASE.parent / 'sources' / dirname
        checks += [(source / p, sha) for p, sha in author['source'][algorithm]['files'].items()]
    verify_files(checks)
    paths = sorted((BASE / 'results/records').rglob('attempt_*.json'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(read_verified, paths))
    frame = pd.DataFrame(rows)
    columns = KEY + ['model']
    expected = set(product(configuration['datasets'], [42, 7, 21, 84, 123],
                           [1, 2, 3], [1, 2, 3, 4, 5], MODELS))
    actual = list(frame[columns].itertuples(index=False, name=None))
    if len(actual) != 5100 or len(set(actual)) != 5100 or set(actual) != expected:
        raise IntegrityError('Incomplete, duplicate or unexpected full-run grid')
    if not frame.configuration_id.eq(configuration_id).all():
        raise IntegrityError('Record configuration mismatch')
    success = frame.execution_status.eq('success')
    if not (success == frame.prediction_available).all():
        raise IntegrityError('Inconsistent prediction availability')
    if not success.all():
        raise IntegrityError('Incomplete predictive coverage: report failures before scientific aggregation')
    if not (frame.depth.le(2) & frame.nodes.le(3)).all():
        raise IntegrityError('Tree constraints violated')
    if not (frame.wall_limit_seconds.eq(120) & frame.job_memory_limit_bytes.eq(4*1024**3)
            & frame.threads.eq(1)).all():
        raise IntegrityError('Budget mismatch')
    for model, status in [('murtree', 'certified_optimal'), ('quant_bnb', 'feasible_uncertified'),
                          ('cart_depth2_murtree', 'heuristic_complete'),
                          ('cart_depth2_quant_bnb', 'heuristic_complete')]:
        if not frame.loc[frame.model.eq(model), 'termination_status'].eq(status).all():
            raise IntegrityError('Unexpected solver status; review before publishing')
    artifacts = {BASE / name: sha for r in rows for name, sha in r['artifact_hashes'].items()}
    print(f'Checking {len(artifacts)} saved fit artifacts; no solver is invoked.', flush=True)
    verify_files(artifacts.items())
    baseline = ROOT / 'experiments/results/official/revision2-five-seed-v1/block2b-five-seed'
    cpaths = [p for dataset in configuration['datasets']
              for p in (baseline / 'records' / dataset / 'decision_tree_default').rglob('attempt_*.json')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        canonical = pd.DataFrame(list(pool.map(read_verified, cpaths)))
    if len(canonical) != 1275 or canonical.duplicated(KEY).any():
        raise IntegrityError('Unrestricted canonical CART coverage mismatch')
    if not (canonical.execution_status.eq('success') & canonical.purpose.eq('official')).all():
        raise IntegrityError('Invalid canonical CART record')
    identities = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths + cpaths}
    audit = dict(configuration_id=configuration_id, records=5100, canonical_cart_records=1275,
                 successful=5100, missing=0, timeouts=0, failures=0,
                 source_and_split_files_verified=len(checks), fit_artifacts_verified=len(artifacts),
                 matched_pairs=2550, source_record_sha256=identities,
                 previous_prediction_audit=json.loads((BASE/'full_execution_validation.json').read_text()),
                 prediction_note='Existing audit recomputed 5100 prediction scores exactly; this analysis verifies the saved prediction artifacts against their record hashes.')
    return frame, canonical, configuration, audit


def paired(frame, baseline, model, reference, representation_matched):
    a = frame.loc[frame.model.eq(model)].set_index(KEY).sort_index()
    b = baseline.set_index(KEY).sort_index()
    if not a.index.equals(b.index) or a.index.has_duplicates or b.index.has_duplicates:
        raise IntegrityError('Unmatched or duplicate folds')
    cols = ['split_id', 'dataset_fingerprint', 'environment_id']
    if representation_matched:
        cols += ['representation_id', 'configuration_id']
    for col in cols:
        if not a[col].equals(b[col]):
            raise IntegrityError('Pairing mismatch: ' + col)
    score = 'validation_macro_f1' if representation_matched else 'macro_f1'
    result = (a.validation_macro_f1 - b[score]).rename('delta').reset_index()
    return result.assign(model=model, reference=reference)


def summarize(frame, canonical, labels):
    """Only dataset means enter stratum summaries; signs use exact float values."""
    pieces = []
    for model in ['murtree', 'quant_bnb']:
        pieces.append(paired(frame, frame[frame.model.eq('cart_depth2_' + model)],
                             model, 'representation_matched_depth2_CART', True))
        pieces.append(paired(frame, canonical, model, 'unrestricted_unified_CART', False))
    fold_deltas = pd.concat(pieces, ignore_index=True)
    delta = fold_deltas.groupby(['dataset', 'model', 'reference']).delta.agg(
        pairs='size', mean_delta='mean').reset_index()
    if not delta.pairs.eq(75).all():
        raise IntegrityError('Expected 75 pairs per dataset/reference')
    # Balanced grid makes the 75-fold mean equal the mean of five 15-fold means.
    seed = frame.groupby(['dataset', 'model', 'seed']).agg(
        mean_macro_f1=('validation_macro_f1','mean'),
        within_cv_sd=('validation_macro_f1',lambda x:x.std(ddof=1)),
        mean_wall_seconds=('wall_seconds','mean'), mean_depth=('depth','mean'),
        mean_decision_nodes=('nodes','mean'), mean_memory_bytes=('peak_job_memory_bytes','mean')).reset_index()
    dataset = seed.groupby(['dataset','model']).mean(numeric_only=True).drop(columns='seed').reset_index()
    dataset['mean_leaves'] = dataset.mean_decision_nodes + 1
    dataset = dataset.merge(labels, on='dataset', validate='many_to_one')
    delta = delta.merge(labels, on='dataset', validate='many_to_one')
    groups, signs = [], []
    for ref in ['F', 'H']:
        grouped = dataset.groupby(['model', 'stratum_' + ref])
        group = grouped.agg(datasets=('dataset','size'), mean_macro_f1=('mean_macro_f1','mean'),
                            mean_wall_seconds=('mean_wall_seconds','mean'),
                            mean_depth=('mean_depth','mean'), mean_leaves=('mean_leaves','mean')).reset_index()
        groups.append(group.rename(columns={'stratum_'+ref:'stratum'}).assign(stratification=ref))
        for (model, baseline, stratum), g in delta.groupby(['model','reference','stratum_'+ref]):
            signs.append(dict(stratification=ref, stratum=stratum, model=model, reference=baseline,
                              datasets=len(g), mean_delta=g.mean_delta.mean(),
                              positive=int((g.mean_delta>0).sum()), zero=int((g.mean_delta==0).sum()),
                              negative=int((g.mean_delta<0).sum())))
    runtime = frame.groupby('model').agg(fits=('dataset','size'),
        wall_min=('wall_seconds','min'), wall_median=('wall_seconds','median'),
        wall_mean=('wall_seconds','mean'), wall_p95=('wall_seconds',lambda x:x.quantile(.95)),
        wall_max=('wall_seconds','max'), peak_job_memory_bytes=('peak_job_memory_bytes','max'),
        depth_min=('depth','min'), depth_max=('depth','max'),
        decision_nodes_min=('nodes','min'), decision_nodes_max=('nodes','max')).reset_index()
    status = frame.groupby(['model','execution_status','termination_status']).size().rename('fits').reset_index()
    return dict(dataset_performance=dataset, dataset_paired_deltas=delta,
                stratum_performance=pd.concat(groups,ignore_index=True),
                stratum_paired_deltas=pd.DataFrame(signs), runtime_summary=runtime,
                solver_status=status, stratum_labels=labels)


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Analysis exports are immutable; use a new directory')
    frame, canonical, configuration, audit = load_records()
    F = load_outputs()['five_seed_primary']
    historical_path = ROOT / 'results/new_characterization_experiments/tables/dataset_experimental_profile.csv'
    H = pd.read_csv(historical_path, float_precision='round_trip').rename(columns=RENAME)
    H = H.loc[H.dataset.isin(F.dataset)]
    labels = F[['dataset','rank','stratum','difficulty_score']].rename(
        columns={'rank':'rank_F','stratum':'stratum_F','difficulty_score':'difficulty_F'}).merge(
        H[['dataset','rank','stratum','difficulty_score']].rename(
        columns={'rank':'rank_H','stratum':'stratum_H','difficulty_score':'difficulty_H'}),
        on='dataset',validate='one_to_one')
    if configuration['canonical_labels'] != F.set_index('dataset').stratum.to_dict():
        raise IntegrityError('Frozen primary labels do not match F')
    tables = summarize(frame, canonical, labels)
    expected = {('F','murtree'):[.007742,.027024,.016812], ('H','murtree'):[.007742,.021977,.022870],
                ('F','quant_bnb'):[.005729,.020069,.015130], ('H','quant_bnb'):[.005729,.018537,.016968]}
    for (ref,model), values in expected.items():
        group = tables['stratum_paired_deltas'].query(
            'stratification == @ref and model == @model and reference == "representation_matched_depth2_CART"')
        np.testing.assert_allclose(group.set_index('stratum').loc[STRATA,'mean_delta'],values,atol=5e-7,rtol=0)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    # No absolute personal paths in the public metadata; the original frozen
    # configuration remains unchanged locally and is bound by its digest.
    public_config = dict(policy=configuration['policy'],
                         source_commits={k:v['commit'] for k,v in configuration['author_implementations']['source'].items()},
                         python=configuration['author_implementations']['canonical_environment']['python'],
                         packages=configuration['author_implementations']['canonical_environment']['packages'],
                         julia='1.10.10', cpp='C++14 via portable Zig 0.13.0',
                         cart_parameters=configuration['cart_parameters'])
    audit['aggregation'] = 'Five 15-fold means per dataset, then equal-weight dataset means within fixed F or H; no independent-fold inference.'
    audit['historical_profile_sha256'] = digest(historical_path.read_bytes())
    audit['primary_profile_sha256'] = digest((ROOT/'experiments/analysis/block3-five-seed-v1/five_seed_primary.csv').read_bytes())
    audit['analysis_source_sha256'] = {n:digest((Path(__file__).parent/n).read_bytes())
        for n in ['block8_scientific.py','block8_scientific_figures.py']}
    (output/'provenance.json').write_bytes(json_bytes(audit))
    (output/'configuration_summary.json').write_bytes(json_bytes(public_config))
    from .block8_scientific_figures import make_figures
    make_figures(tables, output)
    (output/'output_checksums.json').write_bytes(json_bytes(
        {p.name:digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}))
    print(tables['stratum_paired_deltas'].to_string(index=False),flush=True)
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
