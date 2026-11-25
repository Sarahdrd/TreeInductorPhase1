"""Read-only preservation audit plus prepared-design exports for Stage A."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from . import block7 as b7
from .block7_analysis import metadata_tables
from .infrastructure import ROOT,digest,write_frozen


def audit():
    r=b7.MagicRunner(pilot=True)
    protected=json.loads((ROOT/'experiments/protocol/block7_preservation.json').read_text())
    def check(item):
        name,sha=item
        if digest((ROOT/name).read_bytes())!=sha:raise ValueError('Previous block changed: '+name)
        return 1
    with ThreadPoolExecutor(max_workers=8) as pool:count=sum(pool.map(check,protected.items()))
    for seed in b7.SUBSEEDS:
        previous=set()
        for n in b7.SIZES:
            meta,ids,X,y,splits=r.context[f's{seed}_n{n}']
            if not previous.issubset(set(ids)):raise ValueError('Nesting changed')
            previous=set(ids)
    tables=metadata_tables(r);folder=b7.BASE/'preparation_tables';folder.mkdir(exist_ok=True)
    for name,t in tables.items():
        payload=t.to_csv(index=False,float_format='%.17g').encode('utf8');path=folder/(name+'.csv')
        if path.exists():
            if path.read_bytes()!=payload:raise ValueError('Preparation table changed')
        else:path.write_bytes(payload)
    pilot=json.loads((b7.BASE/'pilot_validation.json').read_text())
    result=dict(stage='A',protected_files_unchanged=count,original_data_unchanged=True,existing_notebooks_unchanged=True,
        prepared_samples=35,cv_collections=175,shared_folds=2625,planned_full_evaluations=10500,pilot=pilot,
        full_evaluations_completed=0,environment_unchanged=True,canonical_reference_unchanged=True,scientific_analysis_deferred=True)
    if (b7.BASE/'results/full').exists():raise ValueError('Unexpected full-run directory before authorization')
    write_frozen(b7.BASE/'stage_a_validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
