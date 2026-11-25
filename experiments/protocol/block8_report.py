"""Export validated Phase 8A diagnostics, not scientific stratum conclusions."""
import json
from pathlib import Path
import pandas as pd
from . import block8 as b
from .infrastructure import ROOT,digest,json_bytes,write_frozen
from .checkpoints import Store,read_verified,filesystem_path
from .runner import Runner
from .block3_analysis import load_outputs

OUT=ROOT/'experiments/analysis/block8-feasibility-v1'

def export():
    if OUT.exists():raise FileExistsError(OUT)
    pilot=json.loads((b.BASE/'pilot_configuration.json').read_text());assert pilot['provenance']==b.provenance()
    baseline=Runner();labels=load_outputs()['five_seed_primary'].set_index('dataset');records=[];cart=[]
    for directory,target in [('feasibility-v1',records),('matched-cart-pilot-v1',cart)]:
        root=filesystem_path(b.BASE/'results'/directory);manifest=read_verified(root/'run.json');store=Store(root,manifest)
        models=b.ALGORITHMS if directory=='feasibility-v1' else ['cart_depth2_'+a for a in b.ALGORITHMS]
        for dataset in b.PILOT_DATASETS:
            _,_,_,expected=baseline.context((dataset,'decision_tree_default',42,1,1))
            for model in models:
                rr=store.attempts((dataset,model,42,1,1));assert len(rr)==1;r=rr[-1]
                assert r['execution_status']=='success' and r['dataset_fingerprint']==expected['dataset_fingerprint'] and r['split_id']==expected['split_id']
                assert r['configuration_id']==digest(json_bytes(manifest))
                assert 0<=r['validation_macro_f1']<=1 and r['depth']<=2
                target.append(r)
    rows=pd.DataFrame(records);cart=pd.DataFrame(cart)
    compatibility=pd.read_csv(b.BASE/'algorithm_compatibility.csv');compatibility['pilot_completed']=compatibility.dataset.isin(b.PILOT_DATASETS)
    compatibility['runtime_coverage']=compatibility.pilot_completed.map({True:'one matched saved fold successful',False:'representation validated; solver runtime untested'})
    coverage=compatibility[['dataset','algorithm','canonical_stratum','pilot_completed']].copy();coverage['planned_full_folds']=75;coverage['completed_full_folds']=0;coverage['pilot_folds']=coverage.pilot_completed.astype(int)
    pair=[]
    for r in records:
        c=cart[(cart.dataset==r['dataset']) & (cart.representation==r['algorithm'])].iloc[0]
        pair.append(dict(dataset=r['dataset'],algorithm=r['algorithm'],split_id=r['split_id'],solver_validation_macro_f1=r['validation_macro_f1'],matched_cart_validation_macro_f1=c.validation_macro_f1,delta=r['validation_macro_f1']-c.validation_macro_f1,purpose='pilot diagnostic only'))
    OUT.mkdir()
    for name,t in dict(algorithm_compatibility=compatibility,feasibility_report=rows,matched_cart_pilot=cart,pilot_matched_cart_differences=pd.DataFrame(pair),evaluation_coverage=coverage).items():t.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
    env=dict(python_environment=pilot['provenance']['canonical_environment'],implementations=pilot['provenance']['source'],julia_version='1.10.10',compiler='Zig 0.13.0 c++ -std=c++14 -O2 -DNDEBUG',julia_project=json.loads('{}'),sidecar_location='Block 8 only; no system-wide installations',
        julia_project_sha256=digest((b.BASE/'sidecars/julia-project/Project.toml').read_bytes()),julia_manifest_sha256=digest((b.BASE/'sidecars/julia-project/Manifest.toml').read_bytes()),JSON_version='0.21.4',murtree_license='MIT',quant_bnb_license='No explicit license file in pinned repository; do not redistribute vendor source',
        v1_timeout_test='failed parent cleanup; pilot fits completed normally before budget',v2_timeout_test='Windows job-object parent/descendant/Julia timeout tests passed; memory hard cap; use v2 for subsequent runs')
    (OUT/'algorithm_environment_manifest.json').write_bytes(json_bytes(env))
    for name in ['Project.toml','Manifest.toml']:(OUT/('julia_'+name)).write_bytes((b.BASE/'sidecars/julia-project'/name).read_bytes())
    proposed=dict(status='proposal only; needs explicit Phase 8B approval',depth=2,max_decision_nodes=3,numeric_threshold_cap=16,threads=1,
        external_wall_seconds=120,murtree_internal_seconds=120,memory_limit_bytes=4*1024**3,canonical_cv_seeds=[42,7,21,84,123],repetitions=3,folds=5,
        solver_evaluations=2550,matched_cart_configurations=2,matched_cart_evaluations=2550,total_new_fits=5100,solver_wall_ceiling_hours=85,
        optimality='MurTree solver flag; Quant-BnB feasible but uncertified unless independently established without changing optimizer',full_run_enabled=False)
    (OUT/'proposed_protocol.json').write_bytes(json_bytes(proposed))
    provenance=dict(pilot_manifest=pilot,report_source=digest(Path(__file__).read_bytes()),v2_process_source=digest((ROOT/'experiments/protocol/block8_process.py').read_bytes()),
        v2_adapter_source=digest((ROOT/'experiments/protocol/block8_adapters.py').read_bytes()),record_sha256={str(p.relative_to(filesystem_path(b.BASE))):digest(p.read_bytes()) for p in filesystem_path(b.BASE/'results').rglob('attempt_*.json')},main_run=False)
    # Extended paths cannot be relative to ordinary paths on Windows.
    provenance['record_sha256']={str(p.relative_to(filesystem_path(b.BASE))):digest(p.read_bytes()) for p in filesystem_path(b.BASE/'results').rglob('attempt_*.json')}
    (OUT/'provenance.json').write_bytes(json_bytes(provenance));(OUT/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in OUT.iterdir()}))
    print(rows[['dataset','algorithm','status','wall_seconds','validation_macro_f1']].to_string(index=False))

if __name__=='__main__':export()
