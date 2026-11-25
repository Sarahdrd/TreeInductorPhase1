"""Immutable official depth-2 production runner. Only first-stage is authorized by default."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from . import block8 as pilot
from .block8_process import external
from .infrastructure import ROOT,config,digest,json_bytes,write_frozen
from .runner import Runner,parameters
from .checkpoints import Store,writer_lock,filesystem_path,IntegrityError
from .block3_analysis import load_outputs
from sklearn.tree import DecisionTreeClassifier

BASE=pilot.BASE/'official-depth2-v1'
MODELS=['murtree','quant_bnb','cart_depth2_murtree','cart_depth2_quant_bnb']
POLICY=dict(version='block8-official-depth2-v1',depth=2,max_decision_nodes=3,sparsity_penalty=0,numeric_threshold_cap=16,
    external_wall_seconds=120,murtree_internal_seconds=120,job_committed_memory_bytes=4*1024**3,threads=1,serial=True,
    estimator_seed=42,cv_seeds=[42,7,21,84,123],repetitions=3,folds=5,
    quant=dict(depth=2,s=3,mid_method=2,mid_ratio=.2,treetype='C',upper_bound='original greedy misclassifications * (1+1e-6)',certification='feasible but uncertified'),
    outcomes='default resume skips all committed non-interrupted outcomes; explicit --retry-failed retries non-success',
    budgets='same external wall and job-memory envelope also applied to separate CART worker; preprocessing outside fit wall envelope',
    purpose='official',first_stage=dict(seed=42,repeat=1,fold=1),first_stage_expected=68,full_expected=5100)


def plan(first=False):
    c=config();return list(itertools.product(c['primary_datasets'],MODELS,[42] if first else POLICY['cv_seeds'],[1] if first else range(1,4),[1] if first else range(1,6)))


def should_run(records,retry_failed=False):
    return not records or records[-1]['termination_status']=='interrupted' or (retry_failed and records[-1]['execution_status']!='success')


class Production:
    def __init__(self):
        self.baseline=Runner();self.labels=load_outputs()['five_seed_primary'].set_index('dataset');self.cart={**config()['models']['decision_tree_default'],'max_depth':2}
        files=['block8_full.py','block8_worker.py','block8_process.py','block8.py','runner.py','models.py','preprocessing.py','infrastructure.py','checkpoints.py','datasets.py']
        inputs={};self.expected={}
        for key in plan():
            dataset,model,seed,repeat,fold=key
            _,_,_,expected=self.baseline.context((dataset,'decision_tree_default',seed,repeat,fold));self.expected[key]=expected
        for (dataset,seed),(_,_,path,sha,meta_sha) in self.baseline._splits.items():inputs[str(path.relative_to(ROOT))]=sha;inputs[str(path.with_suffix('.json').relative_to(ROOT))]=meta_sha
        self.manifest=dict(policy=POLICY,models=MODELS,datasets=config()['primary_datasets'],canonical_labels=self.labels.stratum.to_dict(),
            canonical_reference_sha256=digest((ROOT/'experiments/analysis/block3-five-seed-v1/five_seed_primary.csv').read_bytes()),
            saved_split_hashes=inputs,canonical_manifest=self.baseline.data_manifest,cart_parameters=self.cart,cart_resolved_parameters=parameters(DecisionTreeClassifier(**self.cart)),
            author_implementations=pilot.provenance(),source_sha256={f:digest((Path(__file__).parent/f).read_bytes()) for f in files},
            representation='exact approved Phase 8A Representation, train-fold-only; matched CART uses identical representation',
            process_controller='Windows job-object v2; kill-on-close, committed-memory cap, serial; no archived pilot controller')
        self.configuration_id=digest(json_bytes(self.manifest));self.store=Store(BASE/'results',self.manifest);self.store.check_manifest()
        frozen=BASE/'configuration.json'
        if frozen.exists() and json.loads(frozen.read_text())!=self.manifest:raise IntegrityError('Immutable official configuration changed')

    def identity(self,key):
        dataset,model,seed,repeat,fold=key;e=self.expected[key]
        return dict(dataset=dataset,model=model,seed=seed,repeat=repeat,fold=fold,split_id=e['split_id'],dataset_fingerprint=e['dataset_fingerprint'],canonical_stratum=self.labels.loc[dataset,'stratum'],
            configuration_id=self.configuration_id,environment_id=self.baseline.env['environment_id'],purpose='official',representation='murtree' if model.endswith('murtree') else 'quant_bnb',
            wall_limit_seconds=120,job_memory_limit_bytes=4*1024**3,threads=1)

    def attempts(self,key):
        rr=self.store.attempts(key);identity=self.identity(key)
        for r in rr:
            if any(r.get(k)!=v for k,v in identity.items()):raise IntegrityError('Result identity mismatch')
            if r['execution_status']=='success':
                if not r['prediction_available'] or r['termination_status'] not in ['certified_optimal','feasible_uncertified','heuristic_complete']:raise IntegrityError('Invalid successful state')
                for metric in ['training_macro_f1','validation_macro_f1']:
                    if not isinstance(r.get(metric),(int,float)) or not np.isfinite(r[metric]) or not 0<=r[metric]<=1:raise IntegrityError('Invalid score')
                if r.get('depth',3)>2 or r.get('nodes',4)>3:raise IntegrityError('Tree constraint violated')
            elif r['prediction_available'] or r.get('training_macro_f1') is not None or r.get('validation_macro_f1') is not None:raise IntegrityError('Failure has fabricated score')
            for name,sha in r.get('artifact_hashes',{}).items():
                if digest((filesystem_path(BASE)/name).read_bytes())!=sha:raise IntegrityError('Fit artifact changed')
        return rr

    def execute(self,key,record,folder):
        dataset,model,seed,repeat,fold=key;obj,(a,b),_,_=self.baseline.context((dataset,'decision_tree_default',seed,repeat,fold))
        started=time.perf_counter();rep=record['representation'];prep=pilot.Representation(rep).fit(obj['X'].iloc[a]);A=prep.transform(obj['X'].iloc[a]);B=prep.transform(obj['X'].iloc[b]);classes=np.unique(obj['y'][a]);y=np.searchsorted(classes,obj['y'][a]);output=folder/'output.json'
        record.update(preprocessing_seconds=time.perf_counter()-started,transformed_features=A.shape[1])
        (folder/'preprocessing.json').write_bytes(json_bytes(dict(columns=prep.columns,spec=prep.spec,classes=classes.tolist())))
        record['representation_id']=digest((folder/'preprocessing.json').read_bytes())
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',JULIA_NUM_THREADS='1',JULIA_DEPOT_PATH=str(pilot.BASE/'sidecars/julia-depot'))
        if model=='murtree':
            if len(classes)>10:record['termination_status']='unsupported_target_encoding';return
            np.savetxt(folder/'train.txt',np.column_stack([y,A]),fmt='%d',delimiter=' ')
            cmd=[str(pilot.BASE/'build/murtree_bridge.exe'),str(folder/'train.txt'),str(output),'2','120','42']
        elif model=='quant_bnb':
            (folder/'input.json').write_bytes(json_bytes(dict(X_train=A.tolist(),X_valid=B.tolist(),y_train=y.tolist(),classes=len(classes))))
            cmd=[str(pilot.BASE/'sidecars/julia-1.10.10/bin/julia.exe'),'--startup-file=no','--threads=1','--project='+str(pilot.BASE/'sidecars/julia-project'),str(pilot.BASE/'adapters/quant_bridge.jl'),str(pilot.BASE/'sources/Quant-BnB'),str(folder/'input.json'),str(output)]
        else:
            np.savez_compressed(folder/'input.npz',A=A,B=B,y=y);(folder/'parameters.json').write_bytes(json_bytes(self.cart))
            cmd=[sys.executable,'-m','experiments.protocol.block8_worker',str(folder/'input.npz'),str(folder/'parameters.json'),str(output)]
        (folder/'command.json').write_bytes(json_bytes(cmd))
        record['termination_status']='execution_failure';runtime=external(cmd,folder,seconds=120,memory_limit=4*1024**3,env=env);record.update(runtime)
        if runtime['external_status']!='completed':record['termination_status']=runtime['external_status'];return
        if runtime['returncode']!=0:
            record['error']=(folder/'stderr.log').read_text(encoding='utf8')[-6000:];return
        record['termination_status']='output_validation_failure';raw=json.loads(output.read_text(encoding='utf8'))
        record['stderr']=(folder/'stderr.log').read_text(encoding='utf8')[-6000:]
        if not raw['feasible']:
            record['termination_status']='internal_timeout_no_tree' if model=='murtree' else 'no_returned_tree';return
        if model=='murtree':tr=pilot.predict_tree(raw['tree'],A);va=pilot.predict_tree(raw['tree'],B)
        else:tr=np.asarray(raw['train_predictions']);va=np.asarray(raw['valid_predictions'])
        if len(tr)!=len(a) or len(va)!=len(b) or not np.isin(tr,np.arange(len(classes))).all() or not np.isin(va,np.arange(len(classes))).all():raise ValueError('Invalid predictions')
        if raw['depth']>2 or raw['nodes']>3:raise ValueError('Depth/node constraint violated')
        if model in ['murtree','quant_bnb']:
            objective=raw.get('misclassifications',raw.get('objective'))
            if objective is not None and abs(float(objective)-np.sum(tr!=y))>1e-6:raise ValueError('Training objective disagrees with returned predictions')
        np.savez_compressed(folder/'predictions.npz',training=classes[tr.astype(int)],validation=classes[va.astype(int)])
        record.update(execution_status='success',prediction_available=True,optimal=bool(raw['optimal']) if model=='murtree' else False,
            termination_status=('certified_optimal' if raw['optimal'] else 'feasible_uncertified') if model=='murtree' else ('feasible_uncertified' if model=='quant_bnb' else 'heuristic_complete'),
            training_macro_f1=float(f1_score(obj['y'][a],classes[tr.astype(int)],average='macro',zero_division=0)),validation_macro_f1=float(f1_score(obj['y'][b],classes[va.astype(int)],average='macro',zero_division=0)),
            depth=int(raw['depth']),nodes=int(raw['nodes']),solver_seconds=raw.get('solver_seconds'),remaining_rows=raw.get('remaining_rows'),certification=raw.get('certification','MurTree original flag'))

    def run(self,first=True,authorize_remaining=False,retry_failed=False):
        if not first and not authorize_remaining:raise PermissionError('Remaining folds require explicit authorization and --authorize-remaining')
        keys=plan(first);new=0
        with writer_lock(self.store.root):
            write_frozen(BASE/'configuration.json',self.manifest);self.store.initialize()
            for key in keys:
                previous=self.attempts(key)
                if not should_run(previous,retry_failed):continue
                record=dict(**self.identity(key),attempt=len(previous)+1,execution_status='failure',prediction_available=False,optimal=False,training_macro_f1=None,validation_macro_f1=None,termination_status='preprocessing_failure')
                folder=BASE/'work'/(digest(json_bytes(list(key)))[:20]+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
                start=time.perf_counter();interrupted=False
                try:self.execute(key,record,folder)
                except KeyboardInterrupt:record.update(termination_status='interrupted',error='Interrupted; subprocess job closed');interrupted=True
                except Exception as exc:record.update(error_type=type(exc).__name__,error=str(exc))
                record['total_seconds']=time.perf_counter()-start;record['artifact_hashes']={str(p.relative_to(BASE)):digest(p.read_bytes()) for p in folder.iterdir() if p.is_file()}
                self.store.save(key,record);self.attempts(key);new+=1
                print(json.dumps(dict(new=new,dataset=key[0],model=key[1],seed=key[2],repeat=key[3],fold=key[4],status=record['termination_status'],wall=record.get('wall_seconds'),predictions=record['prediction_available'])),flush=True)
                if interrupted:raise KeyboardInterrupt
        print(json.dumps(dict(new_outcomes=new,stage='first' if first else 'full',planned=len(keys))),flush=True)

    def validate(self):
        allowed={self.store.folder(k) for k in plan()}
        for p in self.store.root.glob('records/**/attempt_*.json'):
            if p.parent not in allowed:raise IntegrityError('Unexpected result key')
        rows=[]
        for key in plan():
            rr=self.attempts(key)
            if rr:rows.append(rr[-1])
        return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['dry-run','first-stage','resume','validate']);p.add_argument('--authorize-remaining',action='store_true');p.add_argument('--retry-failed',action='store_true');a=p.parse_args();r=Production()
    if a.action in ['dry-run','validate']:
        rows=r.validate();print(json.dumps(dict(full_plan=len(plan()),first_stage=len(plan(True)),committed=len(rows),successful=sum(v['execution_status']=='success' for v in rows)),indent=2))
    else:r.run(first=a.action=='first-stage',authorize_remaining=a.authorize_remaining,retry_failed=a.retry_failed)

if __name__=='__main__':main()
