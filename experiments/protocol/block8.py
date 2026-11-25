"""Block 8 bounded feasibility only; original external optimizers, saved canonical folds."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .infrastructure import ROOT,digest,json_bytes,write_frozen,environment
from .runner import Runner
from .checkpoints import Store,writer_lock,IntegrityError
from .block3_analysis import load_outputs

BASE=ROOT/'experiments/block8-optimal-trees-v1'
PILOT_DATASETS=['breast_cancer','tic_tac_toe','iris','semeion','haberman']
ALGORITHMS=['murtree','quant_bnb']
POLICY=dict(version='block8-feasibility-v1',depth=2,max_decision_nodes=3,numeric_threshold_cap=16,
    murtree_internal_seconds=30,external_wall_seconds=120,memory_limit_bytes=4*1024**3,threads=1,
    seed=42,cv_seed=42,repeat=1,fold=1,quant_parameters=dict(s=3,mid_method=2,mid_ratio=.2,treetype='C'),
    status='pilot only; full protocol requires approval',full_execution_enabled=False)


class Representation:
    def __init__(self,algorithm):
        if algorithm not in ALGORITHMS:raise ValueError(algorithm)
        self.algorithm=algorithm
    def fit(self,X):
        self.columns=list(X.columns);self.spec=[]
        for col in self.columns:
            x=X[col]
            if pd.api.types.is_numeric_dtype(x):
                a=x.to_numpy(dtype=float);good=a[np.isfinite(a)];fill=float(np.median(good)) if len(good) else 0.
                a=np.where(np.isfinite(a),a,fill);u=np.unique(a)
                if len(u)<2:continue
                if self.algorithm=='murtree':
                    cuts=u[:-1]/2+u[1:]/2;idx=np.unique(np.rint(np.linspace(0,len(cuts)-1,min(16,len(cuts)))).astype(int));cuts=cuts[idx].tolist()
                else:cuts=[]
                self.spec.append(dict(column=col,type='numeric',fill=fill,cuts=cuts))
            else:
                z=x.dropna().map(str);counts=z.value_counts();fill=sorted(counts[counts==counts.max()].index)[0] if len(counts) else '__ALL_MISSING__'
                values=x.map(lambda v:fill if pd.isna(v) else str(v));categories=sorted(values.unique())
                if len(categories)<2:continue
                self.spec.append(dict(column=col,type='categorical',fill=fill,categories=categories))
        if not self.spec:raise ValueError('No varying training features; requires an explicitly specified constant-tree case')
        return self
    def transform(self,X):
        if list(X.columns)!=self.columns:raise ValueError('Schema changed')
        arrays=[]
        for s in self.spec:
            x=X[s['column']]
            if s['type']=='numeric':
                a=x.to_numpy(dtype=float);a=np.where(np.isfinite(a),a,s['fill'])
                arrays.extend([(a>v).astype(float) for v in s['cuts']] if self.algorithm=='murtree' else [a])
            else:
                a=x.map(lambda v:s['fill'] if pd.isna(v) else str(v)).to_numpy();arrays.extend([(a==v).astype(float) for v in s['categories']])
        out=np.column_stack(arrays)
        if not np.isfinite(out).all():raise ValueError('Invalid transformed data')
        if self.algorithm=='murtree' and not np.isin(out,[0,1]).all():raise ValueError('Nonbinary MurTree input')
        return out


class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]


def external(cmd,folder,seconds=120,memory_limit=POLICY['memory_limit_bytes'],env=None):
    started=time.perf_counter();peak=0;status='completed'
    with (folder/'stdout.log').open('w',encoding='utf8') as out,(folder/'stderr.log').open('w',encoding='utf8') as err:
        process=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
                if ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(process._handle)),ctypes.byref(counters),counters.cb):peak=max(peak,counters.PeakWorkingSetSize)
                if time.perf_counter()-started>seconds:status='external_timeout';break
                if peak>memory_limit:status='memory_limit';break
                time.sleep(.05)
        finally:
            if process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                process.wait(timeout=10)
    return dict(external_status=status,returncode=process.returncode,wall_seconds=time.perf_counter()-started,peak_working_set_bytes=peak,memory_note='polled process peak working set; not whole-system or hard OS memory cap')


def provenance():
    source={}
    for alg,folder in [('murtree','murtree'),('quant_bnb','Quant-BnB')]:
        root=BASE/'sources'/folder
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        source[alg]=dict(commit=commit,files={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.suffix in ['.cpp','.h','.jl'] and '.git' not in p.parts})
    paths=[Path(__file__),BASE/'adapters/murtree_bridge.cpp',BASE/'adapters/quant_bridge.jl',BASE/'build/murtree_bridge.exe',BASE/'sidecars/julia-1.10.10/bin/julia.exe',BASE/'sidecars/julia-project/Project.toml',BASE/'sidecars/julia-project/Manifest.toml']
    return dict(source=source,artifacts={str(p.relative_to(ROOT)):digest(p.read_bytes()) if p.exists() else None for p in paths},canonical_environment=environment())


def predict_tree(tree,X):
    def one(node,row):
        while 'label' not in node:node=node['right'] if row[node['feature']]==1 else node['left']
        return node['label']
    return np.array([one(tree,row) for row in X],dtype=int)


def execute(algorithm,X,y,Z,folder):
    folder.mkdir(parents=True,exist_ok=False);prep=Representation(algorithm).fit(X);A=prep.transform(X);B=prep.transform(Z)
    classes=np.unique(y);encoded=np.searchsorted(classes,y)
    (folder/'preprocessing.json').write_bytes(json_bytes(dict(spec=prep.spec,columns=prep.columns,classes=classes.tolist())))
    output=folder/'output.json'
    if algorithm=='murtree':
        # File reader's line-width inference requires one-digit class codes. Current pool has <=10 classes.
        if len(classes)>10:raise ValueError('Original MurTree text reader cannot safely parse multi-digit labels')
        np.savetxt(folder/'train.txt',np.column_stack([encoded,A]),fmt='%d',delimiter=' ')
        cmd=[str(BASE/'build/murtree_bridge.exe'),str(folder/'train.txt'),str(output),'2','30','42'];env=None
    else:
        (folder/'input.json').write_bytes(json_bytes(dict(X_train=A.tolist(),X_valid=B.tolist(),y_train=encoded.tolist(),classes=len(classes))))
        cmd=[str(BASE/'sidecars/julia-1.10.10/bin/julia.exe'),'--startup-file=no','--threads=1','--project='+str(BASE/'sidecars/julia-project'),str(BASE/'adapters/quant_bridge.jl'),str(BASE/'sources/Quant-BnB'),str(folder/'input.json'),str(output)]
        env=os.environ.copy();env.update(JULIA_DEPOT_PATH=str(BASE/'sidecars/julia-depot'),OPENBLAS_NUM_THREADS='1')
    runtime=external(cmd,folder,env=env);result=dict(**runtime,transformed_features=A.shape[1],feasible=False,optimal=False)
    if runtime['external_status']!='completed':return result,None,None
    if runtime['returncode']!=0:result.update(status='execution_failure',error=(folder/'stderr.log').read_text(encoding='utf8')[-3000:]);return result,None,None
    raw=json.loads(output.read_text());result.update({k:v for k,v in raw.items() if k not in ['tree','train_predictions','valid_predictions']})
    if not raw['feasible']:result['status']='internal_timeout_no_tree' if algorithm=='murtree' else 'no_returned_tree';return result,None,None
    if algorithm=='murtree':tr=predict_tree(raw['tree'],A);va=predict_tree(raw['tree'],B)
    else:tr=np.asarray(raw['train_predictions']);va=np.asarray(raw['valid_predictions'])
    if len(tr)!=len(X) or len(va)!=len(Z) or not np.isin(tr,np.arange(len(classes))).all() or not np.isin(va,np.arange(len(classes))).all():raise ValueError('Invalid predictions')
    result['status']='certified_optimal' if raw['optimal'] else 'feasible_uncertified'
    return result,classes[tr],classes[va]


def pilot():
    baseline=Runner();labels=load_outputs()['five_seed_primary'].set_index('dataset');prov=provenance()
    if any(v is None for v in prov['artifacts'].values()):raise RuntimeError('Sidecars not built; pilot cannot execute')
    manifest=dict(policy=POLICY,provenance=prov,canonical_labels=labels.stratum.to_dict(),datasets=PILOT_DATASETS)
    write_frozen(BASE/'pilot_configuration.json',manifest);store=Store(BASE/'results/feasibility-v1',manifest)
    with writer_lock(store.root):
        store.initialize()
        for dataset in PILOT_DATASETS:
            obj,(train,valid),meta,expected=baseline.context((dataset,'decision_tree_default',42,1,1))
            for algorithm in ALGORITHMS:
                key=(dataset,algorithm,42,1,1);old=store.attempts(key)
                if old and old[-1]['execution_status']=='success':continue
                record=dict(dataset=dataset,algorithm=algorithm,canonical_stratum=labels.loc[dataset,'stratum'],dataset_fingerprint=expected['dataset_fingerprint'],split_id=expected['split_id'],seed=42,repeat=1,fold=1,attempt=len(old)+1,configuration_id=digest(json_bytes(manifest)))
                folder=BASE/'work'/f'{dataset}_{algorithm}_attempt{len(old)+1}'
                try:
                    result,tr,va=execute(algorithm,obj['X'].iloc[train],obj['y'][train],obj['X'].iloc[valid],folder)
                    record.update(result,execution_status='success' if result['feasible'] else 'failure')
                    if va is not None:record.update(training_macro_f1=float(f1_score(obj['y'][train],tr,average='macro',zero_division=0)),validation_macro_f1=float(f1_score(obj['y'][valid],va,average='macro',zero_division=0)))
                except Exception as exc:record.update(execution_status='failure',status='adapter_failure',error=str(exc))
                store.save(key,record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pilot','dry-run']);a=p.parse_args()
    if a.action=='pilot':pilot()
    else:print(json.dumps(dict(pilot_fits=10,main_run_enabled=False,policy=POLICY),indent=2))
