"""Separate MAGIC preparation and saved-fold runner. Full execution is opt-in."""
import argparse
import csv
import io
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .infrastructure import ROOT, config, digest, json_bytes, environment, write_frozen, generate_splits, validate_splits
from .checkpoints import Store, writer_lock, read_verified, IntegrityError
from .models import make_model
from .runner import parameters
from . import block3_analysis as b3

BASE=ROOT/'experiments/block7-magic-subsampling-v1'
SOURCE=ROOT/'data/magic04.data'
SHA='e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a'
FEATURES=['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist']
SIZES=[200,500,1000,1500,2000,5000,10000]
SUBSEEDS=[101,202,303,404,505]
CVSEEDS=[42,7,21,84,123]
MODELS=['decision_tree_default','decision_tree_balanced','logistic_regression','logistic_regression_balanced']
VERSION='magic-nested-v1'


def inspect_data(path=SOURCE):
    raw=Path(path).read_bytes()
    if digest(raw)!=SHA: raise IntegrityError('MAGIC bytes differ from inspected source; stop before fitting')
    rows=list(csv.reader(io.StringIO(raw.decode('ascii'))))
    if len(rows)!=19020 or any(len(r)!=11 for r in rows): raise IntegrityError('Malformed MAGIC row/dimensions')
    frame=pd.read_csv(io.BytesIO(raw),header=None,names=FEATURES+['class'],float_precision='round_trip')
    X=frame[FEATURES];y=frame['class']
    if not np.isfinite(X.to_numpy(dtype=float)).all() or frame.isna().any().any(): raise IntegrityError('Missing/nonfinite MAGIC values')
    if y.value_counts().to_dict()!={'g':12332,'h':6688}: raise IntegrityError('Unexpected original target values')
    fingerprint=digest(pd.util.hash_pandas_object(frame,index=True).values.tobytes())
    manifest=dict(dataset='MAGIC Gamma Telescope',source_file=str(Path(path).relative_to(ROOT)),file_sha256=SHA,
        dataset_fingerprint=fingerprint,observations=len(frame),features=FEATURES,target='class',class_counts=y.value_counts().to_dict(),
        missing_values=0,malformed_rows=0,duplicate_full_rows=int(frame.duplicated().sum()),duplicate_feature_rows=int(X.duplicated().sum()),
        source='https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope',doi='10.24432/C52C8B',
        source_verification='Local structure/counts match UCI documentation; no claim of remote byte identity.',
        row_order='Unchanged local physical line order; zero-based row_id = line number minus 1; duplicates retained',
        target_runs=[dict(start=int(i),label=str(y.iloc[i])) for i in range(len(y)) if i==0 or y.iloc[i]!=y.iloc[i-1]],
        parsing='ASCII comma-separated, no header; fixed UCI feature order; pandas round_trip; original g/h targets')
    return X,y,manifest


def reference():
    p=b3.load_outputs()['five_seed_primary']
    return dict(E_min_ref=float(p.error.min()),E_max_ref=float(p.error.max()),V_min_ref=float(p.variability.min()),V_max_ref=float(p.variability.max()),
        reference_sha256=digest((b3.OUTPUT/'five_seed_primary.csv').read_bytes()),weights=[.7,.3],clip=False)


def nested_indices(y,seed):
    # Largest remainder allocation; ties resolved by sorted original label.
    labels=sorted(y.unique());rng=np.random.default_rng(seed)
    permutations={label:rng.permutation(np.flatnonzero(y.to_numpy()==label)) for label in labels}
    counts=np.array([len(permutations[label]) for label in labels]);previous=set();result={}
    for n in SIZES:
        numerators=counts*n;allocation=numerators//len(y)
        order=sorted(range(len(labels)),key=lambda i:(-int(numerators[i]%len(y)),labels[i]))
        for i in order[:int(n-allocation.sum())]:allocation[i]+=1
        ids=np.sort(np.concatenate([permutations[label][:allocation[i]] for i,label in enumerate(labels)])).astype('<i8')
        if not previous.issubset(set(ids)) or len(ids)!=n:raise IntegrityError('Non-nested allocation')
        result[n]=ids;previous=set(ids)
    return result


def source_hashes():
    return {name:digest((Path(__file__).parent/name).read_bytes()) for name in ['block7.py','models.py','preprocessing.py','infrastructure.py','checkpoints.py','runner.py']}


def frozen_configuration():
    env=environment()
    if env!=json.loads((ROOT/config()['runtime_environment']).read_text()):raise IntegrityError('Unified environment changed')
    return dict(experiment_id=VERSION,sizes=SIZES,subsample_seeds=SUBSEEDS,cv_seeds=CVSEEDS,models=MODELS,
        folds=5,repetitions=3,expected_evaluations=10500,optional_full_reference=False,
        sampling='PCG64 default_rng; one permutation per sorted class; largest-remainder integer allocation; label-order ties; sorted original row IDs',
        environment=env,normalization=reference(),benchmark_config_sha256=digest(json_bytes(config())),source_sha256=source_hashes())


def prepare():
    X,y,manifest=inspect_data();cfg=frozen_configuration()
    BASE.mkdir(parents=True,exist_ok=True)
    with writer_lock(BASE):
        write_frozen(BASE/'configuration.json',cfg);write_frozen(BASE/'dataset_manifest.json',manifest)
        samples=[]
        for rep,seed in enumerate(SUBSEEDS,1):
            for n,ids in nested_indices(y,seed).items():
                name=f's{seed}_n{n}';sample_y=y.iloc[ids].reset_index(drop=True)
                meta=dict(experiment_id=VERSION,sample_id=name,subsampling_seed=seed,replicate=rep,requested_size=n,actual_size=len(ids),
                    original_dataset_fingerprint=manifest['dataset_fingerprint'],original_file_sha256=SHA,original_row_ids=ids.tolist(),
                    class_counts=sample_y.value_counts().to_dict(),sample_checksum=digest(ids.tobytes()+manifest['dataset_fingerprint'].encode()),algorithm=cfg['sampling'])
                write_frozen(BASE/'samples'/f'{name}.json',meta);samples.append(name)
                for cvseed in CVSEEDS:
                    path=BASE/'splits'/f'{name}_cv{cvseed}.npz';metadata=path.with_suffix('.json')
                    if path.exists() or metadata.exists():
                        load_partitions(meta,cvseed,sample_y);continue
                    splits=generate_splits(sample_y,cvseed);path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as stream:np.savez_compressed(stream,**{f'{part}_{i}':a for i,pair in enumerate(splits) for part,a in zip(['train','valid'],pair)})
                    write_frozen(metadata,dict(sample_checksum=meta['sample_checksum'],cv_seed=cvseed,file_sha256=digest(path.read_bytes())))
        inventory={str(p.relative_to(BASE)):digest(p.read_bytes()) for sub in ['samples','splits'] for p in (BASE/sub).glob('*')}
        write_frozen(BASE/'prepared.json',dict(samples=samples,files=inventory,configuration_sha256=digest(json_bytes(cfg)),dataset_manifest_sha256=digest(json_bytes(manifest))))
    print(json.dumps(dict(samples=len(samples),split_collections=len(samples)*5,shared_folds=len(samples)*75,evaluations=10500),indent=2))


def load_sample(name,y,manifest):
    meta=json.loads((BASE/'samples'/f'{name}.json').read_text());ids=np.asarray(meta['original_row_ids'],dtype='<i8')
    if meta['sample_id']!=name or len(ids)!=meta['requested_size'] or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(y) or not np.array_equal(ids,np.sort(ids)):raise IntegrityError('Invalid sample indices')
    if meta['original_dataset_fingerprint']!=manifest['dataset_fingerprint'] or meta['original_file_sha256']!=SHA:raise IntegrityError('Sample dataset identity')
    if digest(ids.tobytes()+manifest['dataset_fingerprint'].encode())!=meta['sample_checksum']:raise IntegrityError('Sample checksum')
    if y.iloc[ids].value_counts().to_dict()!=meta['class_counts']:raise IntegrityError('Sample target distribution')
    return meta,ids


def load_partitions(meta,seed,y):
    path=BASE/'splits'/f"{meta['sample_id']}_cv{seed}.npz";m=json.loads(path.with_suffix('.json').read_text())
    if m!={'sample_checksum':meta['sample_checksum'],'cv_seed':seed,'file_sha256':digest(path.read_bytes())}:raise IntegrityError('Partition identity')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={f'{p}_{i}' for i in range(15) for p in ['train','valid']}:raise IntegrityError('Partition inventory')
        splits=[(z[f'train_{i}'],z[f'valid_{i}']) for i in range(15)]
    validate_splits(splits,y)
    return splits,m['file_sha256']


class MagicRunner:
    def __init__(self,pilot=False):
        self.X,self.y,self.dataset=inspect_data();self.cfg=json.loads((BASE/'configuration.json').read_text())
        if self.cfg!=frozen_configuration():raise IntegrityError('Frozen configuration/code/environment/reference changed')
        self.prepared=json.loads((BASE/'prepared.json').read_text())
        if self.prepared['configuration_sha256']!=digest(json_bytes(self.cfg)) or self.prepared['dataset_manifest_sha256']!=digest(json_bytes(self.dataset)):raise IntegrityError('Preparation identity')
        for name,sha in self.prepared['files'].items():
            if digest((BASE/name).read_bytes())!=sha:raise IntegrityError('Prepared artifact changed: '+name)
        self.context={};self.plan=[]
        for name in self.prepared['samples']:
            meta,ids=load_sample(name,self.y,self.dataset);X=self.X.iloc[ids].reset_index(drop=True);y=self.y.iloc[ids].reset_index(drop=True)
            splits={seed:load_partitions(meta,seed,y) for seed in CVSEEDS}
            self.context[name]=(meta,ids,X,y,splits)
            # Interleave models on each fold, then seeds; pilot explicitly covers two sizes.
            self.plan.extend((name,model,seed,i//5+1,i%5+1) for seed in CVSEEDS for i in range(15) for model in MODELS)
        if len(self.plan)!=10500 or len(set(self.plan))!=10500:raise IntegrityError('Workload mismatch')
        self.parameters={model:parameters(make_model(model,self.X,self.y)) for model in MODELS}
        manifest=dict(experiment_id=VERSION,purpose='pilot' if pilot else 'full',configuration=self.cfg,
            prepared_sha256=digest(json_bytes(self.prepared)),model_parameters=self.parameters)
        self.store=Store(BASE/'results'/('pilot' if pilot else 'full'),manifest);self.store.check_manifest();self.pilot=pilot

    def identity(self,key):
        name,model,seed,rep,fold=key;meta,ids,X,y,splits=self.context[name];a,b=splits[seed][0][(rep-1)*5+fold-1]
        return dict(sample_id=name,model=model,cv_seed=seed,repeat=rep,fold=fold,subsampling_seed=meta['subsampling_seed'],replicate=meta['replicate'],sample_size=meta['actual_size'],
            dataset_fingerprint=self.dataset['dataset_fingerprint'],sample_checksum=meta['sample_checksum'],
            split_id=digest(json_bytes([meta['sample_checksum'],seed,rep,fold])+a.tobytes()+b.tobytes()),
            configuration_id=digest(json_bytes(self.parameters[model])),environment_id=self.cfg['environment']['environment_id'],purpose=self.store.manifest['purpose'])

    def completed(self,key):
        records=self.store.attempts(key);identity=self.identity(key)
        for r in records:
            if any(r.get(k)!=v for k,v in identity.items()):raise IntegrityError('Record provenance mismatch')
            if r['execution_status'] not in ['success','failure']:raise IntegrityError('Invalid execution status')
            if r['execution_status']=='success' and (not isinstance(r.get('macro_f1'),(float,int)) or not np.isfinite(r['macro_f1']) or not 0<=r['macro_f1']<=1):raise IntegrityError('Invalid score')
        return records[-1] if records and records[-1]['execution_status']=='success' else None

    def validate_inventory(self):
        expected={self.store.folder(k) for k in self.plan}
        for p in self.store.root.glob('records/**/attempt_*.json'):
            if p.parent not in expected:raise IntegrityError('Unexpected result key')
        records=[self.completed(k) for k in self.plan];return [r for r in records if r is not None]

    def run(self,limit=16,authorize_full=False):
        if not self.pilot and not authorize_full:raise PermissionError('Full run requires explicit user authorization and --authorize-full')
        if self.pilot and not 1<=limit<=20:raise ValueError('Pilot is capped at 20 evaluations')
        keys=[(f's101_n{n}',m,42,1,f) for n in [200,500] for f in [1,2] for m in MODELS] if self.pilot else self.plan
        done=0
        with writer_lock(self.store.root),threadpool_limits(limits=1):
            self.store.initialize();self.validate_inventory()
            for key in keys:
                if self.completed(key):continue
                if self.pilot and done>=limit:break
                name,model,seed,rep,fold=key;meta,ids,X,y,splits=self.context[name];a,b=splits[seed][0][(rep-1)*5+fold-1]
                record=dict(**self.identity(key),attempt=len(self.store.attempts(key))+1);started=time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');estimator=make_model(model,X,y)
                        estimator.fit(X.iloc[a],y.iloc[a]);pred=estimator.predict(X.iloc[b])
                    record.update(execution_status='success',macro_f1=float(f1_score(y.iloc[b],pred,average='macro',zero_division=0)),warnings=[str(w.message) for w in caught])
                except Exception as exc:
                    record.update(execution_status='failure',error_type=type(exc).__name__,error=str(exc))
                record['elapsed_seconds']=time.perf_counter()-started;self.store.save(key,record);done+=1
                if record['execution_status']=='failure':raise RuntimeError('Fold failed and was checkpointed: '+record['error'])
                if done%25==0 or self.pilot:print(f'{done} newly completed; {name} {model} CV={seed} r={rep} f={fold}',flush=True)
        print(json.dumps(dict(new_successes=done,total_successes=len(self.validate_inventory()),expected=16 if self.pilot else 10500)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','dry-run','pilot','run','validate']);p.add_argument('--limit',type=int,default=16);p.add_argument('--authorize-full',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare();return
    runner=MagicRunner(pilot=a.action=='pilot' or a.pilot)
    if a.action in ['dry-run','validate']:print(json.dumps(dict(expected=10500,successes=len(runner.validate_inventory()),models=MODELS,sizes=SIZES,full_reference=False),indent=2))
    else:runner.run(a.limit,a.authorize_full)


if __name__=='__main__':main()
