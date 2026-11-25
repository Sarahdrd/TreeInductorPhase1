"""Depth-2 CART pilot with representations matched to the independent inductors."""
import json
from pathlib import Path
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from . import block8 as b8
from .runner import Runner,parameters
from .infrastructure import config,digest,json_bytes,write_frozen
from .checkpoints import Store,writer_lock


def run():
    cfg=json.loads((b8.BASE/'pilot_configuration.json').read_text());params={**config()['models']['decision_tree_default'],'max_depth':2}
    manifest=dict(version='block8-depth2-cart-pilot-v1',frozen_solver_pilot_configuration_sha256=digest(json_bytes(cfg)),parameters=params,
        resolved_parameters=parameters(DecisionTreeClassifier(**params)),representations=['murtree','quant_bnb'],source_sha256=digest(Path(__file__).read_bytes()),threads=1,
        interpretation='Same depth and same training-only representation per solver; CART Gini heuristic versus misclassification optimization, no equal node-count enforcement beyond depth')
    write_frozen(b8.BASE/'matched_cart_configuration.json',manifest);store=Store(b8.BASE/'results/matched-cart-pilot-v1',manifest);baseline=Runner()
    with writer_lock(store.root),threadpool_limits(limits=1):
        store.initialize()
        for dataset in b8.PILOT_DATASETS:
            obj,(a,b),meta,identity=baseline.context((dataset,'decision_tree_default',42,1,1))
            for representation in b8.ALGORITHMS:
                model='cart_depth2_'+representation;key=(dataset,model,42,1,1);old=store.attempts(key)
                if old and old[-1]['execution_status']=='success':continue
                prep=b8.Representation(representation).fit(obj['X'].iloc[a]);A=prep.transform(obj['X'].iloc[a]);B=prep.transform(obj['X'].iloc[b])
                frozen=json.loads((b8.BASE/'work'/f'{dataset}_{representation}_attempt1/preprocessing.json').read_text())
                assert prep.spec==frozen['spec'] and prep.columns==frozen['columns']
                est=DecisionTreeClassifier(**params);start=time.perf_counter();est.fit(A,obj['y'][a]);elapsed=time.perf_counter()-start
                row=dict(dataset=dataset,algorithm=model,representation=representation,seed=42,repeat=1,fold=1,split_id=identity['split_id'],dataset_fingerprint=identity['dataset_fingerprint'],
                    configuration_id=digest(json_bytes(manifest)),attempt=len(old)+1,execution_status='success',depth=est.get_depth(),decision_nodes=int((est.tree_.children_left!=-1).sum()),
                    transformed_features=A.shape[1],fit_seconds=elapsed,training_macro_f1=float(f1_score(obj['y'][a],est.predict(A),average='macro',zero_division=0)),validation_macro_f1=float(f1_score(obj['y'][b],est.predict(B),average='macro',zero_division=0)))
                store.save(key,row);print(json.dumps(row),flush=True)


if __name__=='__main__':run()
