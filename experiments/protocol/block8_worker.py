"""Isolated representation-matched CART worker for the Block 8 official runner."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

def main():
    data=np.load(sys.argv[1],allow_pickle=False);params=json.loads(Path(sys.argv[2]).read_text())
    with threadpool_limits(limits=1):
        model=DecisionTreeClassifier(**params);start=time.perf_counter();model.fit(data['A'],data['y']);seconds=time.perf_counter()-start
        result=dict(feasible=True,optimal=False,certification='heuristic CART',solver_seconds=seconds,depth=model.get_depth(),nodes=int((model.tree_.children_left!=-1).sum()),train_predictions=model.predict(data['A']).tolist(),valid_predictions=model.predict(data['B']).tolist())
    Path(sys.argv[3]).write_text(json.dumps(result),encoding='utf8')
if __name__=='__main__':main()
