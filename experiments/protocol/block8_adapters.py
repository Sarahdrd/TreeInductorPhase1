"""Versioned adapter dispatcher using tested process-tree containment.
The original Phase-8A module and records remain immutable evidence. No full-run CLI.
"""
from unittest.mock import patch
from . import block8
from .block8_process import external

class ExternalTreeAdapter:
    def __init__(self,algorithm):
        if algorithm not in block8.ALGORITHMS:raise ValueError(algorithm)
        self.algorithm=algorithm
    def prepare(self,X_train,y_train,X_valid,folder):
        self.inputs=(X_train,y_train,X_valid,folder);return self
    def train(self):
        # Serial dispatcher; do not share this module patch across concurrent threads.
        with patch.object(block8,'external',external):
            self.metadata,self.training_predictions,self.validation_predictions=block8.execute(self.algorithm,*self.inputs)
        self.metadata['adapter_version']='block8-adapter-v2-job-object'
        return self
    def predict(self,partition='validation'):
        if partition not in ['training','validation']:raise ValueError(partition)
        prediction=self.training_predictions if partition=='training' else self.validation_predictions
        if prediction is None:raise RuntimeError('No feasible prediction returned; never substitute a zero score')
        return prediction.copy()
    def get_metadata(self):return dict(self.metadata)
