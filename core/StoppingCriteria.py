# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:17:19 2025
"""
import numpy as np


class StoppingCriterion:
    """
    Defines the stopping rules for terminating the growth of a decision tree.
    
    Parameters
    ----------
    n_tot_samples : int
        Total number of samples in the dataset.
    criterion : str
        Stopping strategy name ('max_depth', 'homogeneity', etc.).
    param : int
        Integer controlling the stopping behavior (from gene value).
    """
    def __init__(self, n_tot_samples, criterion, param=2):
        self.n_tot_samples = n_tot_samples
        self.criterion = criterion
        self.param = param

    # --- Individual stopping rules ---
    def _homogeneity(self, y):
        return len(np.unique(y)) == 1

    def _max_depth(self, depth):
        return depth >= (2 + self.param)

    def _min_samples_split(self, y):
        return len(y) < (self.param * 0.19) + 1

    def _min_portion_split(self, y):
        if self.n_tot_samples == 0:
            return True  # nothing to split on
        mapped = ((self.param * 0.09) + 1) / 100
        return (len(y) / self.n_tot_samples) < mapped

    def _predictive_accuracy(self, y):
        if len(y) == 0:
            return True  # safe stop if no samples
        mapped = (self.param % 7) * 5 + 70  # 70–100 range
        majority_class = np.argmax(np.bincount(y))
        accuracy = np.sum(y == majority_class) / len(y)
        return accuracy * 100 >= mapped

    # General stop controller
    def stop(self, y, depth):
        checks = {
            "homogeneity": lambda: self._homogeneity(y),
            "max_depth": lambda: self._max_depth(depth),
            "min_samples_split": lambda: self._min_samples_split(y),
            "min_portion_split": lambda: self._min_portion_split(y),
            "predictive_accuracy": lambda: self._predictive_accuracy(y)
        }
        func = checks.get(self.criterion)
        return func() if func else False
