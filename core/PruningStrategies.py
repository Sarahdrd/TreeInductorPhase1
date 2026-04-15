# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:09:52 2025

"""
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from core.split_utils import safe_train_test_split

class Pruning:
    """
    Implements post-pruning strategies for decision trees.
    
    Supported methods:
        - REP: Reduced Error Pruning
        - PEP: Pessimistic Error Pruning
        - MEP: Minimum Error Pruning
        - CCP: Cost-Complexity Pruning
        - EBP: Error-Based Pruning
    """
    #def __init__(self, method=None, param=None):
    #   self.method = method
    #  self.param = param
    
    def __init__(self, method=None, param=None):
        self.method = method
        self.param = 0 if param is None else param


    # ------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------
    def _predict_from_node(self, node, x):
        while not node.is_leaf:
            node = node.left if x[node.feat_idx] <= node.threshold else node.right
        return node.leaf_value

    def _evaluate_error(self, tree, X, y):
        preds = [self._predict_from_node(tree, xi) for xi in X]
        return np.sum(preds != y)

    def _count_leaves(self, node):
        if node is None:
            return 0
        return 1 if node.is_leaf else self._count_leaves(node.left) + self._count_leaves(node.right)

#    def _collect_reachable_examples(self, root, target, X, y):
#        """Return samples that reach a specific node."""
#        Xr, yr = [], []
#        for xi, yi in zip(X, y):
#            node = root
#            while node and not node.is_leaf:
#                if node == target:
#                    Xr.append(xi); yr.append(yi); break
#                node = node.left if xi[node.feat_idx] <= node.threshold else node.right
#            if node == target:
#                Xr.append(xi); yr.append(yi)
#        return np.array(Xr), np.array(yr)
    

    def _collect_reachable_examples(self, root, target, X, y):
        """Return samples that reach a specific node."""
        Xr, yr = [], []
        for xi, yi in zip(X, y):
            node = root
            # Bajamos por el árbol hasta hoja o target
            while node is not None and not node.is_leaf and node is not target:
                if xi[node.feat_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # Si terminamos exactamente en el target, este ejemplo lo alcanza
            if node is target:
                Xr.append(xi)
                yr.append(yi)
        return np.array(Xr), np.array(yr)


    # ------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------
    def prune(self, tree, X, y):
        methods = {
            "REP": self._REP,
            "PEP": self._PEP,
            "MEP": self._MEP,
            "CCP": self._CCP,
            "EBP": self._EBP
        }
        func = methods.get(self.method)
        return func(tree, X, y) if func else tree

    # ------------------------------------------------------
    # 1. Reduced Error Pruning (REP)
    # ------------------------------------------------------
    def _REP(self, tree, X, y):
        val_size = (self.param * 0.4 + 10) / 100
        
        _, Xv, _, yv = safe_train_test_split( X, y, test_size=val_size, random_state=42, prefer_stratify=True)

        return self._rep_node(tree, tree, Xv, yv)

    def _rep_node(self, node, tree, Xv, yv):
        if node is None or node.is_leaf:
            return node
        node.left = self._rep_node(node.left, tree, Xv, yv)
        node.right = self._rep_node(node.right, tree, Xv, yv)

        err_before = self._evaluate_error(tree, Xv, yv)
        orig = (node.left, node.right, node.is_leaf, node.leaf_value)

        node.left = node.right = None
        node.is_leaf = True
        node.leaf_value = node.majority_class

        err_after = self._evaluate_error(tree, Xv, yv)
        if err_after > err_before:
            node.left, node.right, node.is_leaf, node.leaf_value = orig
        return node

    # ------------------------------------------------------
    # 2. Pessimistic Error Pruning (PEP)
    # ------------------------------------------------------
    def _PEP(self, tree, X, y):
        return self._pep_node(tree, tree, X, y)

    def _pep_node(self, node, tree, X, y):
        if node is None or node.is_leaf:
            return node
        node.left = self._pep_node(node.left, tree, X, y)
        node.right = self._pep_node(node.right, tree, X, y)

        Xn, yn = self._collect_reachable_examples(tree, node, X, y)
        if len(yn) == 0:
            return node

        err_sub = np.sum([self._predict_from_node(node, xi) != yi for xi, yi in zip(Xn, yn)])
        nb_leaves = self._count_leaves(node)
        mapping = [0.5, 1.0, 1.5, 2.0]
        se = mapping[min(int(self.param / 25), 3)]

        err_sub += se * 0.5 * nb_leaves
        majority = np.bincount(yn).argmax()
        err_leaf = np.sum(yn != majority) + se * 0.5

        if err_leaf <= err_sub:
            node.left = node.right = None
            node.is_leaf = True
            node.leaf_value = majority
        return node

    # ------------------------------------------------------
    # 3. Minimum Error Pruning (MEP)
    # ------------------------------------------------------
    def _MEP(self, tree, X, y):
        return self._mep_node(tree, tree, X, y)

    def _mep_node(self, node, tree, X, y):
        if node is None or node.is_leaf:
            return node
        node.left = self._mep_node(node.left, tree, X, y)
        node.right = self._mep_node(node.right, tree, X, y)

        Xn, yn = self._collect_reachable_examples(tree, node, X, y)
        if len(yn) == 0:
            return node

        m = max(1, self.param)
        err_sub = np.sum([self._predict_from_node(node, xi) != yi for xi, yi in zip(Xn, yn)])
        n = len(yn)
        err_est_sub = (err_sub + m * 0.5) / (n + m)

        majority = np.bincount(yn).argmax()
        err_leaf = np.sum(yn != majority)
        err_est_leaf = (err_leaf + m * 0.5) / (n + m)

        if err_est_leaf <= err_est_sub:
            node.left = node.right = None
            node.is_leaf = True
            node.leaf_value = majority
        return node

    # ------------------------------------------------------
    # 4. Cost-Complexity Pruning (CCP)
    # ------------------------------------------------------
    def _CCP(self, tree, X, y):
        val_size = min(max(self.param / 100.0, 0.1), 0.5)
        _, Xv, _, yv = safe_train_test_split( X, y, test_size=val_size, random_state=42, prefer_stratify=True)
        seq = self._ccp_sequence(tree, X, y)
        return self._select_best(seq, Xv, yv)

    def _ccp_sequence(self, tree, X, y):
        seq = [deepcopy(tree)]
        while True:
            candidates = []
            self._collect_alpha(tree, tree, X, y, candidates)
            if not candidates:
                break
            node = min(candidates, key=lambda x: x['alpha'])['node']
            Xn, yn = self._collect_reachable_examples(tree, node, X, y)
            majority = np.bincount(yn).argmax()
            node.left = node.right = None
            node.is_leaf = True
            node.leaf_value = majority
            seq.append(deepcopy(tree))
        return seq

    def _collect_alpha(self, root, node, X, y, cands):
        if node is None or node.is_leaf:
            return
        Xn, yn = self._collect_reachable_examples(root, node, X, y)
        if len(yn) == 0:
            return
        err_Tt = np.sum([self._predict_from_node(node, xi) != yi for xi, yi in zip(Xn, yn)])
        majority = np.bincount(yn).argmax()
        err_t = np.sum(yn != majority)
        leaves = self._count_leaves(node)
        if leaves > 1:
            alpha = (err_t - err_Tt) / (leaves - 1)
            cands.append({'node': node, 'alpha': alpha})
        self._collect_alpha(root, node.left, X, y, cands)
        self._collect_alpha(root, node.right, X, y, cands)

    def _select_best(self, seq, X, y):
        se = [0.5, 1.0, 1.5, 2.0][min(int(self.param / 25), 3)]
        errs = [self._evaluate_error(t, X, y) for t in seq]
        best_err = min(errs)
        std = np.std(errs)
        for t, e in zip(seq, errs):
            if e <= best_err + se * std:
                return t
        return seq[-1]

    # ------------------------------------------------------
    # 5. Error-Based Pruning (EBP)
    # ------------------------------------------------------
    def _EBP(self, tree, X, y):
        val_size = min(max(self.param / 100.0, 0.1), 0.5)
        _, Xv, _, yv = safe_train_test_split( X, y, test_size=val_size, random_state=42, prefer_stratify=True)
        cf = np.clip(self.param / 100, 0.01, 0.5)
        z = norm.ppf(1 - cf)
        return self._ebp_node(tree, tree, Xv, yv, z)

    def _ebp_node(self, node, tree, X, y, z):
        if node is None or node.is_leaf:
            return node
        node.left = self._ebp_node(node.left, tree, X, y, z)
        node.right = self._ebp_node(node.right, tree, X, y, z)

        Xn, yn = self._collect_reachable_examples(tree, node, X, y)
        if len(yn) == 0:
            return node

        majority = np.bincount(yn).argmax()
        n = len(yn)
        err_leaf = np.sum(yn != majority) / n
        err_leaf += z * np.sqrt(err_leaf * (1 - err_leaf) / n)

        err_sub = np.sum([self._predict_from_node(node, xi) != yi for xi, yi in zip(Xn, yn)]) / n
        err_sub += z * np.sqrt(err_sub * (1 - err_sub) / n)

        if err_leaf <= err_sub:
            node.left = node.right = None
            node.is_leaf = True
            node.leaf_value = majority
        return node
