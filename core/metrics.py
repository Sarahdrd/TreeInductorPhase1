# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:38:33 2025

"""

from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    multilabel_confusion_matrix
)
import numpy as np

# Complexity: counting nodes
def count_nodes(node):
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)



def evaluate_tree(tree, X_train, y_train, X_test, y_test):
    try:
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        all_labels = np.unique(y_train)

        f1 = f1_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        precision = precision_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        recall_per_class = recall_score(y_test,y_pred,labels=all_labels,average=None,zero_division=0)
        balanced_acc = np.mean(recall_per_class)
        mcc = matthews_corrcoef(y_test, y_pred)
        specificity = compute_macro_specificity(y_test, y_pred)

        # Tree structure
        root = getattr(tree, "tree_", None)
        if root is None:
            root = getattr(tree, "root", None)

        n_nodes = count_nodes(root)

        depth = tree.get_depth() if hasattr(tree, "get_depth") else None

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "mcc": mcc,
            "n_nodes": n_nodes,
            "depth": depth
        }

    except Exception as e:
        print(f" Error during evaluation: {e}")
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "mcc": 0.0,
            "n_nodes": 9999,
            "depth": None
        }

def compute_macro_specificity(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    specificities = []

    for conf in mcm:
        tn, fp, fn, tp = conf.ravel()
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        specificities.append(spec)

    return np.mean(specificities)
