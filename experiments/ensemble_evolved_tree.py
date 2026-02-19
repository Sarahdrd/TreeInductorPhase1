# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:34:25 2025

@author: Aurora
"""
from evolution.encode_decode import decode 
from Extract_data import extract_data
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from core.metrics import compute_macro_specificity

def majority_vote(trees, X):

    all_preds = np.array([tree.predict(X) for tree in trees])  # shape: (n_trees, n_samples)
    all_preds = all_preds.T  # shape: (n_samples, n_trees)

    majority_preds = []
    for sample_preds in all_preds:
        most_common = Counter(sample_preds).most_common(1)[0][0]
        majority_preds.append(most_common)
    
    return np.array(majority_preds)

X_train, X_test, y_train, y_test = extract_data("winequality-red")
final_population = [[8, 2, 14, 1, 6, 0, 2, 98],
 [6, 2, 90, 1, 5, 0, 4, 73],
 [3, 2, 18, 1, 6, 0, 1, 7],
 [8, 2, 14, 1, 6, 0, 2, 99],
 [6, 2, 90, 1, 3, 0, 1, 7],
 [7, 1, 18, 0, 3, 0, 4, 16],
 [7, 1, 10, 3, 5, 0, 4, 16],
 [11, 1, 10, 3, 6, 1, 4, 11],
 [10, 2, 1, 3, 5, 1, 5, 46],
 [3, 1, 10, 3, 3, 0, 5, 46],
 [11, 1, 18, 0, 3, 0, 4, 16],
 [3, 2, 65, 2, 3, 0, 4, 60],
 [10, 2, 18, 1, 3, 0, 4, 73],
 [8, 3, 98, 0, 3, 0, 5, 46],
 [8, 3, 98, 0, 3, 0, 4, 60],
 [3, 1, 14, 1, 6, 0, 4, 73],
 [10, 2, 1, 3, 3, 1, 5, 44],
 [4, 2, 74, 0, 3, 0, 4, 16],
 [10, 2, 10, 2, 3, 0, 5, 46],
 [4, 2, 14, 0, 2, 0, 0, 80],
 [10, 2, 18, 1, 3, 0, 5, 45],
 [3, 2, 94, 3, 3, 1, 5, 44],
 [3, 1, 14, 1, 6, 0, 5, 46],
 [10, 2, 74, 0, 3, 0, 4, 80],
 [10, 2, 10, 2, 3, 1, 4, 75],
 [10, 3, 6, 1, 2, 0, 4, 80],
 [3, 2, 65, 2, 5, 0, 4, 34],
 [10, 2, 74, 2, 6, 1, 4, 11],
 [3, 2, 18, 1, 6, 0, 2, 75],
 [10, 3, 70, 3, 5, 0, 4, 7],
 [3, 3, 98, 1, 6, 0, 2, 75],
 [3, 2, 58, 3, 5, 0, 4, 2],
 [3, 2, 18, 1, 6, 1, 4, 99],
 [10, 2, 18, 1, 6, 0, 5, 46],
 [10, 2, 14, 3, 6, 0, 5, 46],
 [10, 2, 56, 0, 6, 0, 5, 46],
 [3, 1, 49, 3, 3, 1, 4, 46],
 [4, 2, 74, 0, 3, 0, 4, 99],
 [4, 2, 14, 1, 6, 0, 2, 75],
 [10, 2, 6, 3, 5, 0, 4, 80],
 [8, 1, 31, 0, 5, 0, 4, 2],
 [3, 3, 98, 3, 6, 0, 1, 7],
 [3, 3, 98, 3, 6, 0, 4, 99],
 [4, 2, 14, 1, 3, 0, 4, 73],
 [3, 2, 18, 1, 6, 0, 4, 98],
 [4, 1, 18, 0, 3, 0, 4, 16],
 [10, 2, 10, 1, 6, 0, 4, 80],
 [10, 2, 98, 0, 3, 1, 4, 11],
 [3, 2, 98, 0, 5, 0, 4, 56],
 [4, 3, 10, 0, 3, 1, 4, 16],
 [10, 2, 14, 3, 5, 0, 4, 73],
 [3, 2, 70, 1, 6, 1, 4, 56],
 [8, 1, 14, 1, 6, 0, 5, 46],
 [3, 2, 23, 1, 6, 0, 1, 80],
 [4, 3, 10, 0, 3, 1, 4, 99],
 [10, 2, 14, 3, 5, 1, 5, 44],
 [3, 3, 75, 1, 6, 1, 2, 98],
 [10, 2, 65, 0, 5, 0, 4, 56],
 [3, 2, 83, 3, 5, 1, 4, 60],
 [10, 2, 10, 1, 6, 1, 4, 75],
 [3, 1, 10, 3, 3, 0, 5, 98],
 [8, 1, 49, 3, 3, 1, 4, 46],
 [3, 3, 71, 1, 6, 0, 5, 46],
 [3, 1, 14, 3, 5, 0, 4, 73],
 [10, 2, 14, 3, 5, 1, 4, 75],
 [4, 2, 14, 3, 6, 0, 4, 73],
 [1, 1, 1, 3, 3, 0, 4, 99],
 [10, 2, 10, 2, 5, 0, 4, 73],
 [3, 2, 83, 3, 5, 0, 4, 56],
 [4, 3, 56, 3, 6, 1, 5, 46],
 [3, 1, 14, 3, 5, 0, 4, 34],
 [1, 3, 56, 0, 1, 0, 4, 56],
 [10, 2, 6, 3, 5, 0, 4, 7],
 [4, 2, 14, 0, 2, 0, 0, 60],
 [3, 3, 75, 1, 6, 0, 4, 99],
 [10, 3, 6, 1, 2, 1, 4, 98],
 [4, 3, 56, 3, 6, 0, 4, 98],
 [1, 1, 1, 3, 3, 1, 4, 46],
 [3, 2, 18, 1, 6, 0, 1, 44],
 [3, 1, 17, 2, 5, 0, 4, 34],
 [3, 2, 58, 3, 5, 1, 4, 75],
 [3, 2, 98, 2, 3, 0, 4, 60],
 [3, 3, 98, 0, 3, 1, 5, 45],
 [3, 2, 65, 0, 5, 0, 4, 56],
 [3, 3, 98, 1, 6, 0, 5, 45],
 [4, 3, 23, 0, 3, 0, 4, 2],
 [10, 2, 1, 3, 5, 1, 5, 44],
 [4, 2, 74, 0, 3, 1, 4, 28],
 [10, 2, 74, 0, 3, 0, 4, 99],
 [4, 3, 10, 1, 6, 0, 4, 99],
 [4, 2, 74, 0, 3, 0, 4, 28],
 [1, 3, 98, 2, 3, 0, 5, 46],
 [3, 2, 58, 3, 5, 1, 4, 80],
 [3, 2, 58, 3, 3, 1, 4, 16],
 [3, 1, 14, 3, 0, 1, 5, 73],
 [10, 2, 56, 1, 6, 0, 5, 45],
 [3, 3, 75, 1, 6, 1, 2, 2],
 [10, 2, 18, 1, 6, 1, 5, 46],
 [10, 2, 56, 0, 6, 0, 5, 56],
 [3, 3, 75, 1, 6, 1, 2, 7]]

trees = [decode(indiv) for indiv in final_population]
print("all trees have been decoded")
for tree in trees:
    tree.fit(X_train, y_train)
print("all trees have been fitted")
    

# Ensemble prediction
ensemble_preds = majority_vote(trees, X_test)
print("les arbres ont voté")

# Évaluer la précision
ensemble_f1 = f1_score(y_test, ensemble_preds, average='macro', zero_division=0)
ensemble_recall = recall_score(y_test, ensemble_preds, average='macro', zero_division=0)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_specificity = compute_macro_specificity(y_test, ensemble_preds)
print(f"Ensemble accuracy : {ensemble_acc:.4f}")
print(f"Ensemble f1 : {ensemble_f1:.4f}")
print(f"Ensemble recall : {ensemble_recall:.4f}")
print(f"Ensemble specificity : {ensemble_specificity:.4f}")