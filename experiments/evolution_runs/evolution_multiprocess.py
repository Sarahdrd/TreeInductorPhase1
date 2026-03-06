import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Current working directory
PROJECT_ROOT = Path.cwd()

# Add to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core libraries
from core.metrics import evaluate_tree_new
from evolution.encode_decode import decode
from evolution.genetic_DT_GA_Multi import evolve_trees, _load_dataset_csv

def eval_gene_on_dataset(gene, X, y, k=5, seed=123):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        algo = decode(gene)
        algo.fit(X[tr], y[tr])
        m = evaluate_tree_new(algo, X[te], y[te])
        scores.append(m["f1"])
    return np.array(scores)


def main():
    """
    Main process
    """

    # Build set of datasets
    #iris = load_iris(return_X_y=True)
    #wine = load_wine(return_X_y=True)
    #bc   = load_breast_cancer(return_X_y=True)
    #hepatitis = load_uci_dataset(UCI_DATASETS["hepatitis"])

    data_path = PROJECT_ROOT / "data"
    datasets: dict[str, Path] = {
        "iris": data_path / "iris.csv",
        "wine_red": data_path / "winequality_red.csv",
        "breast_cancer": data_path / "breast_cancer.csv",
        "hepatitis": data_path / "hepatitis.csv"
    }

    # Genetic parameters
    mu = 10
    lambd = 10
    generations = 50

    print("Starting Evolutionary Process...")
    print(f"Experiment with mu={mu}, lambda={lambd} for {generations} generations.")
    print("Datasets to be trained on:", [e for e in datasets.keys()])

    evo_t0 = time.time()
    best_gene, best_fit, history = evolve_trees(datasets, mu=mu, lambd=lambd, generations=generations)
    evo_tf = time.time()

    print("Total time (s):", round(evo_tf - evo_t0, 4))
    print("Best gene:", best_gene)
    print("Fitness (mean - SDD):", best_fit)

    # Report generalization across datasets 
    for name, dataset_path in datasets.items():
        X, y = _load_dataset_csv(dataset_path, target_col="target")
        scores = eval_gene_on_dataset(best_gene, X, y, k=5)
        print(f"{name}: mean F1={scores.mean():.4f}, std F1={scores.std():.4f}, folds={scores}")

    plt.plot(history, marker="o")
    plt.title("Evolution of Multi-Dataset Generalization Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Mean − SDD across datasets)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()