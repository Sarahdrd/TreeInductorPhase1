# General libraries
import os
import math
import time
import random
import traceback
import numpy as np
import pandas as pd

from numpy.typing import NDArray

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import evolutionary functions
from evolution.crossover import single_point_crossover
from evolution.mutation import random_reset_mutation
from evolution.selection import tournament_selection
from evolution.encode_decode import generate_population, decode

from core.metrics import evaluate_tree_new

# Domain
from core.DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES

# Prevent BLAS/OpenMP oversubscription (safe default).
# Put these at top-level BEFORE heavy numeric libs run in workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Define search space per gene 
VAR_DOMAINS = [
    range(len(SPLIT_CRITERIA)),
    range(len(STOPPING_CRITERIA)),
    range(1, 21),  # stopping param domain
    range(len(MV_SPLIT)),
    range(len(MV_DISTRIBUTION)),
    range(len(MV_CLASSIFICATION)),
    range(len(PRUNNING_GENES)),
    range(1, 11),  # pruning param domain
]


# -------------------------
# Types & Interfaces
# -------------------------

#@dataclass(frozen=True)
#class AlgorithmSpec:
#    ...

#


# -------------------------
# Utilities
# -------------------------

def _stratified_kfold_indices(
    y: NDArray[np.float64], n_splits: int, seed: int
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Stratified K-Fold
    """

    # Make random generator
    #rng = np.random.default_rng(seed)

    # Process
    y = np.asarray(y)
    n = y.shape[0]

    # Make K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Split and get indices
    splits = []
    for tr, te in skf.split(np.zeros(n), y):
        splits.append((tr, te))

    return splits


def _load_dataset_csv(path: Path, target_col: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in {path}.\n Columns: {list(df.columns)[:20]}")

    y = LabelEncoder().fit_transform(df[target_col].to_numpy())
    X = df.drop(columns=[target_col]).to_numpy()
    return X, y


# -------------------------
# Worker Job
# -------------------------

def _eval_one_algo_one_dataset(
algo_genome: list[int],
    dataset_name: str,
    dataset_path_str: str,
    target_col: str,
    n_splits: int,
    split_seed: int,
) -> tuple[str, str, list[float]]:
    """
    Evaluate one algorithm on one dataset with stratified K-fold.
    Returns: (algo_name, dataset_name, mean, std)
    """

    start = time.perf_counter()
    pid = os.getpid()

    decode_time = 0
    fit_time = 0
    metrics_time = 0

    dataset_path = Path(dataset_path_str)

    # Load data inside the worker
    X, y = _load_dataset_csv(dataset_path, target_col=target_col)

    # Split data using stratified logic
    splits = _stratified_kfold_indices(y, n_splits=n_splits, seed=split_seed)

    # Main work
    fold_scores: list[float] = []
    n_nodes: list[int] = []
    depths: list[int] = []
    for fold_indx, (tr, va) in enumerate(splits):

        # Instantiate a fresh algorithhm per fold
        t0 = time.perf_counter()
        algo = decode(algo_genome)
        decode_time += time.perf_counter() - t0

        #print("Indices")
        #print(tr, va)

        #print("\nMatrices")
        #print(X, y)

        # Fit algorithm to data
        t0 = time.perf_counter()
        algo.fit(X[tr], y[tr])
        fit_time += time.perf_counter() - t0

        # Predict and get metrics
        t0 = time.perf_counter()
        metrics = evaluate_tree_new(algo, X[va], y[va])
        metrics_time = time.perf_counter() - t0

        # Get F1
        fold_scores.append(metrics["f1"])

        # Get depth and nodes
        n_nodes.append(algo.n_nodes_)
        depths.append(algo.max_depth_)

    duration = time.perf_counter() - start

    # Compute trees stats
    mean_n_nodes = np.mean(n_nodes)
    std_n_nodes = np.std(n_nodes)
    mean_depth = np.mean(depths)
    std_depth = np.std(depths)
    tree_data = [mean_n_nodes, std_n_nodes, mean_depth, std_depth]

    # Right now, the algorithm was testes on the current dataset k times
    #mean = float(np.mean(scores))
    #std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    return algo_genome, dataset_name, fold_scores, tree_data, [duration, decode_time, fit_time, metrics_time], pid


# -------------------------
# Public API
# -------------------------

def evaluation(
    algorithms: list[list[int]],
    datasets: dict[str, Path],
    *,
    target_col: str = "target",
    metric_name: str = "F1",
    n_splits: int = 5,
    global_seed: int = 42,
    n_workers: int | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Parallel evaluation over (algorithm, dataset) jobs.

    Returns:
        results[algo_name][dataset_name] = (mean, std) over K folds
    """

    # Deterministic per-dataset split seeds
    # Same split across all algorithms
    dataset_split_seeds = {
        ds_name: (abs(hash((global_seed, ds_name))) % (2**31 - 1))
        for ds_name in datasets.keys()
    }

    jobs = []
    for algo in algorithms:
        for ds_name, ds_path in datasets.items():
            jobs.append((algo, ds_name, ds_path))

    results = {str(a): {} for a in algorithms}
    job_times = []

    print("------------------------------")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = []
        for algo, ds_name, ds_path in jobs:
            fut = ex.submit(
                _eval_one_algo_one_dataset,
                algo,
                ds_name,
                str(ds_path),
                target_col,
                n_splits,
                dataset_split_seeds[ds_name]
            )
            futs.append(fut)

        for fut in as_completed(futs):
            try:
                algo_name, ds_name, fold_scores, tree_data, times, pid = fut.result()
                results[str(algo_name)][ds_name] = fold_scores
                job_times.append((str(algo_name), ds_name, times, pid, tree_data[0], tree_data[2]))
                print(f"[DONE] {str(algo_name):20s} {ds_name:20s} \t {times[0]:.2f}s")
                print(f"times: decode {times[1]:.2f}s | fit {times[2]:.2f}s | metrics {times[3]:.2f}s (pid={pid})")
                print(f"Mean No. Nodes: {tree_data[0]} +- {tree_data[1]:.2f} | Mean Depth: {tree_data[2]} +- {tree_data[3]:.2f}")

            except Exception as e:
                traceback.print_exc()
                raise
                #print(f"Fail for Algorithm")#: {algo_name}\nOn {ds_name} dataset.")
                #raise

    # Identify slowest job
    job_times.sort(key=lambda x: x[2][0], reverse=True)
    print("\n=== Slowest jobs ===")
    for algo, ds, t, pid, mean_nodes, mean_depth in job_times[:10]:
        print(f"{t[0]:8.2f}s | {algo:20s} | {ds}")

    # Compute fitness based on fold_scores
    # Respect the list order
    N = len(datasets)
    fitness: list[float] = []

    for algo_genome in algorithms:
        print("Alg:", algo_genome)
        scores_per_dataset = results[str(algo_genome)]
        print(scores_per_dataset)

        # Assume best objective value for all instances is F1 = 1.0
        b_i = 1.0

        # Compute x_is
        xi_values = []
        for ds_name, scores in scores_per_dataset.items():
            # Get mean F1
            mean_f1 = np.mean(scores)
            print(ds_name, mean_f1)

            xi = 100 * abs(mean_f1 - b_i) / np.mean([mean_f1, b_i])
            xi_values.append(xi)

        # Get x-bar; i.e., the average of xi values for that individual
        x_bar = np.mean(xi_values)
        diff = [(xi - x_bar)**2 for xi in xi_values]
        print("xi values:", xi_values)
        print("x-bar:", x_bar)
        print("SDD sum:", sum(diff))

        sdd = np.sqrt( sum(diff) / (N - 1) )
        print("SDD:", sdd, "\n")
        fitness.append(-sdd)

    return fitness


def evolve_trees(datasets,
                 mu=10, lambd=10, generations=10,
                 crossover_prob=1.0, mutation_prob=0.1):

    # Generate initial population
    population = generate_population(mu)
    best_history = []

    # Evaluate initial population
    t0 = time.time()
    n_workers = os.cpu_count() - 2
    fitness_values = evaluation(population, datasets, n_workers=n_workers)

    # Best initial individual
    best_val = max(fitness_values)
    best_history.append(best_val)
    best_individual = population[np.argmax(fitness_values)]
    print(f"Gen {0}: best score = {best_val:.4f}, Execution time = {round(time.time() - t0, 4)} seconds.")
    print(f"Best Genome:", best_individual)
    #print("------------------------------")

    for gen in range(1, generations + 1):

        # Set t0
        t0 = time.time()

        # Generate offspring
        offspring = []
        for _ in range(lambd // 2):
            # Parent selection
            p1 = tournament_selection(population, fitness_values, k=2)
            p2 = tournament_selection(population, fitness_values, k=2)

            # Crossover
            c1, c2 = single_point_crossover(
                p1,
                p2,
                VAR_DOMAINS,
                prob=crossover_prob)

            # Mutation (modulo to keep genes in range)
            c1 = [random.choice(list(VAR_DOMAINS[i])) if random.random() < mutation_prob else c1[i] % len(VAR_DOMAINS[i]) for i in range(len(c1))]
            c2 = [random.choice(list(VAR_DOMAINS[i])) if random.random() < mutation_prob else c2[i] % len(VAR_DOMAINS[i]) for i in range(len(c2))]

            offspring.append(c1)
            offspring.append(c2)

        # Evaluate offsprings
        off_fitness_values = evaluation(offspring, datasets, n_workers=n_workers)

        # Combine and evaluate offspring
        candidates = population + offspring
        candidate_fitness = fitness_values + off_fitness_values

        # Update population - select best mu
        best_indices = np.argsort(candidate_fitness)[-mu:]
        population = [candidates[i] for i in best_indices]

        # Update fitness values with previous evaluations
        fitness_values = [candidate_fitness[i] for i in best_indices]

        # Best individual on this generation...
        best_val = max(fitness_values)
        best_individual = population[np.argmax(fitness_values)]
        best_history.append(best_val)
        print(f"Gen {gen}: best score = {best_val:.4f}, Execution time = {round(time.time() - t0, 4)} seconds.")
        print(f"Best Genome:", best_individual)
    print("------------------------------")

    return best_individual, best_val, best_history
