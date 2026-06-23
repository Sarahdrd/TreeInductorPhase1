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
from multiprocessing import Process, Queue

# Import evolutionary functions
from evolution.crossover import single_point_crossover
from evolution.mutation import random_reset_mutation
from evolution.selection import tournament_selection
from evolution.encode_decode import generate_population, decode

from core.metrics import evaluate_tree_new

from sklearn.model_selection import StratifiedKFold, KFold

# Domain
from core.DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES

# Prevent BLAS/OpenMP oversubscription (safe default).
# Put these at top-level BEFORE heavy numeric libs run in workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

DEFAULT_TRAIN_TIMEOUT_S = 15 * 60
DEFAULT_UNSEEN_TIMEOUT_S = 4 * 60

def _safe_xi(mean_f1: float, best_possible: float = 1.0) -> float:
    denom = np.mean([mean_f1, best_possible])
    if not np.isfinite(denom) or denom <= 0:
        return 200.0
    return 100.0 * abs(mean_f1 - best_possible) / denom


def _fitness_from_dataset_scores(scores_by_dataset: dict[str, list[float]]) -> tuple[float, float, list[float]]:
    """
    Fitness to maximize:
        mean(dataset_mean_f1) - std(dataset_mean_f1)

    This keeps the original generalization idea through the variability penalty,
    but also rewards trees that actually perform well. A perfect stable tree
    across datasets approaches fitness=1.0, while unstable or weak trees score lower.

    Returns:
        (fitness, f1_std, dataset_mean_f1_values)
    """
    dataset_mean_f1_values = []
    for scores in scores_by_dataset.values():
        clean_scores = [float(s) for s in scores if np.isfinite(s)]
        mean_f1 = float(np.mean(clean_scores)) if clean_scores else 0.0
        dataset_mean_f1_values.append(mean_f1)

    if not dataset_mean_f1_values:
        return -np.inf, np.inf, []

    mean_perf = float(np.mean(dataset_mean_f1_values))
    f1_std = float(np.std(dataset_mean_f1_values))

    fitness = mean_perf - f1_std
    return fitness, f1_std, dataset_mean_f1_values


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


def _repair_gene_value(value, domain):
    """
    Clamp/repair one gene value so it belongs to its allowed VAR_DOMAINS entry.

    This keeps the search space dynamic: if you add/remove criteria in DT_genes.py,
    the valid categorical ranges update automatically through VAR_DOMAINS.
    """
    # Current domains are range(...) objects, so genes should be integer indices/params.
    value = int(round(value))

    low = min(domain)
    high = max(domain)

    if value < low:
        return low
    if value > high:
        return high
    return value


def repair_gene(gene):
    """
    Repair a full genome after generation, crossover, or mutation.
    """
    if len(gene) != len(VAR_DOMAINS):
        raise ValueError(
            f"Genome length {len(gene)} does not match VAR_DOMAINS length {len(VAR_DOMAINS)}. Gene={gene}"
        )

    return [
        _repair_gene_value(value, VAR_DOMAINS[i])
        for i, value in enumerate(gene)
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

def describe_gene(gene):
    return {
        "raw": gene,
        "split_criterion": SPLIT_CRITERIA.get(gene[0], f"INVALID:{gene[0]}"),
        "stopping_criterion": STOPPING_CRITERIA.get(gene[1], f"INVALID:{gene[1]}"),
        "stopping_param": gene[2],
        "mv_split": MV_SPLIT.get(gene[3], f"INVALID:{gene[3]}"),
        "mv_distribution": MV_DISTRIBUTION.get(gene[4], f"INVALID:{gene[4]}"),
        "mv_classification": MV_CLASSIFICATION.get(gene[5], f"INVALID:{gene[5]}"),
        "pruning": PRUNNING_GENES.get(gene[6], f"INVALID:{gene[6]}"),
        "pruning_param": gene[7],
    }

def _stratified_kfold_indices(
    y: NDArray[np.float64], n_splits: int, seed: int
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Safe CV splitter.

    Uses StratifiedKFold only when every class has at least n_splits samples.
    Otherwise it reduces k when possible. If stratification is still impossible,
    it falls back to KFold. This prevents the "classes don't contain each other"
    / rare-class fold problem from crashing the gene evaluation.
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n < 2:
        raise ValueError("Dataset has fewer than 2 rows; cannot cross-validate.")

    classes, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) else 0

    if len(classes) >= 2 and min_class_count >= 2:
        safe_k = min(int(n_splits), min_class_count)
        skf = StratifiedKFold(n_splits=safe_k, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(np.zeros(n), y)]

    safe_k = min(int(n_splits), n)
    if safe_k < 2:
        raise ValueError("Dataset has too few rows for KFold.")
    kf = KFold(n_splits=safe_k, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.zeros(n))]

##
def _load_dataset_csv(path: Path, target_col=None):

    df = pd.read_csv(path)

    if target_col is None:
        target_col = df.columns[-1]

    elif isinstance(target_col, int):
        target_col = df.columns[target_col]

    elif isinstance(target_col, str):
        if target_col not in df.columns:
            print(f" '{target_col}' not found in {path.name}, using last column instead.")
            target_col = df.columns[-1]

    # Validación final
    if target_col not in df.columns:
        raise ValueError(
            f"target_col='{target_col}' not found in {path}.\n"
            f"Columns: {list(df.columns)}"
        )

    # Target
    y = LabelEncoder().fit_transform(df[target_col].astype(str))

    # Features
    X_df = df.drop(columns=[target_col])

    # Encode categóricos
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_df.to_numpy(dtype=float)

    return X, y

def _evaluate_tree_safe(tree, X_test, y_test, all_labels):
    """
    Evaluate with a fixed label set for the whole dataset.

    This avoids fold-level class mismatch issues where a validation fold or a
    prediction is missing a class that exists elsewhere in the dataset.
    """
    from sklearn.metrics import (
        f1_score,
        recall_score,
        precision_score,
        accuracy_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        multilabel_confusion_matrix,
    )

    y_pred = np.asarray(tree.predict(X_test))
    y_test = np.asarray(y_test)
    all_labels = np.asarray(all_labels)

    def macro_specificity(y_true, y_hat, labels):
        mcm = multilabel_confusion_matrix(y_true, y_hat, labels=labels)
        specs = []
        for conf in mcm:
            tn, fp, fn, tp = conf.ravel()
            specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
        return float(np.mean(specs)) if specs else 0.0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "specificity": macro_specificity(y_test, y_pred, all_labels),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
    }

# -------------------------
# Worker Job
# -------------------------


# evaular no por par de dataset-algoritmo, sino por algoritmo, evaluando ese algoritmo en todos los datasets y obteniendo un fitness global basado en su desempeño en todos los datasets.
# poner un timeout que al ser excedido se descarte ese gen completo y no se evaluen los demas datasets, para evitar que un gen con mal desempeño en un dataset se coma todo el tiempo de evaluación. 
# El timeout se pone por gen completo no por datset

def _eval_dataset_impl(
    q: Queue,
    algo_genome,
    ds_name: str,
    ds_path_str: str,
    target_col,
    n_splits: int,
    split_seed: int,
):
    """
    Child-process worker for one genome on one dataset.

    The parent process owns the timeout. This worker only performs the real
    CV evaluation and returns either ("ok", payload) or ("error", traceback).
    """
    ds_start = time.perf_counter()

    decode_time = 0.0
    fit_time = 0.0
    metrics_time = 0.0
    fold_scores = []
    depths = []
    nodes = []

    try:
        X, y = _load_dataset_csv(Path(ds_path_str), target_col=target_col)
        all_labels = np.unique(y)
        splits = _stratified_kfold_indices(y, n_splits=n_splits, seed=split_seed)

        cv_warning = None
        if len(splits) != n_splits:
            cv_warning = (
                f"{ds_name}: requested {n_splits} folds, using {len(splits)} "
                f"because the rarest class has too few samples."
            )

        for fold_idx, (tr, va) in enumerate(splits, start=1):
            t0 = time.perf_counter()
            algo = decode(algo_genome)
            decode_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            algo.fit(X[tr], y[tr])
            fit_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            metrics = _evaluate_tree_safe(algo, X[va], y[va], all_labels)
            metrics_time += time.perf_counter() - t0

            fold_scores.append(float(metrics["f1"]))
            depths.append(float(getattr(algo, "_depth", np.nan)))
            nodes.append(float(getattr(algo, "_node", np.nan)))

        total_time = time.perf_counter() - ds_start

        q.put((
            "ok",
            {
                "ds_name": ds_name,
                "scores": fold_scores,
                "times": {
                    "total": total_time,
                    "decode": decode_time,
                    "fit": fit_time,
                    "metrics": metrics_time,
                },
                "structure": {
                    "depth": float(np.nanmean(depths)) if depths else np.nan,
                    "nodes": float(np.nanmean(nodes)) if nodes else np.nan,
                },
                "cv_warning": cv_warning,
            },
        ))

    except Exception:
        q.put((
            "error",
            {
                "ds_name": ds_name,
                "reason": traceback.format_exc(),
                "times": {
                    "total": time.perf_counter() - ds_start,
                    "decode": decode_time,
                    "fit": fit_time,
                    "metrics": metrics_time,
                },
            },
        ))


def _eval_one_dataset_with_timeout(
    algo_genome,
    ds_name: str,
    ds_path_str: str,
    target_col,
    n_splits: int,
    split_seed: int,
    timeout_s: int | float | None,
    bad_score: float = 0.0,
):
    """
    Cross-platform hard timeout for one genome/dataset pair.

    This avoids SIGALRM and works on Windows because the evaluation happens in
    a separate process. If the process is still alive after timeout_s, it is
    terminated and that gene is marked infeasible by the caller.
    """
    timeout_s = None if timeout_s is None else float(timeout_s)

    q = Queue()
    p = Process(
        target=_eval_dataset_impl,
        args=(q, algo_genome, ds_name, ds_path_str, target_col, n_splits, split_seed),
    )

    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        try:
            p.close()
        except Exception:
            pass

        return {
            "ds_name": ds_name,
            "scores": [bad_score] * max(1, int(n_splits)),
            "times": {
                "total": float(timeout_s) if timeout_s is not None else 0.0,
                "decode": 0.0,
                "fit": float(timeout_s) if timeout_s is not None else 0.0,
                "metrics": 0.0,
            },
            "structure": {"depth": np.nan, "nodes": np.nan},
            "error": "timeout",
            "cv_warning": None,
        }

    if q.empty():
        return {
            "ds_name": ds_name,
            "scores": [bad_score] * max(1, int(n_splits)),
            "times": {"total": 0.0, "decode": 0.0, "fit": 0.0, "metrics": 0.0},
            "structure": {"depth": np.nan, "nodes": np.nan},
            "error": "worker_died_without_result",
            "cv_warning": None,
        }

    status, payload = q.get()

    if status == "ok":
        payload["error"] = None
        return payload

    return {
        "ds_name": ds_name,
        "scores": [bad_score] * max(1, int(n_splits)),
        "times": payload.get("times", {"total": 0.0, "decode": 0.0, "fit": 0.0, "metrics": 0.0}),
        "structure": {"depth": np.nan, "nodes": np.nan},
        "error": payload.get("reason", "unknown worker error"),
        "cv_warning": None,
    }


def _eval_one_algo_all_datasets(
    algo_genome,
    datasets_info,
    n_splits,
    dataset_split_seeds,
    bad_score=0.0,
    timeout_s: int | float | None = DEFAULT_TRAIN_TIMEOUT_S,
):
    """
    Evaluate one genome across all datasets.

    A dataset evaluation is killed if it exceeds timeout_s. The genome is then
    marked infeasible and remaining datasets are penalized with bad_score.
    """
    pid = os.getpid()

    all_scores = {}
    dataset_times = {}
    dataset_structures = {}

    infeasible = False
    failed_dataset = None
    failure_reason = None

    dataset_names = [ds_name for ds_name, _, _ in datasets_info]

    for ds_name, ds_path_str, target_col in datasets_info:
        result = _eval_one_dataset_with_timeout(
            algo_genome=algo_genome,
            ds_name=ds_name,
            ds_path_str=ds_path_str,
            target_col=target_col,
            n_splits=n_splits,
            split_seed=dataset_split_seeds[ds_name],
            timeout_s=timeout_s,
            bad_score=bad_score,
        )

        all_scores[ds_name] = result["scores"]
        dataset_times[ds_name] = result["times"]
        dataset_structures[ds_name] = result["structure"]

        if result.get("cv_warning"):
            print(f"[CV warning] {result['cv_warning']}")

        if result["error"] is not None:
            infeasible = True
            failed_dataset = ds_name
            failure_reason = result["error"]
            break

    # Penalize remaining datasets not evaluated.
    for ds_name in dataset_names:
        if ds_name not in all_scores:
            all_scores[ds_name] = [bad_score] * max(1, int(n_splits))
            dataset_times[ds_name] = {"total": 0.0, "decode": 0.0, "fit": 0.0, "metrics": 0.0}
            dataset_structures[ds_name] = {"depth": np.nan, "nodes": np.nan}

    return {
        "algo": algo_genome,
        "results": all_scores,
        "times": dataset_times,
        "structures": dataset_structures,
        "infeasible": infeasible,
        "failed_dataset": failed_dataset,
        "failure_reason": failure_reason,
        "pid": pid,
    }

# -------------------------
# Public API
# -------------------------



def evaluation(
    algorithms: list[list[int]],
    datasets: dict[str, Path],
    *,
    target_cols=None,
    metric_name: str = "F1",
    n_splits: int = 5,
    global_seed: int = 42,
    n_workers: int | None = None,
    bad_score: float = 0.0,
    verbose: bool = True,
    timeout_s: int | float | None = DEFAULT_TRAIN_TIMEOUT_S,
) -> list[float]:
    """
    Parallel evaluation over genomes.

    Returns:
        fitness list in the same order as `algorithms`.

    A genome is infeasible only when fitting/evaluation raises an exception;
    infeasible genomes receive -1e9.
    """

    dataset_split_seeds = {
        ds_name: (abs(hash((global_seed, ds_name))) % (2**31 - 1))
        for ds_name in datasets.keys()
    }

    datasets_info = []
    for ds_name, ds_path in datasets.items():
        target_col = None if target_cols is None else target_cols.get(ds_name, None)
        datasets_info.append((ds_name, str(ds_path), target_col))

    results_by_gene = {}
    total = len(algorithms)
    done = 0

    if verbose:
        print(f"\n[Evaluation] {total} genes | {len(datasets_info)} datasets | "
              f"{n_splits} folds | workers={n_workers} | timeout={timeout_s}s")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_gene = {
            ex.submit(
                _eval_one_algo_all_datasets,
                algo,
                datasets_info,
                n_splits,
                dataset_split_seeds,
                bad_score,
                timeout_s,
            ): algo
            for algo in algorithms
        }

        for fut in as_completed(future_to_gene):
            algo = future_to_gene[fut]
            algo_name = str(algo)
            done += 1

            try:
                out = fut.result()
            except Exception as e:
                out = {
                    "algo": algo,
                    "results": {ds_name: [bad_score] * n_splits for ds_name, _, _ in datasets_info},
                    "times": {ds_name: {"total": 0.0, "decode": 0.0, "fit": 0.0, "metrics": 0.0}
                              for ds_name, _, _ in datasets_info},
                    "structures": {ds_name: {"depth": np.nan, "nodes": np.nan}
                                   for ds_name, _, _ in datasets_info},
                    "infeasible": True,
                    "failed_dataset": None,
                    "failure_reason": f"WorkerError: {type(e).__name__}: {e}",
                    "pid": None,
                }

            results_by_gene[algo_name] = out

            fitness_value, f1_std, dataset_mean_f1_values = _fitness_from_dataset_scores(out["results"])
            total_gene_time = sum(v["total"] for v in out["times"].values())
            status = "[INFEASIBLE]" if out["infeasible"] else "[OK]"

            if verbose:
                print(f"\n[Progress] {done}/{total} {status} gene={algo_name} "
                      f"fitness={fitness_value:.4f} mean_f1={float(np.mean(dataset_mean_f1_values)) if dataset_mean_f1_values else 0.0:.4f} f1_std={f1_std:.4f} total={total_gene_time:.2f}s pid={out['pid']}")

                if out["failed_dataset"] is not None:
                    print(f"  stopped_at={out['failed_dataset']} | reason={out['failure_reason']}")

                dataset_means = [
                    float(np.mean(out["results"][ds_name]))
                    for ds_name, _, _ in datasets_info
                ]
                print(
                    f"  dataset mean F1 | best={max(dataset_means):.4f} "
                    f"worst={min(dataset_means):.4f} avg={float(np.mean(dataset_means)):.4f}"
                )
                print(f"  dataset mean F1 values={np.round(dataset_mean_f1_values, 4).tolist()}")

                for ds_name, _, _ in datasets_info:
                    t = out["times"][ds_name]
                    s = out["structures"][ds_name]
                    scores = out["results"][ds_name]
                    mean_score = float(np.mean(scores)) if len(scores) else float("nan")
                    best_score = float(np.max(scores)) if len(scores) else float("nan")
                    worst_score = float(np.min(scores)) if len(scores) else float("nan")

                    print(
                        f"  - {ds_name:20s} F1 mean={mean_score:.4f} "
                        f"best={best_score:.4f} worst={worst_score:.4f} "
                        f"folds={np.round(scores, 4).tolist()}"
                    )
                    print(
                        f"    time total={t['total']:.2f}s decode={t['decode']:.2f}s "
                        f"fit={t['fit']:.2f}s metrics={t['metrics']:.2f}s | "
                        f"depth={s['depth']:.2f} nodes={s['nodes']:.2f}"
                    )

    fitness = []
    for algo in algorithms:
        out = results_by_gene[str(algo)]
        if out["infeasible"]:
            fitness.append(-1e9)
        else:
            fitness_value, _, _ = _fitness_from_dataset_scores(out["results"])
            fitness.append(fitness_value)

    return fitness


def evaluate_on_unseen(
    gene,
    unseen_datasets,
    k=5,
    seed=123,
    target_cols=None,
    timeout_s: int | float | None = DEFAULT_UNSEEN_TIMEOUT_S,
):
    """
    Evaluate a selected gene on unseen datasets with:
    - rare-class-safe CV
    - fixed label space for metrics
    - cross-platform per-dataset timeout
    """
    results = {}

    split_seeds = {
        name: (abs(hash((seed, name))) % (2**31 - 1))
        for name in unseen_datasets.keys()
    }

    for name, dataset_path in unseen_datasets.items():
        target_col = None if target_cols is None else target_cols.get(name, None)

        result = _eval_one_dataset_with_timeout(
            algo_genome=gene,
            ds_name=name,
            ds_path_str=str(dataset_path),
            target_col=target_col,
            n_splits=k,
            split_seed=split_seeds[name],
            timeout_s=timeout_s,
            bad_score=0.0,
        )

        if result.get("cv_warning"):
            print(f"[CV warning][unseen] {result['cv_warning']}")

        if result["error"] == "timeout":
            print(f"[TIMEOUT][unseen] gene={gene} dataset={name} limit={timeout_s}s")
            results[name] = np.nan
            continue

        if result["error"] is not None:
            print(f"[WARN][unseen] {name}: evaluation failed for gene {gene}.")
            print(result["error"])
            results[name] = np.nan
            continue

        scores = result["scores"]
        results[name] = float(np.mean(scores)) if scores else np.nan
        t = result["times"]
        s = result["structure"]
        print(
            f"[DONE][unseen] gene={gene} dataset={name} "
            f"mean_f1={results[name]:.4f} total={t['total']:.2f}s "
            f"depth={s['depth']:.2f} nodes={s['nodes']:.2f}"
        )

    return results




def evolve_trees(datasets,
                 unseen_datasets,
                 target_cols=None,
                 mu=10,
                 lambd=10,
                 generations=10,
                 crossover_prob=1.0,
                 mutation_prob=0.1,
                 train_timeout_s=DEFAULT_TRAIN_TIMEOUT_S,
                 unseen_timeout_s=DEFAULT_UNSEEN_TIMEOUT_S):
    


    # Generate initial population
    population = [repair_gene(gene) for gene in generate_population(mu)]
    best_history = []
    fitness_rank_history = {
        "best": [],
        "median": [],
        "worst": []}

    generalization_history = {
        "best": [],
        "median": [],
        "worst": [] }
    
    generalization_sdd_history = {
        "best": [],
        "median": [],
        "worst": []
        }

    # Evaluate initial population
    t0 = time.time()
    n_workers = max(1, (os.cpu_count() or 1) - 2)
    fitness_values = evaluation(
        population,
        datasets,
        target_cols=target_cols,
        n_workers=n_workers,
        timeout_s=train_timeout_s,)


    # Best initial individual
    best_val = max(fitness_values)
    best_history.append(best_val)
    best_individual = population[np.argmax(fitness_values)]
    print(f"Gen {0}: best score = {best_val:.4f}, Execution time = {round(time.time() - t0, 4)} seconds.")
    print(f"Best Genome:", best_individual)

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

            # Repair after crossover, then mutate inside the same valid domains.
            c1 = repair_gene(c1)
            c2 = repair_gene(c2)

            c1 = [
                random.choice(list(VAR_DOMAINS[i])) if random.random() < mutation_prob
                else c1[i]
                for i in range(len(c1))
            ]
            c2 = [
                random.choice(list(VAR_DOMAINS[i])) if random.random() < mutation_prob
                else c2[i]
                for i in range(len(c2))
            ]

            # Final repair safety net in case mutation/crossover behavior changes later.
            c1 = repair_gene(c1)
            c2 = repair_gene(c2)

            offspring.append(c1)
            offspring.append(c2)

        # Evaluate offsprings
        off_fitness_values = evaluation(
            offspring,
            datasets,
            target_cols=target_cols,
            n_workers=n_workers,
            timeout_s=train_timeout_s,
            )
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

        sorted_idx = np.argsort(fitness_values)

        worst_idx = sorted_idx[0]
        median_idx = sorted_idx[len(sorted_idx)//2]
        best_idx = sorted_idx[-1]

        worst_gene = population[worst_idx]
        median_gene = population[median_idx]
        best_gene = population[best_idx]

        # store SDD fitness
        fitness_rank_history["best"].append(fitness_values[best_idx])
        fitness_rank_history["median"].append(fitness_values[median_idx])
        fitness_rank_history["worst"].append(fitness_values[worst_idx])

        # evaluate generalization
        gen_best = evaluate_on_unseen(best_gene, unseen_datasets, target_cols=target_cols, timeout_s=unseen_timeout_s)
        gen_med  = evaluate_on_unseen(median_gene, unseen_datasets, target_cols=target_cols, timeout_s=unseen_timeout_s)
        gen_worst = evaluate_on_unseen(worst_gene, unseen_datasets, target_cols=target_cols, timeout_s=unseen_timeout_s)



        generalization_history["best"].append(np.mean(list(gen_best.values())))
        generalization_history["median"].append(np.mean(list(gen_med.values())))
        generalization_history["worst"].append(np.mean(list(gen_worst.values())))


        def compute_sdd(scores_dict):
            values = [float(v) for v in scores_dict.values() if np.isfinite(v)]
            if len(values) <= 1:
                return 0.0

            xi_values = [_safe_xi(mean_f1, 1.0) for mean_f1 in values]
            x_bar = np.mean(xi_values)
            diff = [(xi - x_bar) ** 2 for xi in xi_values]

            return float(np.sqrt(sum(diff) / (len(xi_values) - 1)))

        best_sdd_unseen = compute_sdd(gen_best)
        med_sdd_unseen  = compute_sdd(gen_med)
        worst_sdd_unseen = compute_sdd(gen_worst)

        generalization_sdd_history["best"].append(-best_sdd_unseen)
        generalization_sdd_history["median"].append(-med_sdd_unseen)
        generalization_sdd_history["worst"].append(-worst_sdd_unseen)

        print(f"Gen {gen}: best score = {best_val:.4f}, Execution time = {round(time.time() - t0, 4)} seconds.")
        print(f"Best Genome:", best_individual)

    return (
        best_individual,
        best_val,
        best_history,
        fitness_rank_history,
        generalization_history, 
        generalization_sdd_history)