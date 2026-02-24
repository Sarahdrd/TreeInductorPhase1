# genetic_DT_GA.py
import random
import numpy as np
from evolution.crossover import single_point_crossover
from evolution.mutation import random_reset_mutation
from evolution.selection import tournament_selection
from evolution.encode_decode import generate_population, decode
from core.metrics import evaluate_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from core.DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES
# from evolution.encode_decode import VAR_DOMAINS
from evolution.crossover import single_point_crossover
from sklearn.model_selection import StratifiedKFold




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

# Old fitnees function, i swapped for the SDD :*
#def fitness(individual, X_train, X_test, y_train, y_test):
#    """Decode the individual into a tree and return its F1 score."""
#    try:
#        tree = decode(individual)
#        metrics = evaluate_tree(tree, X_train, y_train, X_test, y_test)
#        return metrics["f1"]  # you can swap for recall/specificity
#    except Exception as e:
#        print(f" Error evaluating individual {individual}: {e}")
#        return 0  # penalize invalid configurations


# New and improved fitness function using SDD 
# def fitness(individual, X, y, k_folds=5):
#    """
#    Fitness guided by Standard Deviation of Differences (SDD).
#    Promotes configurations that generalize across folds.
#    """
#    try:
#        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

#        scores = []

        # Evaluate individual on multiple folds to get a distribution of performance
#        for train_idx, test_idx in skf.split(X, y):
#            X_train, X_test = X[train_idx], X[test_idx]
#            y_train, y_test = y[train_idx], y[test_idx]

#            tree = decode(individual)
#            metrics = evaluate_tree(tree, X_train, y_train, X_test, y_test)

#            scores.append(metrics["f1"])

#        scores = np.array(scores)

        # Compute "best" per fold 
        # For now approximate best = max score achieved by this individual
        # maybe swap for population-best or baseline ill see
#        best = np.max(scores)

        # Percentage differences xi 
#        xi = np.abs(scores - best) / ((scores + best) / 2 + 1e-8)

        # SDD 
#        SDD = np.std(xi)

#        mean_perf = np.mean(scores)

        # final fitness
#        fitness_value = mean_perf - SDD

#        return fitness_value
#    except Exception as e:
#        print(f" Error evaluating individual {individual}: {e}")
#        return 0


def _mean_f1_on_dataset(individual, X, y, k_folds=5, seed=42):
    """
    Funcion auxiliar para evaular el o_j 
    Ósea, performance del individuo en un dataset dado,
    usando k-fold cross validation y devolviendo el mean F1 de mujer
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for tr, te in skf.split(X, y):
        tree = decode(individual)
        metrics = evaluate_tree(tree, X[tr], y[tr], X[te], y[te])
        fold_scores.append(metrics["f1"])

    return float(np.mean(fold_scores))




# Even newer fitness function, now SDD approach: mean - std of scores across datasets

def fitness(individual, datasets, k_folds=5):
    """
    datasets: list of tuples (X, y, name) or (X, y)
    Returns: fitness to maximize, based on mean performance across datasets
             penalized by SDD across datasets.
    """
    try:
        perf = []
        for item in datasets:
            if len(item) == 3:
                X, y, _ = item
            else:
                X, y = item

            perf.append(_mean_f1_on_dataset(individual, X, y, k_folds=k_folds))

        perf = np.array(perf)

        # Define "best" per instance set 
        best = np.max(perf)

        # percentage diffs 
        xi = np.abs(perf - best) / ((perf + best) / 2 + 1e-8)

        SDD = float(np.std(xi))
        mean_perf = float(np.mean(perf))

        return mean_perf - SDD

    except Exception as e:
        print(f" Error evaluating individual {individual}: {e}")
        return 0







# ---- SIMPLE GENETIC ALGORITHM ----
def genetic_DT_GA(datasets,
                  mu=10, lambd=20, generations=20,
                  crossover_prob=1.0, mutation_prob=0.1):
    
    # Generate initial population
    population = generate_population(mu)
    best_history = []

    # Evaluate initial population
    fitness_values = [fitness(ind, datasets) for ind in population]

    for gen in range(generations):
        
        # Best individual on this generation...
        best_val = max(fitness_values)
        best_ind = population[np.argmax(fitness_values)]
        best_history.append(best_val)
        print(f"Gen {gen}: best score = {best_val:.4f}")

        # Generate offspring
        offspring = []
        for _ in range(lambd // 2):
            # Parent selection
            p1 = tournament_selection(population, fitness_values, k=3)
            p2 = tournament_selection(population, fitness_values, k=3)

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

        # Combine and evaluate offspring
        candidates = population + offspring
        candidate_fitness = fitness_values + [fitness(ind, datasets) for ind in offspring]

        # Update population - select best mu
        best_indices = np.argsort(candidate_fitness)[-mu:]
        population = [candidates[i] for i in best_indices]

        # Update fitness values with previous evaluations
        fitness_values = [candidate_fitness[i] for i in best_indices]

    best_fitness = max(candidate_fitness)
    best_individual = candidates[np.argmax(candidate_fitness)]
    return best_individual, best_fitness, best_history


# ---- DEMO ----
if __name__ == "__main__":
    print(" Running Simple Genetic Algorithm for Decision Trees 🌳")

    # Load a simple dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Run the GA
    best_ind, best_fit, history = genetic_DT_GA(X_train, X_test, y_train, y_test,
                                                mu=8, lambd=16, generations=10)

    print("\n Best Individual:", best_ind)
    print(" Best F1 Score:", best_fit)
    print(" Fitness history:", history)
