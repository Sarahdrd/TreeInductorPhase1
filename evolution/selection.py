# evolution/selection.py
import random
import numpy as np
from pymoo.core.sampling import Sampling

# Selección simple 
def tournament_selection(population, fitness, k=3):
    """Devuelve el individuo con mejor fitness de una muestra aleatoria de tamaño k."""
    selected = random.sample(list(zip(population, fitness)), k)
    return max(selected, key=lambda x: x[1])[0]

# Sampling compatible con pymoo
class ChoiceRandomSampling(Sampling):
    """
    Genera individuos cuando el problema define vars tipo Choice (pymoo.core.variable.Choice).
    """
    def _do(self, problem, n_samples, **kwargs):
        assert problem.vars is not None

        X = np.full((n_samples, len(problem.vars)), None, dtype=object)
        for i, (_, var) in enumerate(problem.vars.items()):
            X[:, i] = var.sample(n_samples)
        return X

# Dominancia para despues usar en selección multiobjetivo
def dominates(fA, fB):
    """
    fA y fB son dicts con keys: f1, recall, specificity, n_nodes.
    Dominancia: maximizar f1/recall/specificity, minimizar n_nodes.
    """
    better_or_equal = (
        (fA["f1"] >= fB["f1"]) and
        (fA["recall"] >= fB["recall"]) and
        (fA["specificity"] >= fB["specificity"]) and
        (fA["n_nodes"] <= fB["n_nodes"])
    )
    strictly_better = (
        (fA["f1"] > fB["f1"]) or
        (fA["recall"] > fB["recall"]) or
        (fA["specificity"] > fB["specificity"]) or
        (fA["n_nodes"] < fB["n_nodes"])
    )
    return better_or_equal and strictly_better

# Knee point util para resultados de pymoo 
def knee_point(res, objectives, strategy='first'):
    """
    res: resultado de pymoo.minimize con res.F (matriz objetivos) y res.X (soluciones).
    objectives: lista de nombres (solo se usa para longitud).
    strategy: 'first' o 'random' cuando hay empates.
    """
    n_obj = len(objectives)
    W = np.full(n_obj, 1.0 / n_obj)      
    U = np.dot(res.F, W)                  

    min_value = np.min(U)
    best_indices = np.where(U == min_value)[0]

    if strategy == 'random':
        idx = np.random.choice(best_indices)
    else:
        idx = best_indices[0]

    return res.X[idx], idx
