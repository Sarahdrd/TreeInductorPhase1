# encode_decode.py

import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from core.DecisionTreeDecoder import DecisionTree
from core.DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES


def generate_individual():
    """
    Generates a random list of 9 genes representing the configuration of an evolutionary decision tree.
    Each gene is an integer within a defined range corresponding to a specific component of the decision tree.
    """
    gene_bounds = [
        (0, 12),   # split criterion              
        (0, 4),    # stopping criterion            
        (0, 100),  # stopping value                
        (0, 3),    # missing value split           
        (0, 6),    # missing value distribution    
        (0, 1),    # missing value classification  
        (0, 5),    # prunning strategy             
        (0, 100)   # pruning parameter
    ]
    return [random.randint(low, high) for (low, high) in gene_bounds]

# [split, stopping_crit, stopping_param, mv_split, mv_dis, mv_classif, pruning_tech, pruning_param]

def generate_population(n):
    """
    Generate a population of n individuals (lists of 9 genes each).
    """
    return [generate_individual() for _ in range(n)]

def decode(gene_list):
    from core.SplitCriteria import SplitCriterion
    from core.StoppingCriteria import StoppingCriterion
    from core.MissingValues import MissingValues
    from core.PruningStrategies import Pruning as Pruning
    from core.DT_genes import (
        SPLIT_CRITERIA, STOPPING_CRITERIA,
        MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES
    )
    from core.DecisionTreeDecoder import DecisionTree

    split_criterion = SplitCriterion(SPLIT_CRITERIA[gene_list[0]])

    stopping = StoppingCriterion(
        n_tot_samples=1000,
        criterion=STOPPING_CRITERIA[gene_list[1]],
        param=gene_list[2]
    )

    mv = MissingValues(
        mv_split=MV_SPLIT[gene_list[3]],
        mv_distrib=MV_DISTRIBUTION[gene_list[4]],
        mv_classif=MV_CLASSIFICATION[gene_list[5]],
        split_criterion=split_criterion
    )

    pruning = Pruning(
        method=PRUNNING_GENES[gene_list[6]],
        param=gene_list[7]
    )

    # Create the tree with 4 positional args (no keywords)
    return DecisionTree(split_criterion, stopping, mv, pruning)

