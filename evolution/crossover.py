# crossover.py
import random

def single_point_crossover(parent1, parent2, var_domains, prob=1.0):
    """
    Domain-safe 1-point crossover for genes with heterogeneous value ranges.
    Each gene value is wrapped into its valid domain via modulo correction.
    
    Args:
        parent1, parent2 : list[int]
            Parent chromosomes.
        var_domains : list[range or list]
            The valid domain for each gene index.
        prob : float
            Probability of performing crossover.
    Returns:
        child1, child2 : list[int]
            Domain-corrected offspring.
    """
    if random.random() > prob or len(parent1) != len(parent2) or len(parent1) < 2:
        # Return clones if crossover skipped or invalid lengths
        return parent1[:], parent2[:]

    # --- Perform single-point crossover ---
    cx = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cx] + parent2[cx:]
    child2 = parent2[:cx] + parent1[cx:]

    # --- Domain correction (key part) ---
    corrected_child1, corrected_child2 = [], []
    for i, domain in enumerate(var_domains):
        domain_size = len(domain)
        corrected_child1.append(child1[i] % domain_size)
        corrected_child2.append(child2[i] % domain_size)

    return corrected_child1, corrected_child2
