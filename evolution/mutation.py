# mutation.py
import random

def random_reset_mutation(individual, var_domains, prob=0.1):
    """
    individual: lista de genes (ints)
    var_domains: lista de iterables con las opciones válidas por gen
                 ej: [range(13), range(5), range(101), ...]
    """
    mutant = individual[:]
    for i, domain in enumerate(var_domains):
        if random.random() < prob:
            mutant[i] = random.choice(list(domain))
    return mutant
