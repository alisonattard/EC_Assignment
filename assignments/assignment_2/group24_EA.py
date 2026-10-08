"""EA for Assignment 2: evolves the weights of a neural-network controller.

Variants: A = mutation only, B = blend crossover + mutation,
R = random search baseline. Run with --variant, --seed, --pop, --generations.
"""
import argparse
from pathlib import Path
import random

from group24_robot import run_experiment, calculate_num_weights
import numpy as np

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
    config,
    FloatMutator,
    set_seed
)

install()
console = Console()

parser = argparse.ArgumentParser(
    description="EA",
)
parser.add_argument(
    "--generations",
    type=int,
    default=50,
    help="Number of generations",
)
parser.add_argument(
    "--pop", 
    type=int, 
    default=100, 
    help="Population size"
)
parser.add_argument(
    "--variant",
    choices=["A", "B", "R"],
    required=True,
    help="Choose between 'A' (mutation), 'B' (blend crossover + mutation), or 'R' (random search baseline)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
args = parser.parse_args()

# Constants
GENERATIONS: int = args.generations
POP_SIZE: int = args.pop
VARIANT: str = args.variant

# Ensure that the population size is divisible by 4 because blend crossover works in pairs of parents, and half the population becomes parents.
if POP_SIZE % 4 != 0:
    raise ValueError("Population size must be divisible by 4.")

SEED = args.seed
RNG = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

NUM_WEIGHTS = calculate_num_weights()

# Constants for mutation and crossover, can be adjusted for experimentation
MUTATION_PROBABILITY = 0.2  # Probability of mutating each gene
SIGMA = 0.1  # Standard deviation for Gaussian mutation
ALPHA = 0.5  # Blend crossover parameter (standard value is 0.5)


# ── Individual factory ────────────────────────────────────────────────────────

def make_individual() -> Individual:
    ind = Individual()
    ind.genotype = RNG.normal(0, 0.5, NUM_WEIGHTS).tolist()
    return ind


# ── EA steps ─────────────────────────────────────────────────────────────────

NAN_FITNESS = 1e6

def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        fitness = run_experiment(ind.genotype, mode="simple")
        if not np.isfinite(fitness):          # vangt NaN én inf
            console.log(f"[red]Ongeldige fitness ({fitness}) voor individu {ind.id}")
            ind.tags = {"invalid_eval": True}
            fitness = NAN_FITNESS
        ind.fitness = fitness
    return population

def parent_selection(population: Population) -> Population:
    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ <= ind_b.fitness_:
                ind_a.tags = {**ind_a.tags, "selected": True}
                ind_b.tags = {**ind_b.tags, "selected": False}
            else:
                ind_a.tags = {**ind_a.tags, "selected": False}
                ind_b.tags = {**ind_b.tags, "selected": True}

    return shuffled

# For variant A
def make_copies(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for parent in parents:
        child = Individual()
        child.genotype = parent.genotype.copy()
        child.tags = {"mutate": True, "parent_fitness": [parent.fitness]}
        population.append(child)
    return population

# For variant B
def blend_crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        x = np.array(p_a.genotype)
        y = np.array(p_b.genotype)

        lower_weight_value = np.minimum(x, y)
        higher_weight_value = np.maximum(x, y)
        gap = higher_weight_value - lower_weight_value
        interval_lower_bound = lower_weight_value - ALPHA * gap
        interval_upper_bound = higher_weight_value + ALPHA * gap

   
        child_a = Individual()
        child_a.genotype = RNG.uniform(interval_lower_bound, interval_upper_bound).tolist()
        child_a.tags = {"mutate": True, "parent_fitness": [p_a.fitness, p_b.fitness]} 

        child_b = Individual()
        child_b.genotype = RNG.uniform(interval_lower_bound, interval_upper_bound).tolist()
        child_b.tags = {"mutate": True, "parent_fitness": [p_a.fitness, p_b.fitness]} 

        population.extend([child_a, child_b])
    return population

# For variant R
def make_random_children(population: Population) -> Population:
    for _ in range(len(population.where(lambda ind: bool(ind.tags.get("selected", False))))):
        child = make_individual()
        population.append(child)
    return population

def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        ind.genotype = FloatMutator.gaussian(
            individual=ind.genotype,
            std = SIGMA,
            mutation_probability= MUTATION_PROBABILITY,
        )
        ind.tags = {**ind.tags, "mutate": False}
        ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if (ind_a.fitness_ or float('inf')) <= (ind_b.fitness_ or float('inf')):
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # survivor_selection culls down to this; keep it in line with the seed size.
    config.target_population_size = POP_SIZE
    # Set the optimization goal to minimization since we want to minimize the distance to the target.
    config.is_maximisation = False  

    if VARIANT == "A":
        make_children = make_copies
    elif VARIANT == "B":
        make_children = blend_crossover
    elif VARIANT == "R":
        make_children = make_random_children
    else:  
        raise ValueError(f"Unknown variant: {VARIANT}")

    initial = Population([make_individual() for _ in range(POP_SIZE)])
    initial = evaluate(initial)
    print(f"Variant {VARIANT} with {POP_SIZE} individuals, {GENERATIONS} generations, mutation probability {MUTATION_PROBABILITY}, and sigma {SIGMA}")
    print("Best at fitness start: ", min(ind.fitness_ for ind in initial))

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(make_children),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    run_name = f"variant{VARIANT}_pop{POP_SIZE}_gen{GENERATIONS}_seed{SEED}"
    db_path = Path("__data__") / "assignment2_ea_results" / f"{run_name}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ea = EA(initial, ops, num_steps=GENERATIONS, db_file_path=db_path)
    ea.run()

    best = ea.get_solution("best", only_alive=False)
    print("Best at fitness end: ", best.fitness_)


if __name__ == "__main__":
    main()
