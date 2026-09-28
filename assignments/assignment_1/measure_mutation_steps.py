"""Measure how large each mutation operator actually is.

Why: variant A and variant B differ only in how often the point and the subtree
operator are used. That only manipulates the search if the two operators really
produce steps of a different size. This script measures that directly.

Method: bodies are sampled from the databases your runs already wrote. Each
sampled body is mutated once with each operator, and the step size is the
weighted tree edit distance between the parent body and the mutated child. This
is the same distance the fitness function uses, so a step size of 1.0 means
"one module edit away".

Usage (from the repository root, after the runs have finished):
    uv run assignments/assignment_1/measure_mutation_steps.py
    uv run assignments/assignment_1/measure_mutation_steps.py --samples 1000

Nothing here changes the experiment. It only reads the databases.
"""

import argparse
import copy
import csv
import json
import random
import sqlite3
from pathlib import Path

import numpy as np

from ariel.ec.genotypes.tree.operators import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from tree_edit_distance import tree_edit_distance

parser = argparse.ArgumentParser(description="Measure mutation step sizes")
parser.add_argument("--data", type=str, default="__data__/group24_algorithm",
                    help="Folder with the .db files of the runs")
parser.add_argument("--samples", type=int, default=500,
                    help="Number of bodies sampled per database")
parser.add_argument("--max-modules", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
rng = np.random.default_rng(args.seed)

OPERATORS = {
    "point": lambda g: mutate_replace_node(g),
    "subtree": lambda g: mutate_subtree_replacement(g, max_modules=args.max_modules),
    "shrink": lambda g: mutate_shrink(g),
    "hoist": lambda g: mutate_hoist(g),
}


def sample_genotypes(db_path: Path, n: int) -> list[dict]:
    """Return up to n genotypes sampled from one run database."""
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT genotype_ FROM individual WHERE genotype_ IS NOT NULL").fetchall()
    con.close()
    if not rows:
        return []
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [json.loads(rows[i][0]) for i in idx]


def main() -> None:
    data = Path(args.data)
    databases = sorted(data.glob("*_seed*.db"))
    if not databases:
        raise SystemExit(f"No .db files found in {data.resolve()}")

    records: list[tuple[str, str, int, float]] = []
    for db_path in databases:
        variant = db_path.name.split("_seed")[0]
        for genotype in sample_genotypes(db_path, args.samples):
            parent = TreeGenome.from_dict(genotype)
            parent_graph = parent.to_networkx()
            for name, operator in OPERATORS.items():
                child = copy.deepcopy(parent)
                operator(child)
                try:
                    step = tree_edit_distance(parent_graph, child.to_networkx())
                except ValueError:
                    continue  # mutation produced an invalid tree, skip
                records.append((variant, name, len(parent.nodes), step))

    out_csv = data / "mutation_step_sizes.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "operator", "parent_size", "step_size"])
        writer.writerows(records)

    print(f"{len(records)} measurements written to {out_csv}\n")
    print(f"{'operator':<10}{'mean':>8}{'median':>8}{'p90':>8}{'max':>8}{'no-change':>11}{'n':>8}")
    for name in OPERATORS:
        steps = np.array([r[3] for r in records if r[1] == name])
        if steps.size:
            print(f"{name:<10}{steps.mean():>8.2f}{np.median(steps):>8.2f}"
                  f"{np.percentile(steps, 90):>8.2f}{steps.max():>8.2f}"
                  f"{(steps == 0).mean():>10.1%}{steps.size:>8}")
    print("\nReport these numbers in the Methods or Discussion: they show whether "
          "the point and subtree operators differ in step size at all.")


if __name__ == "__main__":
    main()