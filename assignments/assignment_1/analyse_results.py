"""Analyse and plot variant A, B and baseline runs.

Assumes the experiment outputs already stored in the databases:

- variantA_seed{seed}.db
- variantB_seed{seed}.db
- baseline_random_search_seed{seed}.csv

Therefore, the experiments should be run separately before running this script.

It summarises convergence speed and population diversity for all 5 seeds and writes PNG plots to the output directory."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from tree_edit_distance import tree_edit_distance


def parse_genotype(raw: str | None) -> Any:
    """Parse a genotype stored as JSON text in SQLite."""
    if raw is None or raw == "":
        raise ValueError("missing genotype")
    return json.loads(raw)


def db_run(db_path: Path) -> dict[int, dict[str, float | list[float]]]:
    """Return generation-wise summaries for one DB run."""
    print(f"  Loading run: {db_path.name}", flush=True)
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as connection:
        for birth, death, fitness, genotype, alive in connection.execute(
            "SELECT time_of_birth, time_of_death, fitness_, genotype_, alive FROM individual"
        ):
            if genotype in (None, ""):
                continue
            try:
                graph = TreeGenome.from_dict(parse_genotype(genotype)).to_networkx()
            except Exception:
                continue
            rows.append(
                {
                    "birth": int(birth),
                    "death": int(death),
                    "fitness": float(fitness) if fitness is not None else np.nan,
                    "graph": graph,
                    "alive": bool(alive),
                }
            )

    if not rows:
        return {}

    max_gen = max(row["birth"] for row in rows)
    gen_to_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for g in range(int(row["birth"]), int(row["death"]) + 1):
            if row["alive"] or g < int(row["death"]):
                gen_to_rows[g].append(row)
            

    out: dict[int, dict[str, float | list[float]]] = {}
    for gen in range(max_gen + 1):
        members = gen_to_rows.get(gen, [])
        if not members:
            continue
        fits = [float(m["fitness"]) for m in members if np.isfinite(m["fitness"])]
        if not fits:
            continue

        pairwise: list[float] = []
        n = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                d = tree_edit_distance(members[i]["graph"], members[j]["graph"])
                pairwise.append(float(d))

        out[gen] = {
            "best": float(min(fits)),
            "mean": float(np.mean(fits)),
            "diversity": float(np.mean(pairwise)) if pairwise else 0.0,
        }

    return dict(sorted(out.items()))


def load_baseline_csv(csv_path: Path) -> dict[int, float]:
    """Load the best-so-far fitness curve written by the random-search baseline."""
    values: dict[int, float] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = int(row["generation"])
            values[gen] = float(row["best_fitness"])
    return dict(sorted(values.items()))


def summarise_runs(
    run_map: dict[str, list[dict[int, dict[str, float | list[float]]]]]
) -> dict[str, dict[int, dict[str, float]]]:
    """Aggregate per-generation best/mean/diversity across seeds for each variant."""
    summary: dict[str, dict[int, dict[str, float]]] = {}
    for variant, series_list in run_map.items():
        generations = sorted({g for series in series_list for g in series})
        variant_summary: dict[int, dict[str, float]] = {}
        for g in generations:
            values_best = [float(series[g]["best"]) for series in series_list if g in series]
            values_mean = [float(series[g]["mean"]) for series in series_list if g in series]
            values_diversity = [float(series[g]["diversity"]) for series in series_list if g in series]
            variant_summary[g] = {
                "best_median": float(np.median(values_best)) if values_best else np.nan,
                "best_q25": float(np.percentile(values_best, 25)) if values_best else np.nan,
                "best_q75": float(np.percentile(values_best, 75)) if values_best else np.nan,
                "mean_median": float(np.median(values_mean)) if values_mean else np.nan,
                "diversity_median": float(np.median(values_diversity)) if values_diversity else np.nan,
                "diversity_q25": float(np.percentile(values_diversity, 25)) if values_diversity else np.nan,
                "diversity_q75": float(np.percentile(values_diversity, 75)) if values_diversity else np.nan,
            }
        summary[variant] = variant_summary
    return summary


def plot_convergence(summary: dict[str, dict[int, dict[str, float]]], out_dir: Path) -> None:
    """Plot median convergence curves with IQR envelopes."""
    plt.figure(figsize=(8, 5))
    for label, curve in summary.items():
        gens = sorted(curve)
        best = [curve[g]["best_median"] for g in gens]
        q25 = [curve[g]["best_q25"] for g in gens]
        q75 = [curve[g]["best_q75"] for g in gens]
        plt.plot(gens, best, marker="x", linewidth=2.5, label=label)
        plt.fill_between(gens, q25, q75, alpha=0.2)

    plt.title("Convergence Curves (median best fitness over generations)")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png")
    plt.close()


def plot_diversity(summary: dict[str, dict[int, dict[str, float]]], out_dir: Path) -> None:
    """Plot median genotype diversity over generations."""
    plt.figure(figsize=(8, 5))
    for label, curve in summary.items():
        gens = sorted(curve)
        diversity = [curve[g]["diversity_median"] for g in gens]
        q25 = [curve[g]["diversity_q25"] for g in gens]
        q75 = [curve[g]["diversity_q75"] for g in gens]
        plt.plot(gens, diversity, marker="x", linewidth=2.5, label=label)
        plt.fill_between(gens, q25, q75, alpha=0.2)

    plt.title("Genotype Diversity Curves (median population over generations)")
    plt.xlabel("Generation")
    plt.ylabel("Mean Tree Edit Distance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir / "diversity.png")
    plt.close()


def plot_final_best_fitness(
    run_map: dict[str, list[dict[int, dict[str, float | list[float]]]]],
    out_dir: Path,
) -> None:
    """Plot final best fitness for each variant."""
    data: dict[str, list[float]] = {}
    for variant, series_list in run_map.items():
        final_best: list[float] = []
        for series in series_list:
            if not series:
                continue
            final_best.append(min(float(v["best"]) for v in series.values()))
        data[variant] = final_best

    plt.figure(figsize=(8, 5))
    plt.boxplot([data[k] for k in data], labels=list(data))
    plt.title("Final Best Fitness")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir / "final_best_fitness.png")
    plt.close()


def save_results_csv(summary: dict[str, dict[int, dict[str, float]]], out_dir: Path) -> None:
    """Save summary results to CSV file."""
    rows: list[dict[str, Any]] = []
    for variant, curve in summary.items():
        for gen, metrics in curve.items():
            rows.append(
                {
                    "variant": variant,
                    "generation": int(gen),
                    "best_median": metrics["best_median"],
                    "best_q25": metrics["best_q25"],
                    "best_q75": metrics["best_q75"],
                    "diversity_median": metrics["diversity_median"],
                    "diversity_q25": metrics["diversity_q25"],
                    "diversity_q75": metrics["diversity_q75"],
                }
            )

    out_path = out_dir / "summary_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "generation",
                "best_median",
                "best_q25",
                "best_q75",
                "diversity_median",
                "diversity_q25",
                "diversity_q75",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_variant_files(data_dir: Path, variant: str, seeds: list[int]) -> list[Path]:
    """Resolve the file paths for a given variant and seeds."""
    files: list[Path] = []
    for seed in seeds:
        if variant == "baseline":
            path = data_dir / f"baseline_random_search_seed{seed}.csv"
        else:
            path = data_dir / f"{variant}_seed{seed}.db"
        if path.exists():
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse and plot variant A, B and baseline runs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("__data__") / "group24_algorithm",
        help="Folder with experiment outputs.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="Seed list")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("__data__") / "analysis_plots",
        help="Output folder for plots and CSV summary.",
    )
    args = parser.parse_args()

    print(f"Starting analysis with data_dir={args.data_dir}, seeds={args.seeds}, out_dir={args.out_dir}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_map: dict[str, list[dict[int, dict[str, float | list[float]]]]] = {}
    start_total = time.perf_counter()
    for variant in ["variantA", "variantB", "baseline"]:
        files = resolve_variant_files(args.data_dir, variant, args.seeds)
        if not files:
            print(f"No files found for {variant} in {args.data_dir} with seeds {args.seeds}", flush=True)
            continue

        print(f"Processing {variant} with {len(files)} files", flush=True)
        if variant == "baseline":
            baseline_curves = [load_baseline_csv(f) for f in files]
            run_map[variant] = [
                {g: {"best": float(v), "mean": float(v), "diversity": 0.0} for g, v in curve.items()}
                for curve in baseline_curves
            ]
        else:
            run_map[variant] = [db_run(f) for f in files]

    if not run_map:
        raise FileNotFoundError("No output files were found.")

    print("Summarise runs", flush=True)
    summary = summarise_runs(run_map)
    print("Writing results to CSV", flush=True)
    save_results_csv(summary, out_dir)
    print("Generating plots", flush=True)
    plot_convergence(summary, out_dir)
    print("Generating diversity plots", flush=True)
    plot_diversity(summary, out_dir)
    print("Generating final best fitness plots", flush=True)
    plot_final_best_fitness(run_map, out_dir)

    elapsed_time = time.perf_counter() - start_total
    print(f"\nCompleted analysis in {elapsed_time:.2f} seconds. Plots and summary CSV saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
