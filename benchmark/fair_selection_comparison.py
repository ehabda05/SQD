"""Compare accumulated marginal selection and Standard SQD at one budget."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.fermion import solve_sci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.marginal_selection import select_top_strings  # noqa: E402
from chemistry.prepare import prepare_registered  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402


def run_marginal(prepared, counts, args, seed):
    alpha_scores = {}
    beta_scores = {}
    progress = []

    def observe(ci_strings, results):
        for (strings_a, strings_b), result in zip(ci_strings, results):
            weights = np.abs(np.asarray(result.sci_state.amplitudes)) ** 2
            for string, weight in zip(strings_a, np.sum(weights, axis=1)):
                alpha_scores.setdefault(int(string), []).append(float(weight))
            for string, weight in zip(strings_b, np.sum(weights, axis=0)):
                beta_scores.setdefault(int(string), []).append(float(weight))

    start = perf_counter()
    collection = run_sqd(
        prepared,
        counts,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim,
        seed=seed,
        batch_observer=observe,
    )
    elapsed = perf_counter() - start
    progress.append((elapsed, collection.total_energy, "collection"))

    selected_a = select_top_strings(alpha_scores, args.strings_per_spin)
    selected_b = select_top_strings(beta_scores, args.strings_per_spin)
    candidate = solve_sci(
        (selected_a, selected_b),
        prepared.h1,
        prepared.h2,
        norb=prepared.num_orbitals,
        nelec=prepared.num_electrons,
    )
    candidate_energy = float(candidate.energy) + prepared.constant_energy
    elapsed = perf_counter() - start
    progress.append((elapsed, candidate_energy, "candidate"))
    best = min(collection.total_energy, candidate_energy)
    return {
        "energy": best,
        "elapsed": elapsed,
        "progress": progress,
        "solve_count": collection.classical_solve_count + 1,
        "determinants": sum(collection.classical_solve_dimensions)
        + selected_a.size * selected_b.size,
        "selected_alpha": selected_a.tolist(),
        "selected_beta": selected_b.tolist(),
    }


def run_standard(prepared, counts, args, seed, budget):
    start = perf_counter()
    progress = []
    runs = []
    while perf_counter() - start < budget or not runs:
        result = run_sqd(
            prepared,
            counts,
            samples_per_batch=args.samples_per_batch,
            num_batches=args.num_batches,
            max_iterations=args.max_iterations,
            max_dim=args.max_dim,
            seed=seed + len(runs),
        )
        elapsed = perf_counter() - start
        runs.append(result)
        progress.append((elapsed, result.total_energy, "restart"))
    best = min(runs, key=lambda result: result.total_energy)
    return {
        "energy": best.total_energy,
        "elapsed": perf_counter() - start,
        "progress": progress,
        "solve_count": sum(result.classical_solve_count for result in runs),
        "determinants": sum(
            sum(result.classical_solve_dimensions) for result in runs
        ),
        "restart_count": len(runs),
    }


def warm_up(prepared, counts, args, seed):
    run_sqd(
        prepared,
        counts,
        samples_per_batch=1,
        num_batches=1,
        max_iterations=1,
        max_dim=args.max_dim,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--pool-dir", type=Path, default=Path("results/sampling-sweep"))
    parser.add_argument("--schedule-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(45, 55)))
    parser.add_argument("--budget-seconds", type=float, default=0.04)
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--strings-per-spin", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/fair-selection"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            counts = json.loads(
                (args.pool_dir / f"pool_{schedule}_{seed}.json").read_text()
            )
            warm_up(prepared, counts, args, seed)
            pilot = run_marginal(prepared, counts, args, seed)
            budget_target = pilot["elapsed"]
            marginal_first = (schedule + seed) % 2 == 0
            if marginal_first:
                marginal = run_marginal(prepared, counts, args, seed + 2000)
                standard = run_standard(
                    prepared, counts, args, seed + 1000, budget_target
                )
            else:
                standard = run_standard(
                    prepared, counts, args, seed + 1000, budget_target
                )
                marginal = run_marginal(prepared, counts, args, seed + 2000)

            def error(energy):
                return (
                    None
                    if prepared.fci_energy is None
                    else 1000 * abs(energy - prepared.fci_energy)
                )

            rows.append(
                {
                    "schedule": schedule,
                    "seed": seed,
                    "marginal_first": marginal_first,
                    "budget_seconds": budget_target,
                    "marginal_elapsed": marginal["elapsed"],
                    "standard_elapsed": standard["elapsed"],
                    "marginal_overshoot": marginal["elapsed"]
                    - budget_target,
                    "standard_overshoot": standard["elapsed"]
                    - budget_target,
                    "marginal_error_mha": error(marginal["energy"]),
                    "standard_error_mha": error(standard["energy"]),
                    "standard_delta_vs_marginal_mha": 1000
                    * (standard["energy"] - marginal["energy"]),
                    "marginal_solve_count": marginal["solve_count"],
                    "standard_solve_count": standard["solve_count"],
                    "marginal_total_determinants": marginal["determinants"],
                    "standard_total_determinants": standard["determinants"],
                    "standard_restart_count": standard["restart_count"],
                    "marginal_progress": marginal["progress"],
                    "standard_progress": standard["progress"],
                    "marginal_selected_alpha": marginal["selected_alpha"],
                    "marginal_selected_beta": marginal["selected_beta"],
                }
            )

    output = args.out / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
