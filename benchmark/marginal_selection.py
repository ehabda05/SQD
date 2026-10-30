"""Test marginal CI-weight selection collected from restricted SQD solves."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.fermion import solve_sci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402


def mean_present(scores):
    return {
        string: float(np.mean(values))
        for string, values in scores.items()
        if values
    }


def select_top_strings(scores, count):
    means = mean_present(scores)
    return np.array(
        sorted(
            sorted(means, key=lambda string: (-means[string], string))[:count]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--pool-dir", type=Path, default=Path("results/sampling-sweep"))
    parser.add_argument("--schedule-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(45, 55)))
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--strings-per-spin", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/marginal-selection"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            pool_path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(pool_path.read_text())
            alpha_scores = {}
            beta_scores = {}

            def observe(ci_strings, results):
                for (strings_a, strings_b), result in zip(ci_strings, results):
                    weights = np.abs(np.asarray(result.sci_state.amplitudes)) ** 2
                    for string, weight in zip(
                        strings_a, np.sum(weights, axis=1)
                    ):
                        alpha_scores.setdefault(int(string), []).append(float(weight))
                    for string, weight in zip(
                        strings_b, np.sum(weights, axis=0)
                    ):
                        beta_scores.setdefault(int(string), []).append(float(weight))

            collection_start = perf_counter()
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
            collection_seconds = perf_counter() - collection_start

            candidate_start = perf_counter()
            selected_a = select_top_strings(
                alpha_scores, args.strings_per_spin
            )
            selected_b = select_top_strings(
                beta_scores, args.strings_per_spin
            )
            candidate = solve_sci(
                (selected_a, selected_b),
                prepared.h1,
                prepared.h2,
                norb=prepared.num_orbitals,
                nelec=prepared.num_electrons,
            )
            candidate_seconds = perf_counter() - candidate_start
            candidate_energy = float(candidate.energy) + prepared.constant_energy
            best_energy = min(collection.total_energy, candidate_energy)

            rows.append(
                {
                    "molecule": args.molecule,
                    "schedule": schedule,
                    "seed": seed,
                    "pool_file": str(pool_path),
                    "collection_energy": collection.total_energy,
                    "collection_error_mha": collection.error_mha,
                    "collection_seconds": collection_seconds,
                    "candidate_energy": candidate_energy,
                    "candidate_error_mha": (
                        None
                        if prepared.fci_energy is None
                        else 1000 * abs(candidate_energy - prepared.fci_energy)
                    ),
                    "candidate_seconds": candidate_seconds,
                    "total_seconds": collection_seconds + candidate_seconds,
                    "best_energy": best_energy,
                    "best_error_mha": (
                        None
                        if prepared.fci_energy is None
                        else 1000 * abs(best_energy - prepared.fci_energy)
                    ),
                    "collection_solve_count": collection.classical_solve_count,
                    "candidate_solve_count": 1,
                    "total_determinants": (
                        sum(collection.classical_solve_dimensions)
                        + selected_a.size * selected_b.size
                    ),
                    "selected_alpha": selected_a.tolist(),
                    "selected_beta": selected_b.tolist(),
                    "scored_alpha_count": len(alpha_means),
                    "scored_beta_count": len(beta_means),
                    "candidate_improved_collection": candidate_energy
                    < collection.total_energy,
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
