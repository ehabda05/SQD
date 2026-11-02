"""Compare accumulated marginal selection and Standard SQD at one budget."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from qiskit_addon_sqd.fermion import solve_sci  # noqa: E402
from solvers.accumulated_marginal import run_accumulated_marginal  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.provenance import pool_hash  # noqa: E402


def run_marginal(prepared, counts, args, seed):
    start = perf_counter()
    result = run_accumulated_marginal(
        prepared,
        counts,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim,
        strings_per_spin=args.strings_per_spin,
        seed=seed,
    )
    elapsed = perf_counter() - start
    collection = result.collection
    best_state = result.retained.sci_state
    return {
        "energy": result.retained_total_energy,
        "elapsed": elapsed,
        "progress": [
            (
                result.collection_seconds,
                collection.total_energy,
                "collection",
            ),
            (elapsed, result.retained_total_energy, "candidate"),
        ],
        "solve_count": result.classical_solve_count,
        "restart_count": 1,
        "complete_classical_elapsed": result.total_classical_seconds,
        "diagonalization_elapsed": (
            collection.classical_solve_elapsed
            + result.candidate_seconds
        ),
        "determinants": sum(result.classical_solve_dimensions),
        "selected_alpha": result.selected_alpha.tolist(),
        "selected_beta": result.selected_beta.tolist(),
        "best_alpha": np.asarray(best_state.ci_strs_a).tolist(),
        "best_beta": np.asarray(best_state.ci_strs_b).tolist(),
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
        "complete_classical_elapsed": perf_counter() - start,
        "diagonalization_elapsed": sum(
            result.classical_solve_elapsed for result in runs
        ),
        "determinants": sum(
            sum(result.classical_solve_dimensions) for result in runs
        ),
        "restart_count": len(runs),
        "best_alpha": np.asarray(best.sci_result.sci_state.ci_strs_a).tolist(),
        "best_beta": np.asarray(best.sci_result.sci_state.ci_strs_b).tolist(),
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
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--solver-seed-offset", type=int, default=0)
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    references = {}
    if args.reference_file is not None:
        references = {
            (row["schedule"], row["seed"]): row
            for row in json.loads(args.reference_file.read_text())
        }
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            counts = json.loads(
                (args.pool_dir / f"pool_{schedule}_{seed}.json").read_text()
            )
            counts_hash = pool_hash(counts)
            n = prepared.num_orbitals
            pool_alpha = {int(bitstring[n:], 2) for bitstring in counts}
            pool_beta = {int(bitstring[:n], 2) for bitstring in counts}
            reference_row = references.get((schedule, seed))
            if references and reference_row is None:
                raise ValueError(f"Missing reference for schedule={schedule}, seed={seed}")
            if reference_row is None:
                reference = solve_sci(
                    (np.array(sorted(pool_alpha)), np.array(sorted(pool_beta))),
                    prepared.h1,
                    prepared.h2,
                    norb=n,
                    nelec=prepared.num_electrons,
                )
                reference_energy = float(reference.energy) + prepared.constant_energy
            else:
                if reference_row["pool_hash"] != counts_hash:
                    raise ValueError(f"Pool hash mismatch for schedule={schedule}, seed={seed}")
                reference_energy = float(reference_row["energy"])
            warm_up(prepared, counts, args, seed)
            method_seed = seed + args.solver_seed_offset
            budget_target = args.budget_seconds
            marginal_first = (schedule + seed) % 2 == 0
            if marginal_first:
                marginal = run_marginal(prepared, counts, args, method_seed)
                standard = run_standard(
                    prepared, counts, args, method_seed, budget_target
                )
            else:
                standard = run_standard(
                    prepared, counts, args, method_seed, budget_target
                )
                marginal = run_marginal(prepared, counts, args, method_seed)

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
                    "solver_seed_offset": args.solver_seed_offset,
                    "pool_hash": counts_hash,
                    "alpha_string_count": len(pool_alpha),
                    "beta_string_count": len(pool_beta),
                    "reference_error_mha": (
                        None
                        if prepared.fci_energy is None
                        else 1000 * abs(reference_energy - prepared.fci_energy)
                    ),
                    "reference_energy": reference_energy,
                    "fci_energy": prepared.fci_energy,
                    "marginal_first": marginal_first,
                    "budget_seconds": budget_target,
                    "marginal_elapsed": marginal["elapsed"],
                    "standard_elapsed": standard["elapsed"],
                    "marginal_complete_classical_elapsed": marginal[
                        "complete_classical_elapsed"
                    ],
                    "standard_complete_classical_elapsed": standard[
                        "complete_classical_elapsed"
                    ],
                    "marginal_diagonalization_elapsed": marginal[
                        "diagonalization_elapsed"
                    ],
                    "standard_diagonalization_elapsed": standard[
                        "diagonalization_elapsed"
                    ],
                    "marginal_non_diagonalization_elapsed": marginal[
                        "complete_classical_elapsed"
                    ]
                    - marginal["diagonalization_elapsed"],
                    "standard_non_diagonalization_elapsed": standard[
                        "complete_classical_elapsed"
                    ]
                    - standard["diagonalization_elapsed"],
                    "marginal_overshoot": marginal["elapsed"]
                    - budget_target,
                    "standard_overshoot": standard["elapsed"]
                    - budget_target,
                    "marginal_error_mha": error(marginal["energy"]),
                    "standard_error_mha": error(standard["energy"]),
                    "standard_delta_vs_marginal_mha": 1000
                    * (standard["energy"] - marginal["energy"]),
                    "marginal_absent_alpha": sorted(
                        set(marginal["best_alpha"]) - pool_alpha
                    ),
                    "marginal_absent_beta": sorted(
                        set(marginal["best_beta"]) - pool_beta
                    ),
                    "standard_absent_alpha": sorted(
                        set(standard["best_alpha"]) - pool_alpha
                    ),
                    "standard_absent_beta": sorted(
                        set(standard["best_beta"]) - pool_beta
                    ),
                    "marginal_solve_count": marginal["solve_count"],
                    "marginal_restart_count": marginal["restart_count"],
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
