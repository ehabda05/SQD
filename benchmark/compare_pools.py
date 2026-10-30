"""Compare SQD selection policies on identical saved sample pools."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from qiskit_addon_sqd.fermion import solve_sci  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.trim_sqd import run_trim_sqd  # noqa: E402


def sampled_product_solve(prepared, counts):
    n = prepared.num_orbitals
    alpha = {int(bitstring[n:], 2) for bitstring in counts}
    beta = {int(bitstring[:n], 2) for bitstring in counts}
    start = perf_counter()
    result = solve_sci(
        (np.array(sorted(alpha)), np.array(sorted(beta))),
        prepared.h1,
        prepared.h2,
        norb=n,
        nelec=prepared.num_electrons,
    )
    return (
        float(result.energy) + prepared.constant_energy,
        len(alpha) * len(beta),
        perf_counter() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--pool-dir", type=Path, default=Path("results/sampling-sweep"))
    parser.add_argument("--schedule-indices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--carryover-ratio", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/pool-comparison"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            pool_path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(pool_path.read_text())
            reference_energy, reference_dim, reference_seconds = (
                sampled_product_solve(prepared, counts)
            )

            methods = [
                ("unrestricted sampled product", reference_energy, reference_dim,
                 reference_seconds, None),
            ]
            for name, runner, trim_ratio in (
                ("standard SQD", run_sqd, None),
                ("merge-only", run_trim_sqd, 1.0),
                ("pruning", run_trim_sqd, 0.5),
            ):
                start = perf_counter()
                if trim_ratio is None:
                    result = runner(
                        prepared,
                        counts,
                        samples_per_batch=args.samples_per_batch,
                        num_batches=args.num_batches,
                        max_iterations=args.max_iterations,
                        max_dim=args.max_dim,
                        seed=seed,
                    )
                else:
                    result = runner(
                        prepared,
                        counts,
                        samples_per_batch=args.samples_per_batch,
                        num_batches=args.num_batches,
                        max_iterations=args.max_iterations,
                        max_dim=args.max_dim,
                        trim_ratio=trim_ratio,
                        carryover_ratio=args.carryover_ratio,
                        seed=seed,
                    )
                methods.append(
                    (
                        name,
                        result.total_energy,
                        result.subspace_dimension,
                        perf_counter() - start,
                        result,
                    )
                )

            for method, energy, dimension, seconds, result in methods:
                dimensions = (
                    (dimension,)
                    if result is None
                    else result.classical_solve_dimensions
                )
                rows.append(
                    {
                        "molecule": args.molecule,
                        "schedule": schedule,
                        "seed": seed,
                        "pool_file": str(pool_path),
                        "method": method,
                        "max_dim_per_spin": args.max_dim,
                        "total_energy": energy,
                        "fci_energy": prepared.fci_energy,
                        "error_mha": 1000 * abs(energy - prepared.fci_energy),
                        "max_actual_solve_dimension": max(dimensions),
                        "cap_satisfied": all(
                            value <= args.max_dim * args.max_dim
                            for value in dimensions
                        ),
                        "classical_solve_count": len(dimensions),
                        "classical_solve_dimensions": dimensions,
                        "total_determinants_diagonalized": sum(dimensions),
                        "solve_seconds": seconds,
                        "reference_outside_budget": result is None,
                    }
                )

    for row in rows:
        standard = next(
            candidate
            for candidate in rows
            if candidate["schedule"] == row["schedule"]
            and candidate["seed"] == row["seed"]
            and candidate["method"] == "standard SQD"
        )
        row["delta_vs_standard_mha"] = row["error_mha"] - standard["error_mha"]
        row["delta_vs_standard_seconds"] = (
            row["solve_seconds"] - standard["solve_seconds"]
        )

    output = args.out / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    means = []
    for schedule in args.schedule_indices:
        for method in (
            "unrestricted sampled product",
            "standard SQD",
            "merge-only",
            "pruning",
        ):
            selected = [
                row for row in rows
                if row["schedule"] == schedule and row["method"] == method
            ]
            means.append(
                {
                    "schedule": schedule,
                    "method": method,
                    "mean_error_mha": sum(
                        row["error_mha"] for row in selected
                    ) / len(selected),
                    "stddev_error_mha": statistics.stdev(
                        row["error_mha"] for row in selected
                    ),
                    "mean_delta_vs_standard_mha": sum(
                        row["delta_vs_standard_mha"] for row in selected
                    ) / len(selected),
                    "stddev_delta_vs_standard_mha": statistics.stdev(
                        row["delta_vs_standard_mha"] for row in selected
                    ),
                    "mean_solve_seconds": sum(
                        row["solve_seconds"] for row in selected
                    ) / len(selected),
                    "mean_total_determinants": sum(
                        row["total_determinants_diagonalized"]
                        for row in selected
                    ) / len(selected),
                }
            )
    (args.out / "means.json").write_text(json.dumps(means, indent=2))
    print(f"wrote {output}, {output.with_suffix('.csv')}, and means.json")


if __name__ == "__main__":
    main()
