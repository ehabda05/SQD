"""Run matched-pool SQD versus DRIFT-TRIM experiments.

Each seed is sampled once. Every solver configuration then consumes that same
saved pool, making selection losses distinguishable from sampling losses.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.fermion import solve_sci

try:
    import resource
except ImportError:  # pragma: no cover - only used on Windows
    resource = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from quantum.SqDRIFT import sample_sqdrift  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.trim_sqd import run_trim_sqd  # noqa: E402


def parse_cap(value: str) -> tuple[int, int, int]:
    try:
        samples, batches, dimension = (int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "caps must use samples_per_batch,num_batches,max_dim"
        ) from error
    if min(samples, batches, dimension) < 1:
        raise argparse.ArgumentTypeError("caps values must be positive")
    return samples, batches, dimension


def peak_memory_mib() -> float:
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sampled_product_solve(prepared, counts):
    alpha = set()
    beta = set()
    n = prepared.num_orbitals
    for bitstring in counts:
        beta.add(int(bitstring[:n], 2))
        alpha.add(int(bitstring[n:], 2))
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--caps",
        type=parse_cap,
        nargs="+",
        default=[(10, 3, 10), (25, 3, 32), (100, 3, 256)],
    )
    parser.add_argument("--times", type=float, nargs="+", default=[1.0, 5.0, 20.0])
    parser.add_argument("--circuits-per-time", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--trim-ratio", type=float, default=0.5)
    parser.add_argument("--carryover-ratio", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--pool-dir", type=Path)
    parser.add_argument("--reuse-pools", action="store_true")
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    previous_rows = []
    previous_path = (
        (args.pool_dir or args.out) / f"experiment_{args.molecule}.json"
    )
    if args.reuse_pools and previous_path.exists():
        previous_rows = json.loads(previous_path.read_text())

    for seed in args.seeds:
        pool_dir = args.pool_dir or args.out
        pool_path = pool_dir / f"pool_{args.molecule}_{seed}.json"
        if args.reuse_pools:
            counts = json.loads(pool_path.read_text())
            prior = next(
                row for row in previous_rows
                if row["seed"] == seed and row["method"] == "sqd"
            )
            sampling_seconds = prior["sampling_seconds"]
            sampling_cost = {
                key: prior[key] for key in ("shots", "circuits", "cx", "depth")
            }
        else:
            sample_start = perf_counter()
            pool = sample_sqdrift(
                prepared,
                times=tuple(args.times),
                circuits_per_time=args.circuits_per_time,
                num_steps=args.num_steps,
                shots_per_circuit=args.shots,
                seed=seed,
            )
            sampling_seconds = perf_counter() - sample_start
            counts = pool.counts
            pool_path = args.out / f"pool_{args.molecule}_{seed}.json"
            pool_path.write_text(json.dumps(counts, sort_keys=True))
            sampling_cost = {
                "shots": pool.total_shots,
                "circuits": len(pool.circuits),
                "cx": sum(item["cx_count"] for item in pool.metadata),
                "depth": sum(item["depth"] for item in pool.metadata),
            }

        untrimmed_energy, untrimmed_dim, untrimmed_seconds = (
            sampled_product_solve(prepared, counts)
        )

        for samples, batches, max_dim in args.caps:
            methods = (
                ("untrimmed", None, untrimmed_energy, untrimmed_dim, untrimmed_seconds),
                ("sqd", None, None, None, None),
                ("drift-trim", args.trim_ratio, None, None, None),
            )
            for method, trim_ratio, fixed_energy, fixed_dim, fixed_seconds in methods:
                solve_start = perf_counter()
                if fixed_energy is not None:
                    total_energy, subspace_dimension, solve_seconds = (
                        fixed_energy, fixed_dim, fixed_seconds
                    )
                    result = None
                elif method == "sqd":
                    result = run_sqd(
                        prepared,
                        counts,
                        samples_per_batch=samples,
                        num_batches=batches,
                        max_iterations=args.max_iterations,
                        max_dim=max_dim,
                        seed=seed,
                    )
                else:
                    result = run_trim_sqd(
                        prepared,
                        counts,
                        samples_per_batch=samples,
                        num_batches=batches,
                        max_iterations=args.max_iterations,
                        max_dim=max_dim,
                        trim_ratio=trim_ratio,
                        carryover_ratio=args.carryover_ratio,
                        seed=seed,
                    )
                if result is not None:
                    solve_seconds = perf_counter() - solve_start
                    total_energy = result.total_energy
                    subspace_dimension = result.subspace_dimension
                    classical_solve_count = result.classical_solve_count
                    classical_solve_dimensions = result.classical_solve_dimensions
                else:
                    classical_solve_count = 1
                    classical_solve_dimensions = (subspace_dimension,)
                rows.append(
                    {
                        "molecule": args.molecule,
                        "seed": seed,
                        "pool_file": str(pool_path),
                        "method": method,
                        "samples_per_batch": samples,
                        "num_batches": batches,
                        "max_iterations": args.max_iterations,
                        "max_dim": max_dim,
                        "trim_ratio": trim_ratio,
                        "carryover_ratio": (
                            None if trim_ratio is None else args.carryover_ratio
                        ),
                        "total_energy": total_energy,
                        "fci_energy": prepared.fci_energy,
                        "error_mha": (
                            None
                            if prepared.fci_energy is None
                            else 1000 * abs(total_energy - prepared.fci_energy)
                        ),
                        "subspace_dimension": subspace_dimension,
                        "max_actual_solve_dimension": max(classical_solve_dimensions),
                        "cap_satisfied": (
                            max(classical_solve_dimensions) <= max_dim * max_dim
                        ),
                        "classical_solve_count": classical_solve_count,
                        "classical_solve_dimensions": classical_solve_dimensions,
                        "sampling_seconds": sampling_seconds,
                        "solve_seconds": solve_seconds,
                        "peak_memory_mib": peak_memory_mib(),
                        **sampling_cost,
                    }
                )

    output = args.out / f"experiment_{args.molecule}.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
