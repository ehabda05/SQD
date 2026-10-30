"""Compare merge-only with Standard SQD given the same solver-time budget."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.trim_sqd import run_trim_sqd  # noqa: E402


def error_mha(energy, reference):
    return None if reference is None else 1000 * abs(energy - reference)


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
    parser.add_argument("--carryover-ratio", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/equal-cost"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            pool_path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(pool_path.read_text())
            merge_start = perf_counter()
            merge = run_trim_sqd(
                prepared,
                counts,
                samples_per_batch=args.samples_per_batch,
                num_batches=args.num_batches,
                max_iterations=args.max_iterations,
                max_dim=args.max_dim,
                trim_ratio=1.0,
                carryover_ratio=args.carryover_ratio,
                seed=seed,
            )
            merge_seconds = perf_counter() - merge_start

            standard_runs = []
            standard_seconds = 0.0
            restart = 0
            while standard_seconds < merge_seconds or not standard_runs:
                start = perf_counter()
                result = run_sqd(
                    prepared,
                    counts,
                    samples_per_batch=args.samples_per_batch,
                    num_batches=args.num_batches,
                    max_iterations=args.max_iterations,
                    max_dim=args.max_dim,
                    seed=seed + restart,
                )
                elapsed = perf_counter() - start
                standard_seconds += elapsed
                standard_runs.append(result)
                restart += 1

            best = min(standard_runs, key=lambda result: result.total_energy)
            rows.append(
                {
                    "molecule": args.molecule,
                    "schedule": schedule,
                    "seed": seed,
                    "pool_file": str(pool_path),
                    "merge_energy": merge.total_energy,
                    "merge_error_mha": merge.error_mha,
                    "merge_solver_seconds": merge_seconds,
                    "merge_solve_count": merge.classical_solve_count,
                    "merge_total_determinants": sum(
                        merge.classical_solve_dimensions
                    ),
                    "standard_best_energy": best.total_energy,
                    "standard_best_error_mha": error_mha(
                        best.total_energy, prepared.fci_energy
                    ),
                    "standard_solver_seconds": standard_seconds,
                    "standard_restart_count": len(standard_runs),
                    "standard_total_solve_count": sum(
                        result.classical_solve_count for result in standard_runs
                    ),
                    "standard_total_determinants": sum(
                        sum(result.classical_solve_dimensions)
                        for result in standard_runs
                    ),
                    "standard_max_solve_dimension": max(
                        max(result.classical_solve_dimensions)
                        for result in standard_runs
                    ),
                    "standard_delta_vs_merge_mha": 1000
                    * (best.total_energy - merge.total_energy),
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
