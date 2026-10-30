"""Measure sample coverage without trimming at a fixed shot budget."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from quantum.SqDRIFT import sample_sqdrift  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from benchmark.experiment import sampled_product_solve  # noqa: E402


def parse_schedule(value: str) -> tuple[tuple[float, ...], int]:
    try:
        time_text, step_text = value.split(":")
        times = tuple(float(item) for item in time_text.split(","))
        steps = int(step_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "schedule must use time1,time2:steps"
        ) from error
    if not times or steps < 1:
        raise argparse.ArgumentTypeError("schedule values must be positive")
    return times, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--schedule",
        type=parse_schedule,
        action="append",
        default=None,
        help="time1,time2:steps; repeat without changing the shot budget",
    )
    parser.add_argument("--circuits-per-time", type=int, default=10)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/sampling-sweep"))
    args = parser.parse_args()

    schedules = args.schedule or [
        ((1.0, 5.0), 20),
        ((5.0, 20.0), 50),
        ((5.0, 20.0), 100),
        ((20.0, 100.0), 200),
    ]
    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule_index, (times, steps) in enumerate(schedules):
        for seed in args.seeds:
            sample_start = perf_counter()
            pool = sample_sqdrift(
                prepared,
                times=times,
                circuits_per_time=args.circuits_per_time,
                num_steps=steps,
                shots_per_circuit=args.shots,
                seed=seed,
            )
            sampling_seconds = perf_counter() - sample_start
            pool_path = args.out / f"pool_{schedule_index}_{seed}.json"
            pool_path.write_text(json.dumps(pool.counts, sort_keys=True))

            untrimmed_energy, untrimmed_dim, untrimmed_seconds = (
                sampled_product_solve(prepared, pool.counts)
            )
            sqd_start = perf_counter()
            sqd = run_sqd(
                prepared,
                pool.counts,
                samples_per_batch=args.samples_per_batch,
                num_batches=args.num_batches,
                max_iterations=args.max_iterations,
                max_dim=args.max_dim,
                seed=seed,
            )
            sqd_seconds = perf_counter() - sqd_start

            common = {
                "molecule": args.molecule,
                "schedule": schedule_index,
                "times": times,
                "num_steps": steps,
                "seed": seed,
                "pool_file": str(pool_path),
                "shots": pool.total_shots,
                "circuits": len(pool.circuits),
                "cx": sum(item["cx_count"] for item in pool.metadata),
                "cx_counts": [item["cx_count"] for item in pool.metadata],
                "depths": [item["depth"] for item in pool.metadata],
                "depth_sum": sum(item["depth"] for item in pool.metadata),
                "shot_weighted_cx": sum(
                    item["cx_count"] * args.shots for item in pool.metadata
                ),
                "shot_weighted_depth": sum(
                    item["depth"] * args.shots for item in pool.metadata
                ),
                "sampling_seconds": sampling_seconds,
            }
            rows.extend(
                [
                    {
                        **common,
                        "method": "untrimmed",
                        "total_energy": untrimmed_energy,
                        "error_mha": 1000 * abs(
                            untrimmed_energy - prepared.fci_energy
                        ),
                        "solve_seconds": untrimmed_seconds,
                        "subspace_dimension": untrimmed_dim,
                        "max_actual_solve_dimension": untrimmed_dim,
                        "cap_satisfied": untrimmed_dim <= args.max_dim * args.max_dim,
                    },
                    {
                        **common,
                        "method": "sqd",
                        "total_energy": sqd.total_energy,
                        "error_mha": sqd.error_mha,
                        "solve_seconds": sqd_seconds,
                        "subspace_dimension": sqd.subspace_dimension,
                        "max_actual_solve_dimension": max(
                            sqd.classical_solve_dimensions
                        ),
                        "cap_satisfied": all(
                            dimension <= args.max_dim * args.max_dim
                            for dimension in sqd.classical_solve_dimensions
                        ),
                        "classical_solve_count": sqd.classical_solve_count,
                        "classical_solve_dimensions": sqd.classical_solve_dimensions,
                    },
                ]
            )

    output = args.out / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
