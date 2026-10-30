"""Audit solver call history using isolated worker processes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def worker(args: argparse.Namespace) -> None:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chemistry.prepare import prepare_registered
    from benchmark.fair_selection_comparison import run_marginal, run_standard

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    counts = json.loads(args.pool.read_text())
    settings = argparse.Namespace(
        samples_per_batch=25,
        num_batches=3,
        max_iterations=5,
        max_dim=12,
        strings_per_spin=12,
    )
    if args.method == "marginal":
        result = run_marginal(prepared, counts, settings, args.seed)
    else:
        result = run_standard(
            prepared, counts, settings, args.seed, args.budget_seconds
        )
    print(json.dumps({
        "method": args.method,
        "seed": args.seed,
        "energy": result["energy"],
        "elapsed": result["elapsed"],
        "solve_count": result["solve_count"],
        "determinants": result["determinants"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="n2-large")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--method", choices=("marginal", "standard"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.method:
        worker(args)
        return

    if args.reference_file is None:
        raise ValueError("--reference-file is required in audit mode")
    reference = json.loads(args.reference_file.read_text())
    results = []
    for order in ("marginal-first", "standard-first"):
        first = "marginal" if order == "marginal-first" else "standard"
        second = "standard" if first == "marginal" else "marginal"
        pilot = subprocess.run(
            [sys.executable, __file__, args.molecule, "--pool", str(args.pool),
             "--seed", str(args.seed), "--budget-seconds", str(args.budget_seconds),
             "--method", "marginal", "--out", str(args.out)],
            capture_output=True, text=True, check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        pilot_result = json.loads(pilot.stdout)
        budget = pilot_result["elapsed"]
        row = {"order": order, "budget": budget}
        for method in (first, second):
            completed = subprocess.run(
                [sys.executable, __file__, args.molecule, "--pool", str(args.pool),
                 "--seed", str(args.seed), "--budget-seconds", str(budget),
                 "--method", method, "--out", str(args.out)],
                capture_output=True, text=True, check=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            row[method] = json.loads(completed.stdout)
        row["reference_energy"] = next(
            item["energy"] for item in reference
            if item["pool_file"] == str(args.pool)
        )
        results.append(row)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
