"""Compare fixed Standard SQD restart traces after different preludes."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(args: argparse.Namespace) -> None:
    import numpy as np

    from chemistry.prepare import prepare_registered
    from solvers.sqd import run_sqd

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    counts = json.loads(args.pool.read_text())
    rows = []
    if args.prelude == "marginal":
        from benchmark.fair_selection_comparison import run_marginal

        settings = argparse.Namespace(
            samples_per_batch=25,
            num_batches=3,
            max_iterations=5,
            max_dim=args.max_dim,
            strings_per_spin=args.max_dim,
        )
        run_marginal(prepared, counts, settings, args.prelude_seed)
    start = perf_counter()
    for restart in range(args.restarts):
        seed = args.seed + restart
        result = run_sqd(
            prepared,
            counts,
            samples_per_batch=25,
            num_batches=3,
            max_iterations=5,
            max_dim=args.max_dim,
            seed=seed,
        )
        rows.append({
            "restart": restart,
            "seed": seed,
            "energy": result.total_energy,
            "elapsed": perf_counter() - start,
            "solve_count": result.classical_solve_count,
            "solve_dimensions": result.classical_solve_dimensions,
            "alpha": np.asarray(result.sci_result.sci_state.ci_strs_a).tolist(),
            "beta": np.asarray(result.sci_result.sci_state.ci_strs_b).tolist(),
        })
    return {"prelude": args.prelude, "restarts": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="n2-large")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=12)
    parser.add_argument("--prelude", choices=("none", "marginal"), default="none")
    parser.add_argument("--prelude-seed", type=int, default=129)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run(args)))
    else:
        if args.out is None:
            parser.error("--out is required without --worker")
        args.out.write_text(json.dumps(_run_subprocess(args), indent=2, sort_keys=True))


def _run_subprocess(args: argparse.Namespace):
    import subprocess

    command = [
        sys.executable, __file__, args.molecule,
        "--pool", str(args.pool), "--seed", str(args.seed),
        "--restarts", str(args.restarts), "--max-dim", str(args.max_dim),
        "--prelude", args.prelude, "--prelude-seed", str(args.prelude_seed),
        "--worker",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True,
        env={**__import__("os").environ,
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    main()
