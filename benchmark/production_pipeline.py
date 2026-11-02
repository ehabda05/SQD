"""Run one reproducible production SQD pipeline and write JSON/CSV output."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered
from solvers.drift_trim import run_drift_trim


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="n2-large")
    parser.add_argument(
        "--method",
        choices=("standard-sqd", "trim-sqd", "accumulated-marginal"),
        default="accumulated-marginal",
    )
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compute-fci", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--times", type=float, nargs="+", default=[20.0, 100.0])
    parser.add_argument("--circuits-per-time", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--solver-seed", type=int, default=42)
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=32)
    parser.add_argument("--strings-per-spin", type=int)
    parser.add_argument("--recover-invalid", action="store_true")
    args = parser.parse_args()

    prepared = prepare_registered(
        args.molecule,
        compute_fci=args.compute_fci,
        fci_cache_dir=args.cache_dir,
    )
    result = run_drift_trim(
        prepared,
        method=args.method,
        pool_path=args.pool,
        times=tuple(args.times),
        circuits_per_time=args.circuits_per_time,
        num_steps=args.num_steps,
        shots_per_circuit=args.shots,
        sampling_seed=args.sampling_seed,
        solver_seed=args.solver_seed,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim,
        strings_per_spin=args.strings_per_spin,
        recover_invalid=args.recover_invalid,
    )
    solution = result.solution
    row = {
        "molecule": prepared.molecule.name,
        "geometry_angstrom": [
            {"symbol": atom.symbol, "position": list(atom.position)}
            for atom in prepared.molecule.atoms
        ],
        "basis": prepared.molecule.basis,
        "active_orbitals": prepared.active_orbitals,
        "active_electrons": prepared.num_electrons,
        "method": result.method,
        "pool_hash": result.pool_hash,
        "configuration_fingerprint": result.configuration_fingerprint,
        "raw_counts": result.raw_counts,
        "counts": result.counts,
        "recovery": result.recovery,
        "sampling_seconds": result.sampling_seconds,
        "complete_classical_seconds": result.classical_seconds,
        "diagonalization_seconds": solution.classical_solve_elapsed,
        "recovery_seconds": result.recovery.get("seconds", 0.0),
        "total_seconds": result.total_seconds,
        "fci_energy": prepared.fci_energy,
        "final_total_energy": result.total_energy,
        "error_mha": result.error_mha,
        "selected_alpha": getattr(solution, "selected_alpha", None),
        "selected_beta": getattr(solution, "selected_beta", None),
        "retained_alpha": getattr(solution, "retained_alpha", None),
        "retained_beta": getattr(solution, "retained_beta", None),
        "retained_amplitudes": getattr(solution, "retained_amplitudes", None),
        "selected_dimension": (
            solution.selected_dimension
            if hasattr(solution, "selected_dimension")
            else solution.subspace_dimension
        ),
        "classical_solve_count": solution.classical_solve_count,
        "classical_solve_dimensions": solution.classical_solve_dimensions,
        "dependency_versions": result.dependency_versions,
        "thread_settings": result.thread_settings,
    }
    if result.samples is not None:
        row.update(
            {
                "circuits": len(result.samples.circuits),
                "shots": result.samples.total_shots,
                "circuit_metadata": result.samples.metadata,
                "summed_cx": sum(item["cx_count"] for item in result.samples.metadata),
                "summed_depth": sum(item["depth"] for item in result.samples.metadata),
                "shot_weighted_cx": sum(
                    item["cx_count"] * item["shots"] for item in result.samples.metadata
                ),
                "shot_weighted_depth": sum(
                    item["depth"] * item["shots"] for item in result.samples.metadata
                ),
                "circuit_construction_seconds": sum(
                    item["circuit_seconds"] for item in result.samples.metadata
                ),
                "simulator_sampling_seconds": sum(
                    item["sampling_seconds"] for item in result.samples.metadata
                ),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(row), indent=2, sort_keys=True) + "\n")
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow({key: json.dumps(_jsonable(value)) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    print(json.dumps(_jsonable(row), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
