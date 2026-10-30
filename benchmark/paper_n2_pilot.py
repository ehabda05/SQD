"""Feasibility pilot for the paper-aligned N2 problem using the existing method."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from quantum.SqDRIFT import sample_sqdrift  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--simulator-method", default="matrix_product_state")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared = prepare_registered("n2-zenko", compute_fci=False)
    memory_gib = (2 ** (2 * prepared.num_orbitals) * 16) / 2**30
    sample_start = perf_counter()
    samples = sample_sqdrift(
        prepared,
        times=(args.time,),
        circuits_per_time=1,
        num_steps=1,
        shots_per_circuit=args.shots,
        seed=args.seed,
        simulator_method=args.simulator_method,
    )
    sampling_seconds = perf_counter() - sample_start

    solve_start = perf_counter()
    solved = run_sqd(
        prepared,
        samples.counts,
        samples_per_batch=25,
        num_batches=3,
        max_iterations=5,
        max_dim=32,
        seed=args.seed,
    )
    solve_seconds = perf_counter() - solve_start

    row = {
        "seed": args.seed,
        "molecule": "n2-zenko",
        "active_orbitals": prepared.num_orbitals,
        "active_electrons": prepared.num_electrons,
        "constant_energy_offset_hartree": prepared.constant_energy,
        "simulator_method": args.simulator_method,
        "simulator_options": {"max_parallel_threads": 2},
        "evolution_time": args.time,
        "shots": samples.total_shots,
        "circuits": len(samples.circuits),
        "sampling_seconds": sampling_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": sampling_seconds + solve_seconds,
        "statevector_complex128_memory_gib": memory_gib,
        "unique_configurations": len(samples.counts),
        "max_circuit_depth": max(item["depth"] for item in samples.metadata),
        "total_cx": sum(item["cx_count"] for item in samples.metadata),
        "energy": solved.total_energy,
        "classical_solve_count": solved.classical_solve_count,
        "classical_solve_dimensions": solved.classical_solve_dimensions,
        "max_solve_dimension": max(solved.classical_solve_dimensions),
        "reference_energy": -109.04667177831362,
        "error_mha": 1000 * abs(solved.total_energy + 109.04667177831362),
        "reference_method": "PySCF direct_spin1 iterative FCI, conv_tol=1e-10",
        "pilot_scope": "one MPS circuit; not confirmation data",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
