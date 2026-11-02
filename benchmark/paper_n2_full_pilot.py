"""Run the paper-sized SqDRIFT and marginal-selection feasibility pilot."""

import argparse
import csv
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from quantum.SqDRIFT import sample_sqdrift  # noqa: E402
from solvers.accumulated_marginal import run_accumulated_marginal  # noqa: E402


REFERENCE_ENERGY = -109.04667177831362


def peak_memory_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--max-bond-dimension", type=int, default=128)
    parser.add_argument("--truncation-threshold", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared = prepare_registered("n2-zenko", compute_fci=False)
    simulator_options = {
        "max_parallel_threads": 2,
        "matrix_product_state_max_bond_dimension": args.max_bond_dimension,
        "matrix_product_state_truncation_threshold": args.truncation_threshold,
    }

    sampling_start = perf_counter()
    try:
        samples = sample_sqdrift(
            prepared,
            times=(20.0, 100.0),
            circuits_per_time=10,
            num_steps=200,
            shots_per_circuit=args.shots,
            seed=args.seed,
            simulator_method="matrix_product_state",
            simulator_options=simulator_options,
        )
    except Exception as error:
        failure = {
            "seed": args.seed,
            "schedule": {"times": [20.0, 100.0], "num_steps": 200},
            "shots_per_circuit": args.shots,
            "simulator_method": "matrix_product_state",
            "simulator_options": simulator_options,
            "sampling_seconds": perf_counter() - sampling_start,
            "peak_memory_mib": peak_memory_mib(),
            "error_type": type(error).__name__,
            "error": str(error),
            "pilot_scope": "development seed only; confirmation seeds 157-186 untouched",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2))
        return
    sampling_seconds = perf_counter() - sampling_start

    n = prepared.num_orbitals
    alpha_strings = sorted({int(bits[n:], 2) for bits in samples.counts})
    beta_strings = sorted({int(bits[:n], 2) for bits in samples.counts})
    collection_start = perf_counter()
    result = run_accumulated_marginal(
        prepared,
        samples.counts,
        samples_per_batch=25,
        num_batches=3,
        max_iterations=5,
        max_dim=32,
        strings_per_spin=32,
        seed=args.seed,
    )
    collection_seconds = perf_counter() - collection_start
    collection = result.collection
    candidate_energy = float(result.candidate.energy) + prepared.constant_energy
    best_energy = result.retained_total_energy

    row = {
        "seed": args.seed,
        "schedule": {"times": [20.0, 100.0], "num_steps": 200},
        "shots_per_circuit": args.shots,
        "circuits": len(samples.circuits),
        "total_shots": samples.total_shots,
        "simulator_method": "matrix_product_state",
        "simulator_options": simulator_options,
        "sampling_seconds": sampling_seconds,
        "collection_seconds": collection_seconds,
        "candidate_seconds": result.candidate_seconds,
        "complete_classical_seconds": result.total_classical_seconds,
        "total_seconds": sampling_seconds + result.total_classical_seconds,
        "peak_memory_mib": peak_memory_mib(),
        "statevector_complex128_memory_gib": 64.0,
        "unique_configurations": len(samples.counts),
        "unique_alpha_strings": len(alpha_strings),
        "unique_beta_strings": len(beta_strings),
        "circuit_metadata": samples.metadata,
        "collection_energy": collection.total_energy,
        "candidate_energy": candidate_energy,
        "best_energy": best_energy,
        "best_error_mha": 1000 * abs(best_energy - REFERENCE_ENERGY),
        "reference_energy": REFERENCE_ENERGY,
        "reference_method": "PySCF direct_spin1 iterative FCI, conv_tol=1e-10",
        "collection_solve_dimensions": collection.classical_solve_dimensions,
        "candidate_solve_dimension": result.selected_dimension,
        "classical_solve_count": result.classical_solve_count,
        "total_determinants": (
            sum(result.classical_solve_dimensions)
        ),
        "scored_alpha_strings": len(result.alpha_scores),
        "scored_beta_strings": len(result.beta_scores),
        "selected_alpha": result.selected_alpha.tolist(),
        "selected_beta": result.selected_beta.tolist(),
        "pilot_scope": "development seed only; confirmation seeds 157-186 untouched",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
