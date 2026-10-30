"""Audit one unrestricted and selected SCI calculation against explicit matrices."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from benchmark.ci_diagnostic import top_indices  # noqa: E402


def solve(prepared, strings_a, strings_b):
    result = solve_sci(
        (np.asarray(strings_a), np.asarray(strings_b)),
        prepared.h1,
        prepared.h2,
        norb=prepared.num_orbitals,
        nelec=prepared.num_electrons,
    )
    return float(result.energy) + prepared.constant_energy, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default="h2o")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--strings-per-spin", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    counts = json.loads(args.pool.read_text())
    n = prepared.num_orbitals
    pool_a = sorted({int(bitstring[n:], 2) for bitstring in counts})
    pool_b = sorted({int(bitstring[:n], 2) for bitstring in counts})
    unrestricted_energy, unrestricted = solve(prepared, pool_a, pool_b)
    state = unrestricted.sci_state
    vector = np.asarray(state.amplitudes).reshape(-1)
    alpha = np.asarray(state.ci_strs_a)
    beta = np.asarray(state.ci_strs_b)

    if vector.size != alpha.size * beta.size:
        raise RuntimeError("SCI vector shape does not match CI string dimensions.")
    vector = vector / np.linalg.norm(vector)

    na, nb = alpha.size, beta.size
    addr, matrix = fci.direct_spin1.pspace(
        np.asarray(prepared.h1),
        np.asarray(prepared.h2),
        n,
        prepared.num_electrons,
        np=na * nb,
    )
    if not np.array_equal(addr, np.arange(na * nb)):
        raise RuntimeError("Full pspace matrix did not return all addresses in order.")

    strings_a = fci.cistring.addrs2str(n, prepared.num_electrons[0], np.arange(na))
    strings_b = fci.cistring.addrs2str(n, prepared.num_electrons[1], np.arange(nb))
    address_a = {int(value): index for index, value in enumerate(strings_a)}
    address_b = {int(value): index for index, value in enumerate(strings_b)}
    row_indices = [address_a[int(value)] for value in alpha]
    col_indices = [address_b[int(value)] for value in beta]
    full_indices = np.array(
        [row * nb + column for row in row_indices for column in col_indices]
    )

    full_vector = vector
    direct_energy = float(np.vdot(full_vector, matrix @ full_vector).real)
    residual = np.linalg.norm(matrix @ full_vector - direct_energy * full_vector)
    eigenvalues = np.linalg.eigvalsh(matrix)
    matrix_ground = float(eigenvalues[0])
    matrix_maximum = float(eigenvalues[-1])

    weights = np.abs(np.asarray(state.amplitudes)) ** 2
    alpha_indices = top_indices(np.sum(weights, axis=1), args.strings_per_spin)
    beta_indices = top_indices(np.sum(weights, axis=0), args.strings_per_spin)
    selected_indices = np.array(
        [
            row * nb + column
            for row in alpha_indices
            for column in beta_indices
        ]
    )
    selected_vector = full_vector[selected_indices]
    retained_weight = float(np.vdot(selected_vector, selected_vector).real)
    projected_vector = selected_vector / np.sqrt(retained_weight)
    selected_matrix = matrix[np.ix_(selected_indices, selected_indices)]
    projected_energy = float(
        np.vdot(projected_vector, selected_matrix @ projected_vector).real
    )
    selected_ground = float(np.linalg.eigvalsh(selected_matrix)[0])
    selected_strings_a = np.sort(alpha[alpha_indices])
    selected_strings_b = np.sort(beta[beta_indices])
    selected_solver_energy, selected_result = solve(
        prepared, selected_strings_a, selected_strings_b
    )
    selected_solver_electronic = selected_solver_energy - prepared.constant_energy
    returned_state = selected_result.sci_state
    returned_vector = np.asarray(returned_state.amplitudes).reshape(-1)
    returned_vector = returned_vector / np.linalg.norm(returned_vector)
    returned_alpha = np.asarray(returned_state.ci_strs_a)
    returned_beta = np.asarray(returned_state.ci_strs_b)
    returned_rows = [address_a[int(value)] for value in returned_alpha]
    returned_cols = [address_b[int(value)] for value in returned_beta]
    returned_indices = np.array(
        [row * nb + column for row in returned_rows for column in returned_cols]
    )
    returned_matrix = matrix[np.ix_(returned_indices, returned_indices)]
    returned_vector_energy = float(
        np.vdot(returned_vector, returned_matrix @ returned_vector).real
    )
    returned_residual = np.linalg.norm(
        returned_matrix @ returned_vector
        - returned_vector_energy * returned_vector
    )

    output = {
        "pool": str(args.pool),
        "dimension": int(matrix.shape[0]),
        "constant_energy": prepared.constant_energy,
        "unrestricted_solver_total_energy": unrestricted_energy,
        "unrestricted_matrix_ground_total_energy": (
            matrix_ground + prepared.constant_energy
        ),
        "unrestricted_vector_energy_total": (
            direct_energy + prepared.constant_energy
        ),
        "spectral_maximum_total_energy": (
            matrix_maximum + prepared.constant_energy
        ),
        "unrestricted_vector_norm": float(np.vdot(full_vector, full_vector).real),
        "unrestricted_eigenvector_residual": float(residual),
        "selected_dimension": int(selected_matrix.shape[0]),
        "retained_weight": retained_weight,
        "selected_projected_total_energy": (
            projected_energy + prepared.constant_energy
        ),
        "projected_loss_bound": float(
            (1.0 - retained_weight) * (matrix_maximum - matrix_ground)
        ),
        "projected_loss": float(projected_energy - matrix_ground),
        "selected_matrix_ground_total_energy": (
            selected_ground + prepared.constant_energy
        ),
        "selected_solver_total_energy": selected_solver_energy,
        "selected_solver_electronic_energy": selected_solver_electronic,
        "selected_solver_energy_difference": (
            selected_solver_energy
            - (selected_ground + prepared.constant_energy)
        ),
        "selected_returned_vector_energy_total": (
            returned_vector_energy + prepared.constant_energy
        ),
        "selected_returned_vector_residual": float(returned_residual),
        "selected_returned_alpha": returned_alpha.tolist(),
        "selected_returned_beta": returned_beta.tolist(),
        "energy_ordering_ok": (
            unrestricted_energy
            <= selected_solver_energy
            <= projected_energy + prepared.constant_energy + 1e-10
        ),
        "selected_alpha": selected_strings_a.tolist(),
        "selected_beta": selected_strings_b.tolist(),
    }
    output_path = args.pool.with_name(args.pool.stem + "_audit.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
