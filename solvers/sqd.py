from dataclasses import dataclass

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
)

from chemistry.prepare import PreparedHamiltonian


@dataclass
class SQDResult:
    electronic_energy: float
    total_energy: float
    error_mha: float
    subspace_dimension: int
    sci_result: SCIResult


def run_sqd(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 5,
    max_dim: int = 256,
    seed: int = 42,
    policy=None,
) -> SQDResult:
    if not counts:
        raise ValueError("counts must not be empty.")

    n = prepared.num_orbitals

    for bitstring, count in counts.items():
        if (
            len(bitstring) != 2 * n
            or set(bitstring) - {"0", "1"}
        ):
            raise ValueError(f"Invalid bitstring: {bitstring!r}")

        if type(count) is not int or count <= 0:
            raise ValueError("Counts must be positive integers.")

    for name, value in (
        ("samples_per_batch", samples_per_batch),
        ("num_batches", num_batches),
        ("max_iterations", max_iterations),
        ("max_dim", max_dim),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    num_alpha, num_beta = prepared.num_electrons
    ncore = len(prepared.frozen_orbitals)

    # Initial occupations follow the actual active-orbital ordering.
    initial_alpha = np.array([
        float(index < ncore + num_alpha)
        for index in prepared.active_orbitals
    ])
    initial_beta = np.array([
        float(index < ncore + num_beta)
        for index in prepared.active_orbitals
    ])

    bit_array = BitArray.from_counts(counts)

    result = diagonalize_fermionic_hamiltonian(
        prepared.h1,
        prepared.h2,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=n,
        nelec=prepared.num_electrons,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        energy_tol=1e-9,
        occupancies_tol=1e-6,
        initial_occupancies=(initial_alpha, initial_beta),
        symmetrize_spin=False,
        seed=seed,
        policy=policy,
    )

    electronic_energy = float(result.energy)
    total_energy = electronic_energy + prepared.constant_energy

    if result.sci_state is None:
        raise RuntimeError("Expected the default solver to return an SCI state.")

    return SQDResult(
        electronic_energy=electronic_energy,
        total_energy=total_energy,
        error_mha=1000 * abs(total_energy - prepared.fci_energy),
        subspace_dimension=int(result.sci_state.amplitudes.size),
        sci_result=result,
    )