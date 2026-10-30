from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

from chemistry.prepare import PreparedHamiltonian


@dataclass
class SQDResult:
    electronic_energy: float
    total_energy: float
    error_mha: float | None
    subspace_dimension: int
    classical_solve_count: int
    classical_solve_dimensions: tuple[int, ...]
    classical_solve_elapsed: float
    sci_result: SCIResult


def canonicalize_ci_strings(ci_strings):
    """Return CI string pairs in the sorted order required by selected-CI."""
    return [
        (np.sort(np.asarray(strings_a)), np.sort(np.asarray(strings_b)))
        for strings_a, strings_b in ci_strings
    ]


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
    batch_observer: Callable | None = None,
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
    classical_solve_dimensions: list[int] = []
    classical_solve_elapsed = 0.0

    def solve_and_record(ci_strings, one_body_tensor, two_body_tensor, norb, nelec):
        nonlocal classical_solve_elapsed
        ci_strings = canonicalize_ci_strings(ci_strings)
        classical_solve_dimensions.extend(
            int(strings_a.size * strings_b.size)
            for strings_a, strings_b in ci_strings
        )
        solve_start = perf_counter()
        results = solve_sci_batch(
            ci_strings,
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
        )
        classical_solve_elapsed += perf_counter() - solve_start
        if batch_observer is not None:
            batch_observer(ci_strings, results)
        return results

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
        sci_solver=solve_and_record,
    )

    electronic_energy = float(result.energy)
    total_energy = electronic_energy + prepared.constant_energy

    if result.sci_state is None:
        raise RuntimeError("Expected the default solver to return an SCI state.")

    error_mha = (
        None
        if prepared.fci_energy is None
        else 1000 * abs(total_energy - prepared.fci_energy)
    )

    return SQDResult(
        electronic_energy=electronic_energy,
        total_energy=total_energy,
        error_mha=error_mha,
        subspace_dimension=int(result.sci_state.amplitudes.size),
        classical_solve_count=len(classical_solve_dimensions),
        classical_solve_dimensions=tuple(classical_solve_dimensions),
        classical_solve_elapsed=classical_solve_elapsed,
        sci_result=result,
    )