"""Reusable accumulated-marginal SQD selection."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.fermion import SCIResult, solve_sci

from chemistry.prepare import PreparedHamiltonian
from solvers.sqd import SQDResult, run_sqd


@dataclass
class AccumulatedMarginalResult:
    collection: SQDResult
    candidate: SCIResult
    retained: SCIResult
    retained_energy: float
    retained_total_energy: float
    retained_error_mha: float | None
    selected_alpha: np.ndarray
    selected_beta: np.ndarray
    selected_dimension: int
    retained_alpha: np.ndarray
    retained_beta: np.ndarray
    retained_amplitudes: np.ndarray
    alpha_scores: dict[int, tuple[float, ...]]
    beta_scores: dict[int, tuple[float, ...]]
    candidate_seconds: float
    collection_seconds: float
    total_classical_seconds: float
    classical_solve_count: int
    classical_solve_dimensions: tuple[int, ...]

    @property
    def total_energy(self) -> float:
        return self.retained_total_energy

    @property
    def error_mha(self) -> float | None:
        return self.retained_error_mha

    @property
    def sci_result(self) -> SCIResult:
        return self.retained

    @property
    def subspace_dimension(self) -> int:
        return int(self.retained.sci_state.amplitudes.size)

    @property
    def classical_solve_elapsed(self) -> float:
        return (
            self.collection.classical_solve_elapsed
            + self.candidate.classical_solve_elapsed
        )


def mean_present(scores: dict[int, list[float]]) -> dict[int, float]:
    return {
        string: float(np.mean(values))
        for string, values in scores.items()
        if values
    }


def select_top_strings(
    scores: dict[int, list[float] | tuple[float, ...]],
    count: int,
) -> np.ndarray:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer.")
    means = mean_present(scores)
    ranked = sorted(means, key=lambda string: (-means[string], string))
    return np.array(sorted(ranked[:count]), dtype=np.int64)


def run_accumulated_marginal(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 5,
    max_dim: int = 256,
    strings_per_spin: int | None = None,
    seed: int = 42,
) -> AccumulatedMarginalResult:
    """Collect SQD marginal weights, select a capped Cartesian candidate, retain best."""
    if type(max_dim) is not int or max_dim < 1:
        raise ValueError("max_dim must be a positive integer.")
    if strings_per_spin is None:
        strings_per_spin = max_dim
    if type(strings_per_spin) is not int or strings_per_spin < 1:
        raise ValueError("strings_per_spin must be a positive integer.")
    if strings_per_spin > max_dim:
        raise ValueError("strings_per_spin cannot exceed max_dim.")

    alpha_scores: dict[int, list[float]] = {}
    beta_scores: dict[int, list[float]] = {}

    def observe(ci_strings, results):
        for (strings_a, strings_b), result in zip(ci_strings, results):
            if result.sci_state is None:
                raise RuntimeError("SQD collection returned no SCI state.")
            amplitudes = np.asarray(result.sci_state.amplitudes)
            weights = np.abs(amplitudes) ** 2
            for string, weight in zip(strings_a, np.sum(weights, axis=1)):
                alpha_scores.setdefault(int(string), []).append(float(weight))
            for string, weight in zip(strings_b, np.sum(weights, axis=0)):
                beta_scores.setdefault(int(string), []).append(float(weight))

    total_start = perf_counter()
    collection_start = perf_counter()
    collection = run_sqd(
        prepared,
        counts,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        seed=seed,
        batch_observer=observe,
    )
    collection_seconds = perf_counter() - collection_start
    selected_alpha = select_top_strings(alpha_scores, strings_per_spin)
    selected_beta = select_top_strings(beta_scores, strings_per_spin)
    selected_dimension = int(selected_alpha.size * selected_beta.size)
    if selected_dimension > max_dim * max_dim:
        raise RuntimeError("Selected candidate exceeds the per-spin determinant cap.")

    start = perf_counter()
    candidate = solve_sci(
        (selected_alpha, selected_beta),
        prepared.h1,
        prepared.h2,
        norb=prepared.num_orbitals,
        nelec=prepared.num_electrons,
    )
    candidate_seconds = perf_counter() - start
    candidate_energy = float(candidate.energy)
    if candidate_energy < collection.electronic_energy:
        retained = candidate
        retained_energy = candidate_energy
    else:
        retained = collection.sci_result
        retained_energy = collection.electronic_energy
    retained_total = retained_energy + prepared.constant_energy
    error = (
        None
        if prepared.fci_energy is None
        else 1000 * abs(retained_total - prepared.fci_energy)
    )
    return AccumulatedMarginalResult(
        collection=collection,
        candidate=candidate,
        retained=retained,
        retained_energy=retained_energy,
        retained_total_energy=retained_total,
        retained_error_mha=error,
        selected_alpha=selected_alpha,
        selected_beta=selected_beta,
        selected_dimension=selected_dimension,
        retained_alpha=np.asarray(retained.sci_state.ci_strs_a),
        retained_beta=np.asarray(retained.sci_state.ci_strs_b),
        retained_amplitudes=np.asarray(retained.sci_state.amplitudes),
        alpha_scores={key: tuple(value) for key, value in alpha_scores.items()},
        beta_scores={key: tuple(value) for key, value in beta_scores.items()},
        candidate_seconds=candidate_seconds,
        collection_seconds=collection_seconds,
        total_classical_seconds=perf_counter() - total_start,
        classical_solve_count=collection.classical_solve_count + 1,
        classical_solve_dimensions=(
            *collection.classical_solve_dimensions,
            selected_dimension,
        ),
    )
