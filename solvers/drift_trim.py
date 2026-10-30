from dataclasses import dataclass
from time import perf_counter

from chemistry.prepare import PreparedHamiltonian
from quantum.SqDRIFT import SqDriftResult, sample_sqdrift
from solvers.sqd import SQDResult
from solvers.trim_sqd import run_trim_sqd


@dataclass
class DriftTrimResult:
    samples: SqDriftResult
    solution: SQDResult
    sampling_seconds: float
    solver_seconds: float

    @property
    def total_energy(self) -> float:
        return self.solution.total_energy

    @property
    def error_mha(self) -> float | None:
        return self.solution.error_mha

    @property
    def total_seconds(self) -> float:
        return self.sampling_seconds + self.solver_seconds


def run_drift_trim(
    prepared: PreparedHamiltonian,
    *,
    # SqDRIFT sampling
    times: tuple[float, ...] = (1.0, 5.0),
    circuits_per_time: int = 10,
    num_steps: int = 20,
    shots_per_circuit: int = 256,
    sampling_seed: int = 42,
    # SQD iteration
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 5,
    max_dim: int = 256,
    solver_seed: int = 42,
    # Trimming and carryover
    trim_ratio: float = 0.5,
    carryover_ratio: float = 0.5,
    carryover_threshold: float = 1e-4,
    max_strings_per_trim: int = 128,
    max_carryover: int = 128,
) -> DriftTrimResult:
    start = perf_counter()

    samples = sample_sqdrift(
        prepared,
        times=times,
        circuits_per_time=circuits_per_time,
        num_steps=num_steps,
        shots_per_circuit=shots_per_circuit,
        seed=sampling_seed,
    )

    sampling_seconds = perf_counter() - start
    start = perf_counter()

    solution = run_trim_sqd(
        prepared,
        samples.counts,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        trim_ratio=trim_ratio,
        carryover_ratio=carryover_ratio,
        carryover_threshold=carryover_threshold,
        max_strings_per_trim=max_strings_per_trim,
        max_carryover=max_carryover,
        seed=solver_seed,
    )

    solver_seconds = perf_counter() - start

    return DriftTrimResult(
        samples=samples,
        solution=solution,
        sampling_seconds=sampling_seconds,
        solver_seconds=solver_seconds,
    )