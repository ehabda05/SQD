from qiskit_addon_sqd.trim import TrimPolicy

from chemistry.prepare import PreparedHamiltonian
from solvers.sqd import SQDResult, run_sqd


def run_trim_sqd(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 5,
    max_dim: int = 256,
    trim_ratio: float = 0.5,
    carryover_ratio: float = 0.5,
    carryover_threshold: float = 1e-4,
    max_strings_per_trim: int = 128,
    max_carryover: int = 128,
    seed: int = 42,
) -> SQDResult:
    policy = TrimPolicy(
        trim_ratio=trim_ratio,
        carryover_ratio=carryover_ratio,
        carryover_threshold=carryover_threshold,
        max_strings_per_trim=max_strings_per_trim,
        max_carryover=max_carryover,
    )

    return run_sqd(
        prepared,
        counts,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        seed=seed,
        policy=policy,
    )