"""Reusable SqDRIFT plus SQD/Trim-SQD/accumulated-marginal pipeline."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

from chemistry.prepare import PreparedHamiltonian
from quantum.SqDRIFT import SqDriftResult, sample_sqdrift
from solvers.accumulated_marginal import (
    AccumulatedMarginalResult,
    run_accumulated_marginal,
)
from solvers.sqd import SQDResult, run_sqd
from solvers.trim_sqd import run_trim_sqd
from solvers.provenance import pool_hash


Method = Literal["standard-sqd", "trim-sqd", "accumulated-marginal"]


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "qiskit", "qiskit-aer", "qiskit-fermions", "qiskit-addon-sqd", "pyscf")
    values = {}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "unavailable"
    return values


def _array_digest(array) -> str:
    value = __import__("numpy").ascontiguousarray(array)
    return hashlib.sha256(
        json.dumps(
            {"dtype": str(value.dtype), "shape": value.shape},
            sort_keys=True,
        ).encode()
        + value.tobytes()
    ).hexdigest()


def _configuration_fingerprint(
    prepared: PreparedHamiltonian,
    *,
    method: Method,
    times: tuple[float, ...],
    circuits_per_time: int,
    num_steps: int,
    shots_per_circuit: int,
    sampling_seed: int,
    simulator_method: str,
    simulator_options: dict | None,
    recover_invalid: bool,
    recovery_seed: int | None,
    samples_per_batch: int,
    num_batches: int,
    max_iterations: int,
    max_dim: int,
    strings_per_spin: int | None,
    solver_seed: int,
    trim_ratio: float,
    carryover_ratio: float,
    carryover_threshold: float,
    max_strings_per_trim: int,
    max_carryover: int,
) -> str:
    molecule = prepared.molecule
    identity = {
        "molecule": {
            "name": molecule.name,
            "basis": molecule.basis,
            "charge": molecule.charge,
            "spin": molecule.spin,
            "atoms": [
                {"symbol": atom.symbol, "position": list(atom.position)}
                for atom in molecule.atoms
            ],
        },
        "active_orbitals": prepared.active_orbitals,
        "frozen_orbitals": prepared.frozen_orbitals,
        "active_electrons": prepared.num_electrons,
        "constant_energy": prepared.constant_energy,
        "nuclear_repulsion": prepared.nuclear_repulsion,
        "h1_digest": _array_digest(prepared.h1),
        "h2_digest": _array_digest(prepared.h2),
        "mo_coeff_digest": _array_digest(prepared.mo_coeff),
        "protocol": {
            "method": method,
            "times": times,
            "circuits_per_time": circuits_per_time,
            "num_steps": num_steps,
            "shots_per_circuit": shots_per_circuit,
            "sampling_seed": sampling_seed,
            "simulator_method": simulator_method,
            "simulator_options": simulator_options or {},
            "recover_invalid": recover_invalid,
            "recovery_seed": recovery_seed,
            "samples_per_batch": samples_per_batch,
            "num_batches": num_batches,
            "max_iterations": max_iterations,
            "max_dim": max_dim,
            "strings_per_spin": strings_per_spin,
            "solver_seed": solver_seed,
            "trim_ratio": trim_ratio,
            "carryover_ratio": carryover_ratio,
            "carryover_threshold": carryover_threshold,
            "max_strings_per_trim": max_strings_per_trim,
            "max_carryover": max_carryover,
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=list).encode()
    ).hexdigest()


@dataclass
class DriftTrimResult:
    method: Method
    samples: SqDriftResult | None
    solution: SQDResult | AccumulatedMarginalResult
    raw_counts: dict[str, int]
    counts: dict[str, int]
    recovery: dict
    sampling_seconds: float
    classical_seconds: float
    pool_hash: str
    configuration_fingerprint: str
    dependency_versions: dict[str, str]
    thread_settings: dict[str, str | None]

    @property
    def total_energy(self) -> float:
        return self.solution.total_energy

    @property
    def error_mha(self) -> float | None:
        return self.solution.error_mha

    @property
    def total_seconds(self) -> float:
        return self.sampling_seconds + self.classical_seconds


def _solve(
    method: Method,
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int,
    num_batches: int,
    max_iterations: int,
    max_dim: int,
    strings_per_spin: int | None,
    solver_seed: int,
    trim_ratio: float,
    carryover_ratio: float,
    carryover_threshold: float,
    max_strings_per_trim: int,
    max_carryover: int,
):
    if method == "standard-sqd":
        return run_sqd(
            prepared,
            counts,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            max_iterations=max_iterations,
            max_dim=max_dim,
            seed=solver_seed,
        )
    if method == "trim-sqd":
        return run_trim_sqd(
            prepared,
            counts,
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
    return run_accumulated_marginal(
        prepared,
        counts,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        strings_per_spin=strings_per_spin,
        seed=solver_seed,
    )


def run_drift_trim(
    prepared: PreparedHamiltonian,
    *,
    method: Method = "trim-sqd",
    counts: dict[str, int] | None = None,
    pool_path: str | Path | None = None,
    times: tuple[float, ...] = (1.0, 5.0),
    circuits_per_time: int = 10,
    num_steps: int = 20,
    shots_per_circuit: int = 256,
    sampling_seed: int = 42,
    simulator_method: str = "statevector",
    simulator_options: dict | None = None,
    recover_invalid: bool = False,
    recovery_seed: int | None = None,
    samples_per_batch: int = 100,
    num_batches: int = 3,
    max_iterations: int = 5,
    max_dim: int = 256,
    strings_per_spin: int | None = None,
    solver_seed: int = 42,
    trim_ratio: float = 0.5,
    carryover_ratio: float = 0.5,
    carryover_threshold: float = 1e-4,
    max_strings_per_trim: int = 128,
    max_carryover: int = 128,
) -> DriftTrimResult:
    """Run one reproducible pipeline from sampling or an existing count pool."""
    if method not in ("standard-sqd", "trim-sqd", "accumulated-marginal"):
        raise ValueError(f"Unknown method: {method}")
    if counts is not None and pool_path is not None:
        raise ValueError("Provide counts or pool_path, not both.")

    samples = None
    sampling_seconds = 0.0
    if pool_path is not None:
        counts = json.loads(Path(pool_path).read_text())
    elif counts is None:
        start = perf_counter()
        samples = sample_sqdrift(
            prepared,
            times=times,
            circuits_per_time=circuits_per_time,
            num_steps=num_steps,
            shots_per_circuit=shots_per_circuit,
            seed=sampling_seed,
            simulator_method=simulator_method,
            simulator_options=simulator_options,
            recover_invalid=recover_invalid,
            recovery_seed=recovery_seed,
        )
        sampling_seconds = perf_counter() - start
        counts = samples.counts

    if not isinstance(counts, dict) or not counts:
        raise ValueError("A nonempty count mapping is required.")
    if samples is None:
        from quantum.recovery import recover_sample_counts, validate_ideal_counts

        raw_counts = dict(counts)
        if recover_invalid:
            counts, recovery = recover_sample_counts(
                prepared,
                raw_counts,
                seed=sampling_seed if recovery_seed is None else recovery_seed,
            )
        else:
            recovery = validate_ideal_counts(prepared, counts)
    else:
        recovery = samples.recovery or {}
        raw_counts = samples.raw_counts or dict(counts)

    recovery_seconds = float(recovery.get("seconds", 0.0))
    sampling_seconds = max(0.0, sampling_seconds - recovery_seconds)
    fingerprint = _configuration_fingerprint(
        prepared,
        method=method,
        times=times,
        circuits_per_time=circuits_per_time,
        num_steps=num_steps,
        shots_per_circuit=shots_per_circuit,
        sampling_seed=sampling_seed,
        simulator_method=simulator_method,
        simulator_options=simulator_options,
        recover_invalid=recover_invalid,
        recovery_seed=recovery_seed,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        strings_per_spin=strings_per_spin,
        solver_seed=solver_seed,
        trim_ratio=trim_ratio,
        carryover_ratio=carryover_ratio,
        carryover_threshold=carryover_threshold,
        max_strings_per_trim=max_strings_per_trim,
        max_carryover=max_carryover,
    )
    start = perf_counter()
    solution = _solve(
        method,
        prepared,
        counts,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        max_dim=max_dim,
        strings_per_spin=strings_per_spin,
        solver_seed=solver_seed,
        trim_ratio=trim_ratio,
        carryover_ratio=carryover_ratio,
        carryover_threshold=carryover_threshold,
        max_strings_per_trim=max_strings_per_trim,
        max_carryover=max_carryover,
    )
    classical_seconds = perf_counter() - start + recovery_seconds
    return DriftTrimResult(
        method=method,
        samples=samples,
        solution=solution,
        raw_counts=raw_counts,
        counts=dict(counts),
        recovery=recovery,
        sampling_seconds=sampling_seconds,
        classical_seconds=classical_seconds,
        pool_hash=pool_hash(counts),
        configuration_fingerprint=fingerprint,
        dependency_versions=_package_versions(),
        thread_settings={
            "OMP_NUM_THREADS": __import__("os").environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": __import__("os").environ.get("OPENBLAS_NUM_THREADS"),
        },
    )
