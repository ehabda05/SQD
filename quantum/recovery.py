"""Configuration recovery helpers for noisy SqDRIFT sample pools."""

from collections import Counter
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.configuration_recovery import recover_configurations

from chemistry.prepare import PreparedHamiltonian


def _validate_counts(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
) -> tuple[int, list[str]]:
    if not counts:
        raise ValueError("counts must not be empty.")
    width = 2 * prepared.num_orbitals
    alpha_electrons, beta_electrons = prepared.num_electrons
    invalid = []
    total = 0
    for bitstring, count in counts.items():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) != width
            or set(bitstring) - {"0", "1"}
        ):
            raise ValueError(f"Invalid bitstring: {bitstring!r}")
        if type(count) is not int or count <= 0:
            raise ValueError("Counts must be positive integers.")
        total += count
        if (
            bitstring[: prepared.num_orbitals].count("1") != beta_electrons
            or bitstring[prepared.num_orbitals :].count("1") != alpha_electrons
        ):
            invalid.append(bitstring)
    return total, invalid


def _counts_from_probabilities(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    shots: int,
) -> dict[str, int]:
    if bitstrings.size == 0:
        raise ValueError("Configuration recovery produced no configurations.")
    raw = np.asarray(probabilities, dtype=float) * shots
    floors = np.floor(raw).astype(int)
    remainder = shots - int(floors.sum())
    if remainder:
        order = np.argsort(-(raw - floors))
        floors[order[:remainder]] += 1
    counts = Counter()
    for bits, count in zip(bitstrings, floors):
        if count:
            counts["".join("1" if value else "0" for value in bits)] += int(count)
    if sum(counts.values()) != shots:
        raise RuntimeError("Recovered counts do not preserve the shot count.")
    return dict(counts)


def recover_sample_counts(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
    *,
    seed: int = 42,
) -> tuple[dict[str, int], dict]:
    """Recover noisy configurations using the installed Qiskit SQD routine.

    The raw counts are never mutated. Invalid-sector outcomes are repaired into
    fixed-spin configurations; valid-sector outcomes may also be changed by
    the occupancy-guided recovery, so recovery is not equivalent to filtering.
    """
    shots, invalid = _validate_counts(prepared, counts)
    n = prepared.num_orbitals
    bitstrings = np.array(
        [[bit == "1" for bit in string] for string in counts],
        dtype=bool,
    )
    probabilities = np.array(list(counts.values()), dtype=float)
    probabilities /= probabilities.sum()
    average_alpha = np.zeros(n)
    average_beta = np.zeros(n)
    for string, count in counts.items():
        weight = count / shots
        average_beta += weight * np.array(
            [bit == "1" for bit in string[:n][::-1]], dtype=float
        )
        average_alpha += weight * np.array(
            [bit == "1" for bit in string[n:][::-1]], dtype=float
        )

    start = perf_counter()
    recovered_bits, recovered_probabilities = recover_configurations(
        bitstrings,
        probabilities,
        (average_alpha, average_beta),
        prepared.num_electrons[0],
        prepared.num_electrons[1],
        rand_seed=seed,
    )
    recovered = _counts_from_probabilities(
        recovered_bits,
        recovered_probabilities,
        shots,
    )
    recovered_shots_total, recovered_invalid = _validate_counts(prepared, recovered)
    if recovered_shots_total != shots:
        raise RuntimeError("Configuration recovery changed the total shot count.")
    if recovered_invalid:
        raise RuntimeError(
            "Configuration recovery returned configurations outside the requested "
            "spin sector."
        )
    recovered_shots = sum(
        abs(recovered.get(string, 0) - counts.get(string, 0))
        for string in set(recovered) | set(counts)
    ) // 2
    return recovered, {
        "enabled": True,
        "invalid_configurations": len(invalid),
        "invalid_shots": sum(counts[string] for string in invalid),
        "recovered_configurations": len(recovered),
        "recovered_shots": recovered_shots,
        "seconds": perf_counter() - start,
        "raw_shots": shots,
    }


def validate_ideal_counts(
    prepared: PreparedHamiltonian,
    counts: dict[str, int],
) -> dict:
    """Reject invalid sectors for ideal sampling and return diagnostics."""
    shots, invalid = _validate_counts(prepared, counts)
    if invalid:
        invalid_shots = sum(counts[string] for string in invalid)
        raise ValueError(
            f"Found {len(invalid)} invalid configurations ({invalid_shots} shots); "
            "enable configuration recovery for an explicitly noisy workflow."
        )
    return {
        "enabled": False,
        "invalid_configurations": 0,
        "invalid_shots": 0,
        "recovered_configurations": 0,
        "recovered_shots": 0,
        "seconds": 0.0,
        "raw_shots": shots,
    }
