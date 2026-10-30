from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
from qiskit_fermions.operators.terms.ordering import canonical_order
from qiskit_fermions.transpiler import FermionicPassManager
from qiskit_fermions.transpiler.passes import QDriftTrotterization
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

from chemistry.prepare import PreparedHamiltonian
from quantum.hamiltonian import build_fermionic_hamiltonian


@dataclass
class SqDriftResult:
    counts: dict[str, int]
    counts_per_circuit: list[dict[str, int]]
    circuits: list[QuantumCircuit]
    metadata: list[dict]

    @property
    def total_shots(self) -> int:
        return sum(self.counts.values())


def hartree_fock_occupation(
    prepared: PreparedHamiltonian,
) -> list[bool]:
    num_alpha, num_beta = prepared.num_electrons
    ncore = len(prepared.frozen_orbitals)

    if num_alpha != num_beta:
        raise ValueError("This initializer supports closed-shell singlets.")

    total_occupied = ncore + num_alpha

    occupied = [
        original_index < total_occupied
        for original_index in prepared.active_orbitals
    ]

    if sum(occupied) != num_alpha:
        raise ValueError("Active orbitals do not match the HF electron count.")

    return occupied + occupied


def sample_sqdrift(
    prepared: PreparedHamiltonian,
    *,
    times: tuple[float, ...] = (1.0, 5.0),
    circuits_per_time: int = 10,
    num_steps: int = 20,
    shots_per_circuit: int = 256,
    seed: int = 42,
    simulator_method: str = "statevector",
    simulator_options: Mapping[str, Any] | None = None,
) -> SqDriftResult:
    for name, value in (
        ("circuits_per_time", circuits_per_time),
        ("num_steps", num_steps),
        ("shots_per_circuit", shots_per_circuit),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(simulator_method, str) or not simulator_method:
        raise ValueError("simulator_method must be a non-empty string.")
    if simulator_options is not None and not isinstance(simulator_options, Mapping):
        raise ValueError("simulator_options must be a mapping.")

    times = tuple(float(t) for t in times)
    if not times or any(not np.isfinite(t) or t < 0 for t in times):
        raise ValueError("times must contain finite, nonnegative values.")

    n = prepared.num_orbitals
    num_modes = 2 * n
    occupation = hartree_fock_occupation(prepared)

    operator = build_fermionic_hamiltonian(prepared)
    grouped = canonical_order(operator.normal_ordered().simplify(atol=1e-14))

    group_terms_by_electronic_structure(
        grouped,
        num_modes,
        two_body_physicist_order=False,
    )

    options = {"max_parallel_threads": 2}
    if simulator_options is not None:
        options.update(simulator_options)
    simulator = AerSimulator(method=simulator_method, **options)
    rng = np.random.default_rng(seed)

    circuits = []
    counts_per_circuit = []
    metadata = []
    combined = Counter()

    for time in times:
        for replica in range(circuits_per_time):
            circuit_seed, transpiler_seed, simulator_seed = (int(value) for value in rng.integers(0, 2**31 - 1, size=3))

            fermionic_circuit = FermionicCircuit(num_modes)
            fermionic_circuit.append(
                InitializeModes(occupation),
                fermionic_circuit.modes,
            )

            # t=0 is useful for checking HF initialization and bit ordering.
            if time != 0.0:
                fermionic_circuit.append(
                    Evolution(num_modes, grouped, time),
                    fermionic_circuit.modes,
                )

            manager = generate_preset_jw_pass_manager()
            manager.optimization = FermionicPassManager([
                QDriftTrotterization(
                    num_steps,
                    rng=circuit_seed,
                )
            ])

            circuit = manager.run(fermionic_circuit)
            circuit.measure_all()

            # A fixed basis makes recorded two-qubit gate counts interpretable.
            circuit = transpile(
                circuit,
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=1,
                seed_transpiler=transpiler_seed,
            )

            execution = simulator.run(
                circuit,
                shots=shots_per_circuit,
                seed_simulator=simulator_seed,
            ).result()

            counts = dict(execution.get_counts(0))

            if sum(counts.values()) != shots_per_circuit:
                raise RuntimeError("Simulator returned an unexpected shot count.")

            # Check spin-sector populations before introducing any noise.
            num_alpha, num_beta = prepared.num_electrons
            for bitstring in counts:
                if (
                    len(bitstring) != num_modes
                    or bitstring[n:].count("1") != num_alpha
                    or bitstring[:n].count("1") != num_beta
                ):
                    raise RuntimeError(f"Unexpected electron counts in {bitstring!r}.")

            circuits.append(circuit)
            counts_per_circuit.append(counts)
            combined.update(counts)

            metadata.append({
                "time": time,
                "replica": replica,
                "num_steps": num_steps if time != 0.0 else 0,
                "shots": shots_per_circuit,
                "circuit_seed": circuit_seed,
                "transpiler_seed": transpiler_seed,
                "simulator_seed": simulator_seed,
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "cx_count": int(circuit.count_ops().get("cx", 0)),
            })

    return SqDriftResult(
        counts=dict(combined),
        counts_per_circuit=counts_per_circuit,
        circuits=circuits,
        metadata=metadata,
    )