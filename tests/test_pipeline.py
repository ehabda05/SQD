import unittest

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

from chemistry.prepare import prepare_registered
from quantum.SqDRIFT import sample_sqdrift
from quantum.recovery import recover_sample_counts
from solvers.accumulated_marginal import run_accumulated_marginal
from solvers.drift_trim import run_drift_trim
from solvers.sqd import canonicalize_ci_strings, run_sqd


class PipelineTests(unittest.TestCase):
    def test_ci_string_order_is_canonicalized_before_solving(self):
        prepared = prepare_registered("h2", compute_fci=False)
        strings = np.array([1, 2])
        reordered = [(strings[::-1], strings[[1, 0]])]
        canonical = canonicalize_ci_strings(reordered)

        ordered_result = solve_sci(
            canonical[0],
            prepared.h1,
            prepared.h2,
            norb=prepared.num_orbitals,
            nelec=prepared.num_electrons,
        )
        reordered_result = solve_sci(
            canonicalize_ci_strings(reordered)[0],
            prepared.h1,
            prepared.h2,
            norb=prepared.num_orbitals,
            nelec=prepared.num_electrons,
        )
        _, matrix = fci.direct_spin1.pspace(
            prepared.h1,
            prepared.h2,
            prepared.num_orbitals,
            prepared.num_electrons,
            np=4,
        )

        self.assertAlmostEqual(ordered_result.energy, reordered_result.energy)
        self.assertAlmostEqual(ordered_result.energy, np.linalg.eigvalsh(matrix)[0])

    def test_matrix_product_state_without_fci_reference(self):
        prepared = prepare_registered("h2", compute_fci=False)
        sampled = sample_sqdrift(
            prepared,
            times=(0.0,),
            circuits_per_time=1,
            num_steps=1,
            shots_per_circuit=8,
            simulator_method="matrix_product_state",
        )

        self.assertEqual(sampled.total_shots, 8)
        result = run_sqd(
            prepared,
            sampled.counts,
            samples_per_batch=8,
            num_batches=1,
            max_iterations=1,
            max_dim=4,
        )
        self.assertIsNone(result.error_mha)

    def test_accumulated_marginal_returns_retained_solution_metadata(self):
        prepared = prepare_registered("h2", compute_fci=False)
        counts = {
            "0101": 4,
            "0110": 2,
            "1001": 2,
            "1010": 2,
        }
        result = run_accumulated_marginal(
            prepared,
            counts,
            samples_per_batch=2,
            num_batches=1,
            max_iterations=1,
            max_dim=2,
            strings_per_spin=2,
        )
        self.assertLessEqual(result.selected_dimension, 4)
        self.assertEqual(result.classical_solve_count, len(result.classical_solve_dimensions))
        self.assertEqual(
            result.retained_total_energy,
            result.retained_energy + prepared.constant_energy,
        )
        self.assertEqual(
            result.retained.sci_state.amplitudes.size,
            result.subspace_dimension,
        )
        self.assertEqual(
            result.retained_amplitudes.size,
            result.retained_alpha.size * result.retained_beta.size,
        )

    def test_nonzero_sqdrift_evolution_records_circuit_resources(self):
        prepared = prepare_registered("h2", compute_fci=False)
        sampled = sample_sqdrift(
            prepared,
            times=(1.0,),
            circuits_per_time=1,
            num_steps=2,
            shots_per_circuit=8,
            seed=3,
        )
        self.assertEqual(sampled.total_shots, 8)
        self.assertGreater(sampled.metadata[0]["depth"], 0)
        self.assertIn("cx_count", sampled.metadata[0])

    def test_noisy_recovery_preserves_shots_and_sector(self):
        prepared = prepare_registered("h2", compute_fci=False)
        recovered, diagnostics = recover_sample_counts(
            prepared,
            {"0101": 8, "0000": 2},
            seed=7,
        )
        self.assertEqual(sum(recovered.values()), 10)
        self.assertEqual(diagnostics["invalid_shots"], 2)
        for bitstring in recovered:
            self.assertEqual(bitstring[:2].count("1"), 1)
            self.assertEqual(bitstring[2:].count("1"), 1)

    def test_production_runner_loads_saved_counts(self):
        prepared = prepare_registered("h2", compute_fci=False)
        result = run_drift_trim(
            prepared,
            method="accumulated-marginal",
            counts={"0101": 4, "0110": 2, "1001": 2, "1010": 2},
            samples_per_batch=2,
            num_batches=1,
            max_iterations=1,
            max_dim=2,
            strings_per_spin=2,
        )
        self.assertEqual(result.sampling_seconds, 0.0)
        self.assertEqual(result.pool_hash, run_drift_trim(
            prepared,
            method="accumulated-marginal",
            counts={"0101": 4, "0110": 2, "1001": 2, "1010": 2},
            samples_per_batch=2,
            num_batches=1,
            max_iterations=1,
            max_dim=2,
            strings_per_spin=2,
        ).pool_hash)


if __name__ == "__main__":
    unittest.main()
