import unittest

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

from chemistry.prepare import prepare_registered
from quantum.SqDRIFT import sample_sqdrift
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


if __name__ == "__main__":
    unittest.main()
