import unittest

import numpy as np

from chemistry.prepare import prepare_registered
from quantum.hamiltonian import build_fermionic_hamiltonian


def dense_matrix(operator, num_modes):
    if num_modes > 8:
        raise ValueError("This dense test helper is limited to 8 modes.")

    dimension = 1 << num_modes
    matrix = np.zeros((dimension, dimension), dtype=complex)

    for actions, coefficient in operator.iter_terms():
        for column in range(dimension):
            state = column
            amplitude = complex(coefficient)

            for create, mode in reversed(actions):
                mode = int(mode)

                if not 0 <= mode < num_modes:
                    raise ValueError(f"Invalid fermionic mode: {mode}")

                mask = 1 << mode
                occupied = bool(state & mask)

                # Creation on occupied or annihilation on empty gives zero.
                if bool(create) == occupied:
                    amplitude = 0.0
                    break

                # Anticommutation sign from occupied lower-index modes.
                lower_occupations = (state & (mask - 1)).bit_count()
                if lower_occupations % 2:
                    amplitude *= -1

                state ^= mask

            matrix[state, column] += amplitude

    return matrix


class HamiltonianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = prepare_registered("h2")
        cls.operator = build_fermionic_hamiltonian(cls.prepared)

        cls.n = cls.prepared.num_orbitals
        cls.matrix = dense_matrix(cls.operator, 2 * cls.n)

        # Block-spin ordering: alpha modes first, then beta modes.
        alpha_mask = (1 << cls.n) - 1
        dimension = cls.matrix.shape[0]

        cls.alpha_counts = np.array([
            (state & alpha_mask).bit_count()
            for state in range(dimension)
        ])
        cls.beta_counts = np.array([
            (state >> cls.n).bit_count()
            for state in range(dimension)
        ])

    def test_hermitian(self):
        np.testing.assert_allclose(
            self.matrix,
            self.matrix.conj().T,
            rtol=0,
            atol=1e-12,
        )

    def test_conserves_spin_resolved_electron_counts(self):
        # [N, H][i, j] = (N[i] - N[j]) * H[i, j].
        for counts in (self.alpha_counts, self.beta_counts):
            commutator = (
                counts[:, None] - counts[None, :]
            ) * self.matrix

            np.testing.assert_allclose(
                commutator,
                0.0,
                rtol=0,
                atol=1e-12,
            )

    def test_sector_ground_energy_matches_fci(self):
        num_alpha, num_beta = self.prepared.num_electrons

        sector = np.flatnonzero(
            (self.alpha_counts == num_alpha)
            & (self.beta_counts == num_beta)
        )

        # H2: two choices for alpha and two choices for beta.
        self.assertEqual(len(sector), 4)

        sector_matrix = self.matrix[np.ix_(sector, sector)]
        electronic_energy = np.linalg.eigvalsh(sector_matrix)[0]

        total_energy = (
            electronic_energy + self.prepared.constant_energy
        )

        self.assertAlmostEqual(
            total_energy,
            self.prepared.fci_energy,
            places=9,
        )

    def test_hf_expectation_matches_reference(self):
        # This fixture uses H2's default ordering: occupied orbital first.
        num_alpha, num_beta = self.prepared.num_electrons

        alpha_bits = (1 << num_alpha) - 1
        beta_bits = ((1 << num_beta) - 1) << self.n
        hf_state = alpha_bits | beta_bits

        electronic_energy = self.matrix[hf_state, hf_state]

        self.assertAlmostEqual(electronic_energy.imag, 0.0, places=12)

        total_energy = (
            electronic_energy.real + self.prepared.constant_energy
        )

        self.assertAlmostEqual(
            total_energy,
            self.prepared.hf_energy,
            places=9,
        )

    def test_constant_offset_is_excluded(self):
        # An electronic Hamiltonian has zero energy on the vacuum.
        self.assertAlmostEqual(
            abs(self.matrix[0, 0]),
            0.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()