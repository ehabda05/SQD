import unittest

import numpy as np
from pyscf import gto, mcscf, scf

from chemistry.prepare import prepare_registered
from molecules import MOLECULES

class PreparationTests(unittest.TestCase):
    def test_h2_full_space(self):
        result = prepare_registered("h2")

        self.assertEqual(result.num_electrons, (1, 1))
        self.assertEqual(result.active_orbitals, (0, 1))
        self.assertEqual(result.frozen_orbitals, ())

        self.assertAlmostEqual(
            result.constant_energy,
            result.nuclear_repulsion,
            places=10,
        )
        self.assertAlmostEqual(
            result.hf_energy,
            -1.116998996754,
            places=7,
        )
        self.assertAlmostEqual(
            result.fci_energy,
            -1.137306035754,
            places=7,
        )

        self.assert_integrals(result)

    def test_lih_matches_independent_casci(self):
        result = prepare_registered("lih")

        self.assertEqual(result.num_electrons, (1, 1))
        self.assertEqual(result.active_orbitals, (1, 2, 3))
        self.assertEqual(result.frozen_orbitals, (0,))

        molecule = MOLECULES["lih"]["molecule"]

        mol = gto.M(
            atom=molecule.to_pyscf_string(),
            basis=molecule.basis,
            charge=molecule.charge,
            spin=molecule.spin,
            unit="Angstrom",
            verbose=0,
        )

        hf = scf.RHF(mol).run(conv_tol=1e-10)
        self.assertTrue(hf.converged)

        cas = mcscf.CASCI(hf, 3, 2)
        cas.fcisolver.conv_tol = 1e-10
        cas.kernel()
        self.assertTrue(cas.fcisolver.converged)

        self.assertAlmostEqual(
            result.fci_energy,
            cas.e_tot,
            places=8,
        )
        self.assertLessEqual(
            result.fci_energy,
            result.hf_energy + 1e-9,
        )

        self.assert_integrals(result)

        # Default ordering places active occupied orbitals first.
        occupied = range(result.num_electrons[0])

        energy = result.constant_energy
        energy += 2 * sum(
            result.h1[i, i] for i in occupied
        )
        energy += sum(
            2 * result.h2[i, i, j, j]
            - result.h2[i, j, j, i]
            for i in occupied
            for j in occupied
        )

        self.assertAlmostEqual(
            energy,
            result.hf_energy,
            places=8,
        )

    def test_explicit_orbital_order(self):
        default = prepare_registered("lih")
        reordered = prepare_registered(
            "lih",
            orbital_indices=[3, 1, 2],
        )

        self.assertEqual(
            reordered.active_orbitals,
            (3, 1, 2),
        )
        self.assertAlmostEqual(
            default.fci_energy,
            reordered.fci_energy,
            places=8,
        )

        np.testing.assert_allclose(
            np.diag(reordered.h1),
            np.diag(default.h1)[[2, 0, 1]],
            rtol=1e-7,
            atol=1e-9,
        )

    def test_invalid_spaces(self):
        invalid_settings = [
            {"active_electrons": 3},
            {"active_orbitals": 0},
            {"orbital_indices": [1, 1, 2]},
            {"orbital_indices": [0, 1, 2]},
            {"orbital_indices": [3, 4, 5]},
            {"orbital_indices": [1, 2, 9]},
        ]

        for kwargs in invalid_settings:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    prepare_registered("lih", **kwargs)

    def test_unconverged_hf_fails(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Hartree–Fock",
        ):
            prepare_registered("lih", max_cycle=1)

    def assert_integrals(self, result):
        n = result.num_orbitals

        self.assertEqual(result.h1.shape, (n, n))
        self.assertEqual(result.h2.shape, (n, n, n, n))

        self.assertTrue(np.isfinite(result.h1).all())
        self.assertTrue(np.isfinite(result.h2).all())

        np.testing.assert_allclose(
            result.h1,
            result.h1.T,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.h2,
            result.h2.transpose(1, 0, 2, 3),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.h2,
            result.h2.transpose(2, 3, 0, 1),
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()