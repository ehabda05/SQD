"""Shared real, spatial-orbital Hamiltonians for closed-shell experiments."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from .molecule import Molecule


@dataclass(frozen=True)
class PreparedHamiltonian:
    """Integrals in Hartree, in the returned active orbital order.

    h2[p,q,r,s] is the chemist integral (pq|rs). The Hamiltonian is
    E_core + sum h1[p,q] a†[p,σ] a[q,σ] +
    1/2 sum h2[p,q,r,s] a†[p,σ] a†[r,τ] a[s,τ] a[q,σ].
    constant_energy includes nuclear repulsion and frozen-core energy.
    Both reported reference energies are TOTAL energies.
    """

    molecule: Molecule
    active_orbitals: tuple[int, ...]
    frozen_orbitals: tuple[int, ...]
    num_electrons: tuple[int, int]
    h1: np.ndarray
    h2: np.ndarray
    constant_energy: float
    nuclear_repulsion: float
    hf_energy: float
    fci_energy: float
    mo_coeff: np.ndarray

    @property
    def num_orbitals(self) -> int:
        return len(self.active_orbitals)


def prepare_molecule(
    molecule: Molecule,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    *,
    orbital_indices: Sequence[int] | None = None,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
) -> PreparedHamiltonian:
    """Run RHF and exact FCI in a fixed canonical-HF active space.

    Geometries use Angstrom; spin follows PySCF (Nalpha - Nbeta).
    Currently only closed-shell singlets are supported. By default freeze
    the lowest doubly occupied orbitals, then select consecutive active MOs.
    Explicit zero-based canonical MO indices preserve the supplied order;
    the lowest remaining orbitals form the frozen core. No orbital optimization
    is performed. FCI scales exponentially: use small active spaces.
    """
    if molecule.spin != 0:
        raise ValueError('Only closed-shell singlet molecules are supported')
    if conv_tol <= 0 or max_cycle < 1:
        raise ValueError('conv_tol and max_cycle must be positive')
    mol = gto.M(
        atom=molecule.to_pyscf_string(), basis=molecule.basis,
        charge=molecule.charge, spin=molecule.spin, unit='Angstrom', verbose=0,
    )
    if mol.nelectron % 2:
        raise ValueError('Closed-shell preparation requires an even electron count')
    nelec = mol.nelectron if active_electrons is None else active_electrons
    if not isinstance(nelec, int) or nelec <= 0 or nelec % 2 or nelec > mol.nelectron:
        raise ValueError('active_electrons must be positive, even, and no larger than total electrons')
    ncore = (mol.nelectron - nelec) // 2
    nmo = mol.nao_nr()
    ncas = nmo - ncore if active_orbitals is None else active_orbitals
    if not isinstance(ncas, int) or not 1 <= ncas <= nmo - ncore or nelec > 2 * ncas:
        raise ValueError('Invalid active orbital count or insufficient electron capacity')
    indices = tuple(range(ncore, ncore + ncas)) if orbital_indices is None else tuple(orbital_indices)
    if (len(indices) != ncas or len(set(indices)) != ncas
            or any(not isinstance(i, int) or i < 0 or i >= nmo for i in indices)):
        raise ValueError('orbital_indices must contain unique valid MO indices matching active_orbitals')
    frozen = tuple(i for i in range(nmo) if i not in indices)[:ncore]
    # Keep the initial determinant in the space: no occupied orbital may be discarded.
    if not set(range(mol.nelectron // 2)).issubset(set(frozen) | set(indices)):
        raise ValueError('Active space would discard an occupied HF orbital')

    hf = scf.RHF(mol)
    hf.conv_tol = conv_tol
    hf.max_cycle = max_cycle
    hf.kernel()
    if not hf.converged:
        raise RuntimeError(f'Hartree–Fock failed to converge for {molecule.name}')
    cas = mcscf.CASCI(hf, ncas, nelec)
    coeff = cas.sort_mo(list(indices), base=0)
    h1, offset = cas.get_h1eff(coeff)
    h2 = ao2mo.restore(1, cas.get_h2eff(coeff), ncas)
    solver = fci.direct_spin0.FCI(mol)
    solver.conv_tol = conv_tol
    solver.max_cycle = max_cycle
    energy, _ = solver.kernel(h1, h2, ncas, (nelec // 2, nelec // 2), ecore=offset)
    if not solver.converged:
        raise RuntimeError(f'Active-space FCI failed to converge for {molecule.name}')
    return PreparedHamiltonian(
        molecule, indices, frozen, (nelec // 2, nelec // 2),
        np.array(h1), np.array(h2), float(offset), float(mol.energy_nuc()),
        float(hf.e_tot), float(energy), np.array(coeff[:, ncore:ncore + ncas]),
    )


def prepare_registered(name: str, **kwargs) -> PreparedHamiltonian:
    """Prepare one of the registry's molecules with its declared active space."""
    from . import MOLECULES

    config = dict(MOLECULES[name])
    config.update(kwargs)
    return prepare_molecule(**config)
