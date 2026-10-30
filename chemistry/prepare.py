from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from .molecule import Molecule


@dataclass(frozen=True)
class PreparedHamiltonian:
    molecule: Molecule
    active_orbitals: tuple[int, ...]
    frozen_orbitals: tuple[int, ...]
    num_electrons: tuple[int, int]
    h1: np.ndarray
    h2: np.ndarray
    constant_energy: float
    nuclear_repulsion: float
    hf_energy: float
    fci_energy: float | None
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
    compute_fci: bool = True,
    fci_cache_dir: str | Path | None = None,
) -> PreparedHamiltonian:
    if molecule.spin != 0:
        raise ValueError("Only closed-shell singlets are supported.")

    if conv_tol <= 0 or max_cycle < 1:
        raise ValueError("conv_tol and max_cycle must be positive.")
    if not isinstance(compute_fci, bool):
        raise ValueError("compute_fci must be a boolean.")

    mol = gto.M(
        atom=molecule.to_pyscf_string(),
        basis=molecule.basis,
        charge=molecule.charge,
        spin=molecule.spin,
        unit="Angstrom",
        verbose=0,
    )

    if mol.nelectron % 2:
        raise ValueError("Closed-shell preparation needs an even electron count.")

    nelec = (
        mol.nelectron
        if active_electrons is None
        else active_electrons
    )

    if (
        not isinstance(nelec, int)
        or nelec <= 0
        or nelec % 2
        or nelec > mol.nelectron
    ):
        raise ValueError(
            "active_electrons must be positive, even, "
            "and no greater than the total electron count."
        )

    ncore = (mol.nelectron - nelec) // 2
    nmo = mol.nao_nr()

    ncas = (
        nmo - ncore
        if active_orbitals is None
        else active_orbitals
    )

    if (
        not isinstance(ncas, int)
        or not 1 <= ncas <= nmo - ncore
        or nelec > 2 * ncas
    ):
        raise ValueError("Invalid active orbital count or electron capacity.")

    indices = (
        tuple(range(ncore, ncore + ncas))
        if orbital_indices is None
        else tuple(orbital_indices)
    )

    if (
        len(indices) != ncas
        or len(set(indices)) != ncas
        or any(
            not isinstance(i, int) or i < 0 or i >= nmo
            for i in indices
        )
    ):
        raise ValueError(
            "orbital_indices must be unique valid MO indices "
            "matching active_orbitals."
        )

    frozen = tuple(i for i in range(nmo) if i not in indices)[:ncore]

    occupied_hf = set(range(mol.nelectron // 2))

    if not set(frozen).issubset(occupied_hf):
        raise ValueError("Frozen orbitals must be occupied HF orbitals.")

    if not occupied_hf.issubset(set(frozen) | set(indices)):
        raise ValueError("Active space would discard an occupied HF orbital.")

    hf = scf.RHF(mol)
    hf.conv_tol = conv_tol
    hf.max_cycle = max_cycle
    hf.kernel()

    if not hf.converged:
        raise RuntimeError(f"Hartree–Fock failed to converge for {molecule.name}.")

    cas = mcscf.CASCI(hf, ncas, nelec)
    coefficients = cas.sort_mo(list(indices), base=0)

    h1, constant_energy = cas.get_h1eff(coefficients)
    h2 = ao2mo.restore(1, cas.get_h2eff(coefficients), ncas,)

    num_electrons = (nelec // 2, nelec // 2)

    fci_energy = None
    if compute_fci:
        cache_path = _fci_cache_path(
            fci_cache_dir,
            molecule=molecule,
            active_electrons=nelec,
            active_orbitals=ncas,
            orbital_indices=indices,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
        )
        if cache_path is not None and cache_path.exists():
            fci_energy = float(json.loads(cache_path.read_text())["fci_energy"])
        else:
            solver = fci.direct_spin0.FCI(mol)
            solver.conv_tol = conv_tol
            solver.max_cycle = max_cycle
            fci_energy, _ = solver.kernel(
                h1,
                h2,
                ncas,
                num_electrons,
                ecore=constant_energy,
            )
            if not solver.converged:
                raise RuntimeError(
                    f"Active-space FCI failed to converge for {molecule.name}."
                )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"fci_energy": float(fci_energy)}, sort_keys=True)
                )

    return PreparedHamiltonian(
        molecule=molecule,
        active_orbitals=indices,
        frozen_orbitals=frozen,
        num_electrons=num_electrons,
        h1=np.array(h1),
        h2=np.array(h2),
        constant_energy=float(constant_energy),
        nuclear_repulsion=float(mol.energy_nuc()),
        hf_energy=float(hf.e_tot),
        fci_energy=None if fci_energy is None else float(fci_energy),
        mo_coeff=np.array(
            coefficients[:, ncore:ncore + ncas]
        ),
    )


def prepare_registered(name: str, **kwargs) -> PreparedHamiltonian:
    """Prepare a registered molecule with optional overrides."""
    from molecules import MOLECULES

    config = dict(MOLECULES[name])
    config.update(kwargs)
    return prepare_molecule(**config)


def _fci_cache_path(
    cache_dir: str | Path | None,
    *,
    molecule: Molecule,
    active_electrons: int,
    active_orbitals: int,
    orbital_indices: tuple[int, ...],
    conv_tol: float,
    max_cycle: int,
) -> Path | None:
    if cache_dir is None:
        return None
    key_data = {
        "molecule": {
            "name": molecule.name,
            "atoms": [
                {"symbol": atom.symbol, "position": list(atom.position)}
                for atom in molecule.atoms
            ],
            "charge": molecule.charge,
            "spin": molecule.spin,
            "basis": molecule.basis,
        },
        "active_electrons": active_electrons,
        "active_orbitals": active_orbitals,
        "orbital_indices": list(orbital_indices),
        "conv_tol": conv_tol,
        "max_cycle": max_cycle,
    }
    key = hashlib.sha256(
        json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Path(cache_dir) / f"{molecule.name.lower()}-{key}.json"