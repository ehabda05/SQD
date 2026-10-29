# DRIFT-TRIM

Shared chemistry preparation, SqDRIFT sampling, and SQD/TRIM-SQD solver modules
for the research workflow. End-to-end benchmark execution remains in progress.

## Setup and validation

```bash
cd /workspace/SQD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
```

## Prepare a Hamiltonian

Run from the repository root; no PYTHONPATH workaround is required.

```python
from chemistry.prepare import prepare_registered

hamiltonian = prepare_registered('lih')
print(hamiltonian.active_orbitals)  # (1, 2, 3), canonical RHF MO indices
print(hamiltonian.num_electrons)    # (1, 1), alpha and beta counts
print(hamiltonian.fci_energy)       # total active-space reference, Hartree
```

The registry specifies H2 (full space), LiH (2 electrons / 3 orbitals),
H2O (4 / 4), and N2 (6 / 6). Geometries are in Angstrom. Preparation currently
supports closed-shell singlets using canonical restricted Hartree–Fock orbitals.
HF and FCI nonconvergence raise errors rather than returning reference results.

The default active space freezes the lowest occupied orbitals and selects the
next consecutive orbitals. To select a different space, pass `orbital_indices`
(zero-based canonical MO indices, in desired order) and matching
`active_orbitals` / `active_electrons`. Remaining lowest MOs are frozen;
discarding an occupied HF orbital is rejected. Orbitals are not optimized.

`h1` and `h2` are real spatial-orbital integrals; `h2[p,q,r,s]` uses chemist
ordering `(pq|rs)`. See `PreparedHamiltonian` for the second-quantized expression.
`constant_energy` includes both nuclear repulsion and frozen-core contributions;
add it exactly once to an electronic subspace energy. `hf_energy` and
`fci_energy` already include this offset. The returned MO coefficient columns
follow the active orbital order. Qubit/bitstring and spin-orbital ordering must
be specified separately when implementing the sampler.

FCI cost grows exponentially; start with H2 and LiH. The tests verify known H2
energies, integral symmetries, LiH against PySCF CASCI, reconstruction of the HF
energy including the frozen-core offset, and explicit orbital reordering.
