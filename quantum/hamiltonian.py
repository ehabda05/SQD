from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyscf.tools import fcidump
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump

from chemistry.prepare import PreparedHamiltonian


def build_fermionic_hamiltonian(
    prepared: PreparedHamiltonian,
) -> FermionOperator:
    n = prepared.num_orbitals
    h1 = np.asarray(prepared.h1)
    h2 = np.asarray(prepared.h2)

    if h1.shape != (n, n):
        raise ValueError(f"Expected h1 shape {(n, n)}, got {h1.shape}.")

    if h2.shape != (n, n, n, n):
        raise ValueError(f"Expected h2 shape {(n, n, n, n)}, got {h2.shape}.")

    if np.iscomplexobj(h1) or np.iscomplexobj(h2):
        raise ValueError("This converter supports real integrals only.")

    if not np.isfinite(h1).all() or not np.isfinite(h2).all():
        raise ValueError("Integrals must contain only finite values.")

    # FCIDUMP exploits real-integral permutation symmetry.
    for actual, expected in (
        (h1, h1.T),
        (h2, h2.transpose(1, 0, 2, 3)),
        (h2, h2.transpose(2, 3, 0, 1)),
    ):
        if not np.allclose(actual, expected, rtol=0, atol=1e-10):
            raise ValueError("Integrals do not have the required symmetry.")

    with TemporaryDirectory(prefix="drift_trim_") as directory:
        path = Path(directory) / "electronic.fcidump"

        fcidump.from_integrals(
            filename=str(path),
            h1e=h1,
            h2e=h2,
            nmo=n,
            nelec=prepared.num_electrons,
            nuc=0.0,  # Keep nuclear and frozen-core offsets separate.
            tol=0.0,
            float_format=" %.17g",
        )

        data = FCIDump.from_file(str(path))
        operator = FermionOperator.from_fcidump(data)

    return operator