"""Compute an explicit iterative fixed-spin FCI reference for a prepared space."""

import argparse
import json
import os
import sys
from time import perf_counter

from pyscf import fci

sys.path.insert(0, os.fspath(__import__("pathlib").Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", default="n2-zenko", nargs="?")
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, compute_fci=False)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = args.conv_tol
    solver.max_cycle = args.max_cycle
    start = perf_counter()
    electronic_energy, _ = solver.kernel(
        prepared.h1,
        prepared.h2,
        prepared.num_orbitals,
        prepared.num_electrons,
    )
    row = {
        "molecule": args.molecule,
        "active_orbitals": prepared.num_orbitals,
        "active_electrons": prepared.num_electrons,
        "electronic_energy": float(electronic_energy),
        "total_energy": float(electronic_energy + prepared.constant_energy),
        "converged": bool(solver.converged),
        "convergence_tolerance": args.conv_tol,
        "max_cycle": args.max_cycle,
        "elapsed_seconds": perf_counter() - start,
        "reference_method": "PySCF direct_spin1 iterative FCI",
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
            handle.write("\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
