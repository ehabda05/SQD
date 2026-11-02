"""Verify saved comparison winners against explicit Hamiltonian submatrices."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.reference_pools import fixed_sector_matrix
from solvers.provenance import pool_hash


def _pool_path(comparison_dir: Path, value: str) -> Path:
    candidates = (
        Path(value),
        comparison_dir / value,
        comparison_dir.parent.parent / value,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def verify_row(
    row: dict,
    prepared,
    matrix: np.ndarray,
    address_a: dict[int, int],
    address_b: dict[int, int],
    comparison_dir: Path,
    *,
    residual_tol: float,
    energy_tol: float,
) -> dict:
    counts = json.loads(_pool_path(comparison_dir, row["pool_file"]).read_text())
    actual_hash = pool_hash(counts)
    if actual_hash != row["pool_hash"]:
        raise ValueError(f"Pool hash mismatch for {row['pool_file']}")

    alpha = [int(value) for value in row["winning_alpha"]]
    beta = [int(value) for value in row["winning_beta"]]
    amplitudes = np.asarray(row["winning_amplitudes"], dtype=float).reshape(-1)
    indices = [
        address_a[string_a] * len(address_b) + address_b[string_b]
        for string_a in alpha
        for string_b in beta
    ]
    if len(indices) != amplitudes.size:
        raise ValueError(f"Amplitude dimension mismatch for {row['method']}")
    submatrix = matrix[np.ix_(indices, indices)]
    norm = float(np.vdot(amplitudes, amplitudes).real)
    electronic = float(np.vdot(amplitudes, submatrix @ amplitudes).real)
    residual = float(
        np.linalg.norm(submatrix @ amplitudes - electronic * amplitudes)
    )
    total_energy = electronic + prepared.constant_energy
    passed = (
        abs(norm - 1.0) <= energy_tol
        and residual <= residual_tol
        and abs(total_energy - float(row["total_energy"])) <= energy_tol
    )
    return {
        "method": row["method"],
        "schedule": row["schedule"],
        "seed": row["seed"],
        "pool_hash": actual_hash,
        "dimension": int(submatrix.shape[0]),
        "normalization": norm,
        "total_energy": total_energy,
        "reported_total_energy": row["total_energy"],
        "energy_difference": total_energy - float(row["total_energy"]),
        "residual": residual,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("--residual-tol", type=float, default=1e-7)
    parser.add_argument("--energy-tol", type=float, default=1e-8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.comparison_dir / "prepared.json").read_text())
    prepared = SimpleNamespace(
        num_orbitals=len(manifest["active_orbitals"]),
        num_electrons=tuple(manifest["active_electrons"]),
        h1=np.asarray(manifest["h1"]),
        h2=np.asarray(manifest["h2"]),
        constant_energy=float(manifest["constant_energy"]),
    )
    matrix, address_a, address_b = fixed_sector_matrix(prepared)
    rows = json.loads((args.comparison_dir / "results.json").read_text())
    verified = [
        verify_row(
            row,
            prepared,
            matrix,
            address_a,
            address_b,
            args.comparison_dir,
            residual_tol=args.residual_tol,
            energy_tol=args.energy_tol,
        )
        for row in rows
    ]
    output = args.out or args.comparison_dir / "replay_verification.json"
    output.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    if not all(row["passed"] for row in verified):
        raise RuntimeError(f"Replay verification failed; see {output}")
    print(json.dumps({"rows": len(verified), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
