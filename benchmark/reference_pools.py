"""Create deterministic unrestricted fixed-sector references for saved pools."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import fci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402


def pool_hash(counts):
    return hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()


def fixed_sector_matrix(prepared):
    n = prepared.num_orbitals
    nelec_a, nelec_b = prepared.num_electrons
    strings_a = fci.cistring.make_strings(range(n), nelec_a)
    strings_b = fci.cistring.make_strings(range(n), nelec_b)
    h2e = fci.direct_spin1.absorb_h1e(
        np.asarray(prepared.h1),
        np.asarray(prepared.h2),
        n,
        prepared.num_electrons,
        .5,
    )
    dimension = len(strings_a) * len(strings_b)
    matrix = np.zeros((dimension, dimension))
    for column in range(dimension):
        vector = np.zeros((len(strings_a), len(strings_b)))
        vector.flat[column] = 1.0
        matrix[:, column] = fci.direct_spin1.contract_2e(
            h2e, vector, n, prepared.num_electrons
        ).ravel()
    return matrix, {int(value): index for index, value in enumerate(strings_a)}, {int(value): index for index, value in enumerate(strings_b)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule")
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--schedule-indices", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    matrix, address_a, address_b = fixed_sector_matrix(prepared)
    rows = []
    for schedule in args.schedule_indices:
        for seed in args.seeds:
            path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(path.read_text())
            n = prepared.num_orbitals
            alpha = sorted({int(bits[n:], 2) for bits in counts})
            beta = sorted({int(bits[:n], 2) for bits in counts})
            indices = [
                address_a[string_a] * len(address_b) + address_b[string_b]
                for string_a in alpha
                for string_b in beta
            ]
            selected_matrix = matrix[np.ix_(indices, indices)]
            electronic = float(np.linalg.eigvalsh(selected_matrix)[0])
            rows.append(
                {
                    "molecule": args.molecule,
                    "schedule": schedule,
                    "seed": seed,
                    "pool_file": str(path),
                    "pool_hash": pool_hash(counts),
                    "alpha_string_count": len(alpha),
                    "beta_string_count": len(beta),
                    "dimension": len(alpha) * len(beta),
                    "energy": electronic + prepared.constant_energy,
                    "error_mha": (
                        None
                        if prepared.fci_energy is None
                        else 1000
                        * abs(electronic + prepared.constant_energy - prepared.fci_energy)
                    ),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "references.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
