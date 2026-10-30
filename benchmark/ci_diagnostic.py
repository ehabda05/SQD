"""Reference-informed CI selection diagnostic.

This is not a competitive algorithm: it uses the unrestricted sampled-pool
wavefunction to choose the strings that define the capped diagnostic space.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit_addon_sqd.fermion import solve_sci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402


def solve(prepared, strings_a, strings_b):
    start = perf_counter()
    result = solve_sci(
        (np.asarray(strings_a), np.asarray(strings_b)),
        prepared.h1,
        prepared.h2,
        norb=prepared.num_orbitals,
        nelec=prepared.num_electrons,
    )
    return (
        float(result.energy) + prepared.constant_energy,
        result,
        perf_counter() - start,
    )


def top_indices(weights, count):
    return np.argsort(-weights, kind="stable")[:count]


def alternating_selection(weights, count):
    beta = top_indices(np.sum(weights, axis=0), count)
    alpha = np.array([], dtype=int)
    for _ in range(32):
        next_alpha = top_indices(np.sum(weights[:, beta], axis=1), count)
        next_beta = top_indices(np.sum(weights[next_alpha, :], axis=0), count)
        if np.array_equal(next_alpha, alpha) and np.array_equal(next_beta, beta):
            break
        alpha, beta = next_alpha, next_beta
    return alpha, beta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--pool-dir", type=Path, default=Path("results/sampling-sweep"))
    parser.add_argument("--schedule-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(45, 55)))
    parser.add_argument("--strings-per-spin", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/ci-diagnostic"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            pool_path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(pool_path.read_text())
            n = prepared.num_orbitals
            strings_a = sorted({int(bitstring[n:], 2) for bitstring in counts})
            strings_b = sorted({int(bitstring[:n], 2) for bitstring in counts})

            full_energy, full_result, full_seconds = solve(
                prepared, strings_a, strings_b
            )
            state = full_result.sci_state
            amplitudes = np.asarray(state.amplitudes)
            state_alpha = np.asarray(state.ci_strs_a)
            state_beta = np.asarray(state.ci_strs_b)
            alignment_ok = (
                amplitudes.shape == (state_alpha.size, state_beta.size)
                and np.unique(state_alpha).size == state_alpha.size
                and np.unique(state_beta).size == state_beta.size
                and np.all(np.diff(state_alpha) >= 0)
                and np.all(np.diff(state_beta) >= 0)
            )
            if not alignment_ok:
                raise RuntimeError(
                    "SCI amplitudes and returned CI string arrays are misaligned."
                )
            weights = np.abs(amplitudes) ** 2
            alpha_weights = np.sum(weights, axis=1)
            beta_weights = np.sum(weights, axis=0)
            keep = args.strings_per_spin
            independent_a_indices = top_indices(alpha_weights, keep)
            independent_b_indices = top_indices(beta_weights, keep)
            selected_a = np.sort(state_alpha[independent_a_indices])
            selected_b = np.sort(state_beta[independent_b_indices])
            independent_weight = float(
                np.sum(weights[np.ix_(independent_a_indices, independent_b_indices)])
            )

            diagnostic_energy, _, diagnostic_seconds = solve(
                prepared, selected_a, selected_b
            )
            alternating_a_indices, alternating_b_indices = alternating_selection(
                weights, keep
            )
            alternating_a = np.sort(state_alpha[alternating_a_indices])
            alternating_b = np.sort(state_beta[alternating_b_indices])
            alternating_weight = float(
                np.sum(weights[np.ix_(alternating_a_indices, alternating_b_indices)])
            )
            alternating_energy, _, alternating_seconds = solve(
                prepared, alternating_a, alternating_b
            )

            flat_indices = np.argsort(weights.ravel())[::-1][: keep * keep]
            pair_rows, pair_cols = np.unravel_index(flat_indices, weights.shape)
            pair_weight = float(np.sum(weights.ravel()[flat_indices]))
            pair_alpha = np.unique(state_alpha[pair_rows])
            pair_beta = np.unique(state_beta[pair_cols])
            rows.append(
                {
                    "molecule": args.molecule,
                    "schedule": schedule,
                    "seed": seed,
                    "pool_file": str(pool_path),
                    "strings_per_spin": keep,
                    "full_sampled_dimension": len(strings_a) * len(strings_b),
                    "diagnostic_dimension": keep * keep,
                    "full_sampled_error_mha": 1000 * abs(
                        full_energy - prepared.fci_energy
                    ),
                    "diagnostic_error_mha": 1000 * abs(
                        diagnostic_energy - prepared.fci_energy
                    ),
                    "diagnostic_delta_vs_full_mha": 1000 * (
                        diagnostic_energy - full_energy
                    ),
                    "diagnostic_retained_weight": independent_weight,
                    "alternating_error_mha": 1000 * abs(
                        alternating_energy - prepared.fci_energy
                    ),
                    "alternating_delta_vs_full_mha": 1000 * (
                        alternating_energy - full_energy
                    ),
                    "alternating_retained_weight": alternating_weight,
                    "top64_pair_retained_weight": pair_weight,
                    "top64_pair_unique_alpha": pair_alpha.size,
                    "top64_pair_unique_beta": pair_beta.size,
                    "top64_pair_expanded_dimension": (
                        pair_alpha.size * pair_beta.size
                    ),
                    "full_solve_seconds": full_seconds,
                    "diagnostic_solve_seconds": diagnostic_seconds,
                    "alternating_solve_seconds": alternating_seconds,
                    "alignment_ok": bool(alignment_ok),
                    "selected_alpha": selected_a.tolist(),
                    "selected_beta": selected_b.tolist(),
                    "alternating_alpha": alternating_a.tolist(),
                    "alternating_beta": alternating_b.tolist(),
                    "reference_informed": True,
                }
            )

    output = args.out / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
