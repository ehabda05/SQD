"""Diagnose SQD error sources on one saved SqDRIFT sample pool.

Compares, on identical samples:
  oracle     every spin string (equals FCI; sanity check)
  untrimmed  every spin string that appears in the postselected pool
  sqd        standard SQD under the configured batch/dimension caps
  drift-trim TRIM-SQD under the same caps
"""

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from quantum.SqDRIFT import sample_sqdrift  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.trim_sqd import run_trim_sqd  # noqa: E402


def spin_strings(counts, n, num_alpha, num_beta):
    """Return sampled alpha and beta strings as integers, with their counts."""
    alpha, beta = {}, {}
    for bitstring, count in counts.items():
        if bitstring[n:].count("1") != num_alpha or bitstring[:n].count("1") != num_beta:
            continue
        a, b = int(bitstring[n:], 2), int(bitstring[:n], 2)
        alpha[a] = alpha.get(a, 0) + count
        beta[b] = beta.get(b, 0) + count
    return alpha, beta


def solve_strings(prepared, strings_a, strings_b):
    start = perf_counter()
    result = solve_sci(
        (np.array(sorted(strings_a)), np.array(sorted(strings_b))),
        prepared.h1,
        prepared.h2,
        norb=prepared.num_orbitals,
        nelec=prepared.num_electrons,
    )
    total = float(result.energy) + prepared.constant_energy
    return total, len(strings_a) * len(strings_b), perf_counter() - start


def fci_weight_captured(prepared, strings_a, strings_b):
    """Fraction of the exact FCI wavefunction weight inside the sampled product space."""
    n = prepared.num_orbitals
    _, vec = fci.direct_spin0.FCI().kernel(
        prepared.h1, prepared.h2, n, prepared.num_electrons
    )
    addr_a = {
        int(s): i
        for i, s in enumerate(fci.cistring.make_strings(range(n), prepared.num_electrons[0]))
    }
    rows = [addr_a[s] for s in strings_a]
    cols = [addr_a[s] for s in strings_b]
    return float(np.sum(np.asarray(vec)[np.ix_(rows, cols)] ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="n2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--times", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--circuits-per-time", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule)
    n = prepared.num_orbitals
    num_alpha, num_beta = prepared.num_electrons
    mha = lambda e: 1000 * (e - prepared.fci_energy)

    start = perf_counter()
    pool = sample_sqdrift(
        prepared,
        times=tuple(args.times),
        circuits_per_time=args.circuits_per_time,
        num_steps=args.num_steps,
        shots_per_circuit=args.shots,
        seed=args.seed,
    )
    sampling_seconds = perf_counter() - start

    args.out.mkdir(exist_ok=True)
    pool_path = args.out / f"pool_{args.molecule}_{args.seed}.json"
    pool_path.write_text(json.dumps(pool.counts))

    alpha, beta = spin_strings(pool.counts, n, num_alpha, num_beta)
    all_alpha = set(
        int(s) for s in fci.cistring.make_strings(range(n), num_alpha)
    )
    valid_shots = sum(alpha.values())

    print(f"{prepared.molecule.name}: FCI {prepared.fci_energy:.6f} Ha, "
          f"{n} orbitals, electrons {prepared.num_electrons}")
    print(f"shots {pool.total_shots} ({valid_shots} valid), "
          f"unique bitstrings {len(pool.counts)}, sampling {sampling_seconds:.1f}s, "
          f"mean cx/circuit {np.mean([m['cx_count'] for m in pool.metadata]):.0f}")
    print(f"unique alpha strings {len(alpha)}/{len(all_alpha)}, "
          f"beta {len(beta)}/{len(all_alpha)}")

    rows = []

    def record(name, energy, dim, seconds):
        rows.append((name, mha(energy), dim, seconds))

    energy, dim, seconds = solve_strings(prepared, all_alpha, all_alpha)
    record("oracle (all strings)", energy, dim, seconds)

    energy, dim, seconds = solve_strings(prepared, set(alpha), set(beta))
    record("untrimmed (all sampled)", energy, dim, seconds)
    captured = fci_weight_captured(prepared, sorted(alpha), sorted(beta))

    start = perf_counter()
    sqd = run_sqd(prepared, pool.counts)
    record("sqd (default caps)", sqd.total_energy, sqd.subspace_dimension, perf_counter() - start)

    start = perf_counter()
    trim = run_trim_sqd(prepared, pool.counts)
    record("drift-trim (default caps)", trim.total_energy, trim.subspace_dimension,
           perf_counter() - start)

    print(f"FCI weight inside sampled product space: {captured:.6f}")
    print(f"{'method':28s}{'error mHa':>12s}{'dim':>8s}{'sec':>8s}")
    for name, error, dim, seconds in rows:
        print(f"{name:28s}{error:12.4f}{dim:8d}{seconds:8.2f}")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")


if __name__ == "__main__":
    main()
