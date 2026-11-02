"""Compare SQD selection policies on identical saved sample pools."""

import argparse
import csv
import json
import statistics
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemistry.prepare import prepare_registered  # noqa: E402
from benchmark.reference_pools import fixed_sector_matrix  # noqa: E402
from solvers.accumulated_marginal import run_accumulated_marginal  # noqa: E402
from solvers.provenance import pool_hash  # noqa: E402
from solvers.sqd import run_sqd  # noqa: E402
from solvers.trim_sqd import run_trim_sqd  # noqa: E402


def reference_is_outside_budget(method):
    return method == "unrestricted sampled product"


def dependency_versions():
    packages = (
        "numpy",
        "qiskit",
        "qiskit-aer",
        "qiskit-fermions",
        "qiskit-addon-sqd",
        "pyscf",
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def amplitudes_for_json(amplitudes):
    values = np.asarray(amplitudes)
    if np.iscomplexobj(values) and not np.allclose(values.imag, 0.0):
        return [
            [float(value.real), float(value.imag)]
            for value in values.reshape(-1)
        ]
    return np.asarray(values.real if np.iscomplexobj(values) else values).tolist()


def run_marginal(prepared, counts, args, seed):
    start = perf_counter()
    result = run_accumulated_marginal(
        prepared,
        counts,
        samples_per_batch=args.samples_per_batch,
        num_batches=args.num_batches,
        max_iterations=args.max_iterations,
        max_dim=args.max_dim,
        strings_per_spin=args.max_dim,
        seed=seed,
    )
    elapsed = perf_counter() - start
    return {
        "energy": result.retained_total_energy,
        "dimension": result.selected_dimension,
        "seconds": elapsed,
        "solve_count": result.classical_solve_count,
        "dimensions": result.classical_solve_dimensions,
        "selected_alpha": result.selected_alpha.tolist(),
        "selected_beta": result.selected_beta.tolist(),
        "winning_alpha": np.asarray(
            result.retained.sci_state.ci_strs_a
        ).tolist(),
        "winning_beta": np.asarray(
            result.retained.sci_state.ci_strs_b
        ).tolist(),
        "winning_amplitudes": amplitudes_for_json(
            result.retained.sci_state.amplitudes
        ),
    }


def sampled_product_solve(prepared, counts, matrix, address_a, address_b):
    n = prepared.num_orbitals
    alpha = {int(bitstring[n:], 2) for bitstring in counts}
    beta = {int(bitstring[:n], 2) for bitstring in counts}
    alpha = sorted(alpha)
    beta = sorted(beta)
    indices = [
        address_a[string_a] * len(address_b) + address_b[string_b]
        for string_a in alpha
        for string_b in beta
    ]
    sampled_matrix = matrix[np.ix_(indices, indices)]
    start = perf_counter()
    eigenvalues, eigenvectors = np.linalg.eigh(sampled_matrix)
    ground_vector = eigenvectors[:, 0]
    return (
        float(eigenvalues[0]) + prepared.constant_energy,
        len(alpha) * len(beta),
        perf_counter() - start,
        alpha,
        beta,
        ground_vector.tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule", nargs="?", default="h2o")
    parser.add_argument("--pool-dir", type=Path, default=Path("results/sampling-sweep"))
    parser.add_argument("--schedule-indices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--carryover-ratio", type=float, default=0.5)
    parser.add_argument(
        "--retain-best-intermediate",
        action="store_true",
        help="Report the best SCI state observed without changing TrimPolicy search.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("results/fci-cache"))
    parser.add_argument("--out", type=Path, default=Path("results/pool-comparison"))
    args = parser.parse_args()

    prepared = prepare_registered(args.molecule, fci_cache_dir=args.cache_dir)
    matrix, address_a, address_b = fixed_sector_matrix(prepared)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for schedule in args.schedule_indices:
        for seed in args.seeds:
            pool_path = args.pool_dir / f"pool_{schedule}_{seed}.json"
            counts = json.loads(pool_path.read_text())
            counts_hash = pool_hash(counts)
            (
                reference_energy,
                reference_dim,
                reference_seconds,
                reference_alpha,
                reference_beta,
                reference_amplitudes,
            ) = sampled_product_solve(
                prepared, counts, matrix, address_a, address_b
            )

            methods = [
                ("unrestricted sampled product", reference_energy, reference_dim,
                 reference_seconds, None, 1, (reference_dim,)),
            ]
            method_specs = [
                ("standard SQD", run_sqd, None),
                ("merge-only", run_trim_sqd, 1.0),
                ("pruning", run_trim_sqd, 0.5),
                ("accumulated-marginal", None, None),
            ]
            if seed % 2:
                method_specs.reverse()
            execution_order = [spec[0] for spec in method_specs]
            for name, runner, trim_ratio in method_specs:
                start = perf_counter()
                if name == "accumulated-marginal":
                    marginal = run_marginal(prepared, counts, args, seed)
                    result = None
                    energy = marginal["energy"]
                    dimension = marginal["dimension"]
                    seconds = marginal["seconds"]
                    classical_solve_count = marginal["solve_count"]
                    classical_solve_dimensions = marginal["dimensions"]
                    winning_alpha = marginal["winning_alpha"]
                    winning_beta = marginal["winning_beta"]
                    winning_amplitudes = marginal["winning_amplitudes"]
                elif trim_ratio is None:
                    result = runner(
                        prepared,
                        counts,
                        samples_per_batch=args.samples_per_batch,
                        num_batches=args.num_batches,
                        max_iterations=args.max_iterations,
                        max_dim=args.max_dim,
                        seed=seed,
                    )
                    energy = result.total_energy
                    dimension = result.subspace_dimension
                    seconds = perf_counter() - start
                    classical_solve_count = result.classical_solve_count
                    classical_solve_dimensions = result.classical_solve_dimensions
                    winning_alpha = np.asarray(
                        result.sci_result.sci_state.ci_strs_a
                    ).tolist()
                    winning_beta = np.asarray(
                        result.sci_result.sci_state.ci_strs_b
                    ).tolist()
                    winning_amplitudes = amplitudes_for_json(
                        result.sci_result.sci_state.amplitudes
                    )
                else:
                    result = runner(
                        prepared,
                        counts,
                        samples_per_batch=args.samples_per_batch,
                        num_batches=args.num_batches,
                        max_iterations=args.max_iterations,
                        max_dim=args.max_dim,
                        trim_ratio=trim_ratio,
                        carryover_ratio=args.carryover_ratio,
                        retain_best_intermediate=args.retain_best_intermediate,
                        seed=seed,
                    )
                    energy = result.total_energy
                    dimension = result.subspace_dimension
                    seconds = perf_counter() - start
                    classical_solve_count = result.classical_solve_count
                    classical_solve_dimensions = result.classical_solve_dimensions
                    winning_alpha = np.asarray(
                        result.sci_result.sci_state.ci_strs_a
                    ).tolist()
                    winning_beta = np.asarray(
                        result.sci_result.sci_state.ci_strs_b
                    ).tolist()
                    winning_amplitudes = amplitudes_for_json(
                        result.sci_result.sci_state.amplitudes
                    )
                methods.append(
                    (
                        name,
                        energy,
                        dimension,
                        seconds,
                        result,
                        classical_solve_count,
                        classical_solve_dimensions,
                    )
                )

            for (
                method,
                energy,
                dimension,
                seconds,
                result,
                classical_solve_count,
                classical_solve_dimensions,
            ) in methods:
                dimensions = (
                    (dimension,)
                    if result is None and method != "accumulated-marginal"
                    else classical_solve_dimensions
                )
                rows.append(
                    {
                        "molecule": args.molecule,
                        "schedule": schedule,
                        "seed": seed,
                        "pool_file": str(pool_path),
                        "pool_hash": counts_hash,
                        "method": method,
                        "max_dim_per_spin": args.max_dim,
                        "total_energy": energy,
                        "fci_energy": prepared.fci_energy,
                        "error_mha": 1000 * abs(energy - prepared.fci_energy),
                        "max_actual_solve_dimension": max(dimensions),
                        "cap_satisfied": all(
                            value <= args.max_dim * args.max_dim
                            for value in dimensions
                        ),
                        "classical_solve_count": classical_solve_count,
                        "classical_solve_dimensions": dimensions,
                        "total_determinants_diagonalized": sum(dimensions),
                        "complete_classical_seconds": seconds,
                        "winning_alpha": (
                            reference_alpha
                            if method == "unrestricted sampled product"
                            else winning_alpha
                        ),
                        "winning_beta": (
                            reference_beta
                            if method == "unrestricted sampled product"
                            else winning_beta
                        ),
                        "winning_amplitudes": (
                            reference_amplitudes
                            if method == "unrestricted sampled product"
                            else winning_amplitudes
                        ),
                        "execution_order": execution_order,
                        "reached_10_mha": (
                            1000 * abs(energy - prepared.fci_energy) <= 10.0
                        ),
                        "reference_outside_budget": reference_is_outside_budget(method),
                    }
                )

    for row in rows:
        standard = next(
            candidate
            for candidate in rows
            if candidate["schedule"] == row["schedule"]
            and candidate["seed"] == row["seed"]
            and candidate["method"] == "standard SQD"
        )
        row["delta_vs_standard_mha"] = row["error_mha"] - standard["error_mha"]
        row["delta_vs_standard_seconds"] = (
            row["complete_classical_seconds"]
            - standard["complete_classical_seconds"]
        )

    output = args.out / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    manifest = {
        "molecule": args.molecule,
        "active_orbitals": prepared.active_orbitals,
        "active_electrons": prepared.num_electrons,
        "constant_energy": prepared.constant_energy,
        "nuclear_repulsion": prepared.nuclear_repulsion,
        "fci_energy": prepared.fci_energy,
        "h1": np.asarray(prepared.h1).tolist(),
        "h2": np.asarray(prepared.h2).tolist(),
        "geometry_angstrom": [
            {"symbol": atom.symbol, "position": list(atom.position)}
            for atom in prepared.molecule.atoms
        ],
        "basis": prepared.molecule.basis,
        "dependency_versions": dependency_versions(),
        "comparison_settings": vars(args),
    }
    (args.out / "prepared.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
    )
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    means = []
    for schedule in args.schedule_indices:
        for method in (
            "unrestricted sampled product",
            "standard SQD",
            "merge-only",
            "pruning",
            "accumulated-marginal",
        ):
            selected = [
                row for row in rows
                if row["schedule"] == schedule and row["method"] == method
            ]
            means.append(
                {
                    "schedule": schedule,
                    "method": method,
                    "mean_error_mha": sum(
                        row["error_mha"] for row in selected
                    ) / len(selected),
                    "stddev_error_mha": (
                        statistics.stdev(row["error_mha"] for row in selected)
                        if len(selected) > 1
                        else 0.0
                    ),
                    "mean_delta_vs_standard_mha": sum(
                        row["delta_vs_standard_mha"] for row in selected
                    ) / len(selected),
                    "stddev_delta_vs_standard_mha": (
                        statistics.stdev(
                            row["delta_vs_standard_mha"] for row in selected
                        )
                        if len(selected) > 1
                        else 0.0
                    ),
                    "mean_complete_classical_seconds": sum(
                        row["complete_classical_seconds"] for row in selected
                    ) / len(selected),
                    "reached_10_mha": sum(
                        row["reached_10_mha"] for row in selected
                    ),
                    "mean_total_determinants": sum(
                        row["total_determinants_diagonalized"]
                        for row in selected
                    ) / len(selected),
                }
            )
    (args.out / "means.json").write_text(json.dumps(means, indent=2))
    print(f"wrote {output}, {output.with_suffix('.csv')}, and means.json")


if __name__ == "__main__":
    main()
