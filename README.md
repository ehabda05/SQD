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

FCI reference calculation can be skipped for production runs with
`compute_fci=False`. To reuse references between runs, pass `fci_cache_dir`;
cache keys include geometry, basis, active-space selection, and preparation
tolerances. In that mode `PreparedHamiltonian.fci_energy` is `None`, and solver
error fields are also `None`.

The registered H2O case follows the ZenKO-sized setup: STO-3G, 8 active
electrons in 6 active orbitals, 0.958 Angstrom bonds, and a 104.5 degree angle.
Use `benchmark/experiment.py` to sample once per seed and compare standard SQD
with DRIFT-TRIM over identical saved pools and multiple solve caps:

```bash
python benchmark/experiment.py h2o --seeds 42 43 44
```

The cap sweep records the maximum actual Cartesian-product dimension for every
classical solve and marks whether it satisfies the configured per-spin cap.
To study sampling coverage without trimming at a fixed shot budget, use:

```bash
python benchmark/sampling_sweep.py h2o --seeds 42 43 44
```

This compares the unrestricted sampled alpha/beta product with standard SQD
while varying evolution times and Trotter steps. It reports FCI error, CX
count, sampling time, solve time, and actual solve dimensions.

To compare selection policies on the saved 100- and 200-step pools without
resampling:

```bash
python benchmark/compare_pools.py h2o --schedule-indices 2 3
```

The unrestricted sampled-product row is a reference outside the configured
determinant budget. Standard SQD, merge-only (`trim_ratio=1.0`), and pruning
(`trim_ratio=0.5`) use the same per-spin cap and fixed carryover ratio.
The comparison report includes per-seed error deltas, standard deviations,
solver time, diagonalization count, and total determinants diagonalized.

To compare merge-only against Standard SQD at approximately equal classical
solver time, reuse the saved pools with:

```bash
python benchmark/equal_cost.py h2o \
  --pool-dir results/sampling-sweep-45-54 \
  --schedule-indices 0 1 \
  --seeds 45 46 47 48 49 50 51 52 53 54
```

Merge-only establishes the per-pool time target. Standard SQD then performs
additional independent solver restarts until that time is matched or
exceeded, retaining its lowest-energy result. All restart solves and
diagonalized determinant counts are recorded.

An experimental marginal-selection rule can be evaluated with:

```bash
python benchmark/marginal_selection.py h2o \
  --pool-dir results/sampling-sweep-45-54 \
  --schedule-indices 0 1 \
  --seeds 45 46 47 48 49 50 51 52 53 54
```

This collects alpha and beta marginal CI weights from each restricted batch
solve, averages each string's scores over the batches in which it appears,
and solves the resulting sorted 8-by-8 Cartesian candidate. The collection
solve, candidate solve, scoring coverage, and total cost are all reported.
The frozen rule averages each string's marginal weight only over batches in
which that string is present. Absent strings are omitted rather than assigned
zero. Ties are broken by the integer CI-string value, and the final candidate
strings are sorted canonically. The rule is experimental; weights from
different restricted spaces are not assumed to be physically comparable.

For a same-run, fixed-budget comparison against repeated Standard SQD, use:

```bash
python benchmark/fair_selection_comparison.py h2o \
  --pool-dir results/sampling-sweep-45-54 \
  --budget-seconds 0.04 \
  --schedule-indices 0 1 \
  --seeds 45 46 47 48 49 50 51 52 53 54
```

This warms up each pool, uses a marginal-selection pilot to establish the
per-pool target, alternates the actual method execution order, records progress
only after completed solves, and reports budget overshoot and all solve costs.

The frozen held-out protocol is recorded in
`benchmark/heldout_evaluation_config.json`. Generate new pools only after
freezing this configuration, then evaluate the primary 200-step schedule
(index 3) and secondary 100-step schedule (index 2) on seeds 55--84 without
tuning the scoring rule or solver settings.

For saved-pool cap studies, use `benchmark/reference_pools.py` to create one
exact fixed-sector matrix reference per pool. Run reference generation with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for reproducible PySCF matrix
construction. The tool builds the complete sector matrix and extracts the
sampled-product principal submatrix; do not average unrestricted and capped
rows or recompute references through repeated SCI iterations.

The corrected N₂ cap-8 confirmation protocol is frozen in
`benchmark/n2_cap8_evaluation_config.json`. It records the practical solver
seed policy, stopping rule, warm-up and timing procedure, and reference
threading. Reference reproducibility is stabilized by this protocol; practical
solver call-history independence remains an explicit limitation.

For a reference-informed upper-bound diagnostic, use
`benchmark/ci_diagnostic.py`. It ranks alpha and beta strings using the
unrestricted sampled-pool SCI wavefunction, retains the top eight strings per
spin, and solves the resulting 64-determinant space. This uses reference
information and is not a competitive method:

```bash
python benchmark/ci_diagnostic.py h2o --pool-dir results/sampling-sweep-45-54
```

The diagnostic also checks amplitude-to-CI-string alignment, reports retained
joint wavefunction weight, and evaluates alternating alpha/beta selection.
The top-64 individual-pair control is reported as a non-Cartesian reference;
its unique spin-string expansion can exceed the 64-determinant budget.

To audit the selected-space energy against an explicit fixed-spin Hamiltonian,
run the following on one saved pool:

```bash
python benchmark/hamiltonian_audit.py \
  --pool results/sampling-sweep-45-54/pool_1_45.json
```

The audit constructs the full sampled-product matrix, checks the SCI
eigenvector residual, and compares the selected subspace eigenvalue with the
projected unrestricted-vector energy. Selected CI strings are canonicalized
before solving because the selected-CI implementation requires determinant
strings in sorted order.

SqDRIFT uses Aer statevector simulation by default. Larger runs can select
another Aer method, such as `matrix_product_state`, through the
`simulator_method` and `simulator_options` arguments of `sample_sqdrift`.
