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
threading. Reference reproducibility is stabilized by this protocol; the
time-budgeted Standard SQD benchmark showed execution-order dependence. An
isolated five-pool audit is available through
`benchmark/isolated_call_history.py`: it found identical marginal energies in
both execution orders, while one Standard SQD pool changed by about 1.45 mHa
when method order was reversed. This may reflect different completed restart
counts rather than history dependence of an individual fixed-seed solve.

The small-system N₂ studies use the six-electron, six-orbital active space.
For larger-space experiments, `n2-large` selects an 8-orbital, 10-electron
STO-3G N₂ active space without changing either selection method.

The frozen larger-space confirmation protocol is recorded in
`benchmark/n2_large_cap12_evaluation_config.json`. It treats the 200-step,
cap-12 configuration as primary, reserves seeds 127--156 for confirmation,
and declares success only when the paired 95% confidence interval for
Standard minus marginal error is strictly positive and marginal solver time
is no greater. The five exploratory seeds 122--126 remain separate.

The saved-pool accuracy--cost curve uses those same 30 schedule-0 pools at
per-spin caps 12, 16, 24, and 32. `benchmark/accuracy_cost_curve.py` reports
attainment and time-to-target for 50, 25, 10, and 5 mHa absolute error
thresholds. The primary target is 10 mHa measured against each pool's exact
sampled-product reference. The comparison records total elapsed time,
classical solver time, sampling/other time, solve dimensions, and completed
solve progress; a run that never reaches a threshold is counted as a failure.
The curve is exploratory and does not change the frozen cap-12 confirmation
claim.

In this saved-pool analysis, `marginal_elapsed` and `standard_elapsed` are
complete classical processing times: scoring, selection, batching, and all
diagonalization solves. They do not include the original quantum-simulation
pool-generation time. Add the `sampling_seconds` values from the corresponding
sampling results to obtain end-to-end time. For the 30 larger-N2 pools, mean
pool generation took 24.641 seconds. Thus the cap-32 mean end-to-end times
were 24.884 seconds for marginal selection and 25.090 seconds for Standard
SQD. Threshold time-to-target is reported both as classical processing time
and with pool generation added; successful-method averages use only runs that
reached the threshold, while failures remain in the attainment denominator.

## Matched ZenKO benchmark

The attached ZenKO paper defines a larger N2 benchmark than the development
`n2-large` case: 1.0 Angstrom, 6-31G, 16 active orbitals, and 10 active
electrons after freezing one core orbital on each atom. This is registered as
`n2-zenko`; FCI is disabled by default because its fixed-spin space has
19,079,424 configurations. The frozen comparison protocol is recorded in
`benchmark/zenko_matched_benchmark_config.json`.

The protocol uses fresh seeds 157--186, a 10 mHa primary target, a per-spin
cap of 32 for accumulated-marginal selection, and the paper's ZenKO settings:
10,000 shots per quantum evaluation, 0.1 mHa SQD convergence, 30-step T2
discovery, 30-step T1 refinement, and at most 10 outer iterations. Both
methods must use the same prepared Hamiltonian and report circuits, shots,
gate counts, diagonalization time, complete classical processing, optimizer
work, simulator wall-clock time, threshold attainment, and failures.

A paper-aligned reference was computed with the bounded iterative
`benchmark/iterative_reference.py` path using PySCF `direct_spin1.FCI`,
`conv_tol=1e-10`, and single-threaded matrix construction. It converged to
`-109.04667177831362 Ha` for this exact Hamiltonian and active space in about
719 seconds. The generated reference record is
`results/n2-zenko-iterative-reference.json`; its convergence flag and solver
settings must accompany any error claim.

The feasibility pilot in `benchmark/paper_n2_pilot.py` uses one non-confirmation
seed, 10,000 shots, and Aer matrix-product-state simulation. It completed a
32-qubit nonzero-time circuit without allocating the approximately 64 GiB
complex128 statevector. The pilot is only an execution and memory check: its
one-step circuit produced one unique configuration and is not evidence of
sampling accuracy. Seeds 157--186 remain reserved.

The full paper schedule is exercised by `benchmark/paper_n2_full_pilot.py`:
20 and 100 evolution times, 200 QDrift steps, 10 circuits per time, and
10,000 shots per circuit, followed by the frozen cap-32 marginal selector.
The reference energy convention is
`E_total = E_electronic + constant_energy_offset`, with electronic energy
`-32.81556923855211 Ha` and offset `-76.23110253976151 Ha`.

The first full-schedule MPS sensitivity check failed at bond dimension 128
and truncation threshold `1e-8`: truncation produced a bitstring outside the
required `(5,5)` electron sector, and the sampler rejected it rather than
silently filtering it. A retry at bond dimension 512 and threshold `1e-12`
did not complete within approximately 18 minutes, so no accuracy result was
recorded. This means the current MPS settings have not yet demonstrated a
manageable full-schedule pilot; no confirmation seeds were used.

The repository does not yet contain a ZenKO runner or its UCJ/CCSD optimizer
implementation, so this configuration freezes the matched protocol but does
not claim a ZenKO result. Simulator runtime must remain explicitly separate
from quantum-hardware execution time.

## N2/STO-3G baseline comparison

The manageable six-orbital N2 comparison is frozen in
`benchmark/n2_original_baseline_config.json`. It reuses 30 identical saved
SqDRIFT pools (seeds 52--81) and compares Standard SQD, the repository's
`qiskit-addon-sqd` `TrimPolicy` wrapper, and accumulated-marginal selection.
The pool protocol is 20 circuits, 100 QDrift steps, 256 shots per circuit, and
evolution times 5 and 20. `benchmark/compare_pools.py` records paired energy,
target attainment, solve dimensions, solve count, determinant work, and
classical processing time for all three capped methods.

On these development pools at cap 8, the mean errors were 26.254 mHa for
Standard SQD, 21.142 mHa for merge-only TrimPolicy, 24.867 mHa for pruning,
and 18.873 mHa for accumulated-marginal selection. The 10 mHa target was not
reached by any capped method; the 25 mHa target was reached on 19, 24, 19,
and 23 pools respectively. Mean classical processing was 0.078 s, 0.152 s,
0.121 s, and 0.064 s, with mean determinant work 461, 1,279, 948, and 525
respectively. The unrestricted sampled-product reference averaged 2.687 mHa
error and is outside the cap. Accumulated marginal therefore used about 14%
more summed solve dimensions than Standard SQD (525 versus 461), while its
reported classical processing time was about 18% lower (0.064 versus 0.078
seconds). Summed dimensions are a workload indicator, not a direct runtime
or hardware-cost measure. On this small problem, the unrestricted solve was
both more accurate and faster; this experiment isolates capped selection
behavior rather than establishing an overall computational advantage.

This is a same-pool selection comparison, not a full-method reproduction.
No separate author-provided SqDRIFT or Trim-SQD workflow implementation was
found in the repository. The sampler uses the installed
`qiskit-fermions` `QDriftTrotterization` path, and the Trim-SQD baseline uses
the installed Addon SQD `TrimPolicy`; neither result should be described as
an independently verified reproduction of an external paper protocol.
The component-by-component external-source review is frozen in
`benchmark/baseline_fidelity_checklist.json`. It marks the current sampler's
missing diagonal-term filtering and group ordering as adaptations, and
records the TrimPolicy batch, ranking, pruning, merging, and carryover
behavior separately from the accumulated-marginal replacements.
Historical pool summaries contain shots, circuit count, and summed circuit CX
counts but not per-circuit depth. `benchmark/sampling_sweep.py` now records
per-circuit CX counts and depths, summed depth, and shot-weighted gate totals
for fresh sampler audits. A summed CX count counts each circuit once; a
shot-weighted count multiplies each circuit's gate count by its shots and is
the closer proxy for executed gate instances.

A fresh audit with development seed 150 and the N2 schedule `(5, 20)` at 100
QDrift steps recorded 20 circuits, 5,120 shots, 6,655 summed CX counts,
8,923 summed circuit depth, 1,703,680 shot-weighted CX instances, and
2,284,288 shot-weighted depth instances. Sampling took 20.784 seconds in the
current simulator environment. These are simulator measurements, not
hardware execution times, and the audit does not establish fidelity to an
external author's implementation.

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
