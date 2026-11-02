import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmark.accuracy_cost_curve import progress_target_time
from benchmark.accuracy_cost_curve import summarize
from benchmark.compare_pools import reference_is_outside_budget
from benchmark.fair_selection_comparison import run_standard
from benchmark.marginal_selection import select_top_strings
from benchmark.reference_pools import fixed_sector_matrix
from solvers.accumulated_marginal import run_accumulated_marginal
from chemistry.prepare import prepare_registered


class BenchmarkRegressionTests(unittest.TestCase):
    def test_selected_strings_are_canonical_after_weight_ranking(self):
        selected = select_top_strings(
            {4: [0.9], 1: [0.8], 2: [0.8]}, count=3
        )
        np.testing.assert_array_equal(selected, np.array([1, 2, 4]))

    def test_accuracy_target_uses_explicit_fci_energy(self):
        row = {
            "fci_energy": -10.0,
            "reference_energy": -10.025,
            "marginal_progress": [(0.5, -10.025, "solve")],
        }
        self.assertIsNone(progress_target_time(row, "marginal", 10.0))

    def test_accuracy_target_rejects_missing_fci_energy(self):
        with self.assertRaisesRegex(ValueError, "explicit fci_energy"):
            progress_target_time(
                {
                    "reference_energy": -10.0,
                    "marginal_progress": [(0.5, -10.0, "solve")],
                },
                "marginal",
                10.0,
            )

    def test_dense_reference_has_an_allocation_guard(self):
        class OversizedPrepared:
            num_orbitals = 32
            num_electrons = (16, 16)

        with self.assertRaisesRegex(ValueError, "iterative_reference"):
            fixed_sector_matrix(OversizedPrepared())

    def test_schedule_aware_sampling_times_are_joined_by_schedule_and_seed(self):
        row_template = {
            "fci_energy": -10.0,
            "marginal_progress": [(0.1, -10.0, "solve")],
            "standard_progress": [(0.1, -10.0, "solve")],
            "marginal_error_mha": 0.0,
            "standard_error_mha": 0.0,
            "marginal_elapsed": 0.1,
            "standard_elapsed": 0.1,
            "marginal_complete_classical_elapsed": 0.1,
            "standard_complete_classical_elapsed": 0.1,
            "marginal_diagonalization_elapsed": 0.05,
            "standard_diagonalization_elapsed": 0.05,
            "marginal_non_diagonalization_elapsed": 0.05,
            "standard_non_diagonalization_elapsed": 0.05,
            "marginal_solve_count": 1,
            "standard_solve_count": 1,
            "marginal_total_determinants": 1,
            "standard_total_determinants": 1,
        }
        rows = [
            {**row_template, "schedule": 0, "seed": 1},
            {**row_template, "schedule": 1, "seed": 1},
        ]
        summary = summarize(
            rows,
            10.0,
            {(0, 1): (1.0, None), (1, 1): (3.0, None)},
        )
        self.assertEqual(
            summary["marginal"]["mean_end_to_end_elapsed_seconds"], 2.1
        )

    def test_only_unrestricted_reference_is_outside_budget(self):
        self.assertTrue(reference_is_outside_budget("unrestricted sampled product"))
        self.assertFalse(reference_is_outside_budget("accumulated-marginal"))

    def test_declared_budget_controls_standard_restarts(self):
        args = SimpleNamespace(
            samples_per_batch=1,
            num_batches=1,
            max_iterations=1,
            max_dim=1,
        )
        fake_result = SimpleNamespace(
            total_energy=-1.0,
            classical_solve_count=1,
            classical_solve_dimensions=[1],
            classical_solve_elapsed=0.0,
            sci_result=SimpleNamespace(
                sci_state=SimpleNamespace(
                    ci_strs_a=np.array([1]), ci_strs_b=np.array([1])
                )
            ),
        )

        def delayed_solve(*_args, **_kwargs):
            time.sleep(0.004)
            return fake_result

        with patch(
            "benchmark.fair_selection_comparison.run_sqd",
            side_effect=delayed_solve,
        ):
            result = run_standard(object(), {}, args, seed=1, budget=0.006)

        self.assertGreaterEqual(result["restart_count"], 2)
        self.assertGreaterEqual(result["elapsed"], 0.006)

    def test_accumulated_marginal_rejects_candidate_cap_bypass(self):
        prepared = prepare_registered("h2", compute_fci=False)
        with self.assertRaisesRegex(ValueError, "cannot exceed max_dim"):
            run_accumulated_marginal(
                prepared,
                {"0101": 4, "0110": 2, "1001": 2, "1010": 2},
                samples_per_batch=2,
                num_batches=1,
                max_iterations=1,
                max_dim=2,
                strings_per_spin=3,
            )
