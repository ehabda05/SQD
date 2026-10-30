"""Summarize threshold attainment for saved-pool accuracy/cost comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np


THRESHOLDS_MHA = (50.0, 25.0, 10.0, 5.0)


def progress_target_time(row, method, threshold):
    reference = float(row["reference_energy"])
    progress = row[f"{method}_progress"]
    for elapsed, energy, _label in progress:
        if 1000 * abs(float(energy) - reference) <= threshold:
            return float(elapsed)
    return None


def summarize(rows, threshold):
    summary = {}
    for method in ("marginal", "standard"):
        target_times = [
            time
            for row in rows
            if (time := progress_target_time(row, method, threshold)) is not None
        ]
        summary[method] = {
            "runs": len(rows),
            "reached": len(target_times),
            "failed": len(rows) - len(target_times),
            "fraction_reached": len(target_times) / len(rows) if rows else None,
            "mean_time_to_target_seconds": (
                float(np.mean(target_times)) if target_times else None
            ),
            "median_time_to_target_seconds": (
                float(np.median(target_times)) if target_times else None
            ),
            "mean_error_mha": float(
                np.mean([row[f"{method}_error_mha"] for row in rows])
            ),
            "mean_total_elapsed_seconds": float(
                np.mean([row[f"{method}_elapsed"] for row in rows])
            ),
            "mean_classical_elapsed_seconds": (
                float(np.mean([row[f"{method}_classical_elapsed"] for row in rows]))
                if f"{method}_classical_elapsed" in rows[0]
                else None
            ),
            "mean_sampling_elapsed_seconds": (
                float(np.mean([row[f"{method}_sampling_elapsed"] for row in rows]))
                if f"{method}_sampling_elapsed" in rows[0]
                else None
            ),
            "mean_solve_count": float(
                np.mean([row[f"{method}_solve_count"] for row in rows])
            ),
            "mean_total_determinants": float(
                np.mean([row[f"{method}_total_determinants"] for row in rows])
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("CAP", "RESULTS"),
        required=True,
        help="Per-spin cap and results.json path; repeat once per cap.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "thresholds_mha": list(THRESHOLDS_MHA),
        "primary_threshold_mha": 10.0,
        "caps": {},
    }
    for cap, path_text in args.input:
        rows = json.loads(Path(path_text).read_text())
        if not rows:
            raise ValueError(f"No rows in {path_text}")
        report["caps"][str(int(cap))] = {
            "rows": len(rows),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "schedule_indices": sorted({int(row["schedule"]) for row in rows}),
            "thresholds": {
                str(threshold): summarize(rows, threshold)
                for threshold in THRESHOLDS_MHA
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
