"""Parameter recovery for H1 and H2 models.

Simulates data with known parameters, fits the model, and checks
that the recovered parameters are close to the true ones.

Usage:
    python parameter_recovery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from create_task_environment import generate_timeline, timeline_to_matrix, timeline_to_correct
from rlddm import (
    rlddm_simulate, rlddm_log_lik, fill_rlddm_pars,
    fit_model, H1_PARAMS, H2_PARAMS, H1_DEFAULTS, H2_DEFAULTS, REVERSAL_POINTS,
)


RP = REVERSAL_POINTS


def run_recovery(model: str, true_pars: dict,
                 n_restarts: int = 5, seed: int = 42) -> dict:
    """Simulate under model, fit same model, compare."""
    rng = np.random.default_rng(seed)
    timeline = generate_timeline(num_trials=140, seed=seed, reversed_state=True,
                                  reversal_points=RP)
    env = timeline_to_matrix(timeline)
    correct = timeline_to_correct(timeline)

    sim = rlddm_simulate(env, true_pars, rng=rng,
                          correct_bandit=correct, reversal_points=RP)
    data = sim["data"]

    fit = fit_model(model, data, reversal_points=RP, n_restarts=n_restarts, rng=rng)
    recovered = fit["pars"]
    true_filled = fill_rlddm_pars(true_pars)

    params = H1_PARAMS if model == "H1" else H2_PARAMS
    comparison = {}
    for name in params:
        comparison[name] = {
            "true": true_filled[name],
            "recovered": recovered[name],
            "abs_error": abs(recovered[name] - true_filled[name]),
        }

    return {
        "model": model,
        "comparison": comparison,
        "true_neg_ll": -rlddm_log_lik(data, true_pars, reversal_points=RP),
        "fitted_neg_ll": fit["bic"] / 2,  # approximate
    }


def print_results(result: dict, label: str = ""):
    print(f"\n{'=' * 60}")
    print(f"Parameter Recovery: {label} ({result['model']})")
    print(f"{'=' * 60}")
    print(f"{'Parameter':<14} {'True':>10} {'Recovered':>10} {'Abs Error':>10}")
    print("-" * 48)
    for name, vals in result["comparison"].items():
        print(f"{name:<14} {vals['true']:>10.4f} {vals['recovered']:>10.4f} "
              f"{vals['abs_error']:>10.4f}")
    print("-" * 48)
    max_err = max(v["abs_error"] for v in result["comparison"].values())
    status = "PASS" if max_err < 0.25 else "CHECK"
    print(f"{'Max abs error:':<20} {max_err:.4f}  -> {status}")


if __name__ == "__main__":
    print("Fitting H1 recovery...")
    r1 = run_recovery("H1", H1_DEFAULTS, n_restarts=5, seed=42)
    print_results(r1, "H1 (drift-only)")

    print("\nFitting H2 recovery...")
    r2 = run_recovery("H2", H2_DEFAULTS, n_restarts=5, seed=42)
    print_results(r2, "H2 (boundary varies)")