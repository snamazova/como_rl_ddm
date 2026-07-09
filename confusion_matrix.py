"""Confusion matrix model comparison: H1 (drift-only) vs H2 (boundary-only).

Simulates data under each model, fits both models to both datasets, and
produces a 2×2 confusion matrix showing which model wins (by BIC) in each
cell. A perfect confusion matrix is diagonal — each model recovers its own
data-generating process.

Research question:
    Does post-reversal slowing reflect drift collapse (H1) or boundary
    increase (H2)?

Usage:
    python confusion_matrix.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from create_task_environment import (
    generate_timeline,
    timeline_to_matrix,
    timeline_to_correct,
)
from rlddm import rlddm_simulate, rlddm_log_lik, fill_rlddm_pars


# =============================================================================
# Parameter sets
# =============================================================================

H1_TRUE = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a": 3.0, "w": 0.5, "t0": 0.25}
H2_TRUE = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a_base": 3.0,
           "kappa": 2.0, "tau": 5.0, "w": 0.5, "t0": 0.25}

H1_PARAMS = ["alpha", "v_max", "beta", "a", "w", "t0"]
H2_PARAMS = ["alpha", "v_max", "beta", "a_base", "kappa", "tau", "w", "t0"]

BOUNDS = {
    "alpha":   (1e-3, 0.999),
    "v_max":   (1e-3, 5.0),
    "beta":    (1e-3, 5.0),
    "a":       (0.5, 6.0),
    "a_base":  (0.5, 6.0),
    "kappa":   (1e-3, 5.0),
    "tau":     (1e-3, 20.0),
    "w":       (0.01, 0.99),
    "t0":      (1e-3, 1.0),
}


# =============================================================================
# Fitting
# =============================================================================

def _unpack(theta: np.ndarray, param_names: list) -> dict:
    pars = {}
    for i, name in enumerate(param_names):
        lo, hi = BOUNDS[name]
        pars[name] = lo + (hi - lo) / (1.0 + np.exp(-theta[i]))
    return pars


def fit_model(model: str,
              data: pd.DataFrame,
              reversal_points: tuple,
              n_restarts: int = 5,
              rng: np.random.Generator | None = None) -> dict:
    """Fit H1 or H2 to data via multi-start Nelder-Mead optimisation.

    Uses a mix of random starts and informed starts (near plausible
    parameter values) to avoid the optimizer getting stuck.
    """
    rng = rng or np.random.default_rng()

    if model == "H1":
        params = H1_PARAMS
    elif model == "H2":
        params = H2_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    def neg_ll(theta):
        pars = _unpack(theta, params)
        try:
            ll = rlddm_log_lik(data, pars, reversal_points=reversal_points)
        except (ValueError, FloatingPointError):
            return 1e10
        return -ll if np.isfinite(ll) else 1e10

    # Informed starting points (unconstrained space, roughly centered)
    informed_starts = {
        "alpha": 0.0, "v_max": 0.5, "beta": 0.0,
        "a": 0.5, "a_base": 0.5, "kappa": 0.0, "tau": 0.0,
        "w": 0.0, "t0": 0.0,
    }
    start_informed = np.array([informed_starts[p] for p in params])

    best_ll = np.inf
    best_pars = None

    for i in range(n_restarts):
        if i == 0:
            theta0 = start_informed
        elif i == 1:
            theta0 = start_informed + rng.normal(0, 0.3, size=len(params))
        else:
            theta0 = rng.normal(0.0, 1.0, size=len(params))
        result = minimize(neg_ll, theta0, method="Nelder-Mead",
                          options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-6})
        if result.fun < best_ll:
            best_ll = result.fun
            best_pars = _unpack(result.x, params)

    n_trials = len(data)
    bic = 2 * best_ll + len(params) * np.log(n_trials)

    return {"model": model, "pars": best_pars, "log_lik": -best_ll,
            "n_params": len(params), "bic": float(bic)}


# =============================================================================
# Confusion matrix
# =============================================================================

def run_confusion_matrix(n_subjects: int = 10,
                         n_trials: int = 140,
                         n_restarts: int = 5,
                         seed: int = 42) -> dict:
    """Run the full 2×2 confusion matrix experiment.

    1. Simulate data under H1 and H2.
    2. Fit both models to both datasets.
    3. Return BICs and winning model for each cell.
    """
    rp = (36, 56, 71, 86, 106)
    rng = np.random.default_rng(seed)

    # Shared timeline
    timeline = generate_timeline(num_trials=n_trials, seed=seed,
                                 reversed_state=True, reversal_points=rp)
    env = timeline_to_matrix(timeline)
    correct = timeline_to_correct(timeline)

    # Simulate two datasets
    print(f"Simulating {n_subjects} subjects under H1...")
    h1_data = []
    for s in range(n_subjects):
        sim = rlddm_simulate(env, H1_TRUE, rng=np.random.default_rng(seed + s),
                             correct_bandit=correct, reversal_points=rp)
        h1_data.append(sim["data"])

    print(f"Simulating {n_subjects} subjects under H2...")
    h2_data = []
    for s in range(n_subjects):
        sim = rlddm_simulate(env, H2_TRUE, rng=np.random.default_rng(seed + 100 + s),
                             correct_bandit=correct, reversal_points=rp)
        h2_data.append(sim["data"])

    # Fit both models to both datasets
    results = {}
    for data_label, dataset in [("H1_data", h1_data), ("H2_data", h2_data)]:
        print(f"\nFitting models to {data_label}...")
        # Aggregate across subjects (sum BICs)
        bic_h1_total = 0
        bic_h2_total = 0
        ll_h1_total = 0
        ll_h2_total = 0

        for s, data in enumerate(dataset):
            fit_h1 = fit_model("H1", data, rp, n_restarts=n_restarts, rng=rng)
            fit_h2 = fit_model("H2", data, rp, n_restarts=n_restarts, rng=rng)
            bic_h1_total += fit_h1["bic"]
            bic_h2_total += fit_h2["bic"]
            ll_h1_total += fit_h1["log_lik"]
            ll_h2_total += fit_h2["log_lik"]
            print(f"  Subject {s}: H1 BIC={fit_h1['bic']:.1f}, H2 BIC={fit_h2['bic']:.1f}")

        winner = "H1" if bic_h1_total < bic_h2_total else "H2"
        results[data_label] = {
            "H1_bic": bic_h1_total,
            "H2_bic": bic_h2_total,
            "H1_ll": ll_h1_total,
            "H2_ll": ll_h2_total,
            "winner": winner,
        }

    return results


def print_confusion_matrix(results: dict):
    """Print the 2×2 confusion matrix."""
    print(f"\n{'=' * 55}")
    print("Confusion Matrix (BIC, summed across subjects)")
    print(f"{'=' * 55}")
    print(f"{'':>12} {'Fit H1':>15} {'Fit H2':>15} {'Winner':>10}")
    print("-" * 55)

    for data_label in ["H1_data", "H2_data"]:
        r = results[data_label]
        print(f"{data_label:>12} {r['H1_bic']:>15.1f} {r['H2_bic']:>15.1f} {r['winner']:>10}")

    print("-" * 55)

    # Check if diagonal
    h1_correct = results["H1_data"]["winner"] == "H1"
    h2_correct = results["H2_data"]["winner"] == "H2"

    if h1_correct and h2_correct:
        print("Diagonal: models are identifiable ✓")
    elif h1_correct or h2_correct:
        print("Partially diagonal: one model identifiable")
    else:
        print("Off-diagonal: models are confounded")

    for data_label in ["H1_data", "H2_data"]:
        r = results[data_label]
        delta = abs(r["H1_bic"] - r["H2_bic"])
        print(f"  Δ BIC ({data_label}): {delta:.1f}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    results = run_confusion_matrix(n_subjects=10, n_restarts=5, seed=42)
    print_confusion_matrix(results)