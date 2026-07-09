"""Parameter recovery for H1 and H2 models.

Simulates data with known parameters, fits the model, and checks
that the recovered parameters are close to the true ones.

H1 (drift-only):  alpha, v_max, beta, a, w, t0
H2 (boundary):   alpha, v_max, beta, a_base, kappa, tau, w, t0

Usage:
    python parameter_recovery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from create_task_environment import generate_timeline, timeline_to_matrix, timeline_to_correct
from rlddm import rlddm_simulate, rlddm_log_lik, fill_rlddm_pars

RP = (36, 56, 71, 86, 106)

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


def _unpack(theta: np.ndarray, params: list) -> dict:
    pars = {}
    for i, name in enumerate(params):
        lo, hi = BOUNDS[name]
        pars[name] = lo + (hi - lo) / (1.0 + np.exp(-theta[i]))
    return pars


def _pack(pars: dict, params: list) -> np.ndarray:
    """Constrained -> unconstrained (inverse sigmoid)."""
    theta = np.zeros(len(params))
    for i, name in enumerate(params):
        lo, hi = BOUNDS[name]
        val = pars[name]
        ratio = (val - lo) / (hi - lo)
        ratio = np.clip(ratio, 1e-6, 1 - 1e-6)
        theta[i] = np.log(ratio / (1 - ratio))
    return theta


def fit_model(model: str, data: pd.DataFrame,
              n_restarts: int = 5,
              rng: np.random.Generator | None = None) -> dict:
    """Fit H1 or H2 via multi-start Nelder-Mead."""
    rng = rng or np.random.default_rng()
    params = H1_PARAMS if model == "H1" else H2_PARAMS

    def neg_ll(theta):
        pars = _unpack(theta, params)
        try:
            ll = rlddm_log_lik(data, pars, reversal_points=RP)
        except (ValueError, FloatingPointError):
            return 1e10
        return -ll if np.isfinite(ll) else 1e10

    # Informed starting point near default values
    informed = {"alpha": 0.0, "v_max": 0.5, "beta": 0.0,
                "a": 0.5, "a_base": 0.5, "kappa": 0.0,
                "tau": 0.0, "w": 0.0, "t0": 0.0}
    start_informed = np.array([informed[p] for p in params])

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

    return {"pars": best_pars, "neg_ll": best_ll}


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

    fit = fit_model(model, data, n_restarts=n_restarts, rng=rng)
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
        "fitted_neg_ll": fit["neg_ll"],
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
    print(f"{'True neg-LL:':<20} {result['true_neg_ll']:.4f}")
    print(f"{'Fitted neg-LL:':<20} {result['fitted_neg_ll']:.4f}")
    max_err = max(v["abs_error"] for v in result["comparison"].values())
    status = "PASS" if max_err < 0.25 else "CHECK"
    print(f"{'Max abs error:':<20} {max_err:.4f}  -> {status}")


if __name__ == "__main__":
    # H1 recovery
    h1_true = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a": 3.0, "w": 0.5, "t0": 0.25}
    print("Fitting H1 recovery...")
    r1 = run_recovery("H1", h1_true, n_restarts=5, seed=42)
    print_results(r1, "H1 (drift-only)")

    # H2 recovery
    h2_true = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a_base": 3.0,
               "kappa": 2.0, "tau": 5.0, "w": 0.5, "t0": 0.25}
    print("\nFitting H2 recovery...")
    r2 = run_recovery("H2", h2_true, n_restarts=5, seed=42)
    print_results(r2, "H2 (boundary-only)")