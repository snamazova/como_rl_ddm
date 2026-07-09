"""Parameter recovery for the RLDDM.

Simulates synthetic data with known parameters, then fits the model by
maximising the log-likelihood (via scipy.optimize.minimize) and checks
that the recovered parameters are close to the true ones.

Usage:
    python parameter_recovery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from create_task_environment import generate_timeline, timeline_to_matrix
from rlddm import rlddm_simulate, rlddm_log_lik, fill_rlddm_pars


# =============================================================================
# Optimisation helpers
# =============================================================================

# Parameters to optimise, their bounds, and unconstraining transforms.
# We fix sv/sw/st0 at 0 for the basic model (matching the default).
PARAM_NAMES = ["alpha", "v_intercept", "v_scale", "a", "w", "t0"]
PARAM_BOUNDS = {
    "alpha":       (1e-3, 0.999),
    "v_intercept": (-3.0, 3.0),
    "v_scale":     (1e-3, 5.0),
    "a":           (0.5, 6.0),
    "w":           (0.01, 0.99),
    "t0":          (1e-3, 1.0),
}


def _unpack(theta: np.ndarray) -> dict:
    """Convert an unconstrained vector back to a constrained parameter dict."""
    pars = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        # Sigmoid-style transform for bounded params, identity for unbounded
        if name in ("alpha", "w"):
            pars[name] = lo + (hi - lo) / (1.0 + np.exp(-theta[i]))
        elif name in ("v_scale", "a", "t0"):
            pars[name] = lo + (hi - lo) / (1.0 + np.exp(-theta[i]))
        else:
            pars[name] = np.clip(theta[i], lo, hi)
    return pars


def _neg_log_lik(theta: np.ndarray, data: pd.DataFrame) -> float:
    """Negative log-likelihood for scipy.optimize.minimize."""
    pars = _unpack(theta)
    try:
        ll = rlddm_log_lik(data, pars)
    except (ValueError, FloatingPointError):
        return 1e10
    if not np.isfinite(ll):
        return 1e10
    return -ll


def fit_rlddm(data: pd.DataFrame,
               n_restarts: int = 5,
               rng: np.random.Generator | None = None) -> dict:
    """Fit the RLDDM to data via multi-start Nelder-Mead optimisation.

    Returns the best-fitting parameter dict and the negative log-likelihood.
    """
    rng = rng or np.random.default_rng()

    best_ll = np.inf
    best_pars = None

    for _ in range(n_restarts):
        # Random starting point in unconstrained space
        theta0 = rng.normal(0.0, 1.0, size=len(PARAM_NAMES))

        result = minimize(
            _neg_log_lik,
            theta0,
            args=(data,),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )

        if result.fun < best_ll:
            best_ll = result.fun
            best_pars = _unpack(result.x)

    return {"pars": best_pars, "neg_ll": best_ll}


# =============================================================================
# Recovery experiment
# =============================================================================

def run_recovery(true_pars: dict,
                 n_trials: int = 140,
                 n_restarts: int = 5,
                 rng_seed: int = 42) -> dict:
    """Simulate data with true_pars, fit the model, and return comparison."""
    rng = np.random.default_rng(rng_seed)

    # 1. Generate PRLT timeline and simulate data
    timeline = generate_timeline(
        num_trials=n_trials,
        seed=rng_seed,
        reversed_state=True,
    )
    env = timeline_to_matrix(timeline)
    sim = rlddm_simulate(env, true_pars, rng=rng)
    data = sim["data"]

    # 2. Fit the model
    fit = fit_rlddm(data, n_restarts=n_restarts, rng=rng)

    # 3. Compare
    recovered = fit["pars"]
    true_filled = fill_rlddm_pars(true_pars)

    comparison = {}
    for name in PARAM_NAMES:
        comparison[name] = {
            "true": true_filled[name],
            "recovered": recovered[name],
            "abs_error": abs(recovered[name] - true_filled[name]),
        }

    return {
        "comparison": comparison,
        "true_neg_ll": -rlddm_log_lik(data, true_pars),
        "fitted_neg_ll": fit["neg_ll"],
        "n_trials": n_trials,
    }


def print_results(result: dict, label: str = ""):
    """Pretty-print a recovery result."""
    print(f"\n{'=' * 60}")
    print(f"Parameter Recovery: {label}")
    print(f"{'=' * 60}")
    print(f"{'Parameter':<14} {'True':>10} {'Recovered':>10} {'Abs Error':>10}")
    print("-" * 48)
    for name, vals in result["comparison"].items():
        print(f"{name:<14} {vals['true']:>10.4f} {vals['recovered']:>10.4f} "
              f"{vals['abs_error']:>10.4f}")
    print("-" * 48)
    print(f"{'True neg-LL:':<20} {result['true_neg_ll']:.4f}")
    print(f"{'Fitted neg-LL:':<20} {result['fitted_neg_ll']:.4f}")
    print(f"{'LL improvement:':<20} {result['true_neg_ll'] - result['fitted_neg_ll']:.4f}")

    # Overall pass/fail threshold
    max_err = max(v["abs_error"] for v in result["comparison"].values())
    status = "PASS" if max_err < 0.25 else "CHECK"
    print(f"{'Max abs error:':<20} {max_err:.4f}  -> {status}")


# =============================================================================
# Main: run multiple recovery experiments
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        {
            "label": "moderate learning, moderate drift",
            "pars": {"alpha": 0.25, "v_intercept": 0.0, "v_scale": 1.0,
                     "a": 3.0, "w": 0.5, "t0": 0.25},
        },
        {
            "label": "fast learning, strong drift",
            "pars": {"alpha": 0.5, "v_intercept": 0.0, "v_scale": 2.0,
                     "a": 2.0, "w": 0.5, "t0": 0.2},
        },
        {
            "label": "slow learning, weak drift",
            "pars": {"alpha": 0.1, "v_intercept": 0.0, "v_scale": 0.5,
                     "a": 3.0, "w": 0.6, "t0": 0.3},
        },
    ]

    for case in test_cases:
        result = run_recovery(
            true_pars=case["pars"],
            n_trials=140,
            n_restarts=8,
            rng_seed=42,
        )
        print_results(result, label=case["label"])