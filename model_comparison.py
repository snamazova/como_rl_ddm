"""Model comparison: Full RLDDM vs DDM-only vs RL-only.

Simulates data from the full RLDDM, then fits three nested models to the
same data and compares their log-likelihoods (and BIC) to show that both
learning (RL) and response times (DDM) contribute to the model's
explanatory power.

The three models are:
    1. Full RLDDM  — all parameters free (learning + DDM)
    2. DDM-only    — v_scale = 0 (no value-based drift, just a constant bias)
    3. RL-only     — softmax choice probabilities, no DDM / no RTs

Models 2 and 3 are nested within model 1, so the comparison is valid.

Usage:
    python model_comparison.py
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
from rlddm import (
    rlddm_simulate,
    rlddm_log_lik,
    fill_rlddm_pars,
    value_update,
    softmax,
    compute_drift,
    _choice_probability,
)


# =============================================================================
# RL-only model: softmax choices, no RT
# =============================================================================

def rl_log_lik(data: pd.DataFrame,
               pars: dict,
               initial_values: tuple = (0.0, 0.0)) -> float:
    """Log-likelihood of an RL-only (softmax) model.

    Learns values via Rescorla-Wagner, converts to choice probabilities via
    softmax with inverse temperature ``beta``, and sums the log-probability
    of the observed choices.  No DDM, no RTs.
    """
    alpha = pars["alpha"]
    beta = pars["beta"]
    choices = np.asarray(data["choices"], dtype=int)
    outcomes = np.asarray(data["outcomes"], dtype=float)
    n_trials = len(choices)

    values = np.array(initial_values, dtype=float)
    ll = 0.0

    for t in range(n_trials):
        probs = softmax(values, beta)
        choice_idx = choices[t] - 1  # 1/2 -> 0/1
        p = max(probs[choice_idx], 1e-12)
        ll += np.log(p)

        # Rescorla-Wagner update
        values = values.copy()
        values[choice_idx] = value_update(
            value=values[choice_idx],
            outcome=outcomes[t],
            alpha=alpha,
        )

    return float(ll)


# =============================================================================
# RLDDM marginal choice log-likelihood (integrates out RTs)
# =============================================================================

def rlddm_choice_log_lik(data: pd.DataFrame,
                         pars: dict,
                         initial_values: tuple = (0.0, 0.0)) -> float:
    """Marginal choice log-likelihood of the RLDDM (RTs integrated out).

    For each trial, computes P(choice_t | drift_t, a, w) — the probability
    that the DDM hits the observed boundary, ignoring the RT.  This is
    directly comparable to the RL-only model's choice log-likelihood.
    """
    pars = fill_rlddm_pars(pars)
    choices = np.asarray(data["choices"], dtype=int)
    outcomes = np.asarray(data["outcomes"], dtype=float)
    n_trials = len(choices)
    n_options = len(initial_values)
    initial_values = np.asarray(initial_values, dtype=float)

    values = initial_values.copy()
    ll = 0.0

    for t in range(n_trials):
        drift = compute_drift(values, pars)
        p_choice = _choice_probability(drift, pars["a"], pars["w"], choices[t])
        ll += np.log(max(p_choice, 1e-12))

        # Rescorla-Wagner update
        values = values.copy()
        values[choices[t] - 1] = value_update(
            value=values[choices[t] - 1],
            outcome=outcomes[t],
            alpha=pars["alpha"],
        )

    return float(ll)


# =============================================================================
# DDM-only model: constant drift (v_scale = 0), no learning
# =============================================================================

def ddm_only_log_lik(data: pd.DataFrame, pars: dict) -> float:
    """Log-likelihood of a DDM-only model (no value-based learning).

    Uses the full RLDDM likelihood machinery but with ``v_scale = 0``,
    so the drift rate is just ``v_intercept`` (a constant).  Values are
    still reconstructed from the observed choices/outcomes for consistency,
    but they have no effect on the drift.
    """
    pars_fixed = dict(pars)
    pars_fixed["v_scale"] = 0.0
    return rlddm_log_lik(data, pars_fixed)


# =============================================================================
# Fitting helpers
# =============================================================================

# Parameter sets and bounds for each model
# Full RLDDM: alpha, v_intercept, v_scale, a, w, t0
# DDM-only:  v_intercept, a, w, t0 (alpha irrelevant for drift, but still learned for values)
# RL-only:   alpha, beta

FULL_PARAMS = ["alpha", "v_intercept", "v_scale", "a", "w", "t0"]
DDM_ONLY_PARAMS = ["v_intercept", "a", "w", "t0"]
RL_ONLY_PARAMS = ["alpha", "beta"]

BOUNDS = {
    "alpha":       (1e-3, 0.999),
    "v_intercept": (-3.0, 3.0),
    "v_scale":     (1e-3, 5.0),
    "a":           (0.5, 6.0),
    "w":           (0.01, 0.99),
    "t0":          (1e-3, 1.0),
    "beta":        (1e-3, 10.0),
}


def _unpack(theta: np.ndarray, param_names: list) -> dict:
    """Transform unconstrained vector to constrained parameter dict."""
    pars = {}
    for i, name in enumerate(param_names):
        lo, hi = BOUNDS[name]
        pars[name] = lo + (hi - lo) / (1.0 + np.exp(-theta[i]))
    return pars


def _neg_log_lik_factory(model: str, data: pd.DataFrame):
    """Create a negative log-likelihood function for the given model."""
    def neg_ll(theta: np.ndarray, param_names: list) -> float:
        pars = _unpack(theta, param_names)
        try:
            if model == "full":
                ll = rlddm_log_lik(data, pars)
            elif model == "ddm_only":
                ll = ddm_only_log_lik(data, pars)
            elif model == "rl_only":
                ll = rl_log_lik(data, pars)
            elif model == "full_choice":
                ll = rlddm_choice_log_lik(data, pars)
            else:
                raise ValueError(f"Unknown model: {model}")
        except (ValueError, FloatingPointError):
            return 1e10
        if not np.isfinite(ll):
            return 1e10
        return -ll
    return neg_ll


def fit_model(model: str,
              data: pd.DataFrame,
              n_restarts: int = 5,
              rng: np.random.Generator | None = None) -> dict:
    """Fit one of the three models via multi-start Nelder-Mead."""
    rng = rng or np.random.default_rng()

    if model == "full":
        params = FULL_PARAMS
    elif model == "ddm_only":
        params = DDM_ONLY_PARAMS
    elif model == "rl_only":
        params = RL_ONLY_PARAMS
    elif model == "full_choice":
        params = FULL_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    neg_ll = _neg_log_lik_factory(model, data)
    best_ll = np.inf
    best_pars = None

    for _ in range(n_restarts):
        theta0 = rng.normal(0.0, 1.0, size=len(params))
        result = minimize(
            neg_ll,
            theta0,
            args=(params,),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if result.fun < best_ll:
            best_ll = result.fun
            best_pars = _unpack(result.x, params)

    n_params = len(params)
    n_trials = len(data)
    bic = 2 * best_ll + n_params * np.log(n_trials)

    return {
        "model": model,
        "pars": best_pars,
        "log_lik": -best_ll,
        "n_params": n_params,
        "bic": float(bic),
    }


# =============================================================================
# Comparison experiment
# =============================================================================

def run_comparison(true_pars: dict,
                   n_trials: int = 140,
                   n_restarts: int = 8,
                   seed: int = 42) -> dict:
    """Simulate data from the full RLDDM, fit all three models, compare."""
    rng = np.random.default_rng(seed)
    rp = (36, 56, 71, 86, 106)

    # 1. Simulate data from the full model
    timeline = generate_timeline(
        num_trials=n_trials, seed=seed, reversed_state=True,
        reversal_points=rp,
    )
    env = timeline_to_matrix(timeline)
    correct = timeline_to_correct(timeline)
    sim = rlddm_simulate(
        env, true_pars, rng=rng,
        correct_bandit=correct, reversal_points=rp,
    )
    data = sim["data"]

    # 2. Fit all models
    print("Fitting full RLDDM (choice + RT)...")
    fit_full = fit_model("full", data, n_restarts=n_restarts, rng=rng)
    print("Fitting DDM-only (v_scale=0, choice + RT)...")
    fit_ddm = fit_model("ddm_only", data, n_restarts=n_restarts, rng=rng)
    print("Fitting full RLDDM (choice only, RTs integrated out)...")
    fit_full_choice = fit_model("full_choice", data, n_restarts=n_restarts, rng=rng)
    print("Fitting RL-only (softmax choices)...")
    fit_rl = fit_model("rl_only", data, n_restarts=n_restarts, rng=rng)

    results = {
        "true_pars": true_pars,
        "data": data,
        "fits": {
            "full": fit_full,
            "ddm_only": fit_ddm,
            "full_choice": fit_full_choice,
            "rl_only": fit_rl,
        },
    }
    return results


def print_comparison(results: dict):
    """Pretty-print the model comparison table."""
    fits = results["fits"]
    n_trials = len(results["data"])

    labels = {
        "full": "Full RLDDM (choice+RT)",
        "ddm_only": "DDM-only (no RL)",
        "full_choice": "Full RLDDM (choice only)",
        "rl_only": "RL-only (no DDM)",
    }

    # --- Comparison 1: Full (choice+RT) vs DDM-only (choice+RT) ---
    print(f"\n{'=' * 65}")
    print("Comparison 1: Does learning (RL) matter?")
    print("  Both models predict choices AND response times.")
    print(f"{'=' * 65}")
    print(f"{'Model':<28} {'Params':>7} {'Log-lik':>12} {'BIC':>12}")
    print("-" * 63)
    for name in ["full", "ddm_only"]:
        f = fits[name]
        print(f"{labels[name]:<28} {f['n_params']:>7} {f['log_lik']:>12.2f} {f['bic']:>12.2f}")
    print("-" * 63)
    delta = fits["ddm_only"]["bic"] - fits["full"]["bic"]
    print(f"  Δ BIC (DDM-only vs full): {delta:.2f}", end="")
    if delta > 10:
        print(" -> Learning matters strongly")
    elif delta > 0:
        print(" -> Learning matters")
    else:
        print(" -> Learning does not improve fit")

    # --- Comparison 2: Full (choice only) vs RL-only (choice only) ---
    print(f"\n{'=' * 65}")
    print("Comparison 2: Do response times (DDM) matter?")
    print("  Both models predict choices only (RTs integrated out).")
    print(f"{'=' * 65}")
    print(f"{'Model':<28} {'Params':>7} {'Log-lik':>12} {'BIC':>12}")
    print("-" * 63)
    for name in ["full_choice", "rl_only"]:
        f = fits[name]
        print(f"{labels[name]:<28} {f['n_params']:>7} {f['log_lik']:>12.2f} {f['bic']:>12.2f}")
    print("-" * 63)
    delta2 = fits["rl_only"]["bic"] - fits["full_choice"]["bic"]
    print(f"  Δ BIC (RL-only vs full): {delta2:.2f}", end="")
    if delta2 > 10:
        print(" -> RTs matter strongly")
    elif delta2 > 0:
        print(" -> RTs matter")
    else:
        print(" -> RTs do not improve fit")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    true_pars = {
        "alpha": 0.25,
        "v_intercept": 0.0,
        "v_scale": 1.0,
        "a": 3.0,
        "w": 0.5,
        "t0": 0.25,
    }

    results = run_comparison(true_pars, n_trials=140, n_restarts=8, seed=42)
    print_comparison(results)