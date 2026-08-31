"""Parameter recovery for H1 and H2 models.

Simulates data with known parameters, fits the model, and checks
that the recovered parameters are close to the true ones.

Usage:
    python parameter_recovery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_task_environment import generate_timeline, timeline_to_matrix, timeline_to_correct
from rlddm import (
    rlddm_simulate, rlddm_log_lik, fill_rlddm_pars,
    fit_model, H1_PARAMS, H2_PARAMS, H1_DEFAULTS, H2_DEFAULTS, REVERSAL_POINTS,
)
from plotting_utils import get_dynamic_fontsize, save_panel, style_ticks


def _apply_dynamic_fontsize(fig) -> float:
    """Compute and apply a figure-width-scaled base fontsize via rcParams.

    Returns the computed fontsize so callers can also pass it to one-off
    ``fontsize=`` sites (e.g. ``fig.suptitle``) that don't inherit rcParams.
    """
    fontsize = get_dynamic_fontsize(fig_width=fig.get_size_inches()[0])
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })
    return fontsize


RP = REVERSAL_POINTS

# Ranges used to draw random "true" parameters for the batch recovery below.
# Deliberately narrower than rlddm.BOUNDS (the optimiser's search space) --
# these should stay in the region of plausible PRLT behaviour around the
# defaults, not the full space the fitter is allowed to search.
RECOVERY_RANGES = {
    "alpha":  (0.05, 0.5),
    "v_max":  (1.0, 3.0),
    "beta":   (0.3, 2.0),
    "a":      (2.0, 4.5),
    "a_base": (2.0, 4.5),
    "kappa":  (0.5, 3.0),
    "tau":    (2.0, 10.0),
    "w":      (0.35, 0.65),
    "t0":     (0.15, 0.4),
}


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


def sample_true_pars(model: str, rng: np.random.Generator) -> dict:
    """Draw one random 'true' parameter set for the given model."""
    params = H1_PARAMS if model == "H1" else H2_PARAMS
    return {name: float(rng.uniform(*RECOVERY_RANGES[name])) for name in params}


def run_recovery_batch(model: str,
                       n_iterations: int = 15,
                       n_restarts: int = 3,
                       seed: int = 42) -> pd.DataFrame:
    """Repeat parameter recovery over many random true-parameter draws.

    For each iteration: draw a random true parameter set, simulate a fresh
    140-trial PRLT dataset under it, fit the same model, and record true vs.
    recovered values. Returns a long-format DataFrame (one row per
    iteration x parameter) suitable for a true-vs-recovered scatter plot --
    a single (true, recovered) pair per parameter, as in ``run_recovery``,
    can't show whether recovery is reliable across the parameter space or
    just lucky for one setting.
    """
    rng = np.random.default_rng(seed)
    params = H1_PARAMS if model == "H1" else H2_PARAMS
    rows = []

    for it in range(n_iterations):
        true_pars = sample_true_pars(model, rng)
        timeline_seed = int(rng.integers(0, 1_000_000))
        timeline = generate_timeline(num_trials=140, seed=timeline_seed,
                                      reversed_state=True, reversal_points=RP)
        env = timeline_to_matrix(timeline)
        correct = timeline_to_correct(timeline)

        sim = rlddm_simulate(env, true_pars, rng=rng,
                              correct_bandit=correct, reversal_points=RP)
        data = sim["data"]

        fit = fit_model(model, data, reversal_points=RP,
                        n_restarts=n_restarts, rng=rng)
        recovered = fit["pars"]

        # fit_model's neg_ll returns the 1e10 sentinel when every restart
        # fails to converge (see rlddm.fit_model) -- that's an optimiser
        # failure, not a genuine "recovered" value, so it would show up as a
        # wild outlier and corrupt the scatter plot / MAE. Drop it, the same
        # way confusion_matrix.py filters fits with bic >= 1e6.
        if fit["bic"] >= 1e6:
            print(f"  [{model}] iteration {it + 1}/{n_iterations}  "
                  f"FAILED TO CONVERGE -- dropped")
            continue

        for name in params:
            rows.append({
                "model": model,
                "iteration": it,
                "parameter": name,
                "true": true_pars[name],
                "recovered": recovered[name],
            })

        print(f"  [{model}] iteration {it + 1}/{n_iterations}  "
              f"(fitted log-lik={fit['log_lik']:.1f})")

    n_dropped = n_iterations - (len(rows) // len(params) if rows else 0)
    if n_dropped:
        print(f"  [{model}] {n_dropped}/{n_iterations} iterations dropped "
              f"(optimiser failed to converge)")

    return pd.DataFrame(rows)


def plot_parameter_recovery(df: pd.DataFrame, params: list,
                            model_label: str) -> plt.Figure:
    """True-vs-recovered scatter plot, one panel per parameter.

    Points on the dashed identity line = perfect recovery; Pearson r and
    mean absolute error are annotated per panel to make imprecise or
    confounded parameters (e.g. v_max/beta) visually obvious.
    """
    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fontsize = _apply_dynamic_fontsize(fig)
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(params):
        ax = axes[i]
        sub = df[df["parameter"] == name]
        true = sub["true"].to_numpy()
        recovered = sub["recovered"].to_numpy()

        lo = min(true.min(), recovered.min())
        hi = max(true.max(), recovered.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="gray", ls="--", lw=1, zorder=1)
        ax.scatter(true, recovered, color="#0072B2", alpha=0.75,
                   edgecolor="white", s=50, zorder=2)

        if len(true) > 1 and np.std(true) > 0 and np.std(recovered) > 0:
            r = np.corrcoef(true, recovered)[0, 1]
            r_label = f"{r:.2f}"
        else:
            r_label = "n/a"
        mae = np.mean(np.abs(recovered - true))

        ax.set_title(f"{name}    r = {r_label}, MAE = {mae:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Recovered")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        style_ticks(ax)

    for j in range(len(params), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Parameter recovery — {model_label}", fontweight="bold",
                fontsize=fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


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
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameter recovery for the H1/H2 RLDDM (single-point "
                     "sanity check + batch recovery plots).")
    parser.add_argument("--n-iterations", type=int, default=15,
                        help="Random true-parameter draws per model for the "
                             "batch recovery plots. Default: 15.")
    parser.add_argument("--n-restarts", type=int, default=5,
                        help="Nelder-Mead restarts per fit in the batch "
                             "recovery. Default: 5.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open an interactive plot window.")
    args = parser.parse_args()

    print("Fitting H1 recovery (single-point sanity check)...")
    r1 = run_recovery("H1", H1_DEFAULTS, n_restarts=5, seed=42)
    print_results(r1, "H1 (drift-only)")

    print("\nFitting H2 recovery (single-point sanity check)...")
    r2 = run_recovery("H2", H2_DEFAULTS, n_restarts=5, seed=42)
    print_results(r2, "H2 (boundary varies)")

    print(f"\n{'=' * 60}")
    print(f"Batch parameter recovery ({args.n_iterations} random draws per "
          f"model) -- this validates recovery across the parameter space, "
          f"not just at one setting.")
    print(f"{'=' * 60}")

    print("\nH1 batch recovery...")
    df_h1 = run_recovery_batch("H1", n_iterations=args.n_iterations,
                               n_restarts=args.n_restarts, seed=args.seed)
    fig_h1 = plot_parameter_recovery(df_h1, H1_PARAMS, "H1 (drift-only)")
    save_panel(fig_h1, "parameter_recovery_h1.png", figsize=fig_h1.get_size_inches(), dpi=150)
    print("  -> parameter_recovery_h1.png")

    print("\nH2 batch recovery...")
    df_h2 = run_recovery_batch("H2", n_iterations=args.n_iterations,
                               n_restarts=args.n_restarts, seed=args.seed + 1000)
    fig_h2 = plot_parameter_recovery(df_h2, H2_PARAMS, "H2 (boundary varies)")
    save_panel(fig_h2, "parameter_recovery_h2.png", figsize=fig_h2.get_size_inches(), dpi=150)
    print("  -> parameter_recovery_h2.png")

    if not args.no_show:
        plt.show()