"""Parameter sweep: systematically vary one RLDDM parameter at a time and
show how it affects PRLT behaviour (accuracy, post-reversal recovery,
perseveration, RT).

This directly addresses the project guideline:
    "Changes in the parameters of interest produce the predicted changes
     in behaviour."

Usage:
    python parameter_sweep.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_task_environment import (
    generate_timeline,
    timeline_to_matrix,
    timeline_to_correct,
)
from rlddm import rlddm_simulate
from simulator import (
    simulate_group,
    aggregate_group,
    compute_accuracy,
    compute_post_reversal_accuracy,
    compute_perseveration,
    compute_win_stay_lose_shift,
)
from plotting_utils import get_dynamic_fontsize, style_ticks


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


REVERSAL_POINTS = (36, 56, 71, 86, 106)
NUM_TRIALS = 140
N_SUBJECTS = 20

BASELINE_PARS = {
    "alpha": 0.25,
    "v_max": 2.0,
    "beta": 1.0,
    "a": 3.0,
    "w": 0.5,
    "t0": 0.25,
}


# =============================================================================
# Sweep helpers
# =============================================================================

def run_sweep(param_name: str,
              values: list,
              n_subjects: int = N_SUBJECTS,
              base_pars: dict = None) -> pd.DataFrame:
    """Sweep one parameter across a range of values.

    For each value, simulate a group and compute behavioural metrics.
    Returns a DataFrame with one row per (value, subject) pair.
    """
    base_pars = base_pars or BASELINE_PARS
    rp = list(REVERSAL_POINTS)
    rows = []

    for val in values:
        pars = dict(base_pars)
        pars[param_name] = val

        group = simulate_group(
            pars, n_subjects=n_subjects, seed=42,
            reversal_points=rp,
        )

        for sim in group:
            data = sim["data"]
            pra = compute_post_reversal_accuracy(data, rp, window=10)
            pers = compute_perseveration(data, rp, window=5)
            wsls = compute_win_stay_lose_shift(data)

            # Average post-reversal accuracy across all reversal points
            pra_means = [np.mean(v) for v in pra.values()]
            pers_means = list(pers.values())

            rows.append({
                "param": param_name,
                "value": val,
                "subject_id": sim["subject_id"],
                "overall_accuracy": compute_accuracy(data),
                "post_reversal_acc_1_5": np.mean([np.mean(v[:5]) for v in pra.values()]),
                "post_reversal_acc_6_10": np.mean([np.mean(v[5:10]) for v in pra.values() if len(v) >= 10]),
                "perseveration_rate": np.mean(pers_means),
                "win_stay": wsls["win_stay"],
                "lose_shift": wsls["lose_shift"],
                "mean_rt": data["RTs"].mean(),
            })

    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, param_name: str, ylabel: str = "Value") -> plt.Figure:
    """Plot behavioural metrics as a function of the swept parameter."""
    metrics = [
        ("overall_accuracy", "Overall accuracy", "#0072B2"),
        ("post_reversal_acc_1_5", "Post-rev accuracy (trials 1-5)", "#D55E00"),
        ("post_reversal_acc_6_10", "Post-rev accuracy (trials 6-10)", "#009E73"),
        ("perseveration_rate", "Perseveration rate", "#CC79A7"),
        ("win_stay", "Win-stay probability", "#E69F00"),
        ("mean_rt", "Mean response time (s)", "#56B4E9"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fontsize = _apply_dynamic_fontsize(fig)
    axes = axes.ravel()

    for i, (col, title, color) in enumerate(metrics):
        agg = df.groupby("value")[col].agg(["mean", "std", "count"])
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])

        x = agg.index.values
        y = agg["mean"].values
        sem = agg["sem"].values

        axes[i].errorbar(x, y, yerr=sem, color=color, marker="o", lw=2, capsize=3)
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel(title)
        axes[i].set_title(title)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        style_ticks(axes[i])

        # Add baseline reference line if this isn't the baseline value
        baseline_val = BASELINE_PARS.get(param_name)
        if baseline_val is not None and baseline_val in x:
            axes[i].axvline(baseline_val, color="gray", ls=":", lw=0.8)

    fig.suptitle(f"Effect of {param_name} on PRLT behaviour", fontweight="bold",
                fontsize=fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Parameter Sweep: alpha (learning rate)")
    print("=" * 60)
    df_alpha = run_sweep("alpha", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    print(df_alpha.groupby("value")[
        ["overall_accuracy", "post_reversal_acc_1_5", "perseveration_rate", "mean_rt"]
    ].mean().round(3))
    fig_alpha = plot_sweep(df_alpha, "alpha")

    print("\n" + "=" * 60)
    print("Parameter Sweep: a (decision threshold)")
    print("=" * 60)
    df_a = run_sweep("a", [1.5, 2.0, 3.0, 4.0, 5.0])
    print(df_a.groupby("value")[
        ["overall_accuracy", "post_reversal_acc_1_5", "perseveration_rate", "mean_rt"]
    ].mean().round(3))
    fig_a = plot_sweep(df_a, "a")

    print("\n" + "=" * 60)
    print("Parameter Sweep: beta (drift sensitivity to value difference)")
    print("=" * 60)
    df_beta = run_sweep("beta", [0.1, 0.5, 1.0, 2.0, 3.0])
    print(df_beta.groupby("value")[
        ["overall_accuracy", "post_reversal_acc_1_5", "perseveration_rate", "mean_rt"]
    ].mean().round(3))
    fig_beta = plot_sweep(df_beta, "beta")

    plt.show()