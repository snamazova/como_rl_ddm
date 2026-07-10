"""Reversal-locked plots: accuracy and RT time-locked to reversal onset.

Aligns behavioural data to each reversal point (trial −20 to +20) and
plots the average accuracy and RT across subjects and reversal points.
This is the descriptive check from Stage 2.4 of the project plan.

Usage:
    python reversal_locked_plots.py
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


# =============================================================================
# Reversal-locked aggregation
# =============================================================================

def compute_reversal_locked(data: pd.DataFrame,
                             reversal_points: list,
                             window: int = 20) -> dict:
    """Compute accuracy and RT aligned to reversal onset.

    For each reversal point, extracts trials from −window to +window
    relative to the reversal. Returns arrays of shape (n_reversals, 2*window+1).
    """
    n_trials = len(data)
    accuracy_curves = []
    rt_curves = []

    for rev in reversal_points:
        start = max(0, rev - window)
        end = min(n_trials, rev + window + 1)

        trials = np.arange(start, end)
        rel = trials - rev  # relative to reversal (−window to +window)

        if "is_correct" in data.columns:
            acc = data["is_correct"].iloc[start:end].to_numpy()
        else:
            acc = np.full(end - start, np.nan)
        rt = data["RTs"].iloc[start:end].to_numpy()

        # Pad to full window on both sides
        full_len = 2 * window + 1
        if len(rel) < full_len:
            pad_left = max(0, window - rev)
            pad_right = max(0, (rev + window + 1) - n_trials)
            acc = np.pad(acc, (pad_left, pad_right), constant_values=np.nan)
            rt = np.pad(rt, (pad_left, pad_right), constant_values=np.nan)
            rel = np.arange(-window, window + 1)

        accuracy_curves.append(acc)
        rt_curves.append(rt)

    return {
        "relative_trials": np.arange(-window, window + 1),
        "accuracy": np.array(accuracy_curves),
        "rt": np.array(rt_curves),
    }


def aggregate_reversal_locked(subjects: list,
                               reversal_points: list,
                               window: int = 20) -> dict:
    """Aggregate reversal-locked curves across subjects."""
    all_acc = []
    all_rt = []

    for sim in subjects:
        locked = compute_reversal_locked(sim["data"], reversal_points, window)
        all_acc.append(locked["accuracy"])
        all_rt.append(locked["rt"])

    all_acc = np.array(all_acc)  # (n_subjects, n_reversals, 2*window+1)
    all_rt = np.array(all_rt)

    return {
        "relative_trials": np.arange(-window, window + 1),
        "acc_mean": np.nanmean(all_acc, axis=(0, 1)),
        "acc_sem": np.nanstd(all_acc, axis=(0, 1)) / np.sqrt(all_acc.shape[0] * all_acc.shape[1]),
        "rt_mean": np.nanmean(all_rt, axis=(0, 1)),
        "rt_sem": np.nanstd(all_rt, axis=(0, 1)) / np.sqrt(all_rt.shape[0] * all_rt.shape[1]),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_reversal_locked(groups: dict,
                          reversal_points: list,
                          window: int = 20) -> plt.Figure:
    """Plot accuracy and RT time-locked to reversal onset for multiple groups.

    Parameters
    ----------
    groups : dict
        Maps group label -> list of simulation output dicts.
    """
    fig, (ax_acc, ax_rt) = plt.subplots(2, 1, figsize=(10, 8))
    colors = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    for i, (label, subjects) in enumerate(groups.items()):
        agg = aggregate_reversal_locked(subjects, reversal_points, window)
        x = agg["relative_trials"]
        color = colors[i % len(colors)]

        ax_acc.plot(x, agg["acc_mean"], color=color, lw=2, label=label)
        ax_acc.fill_between(x, agg["acc_mean"] - agg["acc_sem"],
                            agg["acc_mean"] + agg["acc_sem"], alpha=0.2, color=color)

        ax_rt.plot(x, agg["rt_mean"], color=color, lw=2, label=label)
        ax_rt.fill_between(x, agg["rt_mean"] - agg["rt_sem"],
                           agg["rt_mean"] + agg["rt_sem"], alpha=0.2, color=color)

    ax_acc.axvline(0, color="gray", ls="--", lw=1, label="reversal")
    ax_acc.axhline(0.5, color="gray", ls=":", lw=0.8, label="chance")
    ax_acc.set_xlabel("Trial relative to reversal")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy time-locked to reversal")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(frameon=False)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    ax_rt.axvline(0, color="gray", ls="--", lw=1, label="reversal")
    ax_rt.set_xlabel("Trial relative to reversal")
    ax_rt.set_ylabel("Response time (s)")
    ax_rt.set_title("RT time-locked to reversal")
    ax_rt.legend(frameon=False)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)

    fig.suptitle("Post-reversal behaviour (trial −20 to +20)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    rp = (36, 56, 71, 86, 106)

    h1_pars = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a": 3.0, "w": 0.5, "t0": 0.25}
    h2_pars = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a_base": 3.0,
               "kappa": 2.0, "tau": 5.0, "w": 0.5, "t0": 0.25}

    timeline = generate_timeline(num_trials=140, seed=42, reversed_state=True,
                                 reversal_points=rp)
    env = timeline_to_matrix(timeline)
    correct = timeline_to_correct(timeline)

    print("Simulating H1 group...")
    h1_subjects = []
    for s in range(20):
        sim = rlddm_simulate(env, h1_pars, rng=np.random.default_rng(42 + s),
                             correct_bandit=correct, reversal_points=rp)
        h1_subjects.append(sim)

    print("Simulating H2 group...")
    h2_subjects = []
    for s in range(20):
        sim = rlddm_simulate(env, h2_pars, rng=np.random.default_rng(142 + s),
                             correct_bandit=correct, reversal_points=rp)
        h2_subjects.append(sim)

    fig = plot_reversal_locked(
        {"H1 (drift-only)": h1_subjects, "H2 (boundary-only)": h2_subjects},
        list(rp), window=20,
    )

    # Print summary stats around reversal
    for label, subjects in [("H1", h1_subjects), ("H2", h2_subjects)]:
        agg = aggregate_reversal_locked(subjects, list(rp), window=20)
        idx_0 = 20  # index of trial 0
        print(f"\n{label}:")
        print(f"  Pre-reversal accuracy (trials -5 to -1): {np.mean(agg['acc_mean'][idx_0-5:idx_0]):.3f}")
        print(f"  Post-reversal accuracy (trials 0 to 4): {np.mean(agg['acc_mean'][idx_0:idx_0+5]):.3f}")
        print(f"  Pre-reversal RT (trials -5 to -1): {np.mean(agg['rt_mean'][idx_0-5:idx_0]):.3f}")
        print(f"  Post-reversal RT (trials 0 to 4): {np.mean(agg['rt_mean'][idx_0:idx_0+5]):.3f}")

    plt.show()