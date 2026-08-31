"""PRLT RLDDM multi-subject simulator and behavioural analysis.

Simulates groups of synthetic participants on the probabilistic reversal
learning task (PRLT), computes behavioural metrics (accuracy, post-reversal
recovery, perseveration, win-stay/lose-shift), and produces group-comparison
plots.

Usage:
    python simulator.py
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


# =============================================================================
# PRLT behavioural metrics
# =============================================================================

def compute_accuracy(data: pd.DataFrame) -> float:
    """Overall proportion of trials where the correct bandit was chosen."""
    if "is_correct" not in data.columns:
        raise ValueError("data must contain 'is_correct' column")
    return float(data["is_correct"].mean())


def compute_accuracy_curve(data: pd.DataFrame, window: int = 5) -> np.ndarray:
    """Rolling accuracy over trials (window size = window)."""
    return data["is_correct"].rolling(window=window, min_periods=1).mean().to_numpy()


def compute_post_reversal_accuracy(data: pd.DataFrame,
                                    reversal_points: list,
                                    window: int = 10) -> dict:
    """Accuracy in the first `window` trials after each reversal point.

    Returns a dict mapping reversal point -> accuracy array of length `window`.
    """
    results = {}
    for rev in reversal_points:
        start = rev
        end = min(rev + window, len(data))
        if start < len(data):
            results[rev] = data["is_correct"].iloc[start:end].to_numpy()
    return results


def compute_perseveration(data: pd.DataFrame,
                          reversal_points: list,
                          window: int = 5) -> dict:
    """Proportion of choices to the *previously* correct bandit after reversal.

    For each reversal at trial `rev`, the previously correct bandit is the one
    that was correct *before* the reversal.  We approximate this by checking
    whether the choice in the first `window` trials after reversal is the
    opposite of what `is_correct` expects — i.e. choosing the old bandit.

    Returns a dict mapping reversal point -> perseveration rate.
    """
    results = {}
    for rev in reversal_points:
        start = rev
        end = min(rev + window, len(data))
        if start >= len(data):
            continue
        # After reversal, the correct bandit has flipped.  Perseveration =
        # choosing the *old* correct bandit = choosing *incorrectly*.
        perseveration_rate = 1.0 - data["is_correct"].iloc[start:end].mean()
        results[rev] = float(perseveration_rate)
    return results


def compute_win_stay_lose_shift(data: pd.DataFrame) -> dict:
    """Win-stay and lose-shift probabilities.

    Win-stay: P(repeat choice | previous trial was rewarded).
    Lose-shift: P(switch choice | previous trial was not rewarded).
    """
    choices = data["choices"].to_numpy()
    outcomes = data["outcomes"].to_numpy()

    win_stay = []
    lose_shift = []
    for t in range(1, len(choices)):
        if outcomes[t - 1] == 1:
            win_stay.append(int(choices[t] == choices[t - 1]))
        else:
            lose_shift.append(int(choices[t] != choices[t - 1]))

    return {
        "win_stay": float(np.mean(win_stay)) if win_stay else np.nan,
        "lose_shift": float(np.mean(lose_shift)) if lose_shift else np.nan,
    }


# =============================================================================
# Multi-subject simulation
# =============================================================================

def simulate_group(pars: dict,
                   n_subjects: int = 50,
                   num_trials: int = 140,
                   reversal_points: tuple = (36, 56, 71, 86, 106),
                   p_correct: float = 0.8,
                   seed: int = 42,
                   shared_timeline: bool = True) -> list:
    """Simulate a group of synthetic PRLT participants.

    Parameters
    ----------
    pars : dict
        RLDDM parameters shared by all subjects in this group.
    n_subjects : int
        Number of simulated subjects.
    shared_timeline : bool
        If True, all subjects get the same reward sequence (same timeline)
        but different RNG for DDM sampling.  If False, each gets their own
        timeline draw.

    Returns a list of simulation output dicts (one per subject).
    """
    rng = np.random.default_rng(seed)

    # Generate the shared timeline once (or per-subject)
    if shared_timeline:
        timeline = generate_timeline(
            num_trials=num_trials,
            seed=seed,
            reversed_state=True,
            reversal_points=reversal_points,
            p_correct=p_correct,
        )
        env = timeline_to_matrix(timeline)
        correct = timeline_to_correct(timeline)

    results = []
    for s in range(n_subjects):
        subject_rng = np.random.default_rng(seed + s + 1)

        if not shared_timeline:
            timeline = generate_timeline(
                num_trials=num_trials,
                seed=seed + s,
                reversed_state=s % 2 == 0,
                reversal_points=reversal_points,
                p_correct=p_correct,
            )
            env = timeline_to_matrix(timeline)
            correct = timeline_to_correct(timeline)

        sim = rlddm_simulate(
            env, pars, rng=subject_rng,
            correct_bandit=correct,
            reversal_points=reversal_points,
        )
        sim["subject_id"] = s
        results.append(sim)

    return results


def aggregate_group(results: list) -> dict:
    """Aggregate simulation results across subjects in a group.

    Returns mean accuracy curves, mean RTs, and behavioural metrics.
    """
    n_subjects = len(results)
    n_trials = len(results[0]["data"])

    accuracy = np.zeros((n_subjects, n_trials))
    rts = np.zeros((n_subjects, n_trials))
    choices = np.zeros((n_subjects, n_trials), dtype=int)
    outcomes = np.zeros((n_subjects, n_trials))

    for i, sim in enumerate(results):
        d = sim["data"]
        if "is_correct" in d.columns:
            accuracy[i] = d["is_correct"].to_numpy()
        rts[i] = d["RTs"].to_numpy()
        choices[i] = d["choices"].to_numpy()
        outcomes[i] = d["outcomes"].to_numpy()

    return {
        "accuracy_mean": accuracy.mean(axis=0),
        "accuracy_sem": accuracy.std(axis=0) / np.sqrt(n_subjects),
        "rt_mean": rts.mean(axis=0),
        "rt_sem": rts.std(axis=0) / np.sqrt(n_subjects),
        "overall_accuracy": accuracy.mean(),
        "n_subjects": n_subjects,
        "n_trials": n_trials,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_group_comparison(group_results: dict,
                          reversal_points: list,
                          labels: list,
                          title: str = "Group Comparison") -> plt.Figure:
    """Compare accuracy and RT curves across groups.

    Parameters
    ----------
    group_results : dict
        Maps group label -> aggregated group dict (from ``aggregate_group``).
    reversal_points : list
        Trial indices where reversals occur.
    labels : list
        Group labels in the same order as group_results keys.
    """
    fig, (ax_acc, ax_rt) = plt.subplots(2, 1, figsize=(10, 8))
    _apply_dynamic_fontsize(fig)

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for i, label in enumerate(labels):
        agg = group_results[label]
        trials = np.arange(1, agg["n_trials"] + 1)
        color = colors[i % len(colors)]

        # Accuracy (smoothed)
        acc_smoothed = pd.Series(agg["accuracy_mean"]).rolling(5, min_periods=1).mean()
        ax_acc.plot(trials, acc_smoothed, color=color, lw=2, label=label)
        ax_acc.fill_between(
            trials,
            acc_smoothed - agg["accuracy_sem"],
            acc_smoothed + agg["accuracy_sem"],
            alpha=0.2, color=color,
        )

        # RT
        rt_smoothed = pd.Series(agg["rt_mean"]).rolling(5, min_periods=1).mean()
        ax_rt.plot(trials, rt_smoothed, color=color, lw=2, label=label)
        ax_rt.fill_between(
            trials,
            rt_smoothed - agg["rt_sem"],
            rt_smoothed + agg["rt_sem"],
            alpha=0.2, color=color,
        )

    for rev in reversal_points:
        ax_acc.axvline(rev, color="gray", ls="--", lw=0.8)
        ax_rt.axvline(rev, color="gray", ls="--", lw=0.8)

    ax_acc.set_xlabel("Trial")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy (rolling mean ± SEM)")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(frameon=False)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    style_ticks(ax_acc)

    ax_rt.set_xlabel("Trial")
    ax_rt.set_ylabel("Response time (s)")
    ax_rt.set_title("Response time (rolling mean ± SEM)")
    ax_rt.legend(frameon=False)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)
    style_ticks(ax_rt)

    fig.suptitle(title, fontweight="bold",
                fontsize=get_dynamic_fontsize(fig_width=fig.get_size_inches()[0],
                                              base_font=14))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_post_reversal(group_results: list,
                       reversal_points: list,
                       labels: list,
                       window: int = 15) -> plt.Figure:
    """Post-reversal accuracy curves aligned by reversal onset.

    Each line shows mean accuracy in the first `window` trials after a reversal,
    averaged across subjects and across reversal points.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dynamic_fontsize(fig)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for i, (label, subjects) in enumerate(zip(labels, group_results)):
        all_curves = []
        for sim in subjects:
            data = sim["data"]
            for rev in reversal_points:
                start = rev
                end = min(rev + window, len(data))
                if start < len(data):
                    curve = data["is_correct"].iloc[start:end].to_numpy()
                    # Pad if needed
                    if len(curve) < window:
                        curve = np.pad(curve, (0, window - len(curve)),
                                       constant_values=np.nan)
                    all_curves.append(curve)

        if all_curves:
            mean_curve = np.nanmean(np.array(all_curves), axis=0)
            trials_since = np.arange(1, window + 1)
            ax.plot(trials_since, mean_curve,
                    color=colors[i % len(colors)], lw=2, label=label)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Trials since reversal")
    ax.set_ylabel("Accuracy")
    ax.set_title("Post-reversal recovery")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_ticks(ax)
    fig.tight_layout()
    return fig


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    reversal_points = (36, 56, 71, 86, 106)

    # Define two groups differing in learning rate
    group_a_pars = {"alpha": 0.30, "v_max": 2.0, "beta": 1.0,
                    "a": 3.0, "w": 0.5, "t0": 0.25}
    group_b_pars = {"alpha": 0.10, "v_max": 2.0, "beta": 1.0,
                    "a": 3.0, "w": 0.5, "t0": 0.25}

    print("Simulating Group A (alpha=0.30)...")
    group_a = simulate_group(group_a_pars, n_subjects=30, seed=42)
    print("Simulating Group B (alpha=0.10)...")
    group_b = simulate_group(group_b_pars, n_subjects=30, seed=100)

    agg_a = aggregate_group(group_a)
    agg_b = aggregate_group(group_b)

    print(f"\nGroup A: overall accuracy = {agg_a['overall_accuracy']:.3f}")
    print(f"Group B: overall accuracy = {agg_b['overall_accuracy']:.3f}")

    # Behavioural metrics for group A
    ws_ls_a = [compute_win_stay_lose_shift(s["data"]) for s in group_a]
    ws_ls_b = [compute_win_stay_lose_shift(s["data"]) for s in group_b]
    print(f"\nGroup A: win-stay = {np.mean([x['win_stay'] for x in ws_ls_a]):.3f}, "
          f"lose-shift = {np.mean([x['lose_shift'] for x in ws_ls_a]):.3f}")
    print(f"Group B: win-stay = {np.mean([x['win_stay'] for x in ws_ls_b]):.3f}, "
          f"lose-shift = {np.mean([x['lose_shift'] for x in ws_ls_b]):.3f}")

    # Perseveration
    pers_a = [compute_perseveration(s["data"], list(reversal_points)) for s in group_a]
    pers_b = [compute_perseveration(s["data"], list(reversal_points)) for s in group_b]
    # Average across subjects and reversal points
    pers_a_mean = np.mean([np.mean(list(p.values())) for p in pers_a])
    pers_b_mean = np.mean([np.mean(list(p.values())) for p in pers_b])
    print(f"\nGroup A: perseveration rate = {pers_a_mean:.3f}")
    print(f"Group B: perseveration rate = {pers_b_mean:.3f}")

    # Plots
    fig1 = plot_group_comparison(
        {"Fast learners (α=0.30)": agg_a, "Slow learners (α=0.10)": agg_b},
        list(reversal_points),
        ["Fast learners (α=0.30)", "Slow learners (α=0.10)"],
        title="Effect of learning rate on PRLT behaviour",
    )

    fig2 = plot_post_reversal(
        [group_a, group_b],
        list(reversal_points),
        ["Fast learners (α=0.30)", "Slow learners (α=0.10)"],
    )

    plt.show()