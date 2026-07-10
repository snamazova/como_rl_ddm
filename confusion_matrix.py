"""Confusion matrix model comparison: H1 (drift-only) vs H2 (boundary varies).

Simulates data under each model, fits both models to both datasets, and
produces a 2×2 confusion matrix showing which model wins (by BIC) in each
cell. A perfect confusion matrix is diagonal — each model recovers its own
data-generating process.

Usage:
    python confusion_matrix.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_task_environment import (
    generate_timeline, timeline_to_matrix, timeline_to_correct,
)
from rlddm import (
    rlddm_simulate, fit_model,
    H1_DEFAULTS, H2_DEFAULTS, REVERSAL_POINTS,
)


RP = REVERSAL_POINTS


def run_confusion_matrix(n_subjects=10, n_restarts=3, seed=42):
    """Run the full 2×2 confusion matrix experiment."""
    rng = np.random.default_rng(seed)

    timeline = generate_timeline(num_trials=140, seed=seed, reversed_state=True,
                                  reversal_points=RP)
    env = timeline_to_matrix(timeline)
    correct = timeline_to_correct(timeline)

    print(f"Simulating {n_subjects} subjects under H1...")
    h1_data = [rlddm_simulate(env, H1_DEFAULTS, rng=np.random.default_rng(seed + s),
                              correct_bandit=correct, reversal_points=RP)["data"]
               for s in range(n_subjects)]

    print(f"Simulating {n_subjects} subjects under H2...")
    h2_data = [rlddm_simulate(env, H2_DEFAULTS, rng=np.random.default_rng(seed + 100 + s),
                              correct_bandit=correct, reversal_points=RP)["data"]
               for s in range(n_subjects)]

    # Fit both models, count wins (only successful fits)
    h1_h1_wins, h1_h2_wins = 0, 0
    h2_h1_wins, h2_h2_wins = 0, 0

    for s, data in enumerate(h1_data):
        f1 = fit_model("H1", data, RP, n_restarts=n_restarts, rng=rng)
        f2 = fit_model("H2", data, RP, n_restarts=n_restarts, rng=rng)
        if f1["bic"] < 1e6 and f2["bic"] < 1e6:
            if f1["bic"] < f2["bic"]:
                h1_h1_wins += 1
            else:
                h1_h2_wins += 1
        print(f"  H1 data s{s}: H1={f1['bic']:.0f} H2={f2['bic']:.0f}")

    for s, data in enumerate(h2_data):
        f1 = fit_model("H1", data, RP, n_restarts=n_restarts, rng=rng)
        f2 = fit_model("H2", data, RP, n_restarts=n_restarts, rng=rng)
        if f1["bic"] < 1e6 and f2["bic"] < 1e6:
            if f1["bic"] < f2["bic"]:
                h2_h1_wins += 1
            else:
                h2_h2_wins += 1
        print(f"  H2 data s{s}: H1={f1['bic']:.0f} H2={f2['bic']:.0f}")

    return {
        "H1_data": {"H1_wins": h1_h1_wins, "H2_wins": h1_h2_wins},
        "H2_data": {"H1_wins": h2_h1_wins, "H2_wins": h2_h2_wins},
    }


def print_confusion_matrix(results):
    print(f"\n{'=' * 50}")
    print("Confusion Matrix (win counts)")
    print(f"{'=' * 50}")
    print(f"{'':>12} {'H1 wins':>10} {'H2 wins':>10}")
    print("-" * 35)
    for label in ["H1_data", "H2_data"]:
        r = results[label]
        print(f"{label:>12} {r['H1_wins']:>10} {r['H2_wins']:>10}")
    print("-" * 35)

    h1_correct = results["H1_data"]["H1_wins"] > results["H1_data"]["H2_wins"]
    h2_correct = results["H2_data"]["H2_wins"] > results["H2_data"]["H1_wins"]
    if h1_correct and h2_correct:
        print("Diagonal: models are identifiable ✓")
    elif h1_correct or h2_correct:
        print("Partially diagonal")
    else:
        print("Off-diagonal: models are confounded")


def plot_confusion_matrix(results, filename="confusion_matrix.png"):
    wins = np.array([
        [results["H1_data"]["H1_wins"], results["H1_data"]["H2_wins"]],
        [results["H2_data"]["H1_wins"], results["H2_data"]["H2_wins"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(wins, cmap="Greens", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fit H1 wins", "Fit H2 wins"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["H1 data", "H2 data"])
    ax.set_title("Confusion Matrix (win counts)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{wins[i, j]}", ha="center", va="center",
                    fontsize=20, fontweight="bold")
    fig.colorbar(im, label="Subjects won")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"{filename} saved")


if __name__ == "__main__":
    results = run_confusion_matrix(n_subjects=10, n_restarts=3, seed=42)
    print_confusion_matrix(results)
    plot_confusion_matrix(results)