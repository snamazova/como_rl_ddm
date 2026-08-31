"""Generate all presentation plots for the PRLT RLDDM project.

Produces:
    1. reversal_locked.png       — accuracy & RT time-locked to reversal (H1 vs H2)
    2. single_subject_h1.png     — single-subject diagnostic (H1 model)
    3. single_subject_h2.png     — single-subject diagnostic (H2 model)
    4. parameter_sweep_alpha.png — effect of alpha on behaviour
    5. parameter_sweep_a.png     — effect of a on behaviour
    6. confusion_matrix.png      — 2x2 model comparison (BIC)
    7. parameter_recovery_h1.png — true-vs-recovered scatter, H1
    8. parameter_recovery_h2.png — true-vs-recovered scatter, H2

Usage:
    python generate_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from create_task_environment import (
    generate_timeline, timeline_to_matrix, timeline_to_correct,
)
from rlddm import rlddm_simulate, rlddm_plot, H1_PARAMS, H2_PARAMS
from reversal_locked_plots import plot_reversal_locked, aggregate_reversal_locked
from confusion_matrix import run_confusion_matrix, print_confusion_matrix
from parameter_recovery import run_recovery_batch, plot_parameter_recovery
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


RP = (36, 56, 71, 86, 106)
N_TRIALS = 140
N_SUBJECTS = 15

H1_PARS = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a": 3.0, "w": 0.5, "t0": 0.25}
H2_PARS = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a_base": 3.0,
           "kappa": 2.0, "tau": 5.0, "w": 0.5, "t0": 0.25}


def make_timeline():
    t = generate_timeline(num_trials=N_TRIALS, seed=42, reversed_state=True,
                          reversal_points=RP)
    return timeline_to_matrix(t), timeline_to_correct(t)


def simulate_group(pars, n_subjects, seed_offset=0):
    env, correct = make_timeline()
    subjects = []
    for s in range(n_subjects):
        sim = rlddm_simulate(env, pars, rng=np.random.default_rng(seed_offset + s),
                             correct_bandit=correct, reversal_points=RP)
        subjects.append(sim)
    return subjects


# =============================================================================
# Plot 1: Reversal-locked accuracy & RT (H1 vs H2)
# =============================================================================
print("1/6: Reversal-locked plots...")
h1_subjects = simulate_group(H1_PARS, N_SUBJECTS, seed_offset=42)
h2_subjects = simulate_group(H2_PARS, N_SUBJECTS, seed_offset=142)

fig1 = plot_reversal_locked(
    {"H1 (drift-only)": h1_subjects, "H2 (boundary varies)": h2_subjects},
    list(RP), window=20,
)
save_panel(fig1, "reversal_locked.png", figsize=fig1.get_size_inches(), dpi=150)
print("   -> reversal_locked.png")

# Print summary
for label, subs in [("H1", h1_subjects), ("H2", h2_subjects)]:
    agg = aggregate_reversal_locked(subs, list(RP), window=20)
    idx = 20
    pre_rt = np.mean(agg["rt_mean"][idx-5:idx])
    post_rt = np.mean(agg["rt_mean"][idx:idx+5])
    pre_acc = np.mean(agg["acc_mean"][idx-5:idx])
    post_acc = np.mean(agg["acc_mean"][idx:idx+5])
    print(f"   {label}: pre-rev RT={pre_rt:.3f}s, post-rev RT={post_rt:.3f}s, "
          f"pre-rev acc={pre_acc:.3f}, post-rev acc={post_acc:.3f}")


# =============================================================================
# Plot 2 & 3: Single-subject diagnostics
# =============================================================================
print("\n2/6: Single-subject H1 diagnostic...")
fit_h1 = h1_subjects[0]
fig_h1_1, fig_h1_2 = rlddm_plot(fit_h1)
save_panel(fig_h1_1, "single_subject_h1_rl.png", figsize=fig_h1_1.get_size_inches(), dpi=150)
save_panel(fig_h1_2, "single_subject_h1_ddm.png", figsize=fig_h1_2.get_size_inches(), dpi=150)
print("   -> single_subject_h1_rl.png, single_subject_h1_ddm.png")

print("\n3/6: Single-subject H2 diagnostic...")
fit_h2 = h2_subjects[0]
fig_h2_1, fig_h2_2 = rlddm_plot(fit_h2)
save_panel(fig_h2_1, "single_subject_h2_rl.png", figsize=fig_h2_1.get_size_inches(), dpi=150)
save_panel(fig_h2_2, "single_subject_h2_ddm.png", figsize=fig_h2_2.get_size_inches(), dpi=150)
print("   -> single_subject_h2_rl.png, single_subject_h2_ddm.png")

# Also plot the boundary over trials for H2
fig_bound, ax_bound = plt.subplots(figsize=(10, 4))
_apply_dynamic_fontsize(fig_bound)
ax_bound.plot(range(1, N_TRIALS+1), fit_h2["boundaries"], color="#D55E00", lw=2)
for rev in RP:
    ax_bound.axvline(rev, color="gray", ls="--", lw=0.8)
ax_bound.set_xlabel("Trial")
ax_bound.set_ylabel("Boundary separation a(t)")
ax_bound.set_title("H2: Time-varying boundary (a_base + κ·exp(-trial_since_reversal/τ))")
ax_bound.spines["top"].set_visible(False)
ax_bound.spines["right"].set_visible(False)
style_ticks(ax_bound)
fig_bound.tight_layout()
save_panel(fig_bound, "h2_boundary_over_trials.png", figsize=fig_bound.get_size_inches(), dpi=150)
print("   -> h2_boundary_over_trials.png")


# =============================================================================
# Plot 4: Parameter sweep (alpha)
# =============================================================================
print("\n4/6: Parameter sweep (alpha)...")
from parameter_sweep import run_sweep, plot_sweep
# Update base pars to use tanh parameterization
BASE = {"alpha": 0.25, "v_max": 2.0, "beta": 1.0, "a": 3.0, "w": 0.5, "t0": 0.25}
df_alpha = run_sweep("alpha", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                     n_subjects=10, base_pars=BASE)
fig_sweep = plot_sweep(df_alpha, "alpha")
save_panel(fig_sweep, "parameter_sweep_alpha.png", figsize=fig_sweep.get_size_inches(), dpi=150)
print("   -> parameter_sweep_alpha.png")


# =============================================================================
# Plot 5: Confusion matrix
# =============================================================================
print("\n5/6: Confusion matrix (this is slow)...")
from confusion_matrix import fit_model

# Use shared timeline
env, correct = make_timeline()

# Simulate small datasets for the confusion matrix
print("   Simulating H1 data (5 subjects)...")
h1_data = [rlddm_simulate(env, H1_PARS, rng=np.random.default_rng(200+s),
                           correct_bandit=correct, reversal_points=RP)["data"]
            for s in range(5)]

print("   Simulating H2 data (5 subjects)...")
h2_data = [rlddm_simulate(env, H2_PARS, rng=np.random.default_rng(300+s),
                           correct_bandit=correct, reversal_points=RP)["data"]
            for s in range(5)]

# Fit both models to both datasets
bic_h1_h1 = 0
bic_h2_h1 = 0
bic_h1_h2 = 0
bic_h2_h2 = 0

for s, data in enumerate(h1_data):
    print(f"   Fitting H1 data subject {s}...")
    f1 = fit_model("H1", data, RP, n_restarts=5, rng=np.random.default_rng(400+s))
    f2 = fit_model("H2", data, RP, n_restarts=5, rng=np.random.default_rng(500+s))
    bic_h1_h1 += f1["bic"]
    bic_h2_h1 += f2["bic"]

for s, data in enumerate(h2_data):
    print(f"   Fitting H2 data subject {s}...")
    f1 = fit_model("H1", data, RP, n_restarts=5, rng=np.random.default_rng(600+s))
    f2 = fit_model("H2", data, RP, n_restarts=5, rng=np.random.default_rng(700+s))
    bic_h1_h2 += f1["bic"]
    bic_h2_h2 += f2["bic"]

# Print and plot confusion matrix
print(f"\n{'='*55}")
print("Confusion Matrix (BIC, summed across subjects)")
print(f"{'='*55}")
print(f"{'':>12} {'Fit H1':>15} {'Fit H2':>15} {'Winner':>10}")
print("-" * 55)
print(f"{'H1_data':>12} {bic_h1_h1:>15.1f} {bic_h2_h1:>15.1f} {'H1' if bic_h1_h1 < bic_h2_h1 else 'H2':>10}")
print(f"{'H2_data':>12} {bic_h1_h2:>15.1f} {bic_h2_h2:>15.1f} {'H1' if bic_h1_h2 < bic_h2_h2 else 'H2':>10}")
print("-" * 55)

# Plot as heatmap
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
_apply_dynamic_fontsize(fig_cm)
bic_matrix = np.array([[bic_h1_h1, bic_h2_h1],
                        [bic_h1_h2, bic_h2_h2]])
# Normalize for visualization (lower BIC = better = darker)
im = ax_cm.imshow(bic_matrix, cmap="RdYlGn_r", aspect="auto")
ax_cm.set_xticks([0, 1])
ax_cm.set_xticklabels(["Fit H1", "Fit H2"])
ax_cm.set_yticks([0, 1])
ax_cm.set_yticklabels(["H1 data", "H2 data"])
ax_cm.set_title("Confusion Matrix (BIC)")
style_ticks(ax_cm)
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{bic_matrix[i,j]:.0f}", ha="center", va="center",
                   fontsize=get_dynamic_fontsize(fig_width=fig_cm.get_size_inches()[0], base_font=12),
                   fontweight="bold")
fig_cm.colorbar(im, label="BIC (lower = better)")
fig_cm.tight_layout()
save_panel(fig_cm, "confusion_matrix.png", figsize=fig_cm.get_size_inches(), dpi=150)
print("   -> confusion_matrix.png")


# =============================================================================
# Plot 6: Parameter recovery (true vs. recovered, H1 and H2)
# =============================================================================
print("\n6/6: Parameter recovery (this is slow)...")

print("   H1 batch recovery (12 draws)...")
df_recovery_h1 = run_recovery_batch("H1", n_iterations=12, n_restarts=5, seed=42)
fig_rec_h1 = plot_parameter_recovery(df_recovery_h1, H1_PARAMS, "H1 (drift-only)")
save_panel(fig_rec_h1, "parameter_recovery_h1.png", figsize=fig_rec_h1.get_size_inches(), dpi=150)
print("   -> parameter_recovery_h1.png")

print("   H2 batch recovery (12 draws)...")
df_recovery_h2 = run_recovery_batch("H2", n_iterations=12, n_restarts=5, seed=1042)
fig_rec_h2 = plot_parameter_recovery(df_recovery_h2, H2_PARAMS, "H2 (boundary varies)")
save_panel(fig_rec_h2, "parameter_recovery_h2.png", figsize=fig_rec_h2.get_size_inches(), dpi=150)
print("   -> parameter_recovery_h2.png")
# For a higher-quality version, run e.g.
#   python parameter_recovery.py --n-iterations 30 --n-restarts 8

print("\nDone! All plots saved.")
print("\nGenerated files:")
import os
for f in sorted(os.listdir(".")):
    if f.endswith(".png"):
        print(f"  {f} ({os.path.getsize(f) // 1024} KB)")