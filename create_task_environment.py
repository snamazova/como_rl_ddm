import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import get_dynamic_fontsize, save_panel, style_ticks

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a probabilistic reversal-learning task (PRLT) "
                     "timeline and plot the resulting reward distribution."
    )
    parser.add_argument(
        "--num-trials", type=int, default=140,
        help="Total number of trials in the timeline. Default: 140.",
    )
    parser.add_argument(
        "--reversal-points", type=int, nargs="+", default=[36, 56, 71, 86, 106],
        help="Trial indices (0-indexed) at which the correct bandit flips. "
             "Space-separated, e.g. --reversal-points 30 60 90. "
             "Default: 36 56 71 86 106.",
    )
    parser.add_argument(
        "--p-correct", type=float, default=0.8,
        help="Reward probability for the currently-correct bandit within each "
             "between-reversal block. Default: 0.8.",
    )
    parser.add_argument(
        "--constant-timeline", action="store_true",
        help="If set, the timeline is generated with a fixed seed (and fixed "
             "reversed_state) so it is identical every time the script is run "
             "-- e.g. to give every participant the same trial sequence. If "
             "not set, a new random timeline is drawn each run.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed used only when --constant-timeline is set. Default: 0.",
    )
    parser.add_argument(
        "--reversed-state", type=lambda x: x.lower() in ("1", "true", "yes"),
        default=None,
        help="Whether bandit_1 starts as the correct bandit (true/false). "
             "Only used together with --constant-timeline; if omitted it is "
             "chosen randomly (or fixed to True when --constant-timeline is set "
             "without this flag).",
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="If given, save the plot to this path instead of (or in addition "
             "to) showing it interactively.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't open an interactive plot window (useful for headless runs).",
    )
    return parser.parse_args()


def generate_timeline(num_trials=140, seed=None, reversed_state=None,
                       reversal_points=(36, 56, 71, 86, 106), p_correct=0.8):
    """Generates a timeline of trials for the slot machine task, matching the
    real PRLT design: fixed reversal points, and within each between-reversal
    block an exact quota of `p_correct` fraction of trials go to the
    currently-correct bandit (not an independent per-trial Bernoulli draw,
    which lets the realized ratio drift from 0.8/0.2 within a block).
    Outcomes are zero-sum per trial: exactly one bandit gets reward=1.

    Args:
        num_trials: The number of trials to generate.
        seed: RNG seed for the within-block shuffle order. Leave as None so
            each call (each simulated participant) gets an independent draw;
            pass an int only if you want that participant's draw reproducible.
        reversed_state: True if bandit_1 starts as the correct bandit, False
            if bandit_2 does. Pass this in explicitly (e.g. from a
            counterbalanced list in `main`) rather than leaving it random per
            call, so the split across participants is controlled.
        reversal_points: Trial indices (0-indexed) at which the correct
            bandit flips.
        p_correct: Reward probability for the currently-correct bandit.

    Returns:
        A list of dicts, one per trial, with "bandit_1"/"bandit_2" reward values
        and a "correct" field indicating which bandit (1 or 2) is the
        currently-correct one for that trial.
    """
    if reversed_state is None:
        reversed_state = random.choice([True, False])

    rng = np.random.default_rng(seed)
    boundaries = [0] + list(reversal_points) + [num_trials]

    timeline = [None] * num_trials
    bandit_1_correct = reversed_state

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block_size = end - start
        n_correct_wins = round(block_size * p_correct)
        outcomes = np.array([1] * n_correct_wins + [0] * (block_size - n_correct_wins))
        rng.shuffle(outcomes)

        for offset, correct_bandit_wins in enumerate(outcomes):
            if bandit_1_correct:
                bandit_1_reward, bandit_2_reward = int(correct_bandit_wins), int(1 - correct_bandit_wins)
            else:
                bandit_2_reward, bandit_1_reward = int(correct_bandit_wins), int(1 - correct_bandit_wins)
            timeline[start + offset] = {
                "bandit_1": {"color": "orange", "value": bandit_1_reward},
                "bandit_2": {"color": "blue", "value": bandit_2_reward},
                "correct": 1 if bandit_1_correct else 2,
            }
        bandit_1_correct = not bandit_1_correct  # reversal flips which bandit is correct

    return timeline


def _bandit_1_prob_curve(num_trials, reversal_points, reversed_state, p_correct,
                          ramp_width=2):
    """(x, y) control points tracing bandit_1's reward probability: flat at
    p_correct/1-p_correct within each block, with a `ramp_width`-trial linear
    ramp centered on each reversal point (rather than an instantaneous jump)
    so the transition reads as a visible diagonal instead of a near-vertical
    step. Mirrors the block structure generate_timeline() uses to actually
    sample rewards, so the schedule shown here always matches the
    trial-level correct/incorrect labels."""
    boundaries = [0] + list(reversal_points) + [num_trials]
    bandit_1_correct = reversed_state
    levels = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        levels.append(p_correct if bandit_1_correct else 1 - p_correct)
        bandit_1_correct = not bandit_1_correct

    half = ramp_width / 2
    xs, ys = [1], [levels[0]]
    for i, rev in enumerate(reversal_points):
        xs += [rev - half, rev + half]
        ys += [levels[i], levels[i + 1]]
    xs.append(num_trials)
    ys.append(levels[-1])
    return np.array(xs), np.array(ys)


def plot_reward_distribution(timeline, reversal_points, num_trials, p_correct,
                              reversed_state, save_path=None, show=True):
    """Plot the sampled reward outcomes (top) and the reward probability
    schedule (bottom) over trials, with reversal points marked.
    """
    bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
    bandit_2_rewards = [trial["bandit_2"]["value"] for trial in timeline]
    figsize = (6, 3)  # inches
    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [0.3, 1]},
    )
    fontsize = get_dynamic_fontsize(fig_width=fig.get_size_inches()[0])
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })
    trials = range(1, num_trials + 1)

    bandit_1_reward_trials = [t for t, r in zip(trials, bandit_1_rewards) if r == 1]
    bandit_2_reward_trials = [t for t, r in zip(trials, bandit_2_rewards) if r == 1]
    ax_top.eventplot([bandit_1_reward_trials], lineoffsets=[1], linelengths=0.8,
                      colors="orange", label="Bandit 1")
    ax_top.eventplot([bandit_2_reward_trials], lineoffsets=[0], linelengths=0.8,
                      colors="blue", label="Bandit 2")
    for spine in ["top", "right","bottom"]:
            ax_top.spines[spine].set_visible(False)
    for rev in reversal_points:
        ax_top.axvline(rev, color="gray", linestyle="--", linewidth=0.8)

    #ax_top.set_title("Ground Truth Reward Outcomes")
    ax_top.set_ylim(-0.5, 1.5)
    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(["Bandit 2", "Bandit 1"])
    style_ticks(ax_top)

    prob_x, p_bandit_1 = _bandit_1_prob_curve(num_trials, reversal_points, reversed_state, p_correct)
    p_bandit_2 = 1 - p_bandit_1
    ax.plot(prob_x, p_bandit_1, color="orange", lw=2.5, solid_capstyle="butt",
            label="Bandit 1")
    ax.plot(prob_x, p_bandit_2, color="blue", lw=2.5, solid_capstyle="butt",
            label="Bandit 2")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for rev in reversal_points:
        ax.axvline(rev, color="gray", linestyle="--", linewidth=0.8)
        #ax.text(rev - figsize[0] * 0.1, figsize[1] * 0.01, 'reversal', rotation=90, va='bottom', ha='center', alpha=0.45)
    ax.set_ylim(0, 1)
    ax.set_yticks([p_correct, 1 - p_correct])
    ax.set_ylabel("Reward Probability")
    
    #place title in the bottom of the plot
    #ax.set_title("Reward Probability Schedule", y=-0.5)
    ax.set_xlabel("Trial")
    ax.set_xticks(range(0, num_trials + 1, 20))
    style_ticks(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.05), frameon=False)

    fig.tight_layout()

    if save_path:
        save_panel(fig, save_path, figsize=fig.get_size_inches())
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def timeline_to_matrix(timeline):
    """Convert a PRLT timeline into an (n_trials, 2) NumPy outcome matrix.

    The first column is bandit_1 (Option 1 / orange), the second column is
    bandit_2 (Option 2 / blue).  This matrix can be passed directly to the
    RLDDM simulator as its `task_environment` argument.
    """
    return np.array([
        [trial["bandit_1"]["value"], trial["bandit_2"]["value"]]
        for trial in timeline
    ])


def timeline_to_correct(timeline):
    """Extract the correct-bandit label (1 or 2) for each trial.

    Returns an integer array of shape (n_trials,).  This is metadata for
    behavioural analysis — the model never sees it; it only learns from rewards.
    """
    return np.array([trial["correct"] for trial in timeline])


def main():
    args = parse_args()

    seed = args.seed if args.constant_timeline else None
    reversed_state = args.reversed_state
    if reversed_state is None:
        reversed_state = True if args.constant_timeline else random.choice([True, False])

    timeline = generate_timeline(
        num_trials=args.num_trials,
        seed=seed,
        reversed_state=reversed_state,
        reversal_points=args.reversal_points,
        p_correct=args.p_correct,
    )

    plot_reward_distribution(
        timeline=timeline,
        reversal_points=args.reversal_points,
        num_trials=args.num_trials,
        p_correct=args.p_correct,
        reversed_state=reversed_state,
        save_path=args.save_plot,
        show=not args.no_show,
    )

    return timeline


if __name__ == "__main__":
    main()
