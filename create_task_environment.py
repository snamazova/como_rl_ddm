import argparse
import random

import numpy as np
import matplotlib.pyplot as plt


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
        A list of dicts, one per trial, with "bandit_1"/"bandit_2" reward values.
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
                "bandit_2": {"color": "blue", "value": bandit_2_reward}
            }
        bandit_1_correct = not bandit_1_correct  # reversal flips which bandit is correct

    return timeline


def plot_reward_distribution(timeline, reversal_points, num_trials, p_correct,
                              reversed_state, save_path=None, show=True):
    bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
    bandit_2_rewards = [trial["bandit_2"]["value"] for trial in timeline]

    fig, (ax_trials, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    trials = range(1, num_trials + 1)
    ax_trials.plot(trials, bandit_1_rewards, label="Bandit 1 (Orange)", color="orange",
                    marker="o", linestyle="None", alpha=0.7)
    ax_trials.plot(trials, bandit_2_rewards, label="Bandit 2 (Blue)", color="blue",
                    marker="o", linestyle="None", alpha=0.7)
    for rev in reversal_points:
        ax_trials.axvline(rev, color="gray", linestyle="--", linewidth=0.8)
    ax_trials.set_title("Reward Outcomes Over Trials")
    ax_trials.set_xlabel("Trial Number")
    ax_trials.set_ylabel("Reward Outcome (0 or 1)")
    ax_trials.set_xticks(range(0, num_trials + 1, max(num_trials // 14, 1)))
    ax_trials.set_yticks([0, 1])
    ax_trials.legend()

    # Sanity-check histogram: realized reward probability for the
    # currently-correct bandit within each between-reversal block, compared
    # against the requested p_correct.
    boundaries = [0] + list(reversal_points) + [num_trials]
    block_labels, realized_probs = [], []
    bandit_1_correct = reversed_state
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = timeline[start:end]
        correct_key = "bandit_1" if bandit_1_correct else "bandit_2"
        wins = sum(t[correct_key]["value"] for t in block)
        realized_probs.append(wins / len(block))
        block_labels.append(f"{start}-{end}")
        bandit_1_correct = not bandit_1_correct

    ax_hist.bar(block_labels, realized_probs, color="seagreen", alpha=0.8)
    ax_hist.axhline(p_correct, color="black", linestyle="--", linewidth=1,
                     label=f"target p_correct = {p_correct}")
    ax_hist.set_title("Realized Reward Probability per Block")
    ax_hist.set_xlabel("Block (trial range)")
    ax_hist.set_ylabel("P(correct bandit rewarded)")
    ax_hist.set_ylim(0, 1)
    ax_hist.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
