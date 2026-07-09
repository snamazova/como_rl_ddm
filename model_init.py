# %%
import numpy as np
import matplotlib.pyplot as plt
import hssm

# %% HELPER FUNCTIONS


# %%
def fill_rl_pars(pars={}, defaults=True):
    default_values = {}
    if defaults:
        default_values = {
            "alpha": 0.25,
            "v_intercept": 0,
            "v_scale": 1,
            "a": 3,
            "w": 0.5,
            "t0": 0.25,
            "sv": 0,
            "sw": 0,
            "st0": 0}

    out = {**default_values, **pars}  # Combine dictionaries
    return out


# %%
fill_rl_pars()


# %%
def value_update(value, outcome, alpha):
    prediction_error = outcome - value
    value += alpha * prediction_error
    # if(not alpha in range(0, 1)):
    #     raise ValueError("Alpha in range")
    return value


# %%
#value_update(10, 15, 0.25)


# %%
def softmax(values, beta):
    numerator = np.exp(beta * (values - max(values)))
    denominator = sum(numerator)
    probabilities = numerator / denominator
    return probabilities


# %%
#softmax(np.array([10, 20, 30, 40]), 0.8)


# %%
def simulate_choice(probabilities):
    n_options = len(probabilities)
    choice = np.random.choice(n_options, p=probabilities)
    return choice


# %%
#simulate_choice(probabilities=[0.3, 0.4, 0.3])


# %%
def simulate_task_environment(n_trials, means, sds, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)
    n_options = len(means)
    outcomes = np.full(shape=(n_trials, n_options), fill_value=np.nan)
    for i, (mean, sd) in enumerate(zip(means, sds)):
        outcomes[:, i] = np.random.normal(mean, sd, n_trials)
    return outcomes


# %%
#simulate_task_environment(n_trials=10, means=[10, 20], sds=[1, 3])
# %%

def ddm_sample_trial(pars):

    sim = hssm.simulate_data(
        model="ddm",
        theta=[
            pars["v"],
            pars["a"],
            pars["w"],
            pars["t0"],
        ],
        size=1,
    )

    choice = 1 if sim["response"].iloc[0] == -1 else 2
    rt = sim["rt"].iloc[0]

    return {
        "choice": choice,
        "rt": rt
    }
#DDM related helper functions
def compute_drift(values, pars):
    value_diff = values[1] - values[0]
    drift = pars["v_intercept"] + pars["v_scale"] * value_diff
    return drift


# %% CORE MODEL FUNCTIONS
def rl_simulate(task_environment, pars, initial_values=(0.0, 0.0)):
    pars = fill_rl_pars(pars=pars)

    # initialise objects
    shape = np.shape(task_environment)
    n_trials = shape[0]
    n_options = shape[1]

    values = np.full(shape=(n_trials, n_options), fill_value=np.nan)
    choice_probs = np.full((n_trials, n_options), np.nan, dtype=float)

    choices = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials, dtype=float)
    prediction_errors = np.zeros(n_trials, dtype=float)

    values[0, :] = np.asarray(initial_values, dtype=float)

    pars = fill_rl_pars(pars)
    task_environment = np.asarray(task_environment, dtype=float)

    values[0, :] = np.asarray(initial_values, dtype=float)

    for t in range(n_trials):
        # choice
       # Compute the drift rate from the current value estimates
        drifts[t] = compute_drift(values[t], pars)

        # Create a copy of the parameters and add the current drift rate
        pars_t = pars.copy()
        pars_t["v"] = drifts[t]

        # Simulate one DDM trial
        sim = ddm_sample_trial(pars_t)

        # Store the simulated choice and response time
        choices[t] = sim["choice"]
        RTs[t] = sim["rt"]

        # outcomes
        outcomes[t] = task_environment[t, choices[t]]

        # learning
        prediction_errors[t] = outcomes[t] - values[t, choices[t]]
        if t < n_trials - 1:
            # for all options, pass old value forwards
            values[t + 1,] = values[t,]
            # specifically for the chosen option, update the value
            values[t + 1, choices[t]] = value_update(
                value=values[t, choices[t]],
                outcome=outcomes[t],
                alpha=pars["alpha"],
            )
    data = {
        "choices": choices,
        "outcomes": outcomes,
    }
    parameters = dict(pars)
    parameters["initial_values"] = np.asarray(initial_values, dtype=float)
    out = {
        "data": data,
        "values": values,
        "choice_probs": choice_probs,
        "prediction_errors": prediction_errors,
        "parameters": parameters,
        "task_environment": task_environment,
    }
    return out


# %%
def _pretty_ticks(n, max_ticks=10):
    if n <= 1:
        return [1]
    idx = np.linspace(0, n - 1, max_ticks)
    ticks = np.unique(np.clip(np.round(idx) + 1, 1, n).astype(int))
    return ticks.tolist()

# %% Plotting functions (LLM generated)
def plot_values(fit, colours, show_legend=False, option_names=None, ax=None):
    values_mat = np.asarray(fit["values"], dtype=float)
    n_trials = values_mat.shape[0]

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(1, n_trials + 1)
    for i in range(values_mat.shape[1]):
        ax.plot(x, values_mat[:, i], color=colours[i], lw=2)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Estimated value")
    ax.set_title("Learned values")
    ax.set_xlim(1, n_trials)
    ax.set_xticks(_pretty_ticks(n_trials))
    ax.legend(option_names, loc="lower right", frameon=False) if show_legend else None
    return ax


def plot_choice_probs(fit, colours, ax=None):
    cp = np.asarray(fit["choice_probs"], dtype=float)
    n_trials = cp.shape[0]

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(1, n_trials + 1)
    for i in range(cp.shape[1]):
        ax.plot(x, cp[:, i], color=colours[i], lw=2)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Probability")
    ax.set_title("Choice probabilities")
    ax.set_xlim(1, n_trials)
    ax.set_xticks(_pretty_ticks(n_trials))

    for h in (0.25, 0.5, 0.75):
        ax.axhline(h, ls="--", lw=1.5, color="gray")
    return ax


def plot_pred_errors(fit, colours, ax=None):
    pe = np.asarray(fit["prediction_errors"], dtype=float)
    n_trials = pe.shape[0]

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(1, n_trials + 1)
    y = pe
    ylim = [float(np.min(y)), float(np.max(y))]
    if ylim[0] == ylim[1]:
        ylim = [ylim[0] - 1, ylim[1] + 1]

    ax.set_xlim(1, n_trials)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Prediction error")
    ax.set_title("Prediction errors")
    ax.set_xticks(_pretty_ticks(n_trials))

    # segments from 0 to prediction error, colored by choice
    choices = np.asarray(fit["data"]["choices"], dtype=int)
    for t in range(n_trials):
        c = choices[t]
        ax.vlines(x[t], 0, y[t], color=colours[c], lw=1)

    ax.scatter(x, y, s=30, marker="o", color=[colours[c] for c in choices], zorder=3)
    ax.axhline(0, ls="--", lw=1.5, color="gray")
    return ax


def plot_outcomes(
    fit,
    colours,
    shapes,
    task_environment=None,
    show_legend=False,
    option_names=None,
    ax=None,
):
    outcomes = np.asarray(fit["data"]["outcomes"], dtype=float)
    n_trials = np.asarray(fit["values"]).shape[0]
    n_options = np.asarray(fit["values"]).shape[1]

    if task_environment is None and "task_environment" in fit:
        task_environment = fit["task_environment"]

    ylim = [float(np.min(outcomes)), float(np.max(outcomes))]
    if task_environment is not None:
        te = np.asarray(task_environment, dtype=float)
        ylim = [min(ylim[0], float(np.min(te))), max(ylim[1], float(np.max(te)))]

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlim(1, n_trials)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Outcome")
    ax.set_title("Observed outcomes")
    ax.set_xticks(_pretty_ticks(n_trials))

    if task_environment is not None:
        te = np.asarray(task_environment, dtype=float)
        x = np.arange(1, n_trials + 1)
        for i in range(n_options):
            ax.plot(x, te[:, i], color="gray", lw=1)

    choices = np.asarray(fit["data"]["choices"], dtype=int)
    x = np.arange(1, n_trials + 1)

    # plot each option's points with its own marker
    for opt in range(n_options):
        mask = choices == opt
        ax.scatter(
            x[mask],
            outcomes[mask],
            s=70,
            marker=["o", "s", "^", "*"][shapes[opt]]
            if shapes[opt] in (0, 1, 2)
            else "P",
            color=colours[opt],
            edgecolor="none",
        )

    # legend (handle markers by option)
    if show_legend:
        handles = []
        labels = option_names
        for opt in range(n_options):
            m = ["o", "s", "^", "*"][shapes[opt]] if shapes[opt] in (0, 1, 2) else "P"
            h = ax.scatter([], [], s=70, marker=m, color=colours[opt])
            handles.append(h)
        ax.legend(handles, labels, loc="lower right", frameon=False)

    return ax


def rl_plot(fit, task_environment=None, show_legends=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    n_options = np.asarray(fit["values"]).shape[1]
    option_names = [f"Option {i + 1}" for i in range(n_options)]
    base_cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    colours = [base_cols[i % len(base_cols)] for i in range(n_options)]
    shapes = [0, 1, 2, 5]
    shapes = [shapes[i % len(shapes)] for i in range(n_options)]

    # learned values
    plot_values(
        fit, colours, show_legend=show_legends, option_names=option_names, ax=axs[0]
    )

    # choice probabilities
    plot_choice_probs(fit, colours, ax=axs[1])

    # prediction errors
    plot_pred_errors(fit, colours, ax=axs[2])

    # outcomes
    plot_outcomes(
        fit,
        colours,
        shapes,
        task_environment=task_environment,
        show_legend=show_legends,
        option_names=option_names,
        ax=axs[3],
    )

    alpha = float(fit["parameters"]["alpha"])
    beta = float(fit["parameters"]["beta"])
    fig.suptitle("Delta-rule reinforcement learning", y=0.98, fontweight="bold")
    fig.text(0.5, 0.94, f"alpha = {alpha:.2f}   beta = {beta:.2f}", ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    # return fig


# %%
rl_plot(rl_simulate(simulate_task_environment(100, [1.5, 1], [1, 1]), fill_rl_pars()))



# %%

