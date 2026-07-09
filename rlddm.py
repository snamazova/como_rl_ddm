"""Python port of cognitive_modelling_course_RLDDM_full.r

This file keeps the same function names and structure as the R original so you
can read the two files side-by-side.  It implements:

  * RL helpers:          value_update, softmax, simulate_choice,
                         simulate_task_environment
  * DDM helpers:         ddm_sample_trial, ddm_logpdf_trial
  * RL-DDM helper:       compute_drift
  * Core model:          rlddm_simulate, rlddm_run, rlddm_log_lik
  * Bayesian helpers:    rlddm_priors, rlddm_log_prior, rlddm_log_posterior
  * Plotting helpers:    rlddm_plot (and the small helpers it needs)

The DDM primitives replace R's WienR::rWDM / WienR::dWDM with a pure-Python
analytic first-passage-time implementation (Navarro & Fuss, 2009).  For the
default case without inter-trial variability (sv = sw = st0 = 0) this is exact.
When inter-trial variability is requested the density is integrated numerically,
which is slower but mathematically equivalent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import integrate, stats
import matplotlib.pyplot as plt
from ssms.basic_simulators.simulator import simulator as ssms_simulator


# =============================================================================
# Default parameters
# =============================================================================

RLDDM_DEFAULTS = {
    "alpha": 0.25,
    "v_intercept": 0.0,
    "v_scale": 1.0,
    "a": 3.0,
    "w": 0.5,
    "t0": 0.25,
    "sv": 0.0,
    "sw": 0.0,
    "st0": 0.0,
}


def fill_rlddm_pars(pars: dict | None = None, defaults: dict | None = None) -> dict:
    """Merge user-supplied parameters with defaults (same logic as R version)."""
    if pars is None:
        pars = {}
    if defaults is None:
        defaults = RLDDM_DEFAULTS.copy()
    out = dict(defaults)
    out.update(pars)
    return out


# =============================================================================
# RL-related helpers
# =============================================================================

def value_update(value: float, outcome: float, alpha: float) -> float:
    """Rescorla-Wagner value update."""
    pred_error = outcome - value
    return value + alpha * pred_error


def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Softmax choice probabilities."""
    values = np.asarray(values)
    numerator = np.exp(beta * (values - np.max(values)))
    return numerator / np.sum(numerator)


def simulate_choice(probabilities: np.ndarray,
                    rng: np.random.Generator | None = None) -> int:
    """Draw a single categorical choice (1-indexed, like R)."""
    rng = rng or np.random.default_rng()
    return int(rng.choice(np.arange(1, len(probabilities) + 1), p=probabilities))


def simulate_task_environment(n_trials: int,
                              means: tuple | list,
                              sds: tuple | list,
                              rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Continuous Gaussian bandit environment (matches R simulate_task_environment)."""
    rng = rng or np.random.default_rng()
    n_options = len(means)
    outcomes = np.zeros((n_trials, n_options))
    for i in range(n_options):
        outcomes[:, i] = rng.normal(loc=means[i], scale=sds[i], size=n_trials)
    cols = [f"option_{i + 1}" for i in range(n_options)]
    return pd.DataFrame(outcomes, columns=cols)


# =============================================================================
# DDM-related helpers  (replaces WienR::rWDM / WienR::dWDM)
# =============================================================================

# ---- internal Wiener math ---------------------------------------------------

def _validate_ddm_pars(a: float, w: float, t0: float) -> None:
    if a <= 0:
        raise ValueError("a must be positive")
    if not 0 < w < 1:
        raise ValueError("w must be in (0, 1)")
    if t0 < 0:
        raise ValueError("t0 must be non-negative")


def _choice_probability(v: float, a: float, w: float, choice: int) -> float:
    """Probability of hitting lower (choice=1) or upper (choice=2) boundary."""
    if abs(v) < 1e-12:
        return 1.0 - w if choice == 1 else w

    exp_2va = np.exp(-2.0 * v * a)
    if choice == 1:
        return (np.exp(-2.0 * v * a * w) - exp_2va) / (1.0 - exp_2va)
    return (1.0 - np.exp(-2.0 * v * a * w)) / (1.0 - exp_2va)


def _pdf_small_time(t: float, v: float, a: float, w: float, choice: int,
                    k_max: int = 50) -> float:
    """Small-time FPT density (Navarro & Fuss, 2009)."""
    if t <= 0:
        return 0.0

    # upper -> lower with flipped parameters
    if choice == 2:
        v, w = -v, 1.0 - w

    pre = a * np.exp(-v * a * w - 0.5 * v * v * t) / np.sqrt(2.0 * np.pi * t ** 3)
    s = 0.0
    scale = (a * a) / t
    for k in range(-k_max, k_max + 1):
        x = w + 2.0 * k
        s += x * np.exp(-0.5 * scale * x * x)
    return float(pre * s)


def _pdf_large_time(t: float, v: float, a: float, w: float, choice: int,
                    k_max: int = 50) -> float:
    """Large-time FPT density (Navarro & Fuss, 2009)."""
    if t <= 0:
        return 0.0

    if choice == 2:
        v, w = -v, 1.0 - w

    pre = (np.pi / (a * a)) * np.exp(-v * a * w - 0.5 * v * v * t)
    s = 0.0
    decay = (np.pi * np.pi * t) / (2.0 * a * a)
    for k in range(1, k_max + 1):
        s += k * np.sin(k * np.pi * w) * np.exp(-k * k * decay)
    return float(pre * s)


def _pdf_decision_time(t: float, v: float, a: float, w: float,
                       choice: int, k_max: int = 50) -> float:
    """FPT density; switches between small- and large-time representations."""
    if t <= 0:
        return 0.0
    tau = t / (a * a)
    if tau < 0.4:
        return _pdf_small_time(t, v, a, w, choice, k_max=k_max)
    return _pdf_large_time(t, v, a, w, choice, k_max=k_max)


# ---- user-facing DDM functions ---------------------------------------------

def ddm_logpdf_trial(rt: float, choice: int, pars: dict,
                     *, eps: float = 1e-12) -> float:
    """Log-density of one observed (rt, choice) pair.

    Equivalent to WienR::dWDM(t=rt, response=choice, a=..., v=..., ...).
    pars must contain a, v, w, t0, sv, sw, st0.
    """
    if choice not in (1, 2):
        raise ValueError("choice must be 1 (lower) or 2 (upper)")

    a = pars["a"]
    v = pars["v"]
    w = pars["w"]
    t0 = pars["t0"]
    sv = pars["sv"]
    sw = pars["sw"]
    st0 = pars["st0"]

    _validate_ddm_pars(a, w, t0)
    if rt <= t0:
        return -np.inf

    dt = rt - t0

    # Fast path: no inter-trial variability.
    if sv == 0.0 and sw == 0.0 and st0 == 0.0:
        return float(np.log(max(_pdf_decision_time(dt, v, a, w, choice), eps)))

    # Slow path: marginalise over the inter-trial-variability dimensions.
    # Only the dimensions with variability are integrated (point masses are
    # used for the rest).  The drift-rate prior is Normal(v, sv); the starting
    # point and non-decision time priors are Uniform.  The Gaussian weight is
    # applied exactly once (inside the integrand), so there is no double
    # counting.
    warnings.warn(
        "Inter-trial variability (sv/sw/st0 > 0) uses numerical quadrature; "
        "this is exact in principle but slow.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Integration domains for the active dimensions only.
    ranges = []
    if sv > 0.0:
        ranges.append((v - 6.0 * sv, v + 6.0 * sv))
    if sw > 0.0:
        ranges.append((max(w - sw / 2.0, 1e-4), min(w + sw / 2.0, 1.0 - 1e-4)))
    if st0 > 0.0:
        ranges.append((max(t0, 0.0), t0 + st0))

    w_range = (max(w - sw / 2.0, 1e-4), min(w + sw / 2.0, 1.0 - 1e-4)) if sw > 0.0 else (w, w)
    t0_range = (max(t0, 0.0), t0 + st0) if st0 > 0.0 else (t0, t0)
    w_span = w_range[1] - w_range[0]
    t0_span = t0_range[1] - t0_range[0]

    def integrand(*args: float) -> float:
        # args are supplied in the order: [v_] [w_] [t0_] for the active dims.
        idx = 0
        v_ = args[idx] if sv > 0.0 else v
        if sv > 0.0:
            idx += 1
        w_ = args[idx] if sw > 0.0 else w
        if sw > 0.0:
            idx += 1
        t0_ = args[idx] if st0 > 0.0 else t0
        dt_ = rt - t0_
        if dt_ <= 0.0:
            return 0.0
        dens = _pdf_decision_time(dt_, v_, a, w_, choice)
        if sv > 0.0:
            dens *= stats.norm.pdf(v_, loc=v, scale=sv)
        if sw > 0.0:
            dens /= w_span
        if st0 > 0.0:
            dens /= t0_span
        return dens

    val, _ = integrate.nquad(integrand, ranges)

    return float(np.log(max(val, eps)))


def ddm_sample_trial(pars: dict,
                     rng: np.random.Generator | None = None) -> dict:
    """Sample one (choice, rt) pair from the 7-parameter DDM.

    Uses the ``full_ddm`` model from ``ssms`` (the simulator backend that
    HSSM wraps).  This supports all inter-trial variability parameters
    (sv, sw, st0) natively, so there is no need for an Euler-Maruyama
    fallback.

    Parameterisation note: ``ssms`` parametrises the DDM with the boundary
    parameter as the *half*-boundary (boundaries at +/-a), so the full
    boundary separation is 2*a.  WienR and the analytic density in this
    module use ``a`` as the *full* boundary separation (boundaries at 0 and a).
    We therefore pass ``a / 2`` to ssms.  The relative starting point ``w``
    maps unchanged to ssms's ``z``.  The variability parameters map as:
        sw -> sz  (starting-point variability, full range)
        st0 -> st (non-decision-time variability, full range)

    Choice encoding matches WienR: 1 = lower boundary, 2 = upper boundary.
    """
    rng = rng or np.random.default_rng()

    a = pars["a"]
    v = pars["v"]
    w = pars["w"]
    t0 = pars["t0"]
    sv = pars["sv"]
    sw = pars["sw"]
    st0 = pars["st0"]

    _validate_ddm_pars(a, w, t0)

    seed = int(rng.integers(0, 2_147_483_647))
    result = ssms_simulator(
        theta={
            "v": v,
            "a": a / 2.0,      # ssms uses half-boundary; WienR uses full
            "z": w,             # relative starting point, same convention
            "t": t0,            # non-decision time
            "sz": sw,           # starting-point variability (full range)
            "sv": sv,           # drift-rate variability (SD)
            "st": st0,          # non-decision-time variability (full range)
        },
        model="full_ddm",
        n_samples=1,
        random_state=seed,
    )
    rt = float(result["rts"][0, 0])
    ssms_resp = int(result["choices"][0, 0])
    # ssms: 1 -> upper, -1 -> lower.  Map to WienR convention.
    choice = 2 if ssms_resp > 0 else 1
    return {"choice": choice, "rt": rt}

# =============================================================================
# RL-DDM helper
# =============================================================================

def compute_drift(values: np.ndarray, pars: dict) -> float:
    """Drift rate = v_intercept + v_scale * (value_2 - value_1)."""
    value_diff = values[1] - values[0]
    return float(pars["v_intercept"] + pars["v_scale"] * value_diff)


# =============================================================================
# Core model functions
# =============================================================================

def rlddm_simulate(task_environment: pd.DataFrame | np.ndarray,
                   pars: dict,
                   initial_values: tuple | list | np.ndarray = (0.0, 0.0),
                   rng: np.random.Generator | None = None) -> dict:
    """Simulate choices and RTs from an RLDDM for a given task environment."""
    rng = rng or np.random.default_rng()
    pars = fill_rlddm_pars(pars)

    task_environment = np.asarray(task_environment)
    n_trials, n_options = task_environment.shape
    initial_values = np.asarray(initial_values, dtype=float)

    values = np.full((n_trials, n_options), np.nan)
    drifts = np.zeros(n_trials)
    choices = np.zeros(n_trials, dtype=int)
    RTs = np.zeros(n_trials)
    outcomes = np.zeros(n_trials)
    prediction_errors = np.zeros(n_trials)

    values[0, :] = initial_values

    for t in range(n_trials):
        drifts[t] = compute_drift(values[t, :], pars)
        pars_t = dict(pars)
        pars_t["v"] = drifts[t]

        sim = ddm_sample_trial(pars_t, rng=rng)
        choices[t] = sim["choice"]
        RTs[t] = sim["rt"]

        # choice is 1/2 -> Python index 0/1
        outcomes[t] = task_environment[t, choices[t] - 1]

        prediction_errors[t] = outcomes[t] - values[t, choices[t] - 1]
        if t < n_trials - 1:
            values[t + 1, :] = values[t, :]
            values[t + 1, choices[t] - 1] = value_update(
                value=values[t, choices[t] - 1],
                outcome=outcomes[t],
                alpha=pars["alpha"],
            )

    data = pd.DataFrame({
        "choices": choices,
        "RTs": RTs,
        "outcomes": outcomes,
    })

    parameters = dict(pars)
    parameters["initial_values"] = initial_values

    return {
        "data": data,
        "values": values,
        "drifts": drifts,
        "prediction_errors": prediction_errors,
        "parameters": parameters,
        "task_environment": task_environment,
    }


def rlddm_run(data: pd.DataFrame,
              pars: dict,
              initial_values: tuple | list | np.ndarray = (0.0, 0.0)) -> dict:
    """Compute trial-by-trial log-likelihoods for observed choices and RTs."""
    pars = fill_rlddm_pars(pars)

    choices = np.asarray(data["choices"], dtype=int)
    RTs = np.asarray(data["RTs"], dtype=float)
    outcomes = np.asarray(data["outcomes"], dtype=float)

    n_trials = len(data)
    n_options = len(initial_values)
    initial_values = np.asarray(initial_values, dtype=float)

    values = np.full((n_trials, n_options), np.nan)
    drifts = np.zeros(n_trials)
    prediction_errors = np.zeros(n_trials)
    log_lik = np.zeros(n_trials)

    values[0, :] = initial_values

    for t in range(n_trials):
        drifts[t] = compute_drift(values[t, :], pars)
        pars_t = dict(pars)
        pars_t["v"] = drifts[t]

        log_lik[t] = ddm_logpdf_trial(
            rt=RTs[t],
            choice=choices[t],
            pars=pars_t,
        )

        prediction_errors[t] = outcomes[t] - values[t, choices[t] - 1]
        if t < n_trials - 1:
            values[t + 1, :] = values[t, :]
            values[t + 1, choices[t] - 1] = value_update(
                value=values[t, choices[t] - 1],
                outcome=outcomes[t],
                alpha=pars["alpha"],
            )

    summed_log_lik = float(np.sum(log_lik))

    return {
        "data": data,
        "values": values,
        "drifts": drifts,
        "prediction_errors": prediction_errors,
        "log_lik": log_lik,
        "summed_log_lik": summed_log_lik,
        "parameters": dict(pars),
    }


def rlddm_log_lik(data: pd.DataFrame, pars: dict,
                  initial_values: tuple | list | np.ndarray = (0.0, 0.0)) -> float:
    """Convenience wrapper returning only the summed log-likelihood."""
    fit = rlddm_run(data, pars, initial_values=initial_values)
    return fit["summed_log_lik"]


# =============================================================================
# Priors and posterior
# =============================================================================

# Mirrors R's rlddm_priors list.  SciPy parameterisations:
#   beta(shape1, shape2)          -> stats.beta(a, b)
#   normal(mean, sd)              -> stats.norm(loc, scale)
#   gamma(shape, rate)            -> stats.gamma(a, scale=1/rate)
#   gamma(shape, scale)           -> stats.gamma(a, scale=scale)
#   beta(shape1=2, shape2=8)      -> stats.beta(2, 8)
RLDDM_PRIORS = {
    "alpha": stats.beta(5, 5),
    "v_intercept": stats.norm(0.0, 0.5),
    "v_scale": stats.gamma(2.0, scale=1.0),   # shape=2, rate=1
    "a": stats.gamma(10.0, scale=0.25),        # shape=10, scale=0.25
    "w": stats.beta(20, 20),
    "t0": stats.gamma(20.0, scale=0.015),     # shape=20, scale=0.015
    "sv": stats.gamma(2.0, scale=0.25),
    "sw": stats.beta(2, 8),
    "st0": stats.gamma(2.0, scale=0.05),
}


def rlddm_log_prior(pars: dict, priors: dict | None = None) -> float:
    """Sum of log-prior densities for the supplied parameters.

    When inter-trial variability parameters (sv, sw, st0) are fixed at 0
    (the default reduced model), they are skipped so that gamma/beta priors
    with zero density at 0 do not produce ``-inf``.
    """
    pars = fill_rlddm_pars(pars)
    priors = priors or RLDDM_PRIORS
    out = 0.0
    for name, dist in priors.items():
        # Skip variability parameters when they are fixed at 0; the gamma/beta
        # priors have zero density at 0, which would yield -inf.
        if name in ("sv", "sw", "st0") and pars[name] == 0.0:
            continue
        out += float(dist.logpdf(pars[name]))
    return out


def rlddm_log_posterior(pars: dict,
                        data: pd.DataFrame,
                        initial_values: tuple | list | np.ndarray = (0.0, 0.0)) -> float:
    """Log-posterior = log-prior + log-likelihood."""
    return rlddm_log_prior(pars) + rlddm_log_lik(data, pars, initial_values)


# =============================================================================
# Plotting helpers (matplotlib equivalents of the R plotting functions)
# =============================================================================

RLDDM_COLOURS = ["#0072B2", "#D55E00"]
RLDDM_SHAPES = ["s", "o"]


def _parameter_label(parameters: dict) -> str:
    pars = parameters
    label = [
        f"alpha = {pars['alpha']:.2f}",
        f"v_0 = {pars['v_intercept']:.2f}",
        f"v_scale = {pars['v_scale']:.2f}",
        f"a = {pars['a']:.2f}",
        f"w = {pars['w']:.2f}",
        f"t0 = {pars['t0']:.2f}",
    ]
    if pars.get("sv", 0.0) > 0.0:
        label.append(f"sv = {pars['sv']:.2f}")
    if pars.get("sw", 0.0) > 0.0:
        label.append(f"sw = {pars['sw']:.2f}")
    if pars.get("st0", 0.0) > 0.0:
        label.append(f"st0 = {pars['st0']:.2f}")
    return "    ".join(label)


def _plot_values(ax, fit, show_legend: bool = False,
                 option_names: list | None = None):
    values_mat = fit["values"]
    n_trials = values_mat.shape[0]
    for i in range(values_mat.shape[1]):
        ax.plot(range(1, n_trials + 1), values_mat[:, i],
                color=RLDDM_COLOURS[i], lw=2, label=option_names[i] if option_names else None)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Estimated value")
    ax.set_title("Learned values")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend and option_names:
        ax.legend(frameon=False)


def _plot_drifts(ax, fit):
    n_trials = len(fit["drifts"])
    ax.plot(range(1, n_trials + 1), fit["drifts"], color="black", lw=2)
    ax.axhline(fit["parameters"]["v_intercept"], color="grey", ls="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Drift rate")
    ax.set_title("Drift scaled by value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_pred_errors(ax, fit):
    x = np.arange(1, len(fit["prediction_errors"]) + 1)
    y = fit["prediction_errors"]
    choices = fit["data"]["choices"].to_numpy()
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [0, y[i]], color=RLDDM_COLOURS[choices[i] - 1], lw=1)
    ax.scatter(x, y, c=[RLDDM_COLOURS[c - 1] for c in choices], zorder=3)
    ax.axhline(0, color="0.5", ls="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Prediction error")
    ax.set_title("Prediction errors")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_outcomes(ax, fit, task_environment=None, show_legend: bool = False,
                   option_names: list | None = None):
    n_trials = fit["values"].shape[0]
    n_options = fit["values"].shape[1]
    if task_environment is None:
        task_environment = fit.get("task_environment")

    if task_environment is not None:
        te = np.asarray(task_environment)
        for i in range(te.shape[1]):
            ax.plot(range(1, n_trials + 1), te[:, i], color="0.8", lw=1)

    choices = fit["data"]["choices"].to_numpy()
    ax.scatter(range(1, n_trials + 1), fit["data"]["outcomes"],
               c=[RLDDM_COLOURS[c - 1] for c in choices],
               marker="o", s=40)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Outcome")
    ax.set_title("Observed outcomes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_ddm_schematic(ax, pars, colours=None, n_traces: int = 10,
                        t_max: float = 5.0, seed: int | None = None):
    """Simple matplotlib version of the R plot_ddm_schematic."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    colours = colours or RLDDM_COLOURS
    a = pars["a"]
    v = pars["v"]
    w = pars["w"]
    t0 = pars["t0"]
    sv = pars.get("sv", 0.0)
    sw = pars.get("sw", 0.0)
    st0 = pars.get("st0", 0.0)

    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05 * a, 1.05 * a)
    ax.set_xlabel("Time")
    ax.set_ylabel("Evidence")
    ax.set_title("Diffusion decision process")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.axhline(a, color="black", ls="--", lw=0.8)
    ax.axvline(t0, color="black", ls=":", lw=0.8)
    ax.text(0, a, "Option 2", ha="right", va="center")
    ax.text(0, 0, "Option 1", ha="right", va="center")

    n_steps = 500
    dt = t_max / n_steps
    time = np.linspace(0, t_max, n_steps + 1)

    for _ in range(n_traces):
        v_i = rng.normal(loc=v, scale=max(sv, 1e-12))
        if sw > 0:
            w_i = rng.uniform(w - sw / 2.0, w + sw / 2.0)
        else:
            w_i = w
        if st0 > 0:
            t0_i = rng.uniform(t0, t0 + st0)
        else:
            t0_i = t0

        evidence = np.full_like(time, np.nan)
        start_idx = np.searchsorted(time, t0_i, side="left")
        evidence[start_idx] = w_i * a
        colour = "0.7"
        hit = False
        for j in range(start_idx + 1, n_steps + 1):
            evidence[j] = evidence[j - 1] + v_i * dt + rng.normal(0.0, np.sqrt(dt))
            if evidence[j] >= a:
                evidence[j] = a
                evidence[j + 1:] = np.nan
                colour = colours[1]
                hit = True
                break
            if evidence[j] <= 0:
                evidence[j] = 0
                evidence[j + 1:] = np.nan
                colour = colours[0]
                hit = True
                break
        ax.plot(time, evidence, color=colour, lw=1)
        if hit:
            last = np.max(np.where(np.isfinite(evidence))[0])
            ax.scatter(time[last], evidence[last], color=colour, s=20, zorder=3)


def _plot_rt_hists(ax, data, colours=None):
    colours = colours or RLDDM_COLOURS
    upper = data["RTs"][data["choices"] == 2].to_numpy()
    lower = data["RTs"][data["choices"] == 1].to_numpy()
    all_rts = data["RTs"].to_numpy()
    breaks = np.linspace(all_rts.min(), all_rts.max(), 26)

    hu, edges_u = np.histogram(upper, bins=breaks)
    hl, edges_l = np.histogram(lower, bins=breaks)
    ymax = max(hu.max(), hl.max()) if (len(hu) and len(hl)) else 1

    ax.set_xlim(breaks[0], breaks[-1])
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Response time")
    ax.set_ylabel("Frequency")
    ax.set_title("Response time distributions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", lw=0.5)

    for i in range(len(hu)):
        ax.bar(edges_u[i], hu[i], width=edges_u[i + 1] - edges_u[i],
               align="edge", color=colours[1], edgecolor="white")
    for i in range(len(hl)):
        ax.bar(edges_l[i], -hl[i], width=edges_l[i + 1] - edges_l[i],
               align="edge", color=colours[0], edgecolor="white")


def rlddm_plot(fit,
               n_traces: int = 10,
               t_max: float = 5.0,
               task_environment=None,
               show_legends: bool = False,
               seed: int | None = None):
    """Create the two-page diagnostic figure set from the R code."""
    pars = fit["parameters"]
    option_names = ["Option 1", "Option 2"]

    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    _plot_values(axes1[0, 0], fit, show_legends, option_names)
    _plot_drifts(axes1[0, 1], fit)
    _plot_pred_errors(axes1[1, 0], fit)
    _plot_outcomes(axes1[1, 1], fit, task_environment, show_legends, option_names)
    fig1.suptitle("RL-DDM\n" + _parameter_label(pars), fontsize=10)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    fig2, axes2 = plt.subplots(3, 2, figsize=(10, 12))
    value_diff = fit["values"][:, 1] - fit["values"][:, 0]
    cuts = np.unique(np.quantile(value_diff, [0.1, 0.4, 0.6, 0.9], method="linear"))
    if len(cuts) < 4:
        cuts = np.linspace(np.nanmin(value_diff), np.nanmax(value_diff), 4)
    groups = pd.cut(value_diff, bins=cuts, include_lowest=True, labels=["low", "medium", "high"])

    for row, level in enumerate(["low", "medium", "high"]):
        idx = np.where(groups == level)[0]
        if len(idx) == 0:
            continue
        values_rep = np.median(fit["values"][idx, :], axis=0)
        drift_rep = compute_drift(values_rep, pars)
        pars_rep = dict(pars)
        pars_rep["v"] = drift_rep

        title = f"value difference = {values_rep[1] - values_rep[0]:.2f}  (drift = {drift_rep:.2f})"
        _plot_ddm_schematic(axes2[row, 0], pars_rep, n_traces=n_traces,
                            t_max=t_max, seed=seed)
        axes2[row, 0].set_title(title)
        _plot_rt_hists(axes2[row, 1], fit["data"].iloc[idx])

    fig2.suptitle("RL-DDM\n" + _parameter_label(pars), fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    return fig1, fig2


# =============================================================================
# Simple CLI / smoke-test
# =============================================================================

if __name__ == "__main__":
    from create_task_environment import generate_timeline, timeline_to_matrix

    rng = np.random.default_rng(42)
    timeline = generate_timeline(num_trials=140, seed=42, reversed_state=True)
    env = timeline_to_matrix(timeline)

    pars = {"alpha": 0.25, "v_intercept": 0.0, "v_scale": 1.0,
            "a": 3.0, "w": 0.5, "t0": 0.25}
    fit = rlddm_simulate(env, pars, rng=rng)
    print("Simulated", len(fit["data"]), "trials from the PRLT environment")
    print("Mean log-lik of simulated data under true pars:",
          rlddm_log_lik(fit["data"], pars) / len(fit["data"]))
    print("First 5 trials:")
    print(fit["data"].head())
    print("\nValue of bandit_1 at trial 10:", fit["values"][9, 0])
    print("Drift at trial 10:", fit["drifts"][9])
